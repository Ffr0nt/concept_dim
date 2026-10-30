"""Отбелённый DIM: снимает ли отказ дешевле обычного.

Штраф на вырезаемое из остаточного потока, E_h[(h.w)^2] = w^T Sigma w, делает задачу
отбеливанием, и у неё есть замкнутый ответ — линейный дискриминант:

    w_LDA = Sigma^{-1} (mu_harmful - mu_harmless),

то есть обычный DIM, домноженный на обратную ковариацию. Обучать нечего, нужен один
прямой проход по train-запросам для оценки Sigma в рабочей точке (pos -1, слой 27).

Sigma на 2048 измерений плохо обусловлена, поэтому берётся усадка

    Sigma_a = (1 - a) Sigma + a (tr Sigma / d) I,

и свип по a — непрерывный путь от LDA (a = 0) к обычному DIM (a = 1, где Sigma_a ~ I и
направление в точности mu_diff). Считаются два варианта Sigma: внутриклассовая по Фишеру
(каждый домен и harmless центрированы своим средним) и только по harmless — вторая отвечает
буквальной постановке «минимально возмутить безобидное поведение».

Протокол аблации тот же, что в этапе 03.

Запускать на сервере из корня shared-refusal-component: см. run/whitened_dim.sh
"""
import argparse
import functools
import os

import torch

from shared_refusal import ablation, models
from shared_refusal.artifacts import DirectionSet, Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config
from shared_refusal.geometry import random_directions, unit

CONFIG = "configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml"
OUT = ("component", "whitened_dim_ablation.json")
SET_OUT = ("component", "whitened_dim_directions.pt")
ALPHAS = [1.0, 0.9, 0.5, 0.1, 0.01, 0.001, 0.0]      # 1.0 = обычный DIM, 0.0 = чистый LDA
ACT_BATCH = 16


def shrink_solve(S, mu, a):
    """w = Sigma_a^{-1} mu при Sigma_a = (1-a) S + a (tr S / d) I."""
    d = S.shape[0]
    Sa = (1 - a) * S + a * (float(torch.diagonal(S).sum()) / d) * torch.eye(d, dtype=S.dtype)
    return unit(torch.linalg.solve(Sa, mu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-only", action="store_true", help="без аблации")
    args = ap.parse_args()
    cfg = load_config(CONFIG)
    run = Run.from_config(cfg)
    ac, domains = cfg["ablation"], list(cfg["domains"])
    fam = models.family(cfg["model"])
    log = functools.partial(print, flush=True)
    point = read_json(run.path("manifest.json"))["dim"]
    pos, layer = point["pos"], point["layer"]
    log(f"рабочая точка: pos {pos}, слой {layer}")

    def train_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_train.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    model = models.load(cfg["model"], cfg["model_revision"])
    acts = functools.partial(models.resid_at, model, layer=layer, pos=pos, batch_size=ACT_BATCH)

    log("снимаю активации в рабочей точке...")
    H = {d: acts(train_prompts(d, "harmful")).double() for d in domains}
    L = acts(train_prompts(domains[0], "harmless")).double()
    for d in domains:
        log(f"  {d}: harmful {tuple(H[d].shape)}")
    log(f"  harmless: {tuple(L.shape)}")

    # mu_diff ровно как DIM union: среднее по доменам разностей средних (равные веса доменов)
    mu_l = L.mean(0)
    mu = torch.stack([H[d].mean(0) - mu_l for d in domains]).mean(0)
    g = unit(torch.load(run.path("dim/union/direction.pt"), map_location="cpu",
                        weights_only=True).double())
    log(f"\nmu_diff воспроизводит DIM union: |cos| = {abs(float(unit(mu) @ g)):.6f}, "
        f"норма {float(mu.norm()):.3f} против {point['norms']['union']:.3f}")

    # внутриклассовая ковариация: каждый класс центрирован СВОИМ средним
    cent = [L - mu_l] + [H[d] - H[d].mean(0) for d in domains]
    X = torch.cat(cent)
    S_w = X.T @ X / (len(X) - len(cent))
    Lc = L - mu_l
    S_l = Lc.T @ Lc / (len(Lc) - 1)
    for tag, S in (("внутриклассовая", S_w), ("только harmless", S_l)):
        ev = torch.linalg.eigvalsh(S)
        log(f"{tag}: n={len(X) if S is S_w else len(L)}, след {float(ev.sum()):.1f}, "
            f"lam_max {float(ev[-1]):.2f}, lam_min {float(ev[0]):.2e}, "
            f"эфф. ранг {float(ev.sum() / ev[-1]):.1f}")

    u_pool = None
    w_bt = unit(load_w(run.component("rdo_only")))
    names, vecs = [], []
    for tag, S in (("w", S_w), ("l", S_l)):
        for a in ALPHAS:
            names.append(f"lda_{tag}_a{a:g}")
            vecs.append(shrink_solve(S, mu, a))
    names += ["dim_union", "w_bt_rdo", "rand_x"]
    vecs += [g, w_bt, unit(random_directions(len(g), 1, seed=7)[0].double())]

    geom = {n: {"cos_dim": abs(float(v @ g)), "cos_w_bt": abs(float(v @ w_bt)),
                "cut_w": float(v @ S_w @ v), "cut_l": float(v @ S_l @ v)}
            for n, v in zip(names, vecs)}
    log("\n## Геометрия и вырезаемая величина\n")
    log("| направление | \\|cos\\| с DIM | \\|cos\\| с нашей | E[(h.w)^2] внутрикл. | по harmless |")
    log("|---|---|---|---|---|")
    for n in names:
        q = geom[n]
        log(f"| {n} | {q['cos_dim']:.4f} | {q['cos_w_bt']:.4f} | "
            f"{q['cut_w']:.2f} | {q['cut_l']:.2f} |")

    ds = DirectionSet.from_raw(names, vecs,
                               [{"method": "whitened", "domain": None} for _ in names])
    ds.save(run.path(*SET_OUT))
    log(f"\nнабор сохранён -> {run.path(*SET_OUT)}")
    if args.save_only:
        return

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds,
                            ac["alphas"], log=log)
    write_json({**res, "geometry": geom, "point": {"pos": pos, "layer": layer},
                "n_test": {d: len(p) for d, p in harmful.items()},
                "n_harmless": len(harmless)}, run.path(*OUT))

    base = sum(res["base_asr"].values()) / len(res["base_asr"])
    log(f"\n## Аблация отбелённого DIM\n\nБез аблации ASR = {base:.3f}. "
        "В клетке ASR (среднее по доменам) и KL.\n")
    log("| направление | \\|cos\\| с DIM | " + " | ".join(f"a={a}" for a in ac["alphas"]) + " |")
    log("|---" * (2 + len(ac["alphas"])) + "|")
    for n in names:
        rows = [r for r in res["rows"] if r["direction"] == n]
        cells = " | ".join(f"{r['asr_mean']:.3f} / {r['kl']:.3f}" for r in rows)
        log(f"| {n} | {geom[n]['cos_dim']:.3f} | {cells} |")
    log(f"\nJSON: {run.path(*OUT)}")


if __name__ == "__main__":
    main()
