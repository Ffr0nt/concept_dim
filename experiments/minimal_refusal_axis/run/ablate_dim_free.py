"""Шаги 1-2 лестницы: снимает ли отказ направление с НУЛЕВОЙ долей DIM.

Два способа построить такое направление из уже обученных векторов, без обучения:

  дуга        w_t000 = normalize(u - (u.g)g) для нескольких опорных осей u;
  собственная задача с ограничением —
              w = top eigvec(P_perp P P_perp), P_perp = I - g g^T,
              то есть лучшее DIM-свободное направление во всей линейной оболочке
              набора, а не в одномерном семействе дуги.

Главная защита от Гудхарта: проекция вычитается против DIM НАШЕГО прогона, а доля DIM
проверяется ещё и против DIM базового прогона, который в построении не участвовал.

Протокол аблации тот же, что в этапе 03: частичная аблация во всех слоях, прокси первого
токена на harmful_test, KL на harmless_test, сетка alphas из конфига.

Запускать на сервере из корня shared-refusal-component: см. run/ablate_dim_free.sh
"""
import argparse
import functools
import os

import torch

from shared_refusal import ablation, models
from shared_refusal.artifacts import REPO, DirectionSet, Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config
from shared_refusal.geometry import random_directions, shared_component, unit

CONFIG = "configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml"
BASE_RUN = os.path.join(REPO, "results", "qwen2.5-3b_salad-o1o2o5o6")
OUT = ("component", "dim_free_ablation.json")
SET_OUT = ("component", "dim_free_directions.pt")
# Что отдать судье, если прокси покажет, что направление работает.
FOR_JUDGE = ("w_perp_pool", "w_constr_pool", "w_constr_bt", "w_ctrl_pool")


def vectors(run_dir, kind):
    names = read_json(os.path.join(run_dir, "manifest.json"))["directions"]["names"]
    out = []
    for n in names:
        if kind == "dim" and n.startswith("dim_"):
            out.append(unit(torch.load(os.path.join(run_dir, "dim", n[4:], "direction.pt"),
                                       map_location="cpu", weights_only=True).double()))
        elif kind == "rdo" and n.startswith("rdo_"):
            _, dom, seed = n.split("_")
            o = torch.load(os.path.join(run_dir, "rdo", dom, f"seed_{seed[1:]}.pt"),
                           map_location="cpu", weights_only=False)
            out.append(unit(o["vector"].double()))
    return torch.stack(out)


def perp(v, g):
    """Направление без доли DIM: проекция на g вычтена полностью."""
    return unit(v - g * (v @ g))


def constrained_top(V, g):
    """argmax_{w ⊥ g} w^T P w при P = (1/N) sum v v^T — лучшее DIM-свободное в оболочке V."""
    Vp = torch.stack([perp(v, g) for v in V])      # P_perp V^T, строки уже ортогональны g
    w, lam, _ = shared_component(Vp, Vp.mean(0))
    return unit(w), float(lam[0])


def rotate_like(u, g, target_cos, seed):
    """Контроль: поворот оси на тот же угол в случайную сторону, доля DIM сохранена.

    Ищем w = normalize(a*u + b*r + c*g) с |w.u| = target_cos и w.g = u.g.
    Случайное r берётся ортогональным и u, и g."""
    r = random_directions(len(u), 1, seed=seed)[0].double()
    r = unit(r - u * (r @ u) - g * (r @ g))
    rho = float(u @ g)
    # в базисе (u_perp_g, g, r): сохраняем компоненту вдоль g, поворачиваем остальное
    u_p = unit(u - g * rho)                        # часть оси вне g, норма sqrt(1-rho^2)
    s = (1 - rho ** 2) ** 0.5
    # w = rho*g + s*(cos(phi)*u_p + sin(phi)*r); тогда w.u = rho^2 + s^2*cos(phi)
    cos_phi = (target_cos - rho ** 2) / s ** 2
    assert -1 <= cos_phi <= 1, f"недостижимый угол: cos_phi={cos_phi:.3f}"
    sin_phi = (1 - cos_phi ** 2) ** 0.5
    return unit(rho * g + s * (cos_phi * u_p + sin_phi * r))


def candidates(run, log):
    rdo_bt, rdo_sd = vectors(run.root, "rdo"), vectors(BASE_RUN, "rdo")
    g_bt = unit(torch.load(os.path.join(run.root, "dim/union/direction.pt"),
                           map_location="cpu", weights_only=True).double())
    g_sd = unit(torch.load(os.path.join(BASE_RUN, "dim/union/direction.pt"),
                           map_location="cpu", weights_only=True).double())
    w_bt = unit(load_w(run.component("rdo_only")))
    w_sd = unit(load_w(os.path.join(BASE_RUN, "component", "all")))
    pool_V = torch.cat([rdo_bt, rdo_sd])
    u_pool = unit(shared_component(pool_V, pool_V.mean(0))[0])

    log(f"опорные оси: pool (40 RDO), наша компонента без DIM, базовая компонента")
    log(f"DIM нашего прогона — для вычитания; DIM базового — отложенная проверка "
        f"(|cos| между ними {abs(float(g_bt @ g_sd)):.3f})\n")

    w_constr_pool, lam_p = constrained_top(pool_V, g_bt)
    w_constr_bt, lam_b = constrained_top(rdo_bt, g_bt)
    log(f"собственная задача с ограничением: lam_1 на пуле 40 {lam_p:.3f}, "
        f"на наших 20 {lam_b:.3f}")

    # контроль: тот же угол от оси, что у дуги t=0, но доля DIM сохранена
    target = abs(float(u_pool @ perp(u_pool, g_bt)))
    w_ctrl = rotate_like(u_pool, g_bt, target, seed=7)
    log(f"контроль: повёрнут от оси pool на тот же |cos| = {target:.3f}, "
        f"доля DIM сохранена ({abs(float(w_ctrl @ g_bt)):.3f} против {abs(float(u_pool @ g_bt)):.3f})")

    names = ["w_perp_pool", "w_perp_bt", "w_perp_sd", "w_constr_pool", "w_constr_bt",
             "w_ctrl_pool", "u_pool", "w_bt_rdo", "rand_x"]
    vecs = [perp(u_pool, g_bt), perp(w_bt, g_bt), perp(w_sd, g_bt),
            w_constr_pool, w_constr_bt, w_ctrl, u_pool, w_bt,
            unit(random_directions(len(w_bt), 1, seed=7)[0].double())]
    geom = {n: {"rho_bt": abs(float(v @ g_bt)), "rho_sd": abs(float(v @ g_sd)),
                "cos_pool": abs(float(v @ u_pool)), "cos_w_bt": abs(float(v @ w_bt))}
            for n, v in zip(names, vecs)}
    ds = DirectionSet.from_raw(names, vecs,
                               [{"method": "dim_free", "domain": None} for _ in names])
    return ds, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-only", action="store_true", help="только геометрия, без GPU")
    args = ap.parse_args()
    cfg = load_config(CONFIG)
    run = Run.from_config(cfg)
    ac, domains = cfg["ablation"], list(cfg["domains"])
    fam = models.family(cfg["model"])
    log = functools.partial(print, flush=True)

    ds, geom = candidates(run, log)
    log("\n## Геометрия кандидатов\n")
    log("| направление | rho (наш DIM) | rho (базовый DIM, отложен) | \\|cos\\| с осью pool |")
    log("|---|---|---|---|")
    for n in ds.names:
        g = geom[n]
        log(f"| {n} | {g['rho_bt']:.4f} | {g['rho_sd']:.4f} | {g['cos_pool']:.4f} |")

    DirectionSet.from_raw([n for n in ds.names if n in FOR_JUDGE],
                          [ds[n] for n in ds.names if n in FOR_JUDGE],
                          [{"method": "component", "domain": None} for _ in FOR_JUDGE]
                          ).save(run.path(*SET_OUT))
    log(f"\nнабор для судьи сохранён -> {run.path(*SET_OUT)}")
    if args.save_only:
        return

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    model = models.load(cfg["model"], cfg["model_revision"])
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds,
                            ac["alphas"], log=log)
    write_json({**res, "geometry": geom, "base_run": BASE_RUN,
                "n_test": {d: len(p) for d, p in harmful.items()},
                "n_harmless": len(harmless)}, run.path(*OUT))

    base = sum(res["base_asr"].values()) / len(res["base_asr"])
    log(f"\n## Аблация DIM-свободных направлений\n\nБез аблации ASR = {base:.3f}. "
        "В клетке ASR (среднее по доменам) и KL.\n")
    log("| направление | rho | " + " | ".join(f"a={a}" for a in ac["alphas"]) + " |")
    log("|---" * (2 + len(ac["alphas"])) + "|")
    for n in ds.names:
        rows = [r for r in res["rows"] if r["direction"] == n]
        cells = " | ".join(f"{r['asr_mean']:.3f} / {r['kl']:.3f}" for r in rows)
        log(f"| {n} | {geom[n]['rho_bt']:.3f} | {cells} |")
    log(f"\nJSON: {run.path(*OUT)}")


if __name__ == "__main__":
    main()
