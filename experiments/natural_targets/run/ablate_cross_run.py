"""Снимает ли отказ общая компонента, посчитанная по векторам ОБОИХ прогонов.

«Пересечение» здесь — та же операция, что в этапе 02: векторы двух прогонов складываются в
один набор, берётся первый собственный вектор P = (1/N) sum v v^T. Проверяются:

  w_pool_rdo   — общая компонента 20 наших RDO + 20 базовых RDO (ни одного DIM);
  w_pool_all   — то же по всем 24 + 24 направлениям;
  w_bisect     — общая компонента пары из двух готовых компонент (N = 2);
  w_bt⊥pool    — наша компонента с вычтенной общей осью двух прогонов;
  w_sd⊥pool    — базовая компонента с вычтенной той же осью;
  контроли     — сами компоненты обоих прогонов и одно случайное направление.

Главный вопрос: снимают ли отказ ОБА ортогональных остатка. Если да — осей, снимающих отказ,
больше одной, и защита одной оси аблитерацию не остановит.

Протокол тот же, что в этапе 03: частичная аблация во всех слоях, прокси первого токена на
harmful_test, KL на harmless_test, сетка alphas из конфига.

Запускать на сервере из корня shared-refusal-component: см. run/ablate_cross_run.sh
"""
import argparse
import functools
import os

import torch

from shared_refusal import ablation, models
from shared_refusal.artifacts import REPO, DirectionSet, Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config
from shared_refusal.geometry import null_lam1, random_directions, shared_component, unit

CONFIG = "configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml"
BASE_RUN = os.path.join(REPO, "results", "qwen2.5-3b_salad-o1o2o5o6")   # с чем сравниваем
OUT = ("component", "cross_run_ablation.json")
SET_OUT = ("component", "cross_run_directions.pt")
# Что отдаём судье через generate.extra: общая ось двух прогонов, два остатка компонент после
# её вычитания и сама базовая компонента (её ответы на StrongREJECT судья ещё не видел).
FOR_JUDGE = {"w_pool_rdo": "component", "w_bt⊥pool": "residual",
             "w_sd⊥pool": "residual", "w_sd_all": "component"}


def vectors(run_dir, kind):
    """Единичные направления прогона: kind = 'rdo' или 'dim'."""
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


def pool(V, n_null, null_seed, tag, log):
    """Общая компонента набора V — ровно как этап 02, с нулевым уровнем для своего N."""
    w, lam, cos = shared_component(V, V.mean(0))
    null = null_lam1(len(V), V.shape[1], n_null, null_seed)
    log(f"  {tag:12s} N={len(V):3d}  lam_1={float(lam[0]):.3f}  "
        f"нулевой p95={float(null.quantile(0.95)):.3f}  lam_2={float(lam[1]):.3f}")
    return unit(w), {"N": len(V), "lam1": float(lam[0]), "lam2": float(lam[1]),
                     "null_p95": float(null.quantile(0.95)),
                     "cos_median": float(cos.abs().median())}


def candidates(run, cc, log):
    rdo_bt, rdo_sd = vectors(run.root, "rdo"), vectors(BASE_RUN, "rdo")
    dim_bt, dim_sd = vectors(run.root, "dim"), vectors(BASE_RUN, "dim")
    w_bt = unit(load_w(run.component("rdo_only")))
    w_sd = unit(load_w(os.path.join(BASE_RUN, "component", "all")))
    n_null, seed = cc["n_null"], cc["null_seed"]

    log("общие компоненты объединённых наборов (как этап 02):")
    w_pool_rdo, s_rdo = pool(torch.cat([rdo_bt, rdo_sd]), n_null, seed, "pool_rdo", log)
    w_pool_all, s_all = pool(torch.cat([rdo_bt, dim_bt, rdo_sd, dim_sd]), n_null, seed, "pool_all", log)
    w_bisect, s_bis = pool(torch.stack([w_bt, w_sd]), n_null, seed, "bisect", log)
    # для сравнения — свои компоненты каждого прогона по тем же 20 RDO
    _, s_bt = pool(rdo_bt, n_null, seed, "только наши", log)
    _, s_sd = pool(rdo_sd, n_null, seed, "только базовые", log)

    log("\nсколько векторов каждого прогона согласны с общей осью pool_rdo (|cos|, медиана):")
    for tag, V in [("наши 20 RDO", rdo_bt), ("базовые 20 RDO", rdo_sd)]:
        c = (V @ w_pool_rdo).abs()
        log(f"  {tag:16s} медиана {float(c.median()):.3f}  от {float(c.min()):.3f} до {float(c.max()):.3f}")

    names = ["w_pool_rdo", "w_pool_all", "w_bisect", "w_bt⊥pool", "w_sd⊥pool",
             "w_bt_rdo", "w_sd_all", "rand_x"]
    vecs = [w_pool_rdo, w_pool_all, w_bisect,
            unit(w_bt - w_pool_rdo * (w_bt @ w_pool_rdo)),
            unit(w_sd - w_pool_rdo * (w_sd @ w_pool_rdo)),
            w_bt, w_sd, random_directions(len(w_bt), 1, seed=7)[0].double()]
    geom = {n: {"cos_w_bt": abs(float(v @ w_bt)), "cos_w_sd": abs(float(v @ w_sd)),
                "cos_pool": abs(float(v @ w_pool_rdo))} for n, v in zip(names, vecs)}
    geom["_spectra"] = {"pool_rdo": s_rdo, "pool_all": s_all, "bisect": s_bis,
                        "bt_rdo_only": s_bt, "sd_rdo_only": s_sd}
    ds = DirectionSet.from_raw(names, vecs, [{"method": "cross_run", "domain": None} for _ in names])
    return ds, geom


def save_for_judge(ds, run, log):
    """Направления для этапа 06 через generate.extra: судья видит только то, что сгенерировано."""
    names = [n for n in ds.names if n in FOR_JUDGE]
    assert len(names) == len(FOR_JUDGE), f"не нашлись: {set(FOR_JUDGE) - set(names)}"
    DirectionSet.from_raw(names, [ds[n] for n in names],
                          [{"method": FOR_JUDGE[n], "domain": None} for n in names]
                          ).save(run.path(*SET_OUT))
    log(f"\nдля судьи сохранено {len(names)} направлений -> {run.path(*SET_OUT)}")
    log("добавь в конфиг: generate.extra: [" + "/".join(SET_OUT) + "]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-only", action="store_true",
                    help="только посчитать направления и сохранить набор для судьи, без GPU")
    args = ap.parse_args()
    cfg = load_config(CONFIG)
    run = Run.from_config(cfg)
    ac, domains = cfg["ablation"], list(cfg["domains"])
    fam = models.family(cfg["model"])
    log = functools.partial(print, flush=True)

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    ds, geom = candidates(run, cfg["component"], log)
    log("\nгеометрия проверяемых направлений:")
    for n in ds.names:
        g = geom[n]
        log(f"  {n:11s} |cos| с нашей {g['cos_w_bt']:.3f}, с базовой {g['cos_w_sd']:.3f}, "
            f"с pool_rdo {g['cos_pool']:.3f}")

    save_for_judge(ds, run, log)
    if args.save_only:
        return

    model = models.load(cfg["model"], cfg["model_revision"])
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds, ac["alphas"], log=log)
    write_json({**res, "geometry": geom, "base_run": BASE_RUN,
                "n_test": {d: len(p) for d, p in harmful.items()}, "n_harmless": len(harmless)},
               run.path(*OUT))

    base = sum(res["base_asr"].values()) / len(res["base_asr"])
    log(f"\n## Аблация общей компоненты двух прогонов\n\nБез аблации ASR = {base:.3f}. "
        "В клетке ASR (среднее по доменам) и KL.\n")
    log("| направление | \\|cos\\| наша | \\|cos\\| базовая | " +
        " | ".join(f"a={a}" for a in ac["alphas"]) + " |")
    log("|---" * (3 + len(ac["alphas"])) + "|")
    for n in ds.names:
        rows = [r for r in res["rows"] if r["direction"] == n]
        cells = " | ".join(f"{r['asr_mean']:.3f} / {r['kl']:.3f}" for r in rows)
        log(f"| {n} | {geom[n]['cos_w_bt']:.3f} | {geom[n]['cos_w_sd']:.3f} | {cells} |")
    log(f"\nJSON: {run.path(*OUT)}")


if __name__ == "__main__":
    main()
