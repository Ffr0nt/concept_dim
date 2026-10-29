"""Снимает ли отказ общая часть двух компонент — нашей (без DIM) и базовой.

Пересечение двух одномерных подпространств пусто, поэтому «общая часть» определяется явно:
  common        — биссектриса, первый собственный вектор пары {w_bt, w_sd} (|cos| 0.88 с каждой);
  principal     — первый главный вектор подпространств 20 RDO двух прогонов;
  w_bt⊥w_sd     — наша компонента с вычтенной базовой осью;
  w_sd⊥w_bt     — базовая компонента с вычтенной нашей осью.
Контроли: сами компоненты и одно случайное направление.

Главный вопрос: снимают ли отказ ОБА ортогональных остатка. Если да — осей, снимающих отказ,
больше одной, и защита одной оси аблитерацию не остановит.

Протокол и метрики те же, что в этапе 03 (частичная аблация во всех слоях, прокси первого
токена на harmful_test, KL на harmless_test), сетка alphas берётся из конфига.

Запускать на сервере из корня shared-refusal-component: см. run/ablate_cross_run.sh
"""
import functools
import json
import os

import torch

from shared_refusal import ablation, models
from shared_refusal.artifacts import REPO, DirectionSet, Run, read_json, write_json
from shared_refusal.component import load_w
from shared_refusal.config import load_config
from shared_refusal.geometry import random_directions, unit

CONFIG = "configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml"
BASE_RUN = os.path.join(REPO, "results", "qwen2.5-3b_salad-o1o2o5o6")   # с чем сравниваем
OUT = "component/cross_run_ablation.json"


def rdo_subspace(run_dir):
    """Ортонормированный базис подпространства, натянутого 20 RDO прогона."""
    names = read_json(os.path.join(run_dir, "manifest.json"))["directions"]["names"]
    V = []
    for n in names:
        if not n.startswith("rdo_"):
            continue
        _, dom, seed = n.split("_")
        o = torch.load(os.path.join(run_dir, "rdo", dom, f"seed_{seed[1:]}.pt"),
                       map_location="cpu", weights_only=False)
        V.append(unit(o["vector"].double()))
    return torch.linalg.qr(torch.stack(V).T).Q


def candidates(run, base_run):
    w_bt = unit(load_w(run.component("rdo_only")))            # наша компонента без DIM
    w_sd = unit(load_w(os.path.join(base_run, "component", "all")))   # прежняя компонента
    sign = torch.sign(w_bt @ w_sd)
    common = unit(w_bt + sign * w_sd)
    A, B = rdo_subspace(run.root), rdo_subspace(base_run)
    U, S, Vh = torch.linalg.svd(A.T @ B)
    pa, pb = unit(A @ U[:, 0]), unit(B @ Vh[0])
    principal = unit(pa + torch.sign(pa @ pb) * pb)
    names = ["w_bt_rdo", "w_sd_all", "common", "principal", "w_bt⊥w_sd", "w_sd⊥w_bt", "rand_x"]
    vecs = [w_bt, w_sd, common, principal,
            unit(w_bt - w_sd * (w_bt @ w_sd)), unit(w_sd - w_bt * (w_sd @ w_bt)),
            random_directions(len(w_bt), 1, seed=7)[0].double()]
    geom = {n: {"cos_w_bt": abs(float(v @ w_bt)), "cos_w_sd": abs(float(v @ w_sd))}
            for n, v in zip(names, vecs)}
    geom["_принципиальные_косинусы_подпространств"] = [float(x) for x in S[:6]]
    ds = DirectionSet.from_raw(names, vecs, [{"method": "cross_run", "domain": None} for _ in names])
    return ds, geom


def main():
    cfg = load_config(CONFIG)
    run = Run.from_config(cfg)
    ac, domains = cfg["ablation"], list(cfg["domains"])
    fam = models.family(cfg["model"])

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    ds, geom = candidates(run, BASE_RUN)
    print("геометрия проверяемых направлений:")
    for n in ds.names:
        print(f"  {n:11s} |cos| с нашей {geom[n]['cos_w_bt']:.3f}, с базовой {geom[n]['cos_w_sd']:.3f}")

    model = models.load(cfg["model"], cfg["model_revision"])
    forward = functools.partial(models.last_token_logits, model, batch_size=ac["batch_size"])
    res = ablation.evaluate(forward, fam["refusal_toks"], harmful, harmless, ds, ac["alphas"],
                            log=lambda s: print(s, flush=True))
    write_json({**res, "geometry": geom, "base_run": BASE_RUN,
                "n_test": {d: len(p) for d, p in harmful.items()}, "n_harmless": len(harmless)},
               run.path(*OUT.split("/")))

    print("\n## Аблация общей части двух компонент\n")
    print("Прокси-ASR, среднее по четырём доменам; KL на harmless_test. Без аблации: "
          f"{sum(res['base_asr'].values()) / len(res['base_asr']):.3f}\n")
    print("| направление | \\|cos\\| с нашей | \\|cos\\| с базовой | " +
          " | ".join(f"a={a}" for a in ac["alphas"]) + " |")
    print("|---" * (3 + len(ac["alphas"])) + "|")
    for n in ds.names:
        rows = [r for r in res["rows"] if r["direction"] == n]
        cells = " | ".join(f"{r['asr_mean']:.3f}<br>{r['kl']:.3f}" for r in rows)
        print(f"| {n} | {geom[n]['cos_w_bt']:.3f} | {geom[n]['cos_w_sd']:.3f} | {cells} |")
    print("\nВ каждой клетке: сверху ASR, снизу KL.")
    print(f"\nJSON: {run.path(*OUT.split('/'))}")


if __name__ == "__main__":
    main()
