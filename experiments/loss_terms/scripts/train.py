"""Шаги 0–1 — обучение одномерных RDO с изменённым лоссом (GPU).

Данные, DIM, точка добавления, кэш целей и гиперпараметры — из базового запуска
shared-refusal-component; от base отличаются только веса лосса (common.VARIANTS). Векторы
пишутся в <out>/<вариант>/<домен>/seed_XX.pt. Готовые сиды пропускаются, так что прерванный
прогон можно просто перезапустить.

Калибровка μ для free_norm (шаг 0): --calibrate μ1 μ2 … — один домен и один сид на каждое μ,
в <out>/calibrate/mu_<μ>/<домен>/; правило выбора μ — в idea.md.
"""
import argparse
import datetime
import os

from common import N_SEEDS, VARIANTS, base, seeds
from shared_refusal import models, provenance
from shared_refusal.artifacts import FORK, REPO, read_json, write_json
from shared_refusal.directions import rdo

LOSS_KEYS = ("ablation_lambda", "addition_lambda", "retain_lambda")


def command(cfg, run, fam, domain, out_dir, todo, spec):
    hp = dict(cfg["rdo"]["hparams"])
    hp.update({k: v for k, v in spec.items() if k in LOSS_KEYS})
    cmd = rdo.command(cfg["model"], fam, domain, run.dim(domain), run.splits(domain),
                      os.path.join(run.rdo(domain), "targets"), out_dir, todo, hp)
    if spec.get("free_norm"):
        assert spec.get("norm_lambda") is not None, \
            "free_norm: norm_lambda не задан — сначала калибровка (run/0_calibrate_norm.sh)"
        cmd += ["--free_norm", "--norm_lambda", str(spec["norm_lambda"])]
    if spec.get("addition_mode"):
        cmd += ["--addition_mode", spec["addition_mode"]]
    return cmd, hp


def check_targets(run, domain):
    """Цели обязаны браться из кэша базового запуска: иначе rdo.py сгенерирует их заново и
    запишет в каталог базового запуска."""
    t = os.path.join(run.rdo(domain), "targets")
    for f in ("harmful_targets.json", "harmless_targets.json"):
        assert os.path.exists(os.path.join(t, f)), f"нет кэша целей {t}/{f}"


def record(out_dir, entry):
    """run.json рядом с seed_XX.pt — свой файл у каждой пары (вариант, домен), поэтому
    параллельные процессы на разных доменах друг другу не мешают."""
    p = os.path.join(out_dir, "run.json")
    runs = read_json(p) if os.path.exists(p) else []
    write_json(runs + [entry], p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="корень результатов эксперимента")
    ap.add_argument("--variants", nargs="+", default=None, help="по умолчанию — все обучаемые")
    ap.add_argument("--domains", nargs="+", default=None, help="по умолчанию — все домены базового запуска")
    ap.add_argument("--calibrate", type=float, nargs="+", default=None,
                    help="калибровка free_norm: список μ; домен — первый из --domains, сид — первый сид домена")
    args = ap.parse_args()

    cfg, run = base()
    fam = models.family(cfg["model"])
    domains = args.domains or list(cfg["domains"])
    for d in domains:
        check_targets(run, d)
    provenance.check_revision(cfg["model"], cfg["model_revision"])
    prov = {"repo": provenance.git_state(REPO), "fork": provenance.git_state(FORK),
            "base_run": run.root, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    if prov["fork"]["dirty"]:
        print("[предупреждение: рабочее дерево форка не закоммичено — результат не воспроизводится по коммиту]")

    if args.calibrate:
        d = domains[0]
        s = seeds(cfg, d)[:1]
        for mu in args.calibrate:
            key = f"calibrate/mu_{mu:g}"
            out_dir = os.path.join(args.out, key, d)
            todo = rdo.pending(out_dir, s)
            if not todo:
                print(f"{key} {d}: готово, пропуск")
                continue
            cmd, hp = command(cfg, run, fam, d, out_dir, todo, {**VARIANTS["free_norm"], "norm_lambda": mu})
            print(f"=== калибровка μ={mu:g}: {d} сид {s[0]} ===", flush=True)
            rdo.train(cmd, save_dir=args.out)
            record(out_dir, {"seeds": todo, "hparams": hp, "norm_lambda": mu, **prov})
        return

    variants = args.variants or [v for v, spec in VARIANTS.items() if spec is not None]
    for v in variants:
        spec = VARIANTS[v]
        assert spec is not None, f"{v} не обучается (векторы берутся из базового запуска)"
        if spec.get("free_norm") and spec.get("norm_lambda") is None:
            p = os.path.join(args.out, "calibration.json")
            assert os.path.exists(p), f"free_norm: нет {p} — сначала калибровка (run/0_calibrate_norm.sh)"
            mu = read_json(p)["norm_lambda"]
            assert mu is not None, f"free_norm: по {p} ни одно μ не прошло правило"
            print(f"free_norm: μ = {mu:g} по правилу калибровки ({p})")
            spec = {**spec, "norm_lambda": mu, "norm_lambda_source": p}
        for d in domains:
            out_dir = os.path.join(args.out, v, d)
            todo = rdo.pending(out_dir, seeds(cfg, d))
            if not todo:
                print(f"{v} {d}: все сиды готовы, пропуск")
                continue
            cmd, hp = command(cfg, run, fam, d, out_dir, todo, spec)
            print(f"=== {v} {d}: сиды {' '.join(map(str, todo))} ===", flush=True)
            rdo.train(cmd, save_dir=args.out)
            record(out_dir, {"seeds": todo, "n_seeds": N_SEEDS, "hparams": hp, "spec": spec, **prov})


if __name__ == "__main__":
    main()
