"""Шаг 0б — сводка калибровки μ для free_norm (CPU, модель не грузится).

Читает <out>/calibrate/mu_<μ>/<домен>/seed_XX.pt и печатает по каждому μ: траекторию ‖z‖,
bypass и induce score на train у сохранённого вектора, близость к парному вектору base.

Правило выбора (зафиксировано в idea.md до запуска): наибольшее μ, при котором у
сохранённого вектора induce score > 0 и bypass score < 0. Score — refusal score на train-батчах
шагов обучения (шумный), поэтому берём среднее по 3 шагам, заканчивающимся сохранённым.
"""
import argparse
import glob
import os
import re

import torch

from common import base, variant_dir
from shared_refusal.artifacts import write_json
from shared_refusal.directions import rdo
from shared_refusal.report import table

WINDOW = 3


def at_saved(series, idx):
    xs = [x[0] for x in series[max(0, idx - WINDOW + 1): idx + 1]]
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    cfg, run = base()

    rows, ok = [], []
    for path in sorted(glob.glob(os.path.join(args.out, "calibrate", "mu_*", "*", "seed_*.pt")),
                       key=lambda p: float(re.search(r"mu_([^/]+)", p).group(1))):
        mu = float(re.search(r"mu_([^/]+)", path).group(1))
        d = os.path.basename(os.path.dirname(path))
        s = int(re.search(r"seed_(\d+)", path).group(1))
        r = torch.load(path, map_location="cpu", weights_only=True)
        b = rdo.load_seed(variant_dir(None, "base", d), s)
        assert torch.equal(r["init_vector"], b["init_vector"]), f"{path}: старт не совпадает с base — пары нет"
        i = r["lowest_loss_index"]
        norms = [x[0] for x in r["vector_norms"]]
        byp, ind = at_saved(r["bypass_scores"], i), at_saved(r["induce_scores"], i)
        z = r["vector"].double()
        cos_base = float(z @ b["vector"].double() / z.norm() / b["vector"].double().norm())
        cos_init = float(z @ r["init_vector"].double() / z.norm())
        rows.append([f"{mu:g}", d, s, len(norms), f"{norms[i]:.3f}", f"{min(norms):.3f}–{max(norms):.3f}",
                     f"{byp:+.2f}", f"{ind:+.2f}", f"{r['lowest_loss']:.3f}", f"{cos_base:+.3f}", f"{cos_init:+.3f}"])
        if ind > 0 and byp < 0:
            ok.append(mu)

    lines = ["# Калибровка μ (free_norm)", "",
             f"Базовый запуск: `{os.path.basename(run.root)}`. Score — среднее по {WINDOW} шагам, "
             "заканчивающимся сохранённым вектором (train-батчи).", ""]
    lines += table(["μ", "домен", "сид", "шагов", "‖z‖ сохр.", "‖z‖ мин–макс", "bypass", "induce",
                    "lowest loss", "cos с base", "cos со стартом"], rows)
    mus = sorted({float(r[0]) for r in rows})
    chosen = max(ok) if ok else None
    lines += ["", f"Правило: наибольшее μ с induce > 0 и bypass < 0 → **{chosen:g}**" if ok
              else "Правило: ни одно μ не прошло — сетку нужно сдвинуть вниз."]
    if ok and chosen == max(mus):
        lines += ["", "Внимание: выбрано верхнее μ сетки — граница не найдена, сетку стоит расширить вверх."]
    # решение правила — его читает train.py для free_norm (если NORM_LAMBDA в common.py не задан)
    write_json({"norm_lambda": chosen, "mus": mus, "passed": sorted(ok),
                "at_upper_edge": bool(ok and chosen == max(mus))},
               os.path.join(args.out, "calibration.json"))
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
