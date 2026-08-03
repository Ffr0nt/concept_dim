"""Этап 2b — разброс входа: среднее попарное 1 − cos из дампа активаций. CPU.

Читает дампы acts_<splits>_L<layer>.pt (из extract_activations.py) и считает по каждому
одно число — среднее попарное `1 − cos` активаций (мера разброса корпуса ступени).

Запуск (CPU, можно локально с дампами или на сервере):
    uv run --with torch python experiments/mvp/scripts/input_spread.py
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F


def mean_pairwise_one_minus_cos(acts: torch.Tensor) -> float:
    """Среднее (1 − cos) по всем упорядоченным парам i≠j = 1 − mean_offdiag(cos)."""
    x = F.normalize(acts.float(), dim=1)
    n = x.shape[0]
    sims = x @ x.t()                      # [N, N] косинусы
    off_mean = (sims.sum() - sims.diagonal().sum()) / (n * (n - 1))
    return float(1.0 - off_mean)


def centered_one_minus_cos(acts: torch.Tensor) -> float:
    """То же, но после вычитания среднего вектора ступени — убирает общую компоненту,
    меряет разброс концепта вокруг его центра (а не общий сдвиг активаций)."""
    return mean_pairwise_one_minus_cos(acts.float() - acts.float().mean(dim=0, keepdim=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=None,
                    help="папка с дампами acts_*.pt (по умолчанию experiments/mvp/artifacts)")
    ap.add_argument("--glob", default="acts_*_L*.pt")
    args = ap.parse_args()

    adir = args.artifacts or os.path.join(os.path.dirname(__file__), "..", "artifacts")
    adir = os.path.abspath(adir)
    files = sorted(glob.glob(os.path.join(adir, args.glob)))
    print(f"дампов найдено: {len(files)} в {adir}\n")

    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        raw = mean_pairwise_one_minus_cos(d["acts"])
        cen = centered_one_minus_cos(d["acts"])
        rows.append((d.get("splits"), d.get("layer"), d["acts"].shape[0], raw, cen))
        print(f"{str(d.get('splits')):20s} L={d.get('layer')} "
              f"N={d['acts'].shape[0]:4d}  1-cos(raw)={raw:.4f}  1-cos(centered)={cen:.4f}")
    return rows


if __name__ == "__main__":
    main()
