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


def participation_ratio(acts: torch.Tensor) -> float:
    """Эффективный ранг ковариации центрированных активаций: (Σλ)² / Σλ².
    Сколько направлений реально задействовано разбросом корпуса (мера размерности)."""
    x = acts.float() - acts.float().mean(dim=0, keepdim=True)
    ev = torch.linalg.svdvals(x) ** 2          # собств. значения ковариации (∝)
    return float((ev.sum() ** 2) / ev.pow(2).sum())


def vendi_score(acts: torch.Tensor) -> float:
    """Vendi Score с косинусным ядром — эффективное число различимых направлений
    (exp энтропии Шеннона нормированных собственных значений ядра K/N)."""
    x = F.normalize(acts.float(), dim=1)
    n = x.shape[0]
    K = (x @ x.t()) / n                          # trace(K) = 1
    ev = torch.linalg.eigvalsh(K).clamp_min(0.0)
    ev = ev / ev.sum()
    nz = ev[ev > 1e-12]
    entropy = -(nz * nz.log()).sum()
    return float(torch.exp(entropy))


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
        acts = d["acts"]
        raw = mean_pairwise_one_minus_cos(acts)
        cen = centered_one_minus_cos(acts)
        pr = participation_ratio(acts)
        vs = vendi_score(acts)
        rows.append((d.get("splits"), d.get("layer"), acts.shape[0], raw, cen, pr, vs))
        print(f"{str(d.get('splits')):20s} L={d.get('layer')} N={acts.shape[0]:4d}  "
              f"1-cos(raw)={raw:.4f}  1-cos(cen)={cen:.4f}  PR={pr:7.2f}  Vendi={vs:7.2f}")
    return rows


if __name__ == "__main__":
    main()
