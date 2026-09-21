"""Попарное сходство выученных реперов конуса между сидами + агрегаты. CPU, GPU не нужен.

Для каждой ступени и каждой размерности d берёт реперы A, B ([d, hidden], строки — рёбра
конуса, ортонормированы) из cones/<rung>/seed_<s>/dim_<d>.pt и считает M = A @ B.T ([d,d]).
Метрики попарного сходства (все симметричны по A/B):

  rho_span   = ||M||_F^2 / d      — доля энергии генераторов A, лежащая в span(B).
                                    1.0 = тот же span, d/hidden = случайные реперы.
  cos_match  = mean |M[i,pi(i)]|  — оси сопоставлены венгерским алгоритмом по |M|;
                                    «насколько похожи сами вектора», а не только span.
  cos_max    = max |M|            — лучшее совпадение одной оси из d.
  cos_1      = max sv(M)          — косинус первого главного угла между подпространствами
                                    (верхняя граница: лучшая пара направлений из span-ов).
  neg_frac   = доля сопоставленных пар с ОТРИЦАТЕЛЬНЫМ косинусом. Для конуса знак важен:
                                    ось, совпавшая с точностью до знака, лежит в другом ортанте.

Случайный уровень (`rand`) оценивается эмпирически на случайных ортонормированных реперах
той же формы — иначе числа не с чем сравнивать (при hidden=2048 rho_span=0.05 это ещё
в ~10 раз выше шума, а cos_match=0.05 — уже почти шум).

Выводит: попарные таблицы по каждой ступени + четыре агрегата (по рангу d, по концепту).

ВАЖНО: нужен scipy (в venv форка его нет). Запускать через run/1_seed_agreement.sh,
который подтягивает его разово через `uv run --with scipy`, не трогая зависимости репо.
Жадный фолбэк намеренно НЕ реализован: он молча занижает cos_match.

Запуск:
  bash experiments/learned_directions/run/1_seed_agreement.sh
  python seed_agreement.py --base <fork>/results/cones --out reports/seed-agreement.md
"""
import argparse
import itertools
import os
import statistics as st

import torch

RUNGS = ["theft", "illegal_activities", "malicious_use", "all"]
SEEDS = [21, 3, 7]
PAIRS = list(itertools.combinations(SEEDS, 2))
METRICS = ["rho_span", "cos_match", "cos_max", "cos_1", "neg_frac"]
TITLES = {
    "rho_span": ("rho_span — доля энергии генераторов в чужом span", "1.0 = тот же span"),
    "cos_match": ("cos_match — среднее |cos| по сопоставленным осям", "1.0 = те же вектора"),
    "cos_max": ("cos_max — лучшее совпадение одной оси", ""),
    "cos_1": ("cos_1 — косинус первого главного угла", "верхняя граница сходства span-ов"),
    "neg_frac": ("neg_frac — доля сопоставленных осей с отрицательным cos", "знак важен для конуса"),
}


def load(base, rung, seed, dim):
    p = os.path.join(base, rung, f"seed_{seed}", f"dim_{dim}.pt")
    if not os.path.exists(p):
        return None
    v = torch.load(p, map_location="cpu")["vectors"]
    return (v if torch.is_tensor(v) else torch.stack(list(v))).double()


def dims_of(base, rung, seed):
    p = os.path.join(base, rung, f"seed_{seed}")
    if not os.path.isdir(p):
        return set()
    return {int(f[4:-3]) for f in os.listdir(p) if f.startswith("dim_") and f.endswith(".pt")}


def assign(absM):
    """Венгерское сопоставление осей по |M|. scipy обязателен — см. докстринг модуля."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise SystemExit(
            "scipy не найден. Запускай через run/1_seed_agreement.sh "
            "(он использует `uv run --with scipy`). Жадное сопоставление занижает cos_match, "
            "поэтому фолбэка нет."
        )
    r, c = linear_sum_assignment(-absM.numpy())
    return list(zip(r.tolist(), c.tolist()))


def metrics(A, B):
    M = A @ B.T
    d = M.shape[0]
    matched = torch.tensor([M[i, j] for i, j in assign(M.abs())])
    return {
        "rho_span": ((M ** 2).sum() / d).item(),
        "cos_match": matched.abs().mean().item(),
        "cos_max": M.abs().max().item(),
        "cos_1": torch.linalg.svdvals(M)[0].item(),
        "neg_frac": (matched < 0).double().mean().item(),
    }


def random_baseline(d, hidden, n, g):
    """Случайный уровень: пары независимых ортонормированных реперов [d, hidden]."""
    acc = []
    for _ in range(n):
        F = []
        for _ in range(2):
            Q, _ = torch.linalg.qr(torch.randn(hidden, d, generator=g, dtype=torch.float64))
            F.append(Q.T)
        acc.append(metrics(F[0], F[1]))
    return {k: st.mean(a[k] for a in acc) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/cones форка")
    ap.add_argument("--n_rand", type=int, default=200, help="пар случайных реперов для rand")
    ap.add_argument("--seed", type=int, default=21, help="сид ГСЧ для rand")
    ap.add_argument("--out", default=None, help="файл для markdown-отчёта")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # --- сбор: per-pair значения и среднее по трём парам сидов ---
    pairwise, cell, hidden = {}, {}, None
    for rung in RUNGS:
        dims = sorted(set.intersection(*(dims_of(args.base, rung, s) for s in SEEDS))
                      ) if all(dims_of(args.base, rung, s) for s in SEEDS) else []
        for d in dims:
            fr = {s: load(args.base, rung, s, d) for s in SEEDS}
            hidden = fr[SEEDS[0]].shape[1]
            ms = [metrics(fr[a], fr[b]) for a, b in PAIRS]
            pairwise[(rung, d)] = ms
            cell[(rung, d)] = {k: st.mean(m[k] for m in ms) for k in METRICS}

    all_dims = sorted({d for _, d in cell})
    rand = {d: random_baseline(d, hidden, args.n_rand, g) for d in all_dims}

    emit("# Попарное сходство выученных реперов между сидами")
    emit()
    emit(f"Сиды {'/'.join(map(str, SEEDS))}, hidden={hidden}. Три столбца пар = полная")
    emit("симметричная матрица 3x3 (диагональ = 1 тривиально). `rand` — случайный уровень")
    emit(f"для той же формы ({args.n_rand} пар независимых ортонормированных реперов).")
    emit()

    # --- часть 1: попарные таблицы по ступеням ---
    emit("## Часть 1. Попарные значения по ступеням")
    hdr = " | ".join(f"{a}-{b}" for a, b in PAIRS)
    for rung in RUNGS:
        dims = [d for r, d in cell if r == rung]
        if not dims:
            continue
        emit()
        emit(f"### {rung}")
        for m in METRICS:
            title, note = TITLES[m]
            emit()
            emit(f"**{title}**" + (f" ({note})" if note else ""))
            emit()
            emit(f"| d | {hdr} | среднее | rand |")
            emit("|---|" + "---|" * (len(PAIRS) + 2))
            for d in sorted(dims):
                cells = " | ".join(f"{x[m]:.4f}" for x in pairwise[(rung, d)])
                emit(f"| {d} | {cells} | {cell[(rung,d)][m]:.4f} | {rand[d][m]:.4f} |")

    # --- часть 2: агрегаты ---
    mh = " | ".join(METRICS)
    emit()
    emit("## Часть 2. Агрегаты")
    emit()
    emit("### A. Среднее по концептам для каждого ранга d")
    emit()
    emit("ВНИМАНИЕ: состав концептов меняется с d (не все ступени обучены на всех d),")
    emit("поэтому тренд вдоль таблицы смешан с изменением состава — см. столбец «концепты».")
    emit()
    emit(f"| d | концепты (n) | {mh} |")
    emit("|---|---|" + "---|" * len(METRICS))
    for d in all_dims:
        rs = [r for r in RUNGS if (r, d) in cell]
        vals = " | ".join(f"{st.mean(cell[(r,d)][k] for r in rs):.4f}" for k in METRICS)
        emit(f"| {d} | {', '.join(rs)} ({len(rs)}) | {vals} |")
    emit()
    emit(f"| d | случайный уровень | {mh} |")
    emit("|---|---|" + "---|" * len(METRICS))
    for d in all_dims:
        emit(f"| {d} | rand | " + " | ".join(f"{rand[d][k]:.4f}" for k in METRICS) + " |")

    for title, flt in [
        ("B. Среднее по рангам для каждого концепта", lambda d: True),
        ("C. То же, без вырожденного d=1 (одна ось: метрики бинарны)", lambda d: d >= 2),
    ]:
        emit()
        emit(f"### {title}")
        emit()
        emit(f"| концепт | d | {mh} |")
        emit("|---|---|" + "---|" * len(METRICS))
        for r in RUNGS:
            ds = sorted(d for rr, d in cell if rr == r and flt(d))
            if not ds:
                continue
            vals = " | ".join(f"{st.mean(cell[(r,d)][k] for d in ds):.4f}" for k in METRICS)
            emit(f"| {r} | {min(ds)}–{max(ds)} | {vals} |")

    # D: срез с ОДИНАКОВЫМ набором d — единственное честное сравнение концептов между собой
    common = sorted(set.intersection(*({d for rr, d in cell if rr == r} for r in RUNGS
                                       if any(rr == r for rr, _ in cell))))
    if common:
        emit()
        emit(f"### D. Сопоставимый срез: одинаковые d={min(common)}–{max(common)} у всех концептов")
        emit()
        emit(f"| концепт | {mh} |")
        emit("|---|" + "---|" * len(METRICS))
        for r in RUNGS:
            ds = [d for d in common if (r, d) in cell]
            if not ds:
                continue
            vals = " | ".join(f"{st.mean(cell[(r,d)][k] for d in ds):.4f}" for k in METRICS)
            emit(f"| {r} | {vals} |")
    else:
        emit()
        emit("### D. Сопоставимый срез")
        emit()
        emit("Общего набора d у всех четырёх концептов нет "
             f"({', '.join(f'{r}: {sorted(d for rr,d in cell if rr==r)}' for r in RUNGS)}) — "
             "сравнивать концепты между собой можно только внутри пересекающихся d.")
        narrow = [r for r in RUNGS if r != "all"]
        cn = sorted(set.intersection(*({d for rr, d in cell if rr == r} for r in narrow)))
        if cn:
            emit()
            emit(f"Срез по трём узким концептам, d={min(cn)}–{max(cn)}:")
            emit()
            emit(f"| концепт | {mh} |")
            emit("|---|" + "---|" * len(METRICS))
            for r in narrow:
                vals = " | ".join(f"{st.mean(cell[(r,d)][k] for d in cn):.4f}" for k in METRICS)
                emit(f"| {r} | {vals} |")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[markdown -> {args.out}]")


if __name__ == "__main__":
    main()
