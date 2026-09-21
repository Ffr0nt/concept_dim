"""Конфаундер общей инициализации + бюджет обучения. CPU, GPU не нужен.

Зачем. В rdo.py сид ставится как `torch.manual_seed(base_seed*1000 + i*10 + k)`
(rdo.py:1126) — БЕЗ участия ступени. Значит `theft/dim_5/seed_21` и
`illegal_activities/dim_5/seed_21` стартуют из ОДНОЙ случайной инициализации.
Поэтому любое сравнение ступеней внутри одного сида завышено: часть сходства —
общий старт, а не общий концепт.

Скрипт разводит три режима сравнения и печатает их рядом:
  A. одна ступень, разные сиды      — истинная воспроизводимость концепта;
  B. разные ступени, ОДИН сид       — заражено общей инициализацией;
  C. разные ступени, разные сиды    — общего старта нет, чистое сравнение концептов.
Если B >> C, конфаундер реальный. Если A <= C, концепт не отличим от run-шума.

Плюс печатает бюджет обучения: len(train_losses) = число шагов оптимизатора на конус.
При epochs=1 и effective_batch_size=16 (rdo.py DEFAULT_CONFIG) это ~20-50 шагов —
объясняет, почему реперы не уходят далеко от инициализации.

Запуск:
  bash experiments/learned_directions/run/3_init_confound.sh
  python init_confound.py --base <fork>/results/cones --dims 3 5
"""
import argparse
import itertools
import os
import statistics as st

import torch

RUNGS = ["theft", "illegal_activities", "malicious_use"]  # у 'all' нет малых d — вне сравнения
SEEDS = [21, 3, 7]
HIDDEN = 2048


def load(base, rung, seed, dim):
    p = os.path.join(base, rung, f"seed_{seed}", f"dim_{dim}.pt")
    if not os.path.exists(p):
        return None
    v = torch.load(p, map_location="cpu")["vectors"]
    return (v if torch.is_tensor(v) else torch.stack(list(v))).double()


def rho(A, B):
    """rho_span на генератор = mean_i ||P_B g_i||^2, в [0,1]. None-safe."""
    if A is None or B is None:
        return None
    return ((A @ B.T) ** 2).sum(1).mean().item()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 5])
    args = ap.parse_args()

    for d in args.dims:
        print(f"\n===== d={d}: rho_span (1.0 = тот же span, {d/HIDDEN:.4f} = случайный уровень) =====")

        print("  A. ОДНА ступень, РАЗНЫЕ сиды  (истинная воспроизводимость):")
        a_vals = []
        for r in RUNGS:
            v = [rho(load(args.base, r, x, d), load(args.base, r, y, d))
                 for x, y in itertools.combinations(SEEDS, 2)]
            a_vals += [x for x in v if x is not None]
            print(f"     {r:<20} {[round(x,4) for x in v if x is not None]}  mean={mean(v):.4f}")

        print("  B. РАЗНЫЕ ступени, ОДИН сид  (общая инициализация!):")
        b = {}
        for r1, r2 in itertools.combinations(RUNGS, 2):
            v = [rho(load(args.base, r1, s, d), load(args.base, r2, s, d)) for s in SEEDS]
            b[(r1, r2)] = mean(v)
            print(f"     {r1[:12]:<13}/{r2[:12]:<13} {[round(x,4) for x in v if x is not None]}  mean={mean(v):.4f}")

        print("  C. РАЗНЫЕ ступени, РАЗНЫЕ сиды  (общего старта нет):")
        c = {}
        for r1, r2 in itertools.combinations(RUNGS, 2):
            v = [rho(load(args.base, r1, x, d), load(args.base, r2, y, d))
                 for x, y in itertools.permutations(SEEDS, 2)]
            c[(r1, r2)] = mean(v)
            print(f"     {r1[:12]:<13}/{r2[:12]:<13} mean={mean(v):.4f}"
                  f"   (B/C = {b[(r1,r2)]/c[(r1,r2)]:.2f}x)")

        print(f"  ИТОГ d={d}: A(один концепт)={mean(a_vals):.4f}  "
              f"B(общий старт)={mean(list(b.values())):.4f}  "
              f"C(чистое сравнение)={mean(list(c.values())):.4f}")
        if mean(a_vals) <= mean(list(c.values())):
            print("    !! A <= C: один концепт на разных сидах похож на себя не больше,")
            print("       чем на соседний концепт. Сигнал концепта под run-шумом.")

    print("\n===== бюджет обучения: len(train_losses) = шагов оптимизатора =====")
    for r in RUNGS + ["all"]:
        for s in SEEDS[:1]:
            p = f"{args.base}/{r}/seed_{s}"
            if not os.path.isdir(p):
                continue
            fs = sorted((f for f in os.listdir(p) if f.startswith("dim_")),
                        key=lambda x: int(x[4:-3]))
            steps = [(f[4:-3], len(torch.load(f"{p}/{f}", map_location="cpu")["train_losses"]))
                     for f in fs]
            print(f"  {r:<20} seed_{s}: " + "  ".join(f"d{k}:{v}" for k, v in steps))


if __name__ == "__main__":
    main()
