"""Этап 4 (финал) — из eval_test.json каждой ступени находит размерность конуса d*.

Критерий d* (совпадает с определением конуса «сколько ортогональных направлений
всё ещё работают как направления отказа»):
  d*_weakest : наибольшая d, при которой ВСЕ d осей конуса ещё наводят отказ на
               held-out harmless — т.е. слабейшая ось weakest_induce(d)=min_k induce_k > 0.
               Как только среди d осей появляется «пустая» (переобученный шум,
               на тесте induce<0) — конус исчерпан, d* = d-1.
               Аблация (bypass/ASR) для d* бесполезна: насыщается уже на d=1
               (одной оси хватает, чтобы снять отказ), ступени не различает.
Кросс-проверка:
  d*_knee    : «локоть» кривой bypass-ASR(d) (kneedle) — для контроля, обычно ~1.

CPU. Запуск (на сервере или на Mac, если скопировать eval_test.json):
  python elbow.py --base <fork>/results/cones
Пишет reports/output-axis.md-совместимую таблицу в stdout и (если есть matplotlib)
график PR->d* в reports/.
"""
import argparse
import json
import os

RUNGS = ["theft", "illegal_activities", "malicious_use"]
LABELS = {"theft": "лист", "illegal_activities": "задача", "malicious_use": "домен"}
# PR входной оси (L=12) из reports/input-axis.md — для финальной корреляции
PR_L12 = {"theft": 11.25, "illegal_activities": 12.01, "malicious_use": 12.56}


def d_star_weakest(dims, margin=0.0):
    """d* = длина непрерывного префикса d, где слабейшая ось конуса ещё наводит
    отказ: weakest_induce(d) = per_basis_induce_min > margin."""
    if not dims:
        return None
    best = 0
    for r in dims:
        if r["per_basis_induce_min"] > margin:
            best = r["dim"]
        else:
            break
    return best


def d_star_knee(dims):
    """kneedle по bypass_asr(d): точка макс. расстояния до хорды между концами."""
    xs = [r["dim"] for r in dims]
    ys = [r["bypass_asr"] for r in dims]
    if len(xs) < 3:
        return xs[-1] if xs else None
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    best_d, best_dist = xs[0], -1.0
    for x, y in zip(xs, ys):
        # расстояние точки до прямой (x0,y0)-(x1,y1)
        num = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
        den = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5 or 1.0
        dist = num / den
        if dist > best_dist:
            best_dist, best_d = dist, x
    return best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../geometry-of-refusal/results/cones",
                    help="каталог с <rung>/eval_test.json")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="порог для слабейшей оси: weakest_induce > margin")
    ap.add_argument("--out", default="experiments/mvp/reports")
    args = ap.parse_args()

    summary = {}
    print(f"{'rung':<20} {'PR(L12)':>8}  {'d*_weakest':>10} {'d*_knee':>8}")
    for r in RUNGS:
        f = os.path.join(args.base, r, "eval_test.json")
        if not os.path.exists(f):
            print(f"{r:<20} нет {f}")
            continue
        dims = json.load(open(f))["dims"]
        ds_i = d_star_weakest(dims, args.margin)
        ds_k = d_star_knee(dims)
        summary[r] = {"d_induce": ds_i, "d_knee": ds_k, "dims": dims}
        weak = " ".join(f"{x['per_basis_induce_min']:+.1f}" for x in dims)
        byp = " ".join(f"{x['bypass_mean']:+.1f}" for x in dims)
        capped = "≥" if ds_i == len(dims) else ""
        print(f"{r:<20} {PR_L12.get(r, float('nan')):>8.2f}  {capped + str(ds_i):>10} {str(ds_k):>8}")
        print(f"{'':<20} {'':>8}  weakest_induce: {weak}")
        print(f"{'':<20} {'':>8}  bypass_mean:    {byp}")

    # финальная сводка PR -> d*
    print("\n=== PR (вход) -> d* (выход) ===")
    print(f"{'ступень':<10} {'PR(L12)':>8} {'d*':>4}")
    order_ok = []
    for r in RUNGS:
        if r in summary:
            ds = summary[r]["d_induce"]
            order_ok.append(ds)
            print(f"{LABELS[r]:<10} {PR_L12[r]:>8.2f} {str(ds):>4}")
    if len(order_ok) == 3 and all(x is not None for x in order_ok):
        mono = order_ok[0] <= order_ok[1] <= order_ok[2]
        print(f"\nлестница d*: {order_ok}  ->  "
              f"{'✅ монотонно (лист<=задача<=домен)' if mono else '✗ немонотонно'}")

    # график, если есть matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        prs = [PR_L12[r] for r in RUNGS if r in summary]
        dss = [summary[r]["d_induce"] for r in RUNGS if r in summary]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(prs, dss, "o-", color="#c0392b")
        for r in RUNGS:
            if r in summary:
                ax.annotate(LABELS[r], (PR_L12[r], summary[r]["d_induce"]),
                            textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("PR входной оси (L=12)")
        ax.set_ylabel("d* конуса отказа")
        ax.set_title("Разброс концепта (вход) -> размерность отказа (выход)")
        fig.tight_layout()
        p = os.path.join(args.out, "pr_vs_dstar.png")
        fig.savefig(p, dpi=140)
        print(f"\nграфик: {p}")
    except ImportError:
        print("\n(matplotlib нет — график пропущен)")


if __name__ == "__main__":
    main()
