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


def _prefix(dims, ok):
    """длина непрерывного префикса d (с d=1), где ok(row) истинно."""
    best = 0
    for r in dims:
        if ok(r):
            best = r["dim"]
        else:
            break
    return best


# три операционализации «размерности конуса»:
def d_star_induce(dims):
    """достаточность: все d осей наводят отказ при добавлении (weakest_induce>0)."""
    return _prefix(dims, lambda r: r["per_basis_induce_min"] > 0)


def d_star_single(dims):
    """необходимость (одиночная): каждая ось сама по себе снимает отказ при аблации
    (слабейшая одиночная аблация per_basis_bypass_max < 0)."""
    return _prefix(dims, lambda r: r["per_basis_bypass_max"] < 0)


def d_star_loo(dims):
    """необходимость (leave-one-out): каждая ось независимо нужна — при аблации конуса
    без неё отказ восстанавливается (loo_bypass_min > 0)."""
    return _prefix(dims, lambda r: r.get("loo_bypass_min", -1.0) > 0)


def d_star_sample(dims, tau=0.8):
    """метрика Wollschläger (Fig.3-4): размерность конуса = наибольшая d, при которой
    НИЖНЯЯ ГРАНИЦА ablation-ASR сэмплов конуса (p05) ещё >= tau. Плато́ = исчерпание конуса."""
    return _prefix(dims, lambda r: r.get("sample_asr_p05", 0.0) >= tau)


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


CRITERIA = [
    ("sample", "СТАТЬЯ: ablation-ASR сэмплов, нижняя граница p05>=0.8", d_star_sample, "sample_asr_p05"),
    ("induce", "достаточность (add→отказ)", d_star_induce, "per_basis_induce_min"),
    ("single", "необходимость одиночн. (ablate 1→нет отказа)", d_star_single, "per_basis_bypass_max"),
    ("loo", "необходимость LOO (ablate all−k→отказ)", d_star_loo, "loo_bypass_min"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../geometry-of-refusal/results/cones",
                    help="каталог с <rung>/eval_test.json")
    ap.add_argument("--out", default="experiments/mvp/reports")
    args = ap.parse_args()

    summary = {}
    for r in RUNGS:
        f = os.path.join(args.base, r, "eval_test.json")
        if not os.path.exists(f):
            print(f"{r:<20} нет {f}")
            continue
        dims = json.load(open(f))["dims"]
        ds = {name: fn(dims) for name, _, fn, _ in CRITERIA}
        summary[r] = {"d": ds, "dims": dims, "n": len(dims)}
        print(f"\n## {r}  (PR={PR_L12.get(r, float('nan')):.2f}, n={len(dims)})")
        for name, desc, _, key in CRITERIA:
            cap = "≥" if ds[name] == len(dims) else " "
            curve = " ".join(f"{x[key]:+.1f}" for x in dims)
            print(f"  d*_{name:<7}={cap}{ds[name]}  [{desc}]")
            print(f"           {key}: {curve}")

    # сводная таблица: PR -> d* по трём критериям
    print("\n=== PR (вход) -> d* (выход), три критерия ===")
    hdr = f"{'ступень':<8} {'PR':>6} " + " ".join(f"{n:>8}" for n, *_ in CRITERIA)
    print(hdr)
    for r in RUNGS:
        if r not in summary:
            continue
        cells = []
        for name, *_ in CRITERIA:
            v = summary[r]["d"][name]
            cells.append(("≥" if v == summary[r]["n"] else "") + str(v))
        print(f"{LABELS[r]:<8} {PR_L12[r]:>6.2f} " + " ".join(f"{c:>8}" for c in cells))
    # монотонность по каждому критерию
    print()
    for name, *_ in CRITERIA:
        vals = [summary[r]["d"][name] for r in RUNGS if r in summary]
        if len(vals) == 3:
            mono = vals[0] <= vals[1] <= vals[2] and vals[0] < vals[2]
            print(f"  {name:<7}: {vals}  {'✅ монотонно растёт' if mono else '✗ не поддерживает лестницу'}")

    # непрерывный показатель: среднее p05 по всем d (площадь под нижней границей ablation-ASR)
    print("\n=== непрерывно: mean p05 ablation-ASR по всем d (робастность конуса) ===")
    mp = []
    for r in RUNGS:
        if r in summary:
            vs = [x.get("sample_asr_p05", 0.0) for x in summary[r]["dims"]]
            m = sum(vs) / len(vs)
            mp.append(m)
            print(f"  {LABELS[r]:<8} PR={PR_L12[r]:.2f}  mean_p05={m:.3f}")
    if len(mp) == 3:
        mono = mp[0] < mp[1] < mp[2]
        print(f"  лестница mean_p05: {[round(x,3) for x in mp]}  "
              f"{'✅ строго монотонно растёт' if mono else '✗'}")

    # график: три критерия vs PR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        prs = [PR_L12[r] for r in RUNGS if r in summary]
        colors = {"sample": "#8e44ad", "induce": "#c0392b", "single": "#2980b9", "loo": "#27ae60"}
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, desc, _, _ in CRITERIA:
            ys = [summary[r]["d"][name] for r in RUNGS if r in summary]
            ax.plot(prs, ys, "o-", color=colors.get(name), label=f"d*_{name}")
        ax.set_xlabel("PR входной оси (L=12)")
        ax.set_ylabel("d* конуса отказа")
        ax.set_title("Разброс концепта (вход) → размерность отказа (выход)")
        ax.legend(fontsize=8)
        for r in RUNGS:
            if r in summary:
                ax.annotate(LABELS[r], (PR_L12[r], summary[r]["d"]["sample"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)
        fig.tight_layout()
        p = os.path.join(args.out, "pr_vs_dstar.png")
        fig.savefig(p, dpi=140)
        print(f"\nграфик: {p}")
    except ImportError:
        print("\n(matplotlib нет — график пропущен)")


if __name__ == "__main__":
    main()
