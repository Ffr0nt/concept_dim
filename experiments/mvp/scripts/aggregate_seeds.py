"""Усреднение метрики размерности конуса по нескольким сидам (mean±std).

Читает results/cones/<rung>/seed_<s>/eval_test.json (созданы run/multiseed.sh) и по КАЖДОМУ
сиду считает mean_p05 и d*(порог); затем усредняет ПО СИДАМ (корректно — усредняем скаляр,
а не конус). Печатает лестницу mean±std и рисует её с усами. CPU.

Запуск:
  python aggregate_seeds.py --base <fork>/results/cones --out experiments/mvp/reports
"""
import argparse
import glob
import json
import os
import statistics as st

RUNGS = ["theft", "illegal_activities", "malicious_use"]
LABELS = {"theft": "лист", "illegal_activities": "задача", "malicious_use": "домен"}
PR_L12 = {"theft": 11.25, "illegal_activities": 12.01, "malicious_use": 12.56}


def d_star(p05, tau):
    b = 0
    for i, v in enumerate(p05, 1):
        if v >= tau:
            b = i
        else:
            break
    return b


def load_seed_runs(base, rung):
    """seed -> list p05 по d, из seed_*/eval_test.json."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(base, rung, "seed_*", "eval_test.json"))):
        seed = os.path.basename(os.path.dirname(f)).replace("seed_", "")
        dims = json.load(open(f))["dims"]
        runs[seed] = [x["sample_asr_p05"] for x in dims]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../geometry-of-refusal/results/cones")
    ap.add_argument("--out", default="experiments/mvp/reports")
    ap.add_argument("--tau", type=float, default=0.8)
    args = ap.parse_args()

    summary = {}
    for r in RUNGS:
        runs = load_seed_runs(args.base, r)
        if not runs:
            print(f"{r}: нет seed_*/eval_test.json")
            continue
        seeds = sorted(runs)
        mean_p05_per_seed = [sum(p) / len(p) for p in runs.values()]
        dstar_hi = [d_star(p, args.tau) for p in runs.values()]
        dstar_lo = [d_star(p, 0.6) for p in runs.values()]
        # per-d среднее p05 по сидам (для «сглаженной» кривой)
        L = min(len(p) for p in runs.values())
        per_d_mean = [st.mean(runs[s][d] for s in seeds) for d in range(L)]
        per_d_std = [st.pstdev(runs[s][d] for s in seeds) for d in range(L)]
        summary[r] = {
            "seeds": seeds, "n": len(seeds),
            "mean_p05_mean": st.mean(mean_p05_per_seed),
            "mean_p05_std": st.pstdev(mean_p05_per_seed) if len(seeds) > 1 else 0.0,
            "mean_p05_list": mean_p05_per_seed,
            "dstar_hi": dstar_hi, "dstar_lo": dstar_lo,
            "per_d_mean": per_d_mean, "per_d_std": per_d_std,
        }

    if not summary:
        print("нет данных по сидам — сначала run/multiseed.sh")
        return

    print(f"\n=== усреднение mean_p05 по сидам (порог d*={args.tau}) ===")
    print(f"{'ступень':<8} {'PR':>6} {'N':>2}  {'mean_p05 (mean±std)':>22}  {'d*(0.8)':>16} {'d*(0.6)':>16}")
    for r in RUNGS:
        if r not in summary:
            continue
        s = summary[r]
        dhi = f"{st.mean(s['dstar_hi']):.1f}±{st.pstdev(s['dstar_hi']) if s['n']>1 else 0:.1f} {s['dstar_hi']}"
        dlo = f"{st.mean(s['dstar_lo']):.1f}±{st.pstdev(s['dstar_lo']) if s['n']>1 else 0:.1f} {s['dstar_lo']}"
        print(f"{LABELS[r]:<8} {PR_L12[r]:>6.2f} {s['n']:>2}  "
              f"{s['mean_p05_mean']:>10.3f} ± {s['mean_p05_std']:<8.3f}  {dhi:>16} {dlo:>16}")

    # монотонность лестницы mean_p05
    vals = [summary[r]["mean_p05_mean"] for r in RUNGS if r in summary]
    if len(vals) == 3:
        mono = vals[0] < vals[1] < vals[2]
        print(f"\nлестница mean_p05: {[round(v,3) for v in vals]}  "
              f"{'✅ строго монотонно растёт' if mono else '✗'}")

    # график: mean_p05 ± std vs PR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        rr = [r for r in RUNGS if r in summary]
        xs = [PR_L12[r] for r in rr]
        ys = [summary[r]["mean_p05_mean"] for r in rr]
        es = [summary[r]["mean_p05_std"] for r in rr]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#8e44ad", capsize=5, lw=2)
        for r in rr:
            ax.annotate(f"{LABELS[r]} (N={summary[r]['n']})",
                        (PR_L12[r], summary[r]["mean_p05_mean"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.set_xlabel("PR входной оси (L=12)")
        ax.set_ylabel("mean p05 ablation-ASR")
        ax.set_title("Размерность конуса (mean p05 ± std по сидам) vs разброс концепта")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out, "pr_vs_meanp05_seeds.png")
        fig.savefig(p, dpi=140)
        print(f"\nграфик: {p}")
    except ImportError:
        print("\n(matplotlib нет — график пропущен)")


if __name__ == "__main__":
    main()
