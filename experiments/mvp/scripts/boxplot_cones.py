"""Box-плот ablation-ASR сэмплов конуса по размерностям — как Fig. 3 в Wollschläger et al.

Читает results/cones/<rung>/eval_test.json (нужен ключ sample_asr_all — сырой массив ASR
по 256 сэмплам на каждую d; появляется после перезапуска eval с новой версией eval_cones.py)
и рисует boxplot ASR по d, отдельная панель на ступень. CPU, GPU не нужен.

Запуск:
  python boxplot_cones.py --base <fork>/results/cones --out experiments/mvp/reports
"""
import argparse
import json
import os

RUNGS = ["theft", "illegal_activities", "malicious_use"]
LABELS = {"theft": "лист (theft)", "illegal_activities": "задача (illegal)",
          "malicious_use": "домен (malicious)"}
PR_L12 = {"theft": 11.25, "illegal_activities": 12.01, "malicious_use": 12.56}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../geometry-of-refusal/results/cones")
    ap.add_argument("--out", default="experiments/mvp/reports")
    ap.add_argument("--tau", type=float, default=0.8, help="порог нижней границы (линия)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for r in RUNGS:
        f = os.path.join(args.base, r, "eval_test.json")
        if not os.path.exists(f):
            print(f"нет {f}")
            continue
        dims = json.load(open(f))["dims"]
        if not dims or "sample_asr_all" not in dims[0]:
            print(f"{r}: нет sample_asr_all — перезапусти eval с новой eval_cones.py")
            continue
        data[r] = dims

    if not data:
        print("нет данных с sample_asr_all — сначала перезапусти 4_eval_cones.sh")
        return

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, [x for x in RUNGS if x in data]):
        dims = data[r]
        series = [x["sample_asr_all"] for x in dims]
        positions = [x["dim"] for x in dims]
        ax.boxplot(series, positions=positions, widths=0.6, showfliers=True,
                   medianprops=dict(color="#c0392b"),
                   flierprops=dict(marker=".", markersize=3, alpha=0.4))
        ax.axhline(args.tau, ls="--", lw=1, color="#888",
                   label=f"порог p05={args.tau}")
        ax.set_title(f"{LABELS[r]}\nPR={PR_L12[r]:.2f}", fontsize=10)
        ax.set_xlabel("размерность конуса d")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("ablation-ASR сэмплов конуса")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Распределение ablation-ASR по направлениям конуса (box = 256 сэмплов) — "
                 "нижняя граница падает быстрее у узких концептов", fontsize=11)
    fig.tight_layout()
    p = os.path.join(args.out, "cone_asr_boxplot.png")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(p, dpi=140)
    print(f"box-плот: {p}")


if __name__ == "__main__":
    main()
