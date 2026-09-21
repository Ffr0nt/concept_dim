"""График read/write по глубине: AUROC каждого направления против номера слоя. CPU.

Рисует то, что таблица §8 показывает числами: до какого слоя вредность запроса уже
декодируема, а решение об отказе ещё нет, и где они меняются местами.

Три содержательные серии цветом, две опорные — нейтральным серым (потолок слоя и случайное
направление): опорные линии не «ещё две категории», их роль другая, и цвет для них был бы
лишним. Палитра — первые три слота референсной категориальной палитры, они проходят все
попарные проверки без пересчёта.

Запуск:
  python plot_read_probe.py --json <...>/read-probe-all.json --out reports/read-probe-layers.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SERIES = [
    ("dim", "#2a78d6", "DIM целиком"),
    ("dim_perp_cone", "#eb6834", "остаток DIM (read)"),
    ("cone_perp_dim", "#1baf7a", "ось без DIM (write)"),
]
REFS = [("meandiff_layer", "потолок слоя", (0, (5, 3))),
        ("rand", "случайное", (0, (1, 2)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mark_layer", type=int, default=22, help="слой вмешательства")
    args = ap.parse_args()

    rows = json.load(open(args.json))["rows"]
    n = max(r["layer"] for r in rows) + 1
    series = lambda name: [next(r["auroc"] for r in rows
                                if r["layer"] == l and r["direction"] == name)
                           for l in range(n)]
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvline(args.mark_layer, color=INK3, lw=1, alpha=0.5, zorder=1)
    # подпись — в свободной полосе под всеми сериями (минимум по данным ~0.50)
    ax.annotate(f"вертикаль — слой {args.mark_layer}:\nздесь вмешательство наводит и снимает отказ",
                xy=(0.4, 0.455), color=INK2, fontsize=8.5,
                ha="left", va="center", linespacing=1.5)

    for name, label, dash in REFS:
        ax.plot(xs, series(name), color=INK3, lw=1.4, linestyle=dash, zorder=2, label=label)
    for name, color, label in SERIES:
        ys = series(name)
        ax.plot(xs, ys, color=color, lw=2.0, zorder=3, solid_capstyle="round")
        ax.annotate(label, xy=(n - 1, ys[-1]), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center", fontweight="medium")

    ax.set_xlim(0, n - 1 + 0.5)
    ax.set_ylim(0.42, 1.02)
    ax.set_xlabel("слой", color=INK2, fontsize=9.5)
    ax.set_ylabel("AUROC: harmful против harmless (test)", color=INK2, fontsize=9.5)
    ax.set_title("Вредность декодируется рано, отказ формируется к слою 22",
                 color=INK, fontsize=12.5, loc="left", pad=14)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dcdbd6")
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    # цветные серии подписаны прямо у конца линии — в легенде остаются только опорные
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5,
                    labelcolor=INK2, handlelength=2.6)
    leg.set_zorder(5)

    fig.subplots_adjust(right=0.80, left=0.09, top=0.88, bottom=0.12)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"[записано: {args.out}]")


if __name__ == "__main__":
    main()
