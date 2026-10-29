"""Сводка этапа 03 прогона natural_targets: по каждому направлению лучший прокси-ASR
и ASR в точке сопоставимой цены (KL <= KL_BUDGET).

Печатает markdown-таблицы для reports/results.md. Запускается через
run/summarize_ablation.sh (на сервере, где лежит ablation.json)."""

import json

RUN = "results/qwen2.5-3b_beavertails-b1b5b6bv"
KL_BUDGET = 0.055          # «при равном KL»: самая сильная аблация дешевле этого порога

# Порядок вывода: сначала компоненты и их остатки, потом DIM, потом контроли.
ORDER = [
    "w:all", "w:rdo_only",
    "dim_union", "dim_B6", "dim_B5", "dim_B1", "dim_BV",
    "dim_union⊥w:all", "dim_union⊥w:rdo_only",
    "rand_0", "rand_1", "rand_2",
]


def best(rows):
    """Максимум ASR по силе аблации и значение при бюджете KL."""
    top = max(rows, key=lambda r: r["asr_mean"])
    cheap = [r for r in rows if r["kl"] <= KL_BUDGET]
    at_budget = max(cheap, key=lambda r: r["asr_mean"]) if cheap else None
    return top, at_budget


def row(name, top, at_budget):
    cell = "—" if at_budget is None else f"{at_budget['asr_mean']:.3f} (α={at_budget['alpha']:.1f}, KL={at_budget['kl']:.3f})"
    return f"| {name} | {cell} | {top['asr_mean']:.3f} | {top['alpha']:.1f} | {top['kl']:.3f} |"


def table(names, by_dir):
    head = [
        f"| направление | ASR при KL ≤ {KL_BUDGET:.3f} | лучший ASR | α | KL |",
        "|---|---|---|---|---|",
    ]
    out = []
    for n in names:
        if n not in by_dir:
            continue
        out.append(row(n, *best(by_dir[n])))
    return head + out


def main():
    d = json.load(open(f"{RUN}/ablation/ablation.json"))
    by_dir = {}
    for r in d["rows"]:
        by_dir.setdefault(r["direction"], []).append(r)

    base = d["base_asr"]
    base_mean = sum(base.values()) / len(base)
    print(f"Без аблации прокси-ASR {base_mean:.3f} "
          f"({', '.join(f'{k} {v:.3f}' for k, v in base.items())}), "
          f"по {min(d['n_test'].values())} запросов на домен, "
          f"KL на {d['n_harmless']} harmless.\n")

    print("### Компоненты, DIM и контроли\n")
    print("\n".join(table(ORDER, by_dir)))

    rdo = sorted(n for n in by_dir if n.startswith("rdo_"))
    print("\n### Отдельные RDO: разброс\n")
    tops = {n: best(by_dir[n])[0]["asr_mean"] for n in rdo}
    lo, hi = min(tops, key=tops.get), max(tops, key=tops.get)
    med = sorted(tops.values())[len(tops) // 2]
    print(f"- направлений: {len(rdo)}")
    print(f"- лучший ASR: медиана {med:.3f}, минимум {tops[lo]:.3f} ({lo}), "
          f"максимум {tops[hi]:.3f} ({hi})")

    print("\n### Все направления, лучший ASR\n")
    print("\n".join(table(sorted(by_dir), by_dir)))


if __name__ == "__main__":
    main()
