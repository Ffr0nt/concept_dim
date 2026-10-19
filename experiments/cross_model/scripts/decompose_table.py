"""Раздел «Разложение DIM» сводки по моделям — из decompose.json этапа 08 shared-refusal-component.

    python3 decompose_table.py > section.md
"""
import json
from pathlib import Path

RES = Path("/Users/f.zakharov/Desktop/disser/shared-refusal-component/results")
MODELS = [("qwen2.5-1.5b", "Qwen2.5-1.5B"), ("qwen2.5-3b", "Qwen2.5-3B"), ("qwen2.5-7b", "Qwen2.5-7B"),
          ("qwen2.5-14b", "Qwen2.5-14B"), ("llama3.2-1b", "Llama-3.2-1B"), ("llama3.2-3b", "Llama-3.2-3B"),
          ("llama3-8b", "Meta-Llama-3-8B"), ("llama3.1-8b", "Llama-3.1-8B")]
DOMS = ["O1", "O2", "O5", "O6"]
f = lambda x: f"{x:.3f}"


def load(run):
    r = json.load(open(RES / f"{run}_salad-o1o2o5o6" / "decompose" / "decompose.json"))
    r["by"] = {x["direction"]: x for x in r["rows"]}
    r["rand"] = [x for x in r["rows"] if x["kind"] == "random"]
    return r


def mean(xs):
    return sum(xs) / len(xs)


def main(variant="all"):
    rs = {name: load(run) for run, name in MODELS}
    w = f"w:{variant}"
    P = print
    P("## 4. Разложение DIM: что из DIM снимает отказ, наводит его и читает вредность\n")
    P("Этап 08 `shared-refusal-component` (ветка `feat/dim-decomposition`, 19.10.2026), все 8 моделей. "
      "Для Qwen2.5-3B — свои DIM и RDO; для остальных 7 — DIM и RDO boroday, импортированные без "
      "переобучения (`scripts/import_run.py`), компонента пересчитана нашим этапом 02 (λ1 совпадает с отчётами boroday).\n")
    P("- **DIM** — объединённый (union) в общей точке; **остаток** DIM⊥w = DIM − (DIM·ŵ)ŵ; **w** — общая компонента "
      f"`{variant}` (24 вектора: 4 DIM + 20 RDO); **w⊥DIM** — часть w вне DIM.")
    P("- **bypass** — ASR (refusal_score < 0) на harmful_test (4 × 128) при аблации a = 1 во всех слоях; "
      "**induce** — доля harmless_test (128) с refusal_score > 0 при h + s·α0·v̂ на входе слоя DIM, α0 = ‖DIM union‖; "
      "**read** — AUROC проекции активации в точке DIM, max(auc, 1 − auc), 95% ДИ — бутстрэп 1000.")
    P("- Всё — прокси первого токена, без судьи. «База» — без вмешательства.\n")

    P("| модель | доля DIM вдоль w | bypass: DIM / **остаток** / база | induce s=1: DIM / **остаток** / база | "
      "read: DIM / **остаток** / rand |")
    P("|---|---|---|---|---|")
    for name, r in rs.items():
        by, b = r["by"], r["base"]
        D, R = by["dim_union"], by[f"dim_union⊥{w}"]
        P(f"| {name} | {f(r['energy'][variant]['dim_union'])} | "
          f"{f(D['bypass'])} / **{f(R['bypass'])}** / {f(b['asr_mean'])} | "
          f"{f(D['induce']['1'])} / **{f(R['induce']['1'])}** / {f(b['induce'])} | "
          f"{f(D['read']['auroc'])} / **{f(R['read']['auroc'])}** / {f(mean([x['read']['auroc'] for x in r['rand']]))} |")

    P("\nПодробнее: KL, наведение при s = 2, сама компонента и её часть вне DIM.\n")
    P("| модель | KL: DIM / остаток / w | induce s=2: остаток | bypass w | induce s=1 w | read w⊥DIM [95% ДИ] |")
    P("|---|---|---|---|---|---|")
    for name, r in rs.items():
        by = r["by"]
        D, R, W, WP = by["dim_union"], by[f"dim_union⊥{w}"], by[w], by[f"{w}⊥dim_union"]
        ci = WP["read"]["ci"]
        P(f"| {name} | {D['kl']:.3f} / {R['kl']:.3f} / {W['kl']:.3f} | {f(R['induce']['2'])} | {f(W['bypass'])} | "
          f"{f(W['induce']['1'])} | {f(WP['read']['auroc'])} [{f(ci[0])}, {f(ci[1])}] |")

    P("\nУстойчивость: остатки DIM доменов (диапазон по O1, O2, O5, O6) и w, построенная только по 20 RDO.\n")
    P("| модель | bypass остатков доменов | induce s=1 остатков доменов | остаток при w только по RDO: доля DIM вдоль w / bypass / induce s=1 / read |")
    P("|---|---|---|---|")
    for name, r in rs.items():
        by = r["by"]
        dom = [by[f"dim_{d}⊥{w}"] for d in DOMS]
        bp = [x["bypass"] for x in dom]
        ind = [x["induce"]["1"] for x in dom]
        R2 = by["dim_union⊥w:rdo_only"]
        P(f"| {name} | {f(min(bp))}–{f(max(bp))} | {f(min(ind))}–{f(max(ind))} | "
          f"{f(r['energy']['rdo_only']['dim_union'])} / {f(R2['bypass'])} / {f(R2['induce']['1'])} / {f(R2['read']['auroc'])} |")

    P("\n**Выводы**\n")
    P("1. Около половины энергии DIM (0.47–0.54) приходится на остаток, который **не снимает отказ и не наводит его** "
      "ни на одной из 8 моделей (исключение — слабое наведение у Qwen2.5-1.5B: 0.16 против базы 0.08 при s = 1, 0.30 при s = 2).")
    P("2. Остаток **читает вредность так же, как целый DIM**: AUROC 0.993–1.000 против 0.995–1.000.")
    P("3. **Аблация остатка делает модель осторожнее**: bypass остатка ниже базы на всех 8 моделях "
      "(сильнее всего Qwen2.5-14B 0.38 против 0.55 и Qwen2.5-1.5B 0.10 против 0.24).")
    P("4. Часть w вне DIM (w⊥DIM) читает вредность на уровне случайного направления (AUROC 0.53–0.65 против 0.59–0.92), "
      "это write-направление без read.")
    P("5. С w только по 20 RDO картина та же, кроме Qwen2.5-1.5B: там остаток снимает отказ (0.45 против базы 0.24) — "
      "компонента без DIM не забирает всю причинную часть. Основной вариант — w по всем 24 векторам.")
    P("\nОграничения: прокси без судьи; одна точка на модель (induce на слое DIM, read в точке DIM); RDO семи моделей — "
      "один прогон boroday с параметрами `rdo.py` по умолчанию. Исходники: "
      "`shared-refusal-component/results/<модель>_salad-o1o2o5o6/{decompose/decompose.json, reports/decompose.md}` "
      "(локально и на сервере в `~/f.zakharov/shared-refusal-component/results/`).")


if __name__ == "__main__":
    main()
