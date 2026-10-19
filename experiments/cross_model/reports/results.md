# Общая компонента отказа по моделям — сводка

Собрано 19.10.2026 из отчётов `shared-refusal-component`. Qwen2.5-3B — твой прогон, остальные 7 моделей — прогоны boroday (форк BorodAILA/shared-refusal-component, ветка `main`, коммит `d72f0a5`).

Условия у всех одинаковые: SALAD-Bench base_set, домены O1, O2, O5, O6; 900/128/128 на домен; harmless — Alpaca из форка; 4 DIM + 20 одномерных RDO (сиды 1–20, по 5 на домен); компонента w — первый главный вектор, равные веса. **Сплиты совпадают побайтно** (sha256 в `manifest.json` у всех 8 моделей одинаковые). Гиперпараметры RDO у boroday в конфиге не заданы, значит взяты дефолты `rdo.py`; у тебя они записаны явно. Для 14B уменьшены только батчи фильтра (из-за памяти).

## 1. Компонента и её доля в DIM

λ1 — доля разброса векторов вдоль w; нуль p95 — 95-й перцентиль λ1 для случайных векторов; «выбросы RDO» — сиды с cos(w, RDO) < 0.3. Доля энергии DIM — среднее по 4 доменам cos²(DIM, w).

| модель | hidden | слоёв | точка DIM (pos, layer) | λ1 (DIM+RDO) | λ2 | нуль p95 | λ1 (только RDO) | выбросы RDO | доля DIM вдоль w | то же, w только по RDO | min cos(DIM_i, DIM_j) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1536 | 28 | (-1, 21) | **0.414** | 0.107 | 0.053 | 0.426 | s12, s16 | 0.477 | 0.252 | 0.947 |
| Qwen2.5-3B | 2048 | 36 | (-1, 27) | **0.419** | 0.107 | 0.051 | 0.422 | s08, s12, s18 | 0.526 | 0.292 | 0.952 |
| Qwen2.5-7B | 3584 | 28 | (-2, 19) | **0.401** | 0.107 | 0.049 | 0.410 | s08, s14, s17 | 0.478 | 0.242 | 0.971 |
| Qwen2.5-14B | 5120 | 48 | (-5, 25) | **0.429** | 0.096 | 0.048 | 0.443 | s14, s15 | 0.466 | 0.270 | 0.847 |
| Llama-3.2-1B | 2048 | 16 | (-5, 9) | **0.454** | 0.100 | 0.051 | 0.476 | — | 0.438 | 0.257 | 0.889 |
| Llama-3.2-3B | 3072 | 28 | (-4, 12) | **0.506** | 0.095 | 0.049 | 0.529 | — | 0.480 | 0.308 | 0.919 |
| Meta-Llama-3-8B | 4096 | 32 | (-2, 12) | **0.535** | 0.097 | 0.048 | 0.561 | — | 0.490 | 0.319 | 0.952 |
| Llama-3.1-8B | 4096 | 32 | (-5, 12) | **0.579** | 0.096 | 0.048 | 0.620 | — | 0.453 | 0.308 | 0.909 |

## 2. Аблация (a = 1.00)

ASR — доля harmful_test с refusal_metric < 0, среднее по 4 доменам; KL — на harmless_test. DIM и RDO — среднее по 4 DIM и по 20 RDO. **Сравнивать при близком KL**: у w KL обычно ниже, чем у DIM.

| модель | без аблации | w: ASR | w: KL | w (только RDO): ASR | w (только RDO): KL | DIM: ASR | DIM: KL | RDO: ASR | RDO: KL | rand: ASR |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.240 | **0.980** | 0.0949 | 0.975 | 0.0592 | 0.990 | 0.1459 | 0.885 | 0.0291 | 0.262 |
| Qwen2.5-3B | 0.317 | **0.979** | 0.0551 | — | — | 0.970 | 0.1046 | 0.924 | 0.0303 | 0.313 |
| Qwen2.5-7B | 0.486 | **0.977** | 0.0176 | 0.975 | 0.0132 | 0.960 | 0.0930 | 0.944 | 0.0100 | 0.503 |
| Qwen2.5-14B | 0.550 | **0.973** | 0.0217 | 0.975 | 0.0135 | 0.940 | 0.1166 | 0.942 | 0.0093 | 0.553 |
| Llama-3.2-1B | 0.379 | **0.957** | 0.0168 | 0.945 | 0.0119 | 0.769 | 0.1459 | 0.733 | 0.0082 | 0.367 |
| Llama-3.2-3B | 0.455 | **0.967** | 0.0071 | 0.961 | 0.0034 | 0.815 | 0.0868 | 0.825 | 0.0027 | 0.454 |
| Meta-Llama-3-8B | 0.260 | **0.973** | 0.0060 | 0.969 | 0.0045 | 0.923 | 0.0275 | 0.912 | 0.0043 | 0.258 |
| Llama-3.1-8B | 0.494 | **0.982** | 0.0112 | 0.982 | 0.0077 | 0.936 | 0.0333 | 0.925 | 0.0069 | 0.484 |

## 3. Read-probe в точке DIM (AUROC harmful vs harmless)

Последний токен, вход слоя DIM; пул 1200 harmful + 1156 harmless; w — только по 20 RDO; w⊥DIM — часть w, ортогональная DIM. У Qwen2.5-3B этого шага в shared-refusal-component нет (аналог — `concept_dim/experiments/common_axis/reports/read-probe.md`).

| модель | слой | dim | w | w⊥DIM | rand |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 21 | 0.997 | 0.999 | 0.547 | 0.831 |
| Qwen2.5-3B | 27 | — | — | — | — |
| Qwen2.5-7B | 19 | 0.995 | 0.997 | 0.973 | 0.568 |
| Qwen2.5-14B | 25 | 0.999 | 0.999 | 0.537 | 0.507 |
| Llama-3.2-1B | 9 | 0.999 | 0.998 | 0.884 | 0.857 |
| Llama-3.2-3B | 12 | 0.999 | 0.998 | 0.508 | 0.656 |
| Meta-Llama-3-8B | 12 | 0.999 | 0.999 | 0.802 | 0.539 |
| Llama-3.1-8B | 12 | 0.999 | 0.999 | 0.852 | 0.581 |

## 4. Разложение DIM: что из DIM снимает отказ, наводит его и читает вредность

Этап 08 `shared-refusal-component` (ветка `feat/dim-decomposition`, 19.10.2026), все 8 моделей. Для Qwen2.5-3B — свои DIM и RDO; для остальных 7 — DIM и RDO boroday, импортированные без переобучения (`scripts/import_run.py`), компонента пересчитана нашим этапом 02 (λ1 совпадает с отчётами boroday).

- **DIM** — объединённый (union) в общей точке; **остаток** DIM⊥w = DIM − (DIM·ŵ)ŵ; **w** — общая компонента `all` (24 вектора: 4 DIM + 20 RDO); **w⊥DIM** — часть w вне DIM.
- **bypass** — ASR (refusal_score < 0) на harmful_test (4 × 128) при аблации a = 1 во всех слоях; **induce** — доля harmless_test (128) с refusal_score > 0 при h + s·α0·v̂ на входе слоя DIM, α0 = ‖DIM union‖; **read** — AUROC проекции активации в точке DIM, max(auc, 1 − auc), 95% ДИ — бутстрэп 1000.
- Всё — прокси первого токена, без судьи. «База» — без вмешательства.

| модель | доля DIM вдоль w | bypass: DIM / **остаток** / база | induce s=1: DIM / **остаток** / база | read: DIM / **остаток** / rand |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.488 | 0.986 / **0.100** / 0.240 | 0.852 / **0.156** / 0.078 | 0.995 / **0.993** / 0.739 |
| Qwen2.5-3B | 0.537 | 0.971 / **0.184** / 0.316 | 0.867 / **0.031** / 0.047 | 0.995 / **0.994** / 0.706 |
| Qwen2.5-7B | 0.484 | 0.961 / **0.389** / 0.486 | 0.750 / **0.031** / 0.039 | 0.996 / **0.996** / 0.915 |
| Qwen2.5-14B | 0.500 | 0.947 / **0.379** / 0.551 | 0.742 / **0.047** / 0.055 | 0.999 / **0.997** / 0.593 |
| Llama-3.2-1B | 0.468 | 0.805 / **0.371** / 0.379 | 0.641 / **0.008** / 0.016 | 0.997 / **0.996** / 0.718 |
| Llama-3.2-3B | 0.504 | 0.859 / **0.416** / 0.455 | 0.977 / **0.031** / 0.016 | 0.998 / **0.998** / 0.777 |
| Meta-Llama-3-8B | 0.501 | 0.938 / **0.188** / 0.260 | 0.992 / **0.016** / 0.016 | 1.000 / **1.000** / 0.684 |
| Llama-3.1-8B | 0.475 | 0.957 / **0.381** / 0.494 | 0.984 / **0.000** / 0.000 | 0.999 / **0.998** / 0.701 |

Подробнее: KL, наведение при s = 2, сама компонента и её часть вне DIM.

| модель | KL: DIM / остаток / w | induce s=2: остаток | bypass w | induce s=1 w | read w⊥DIM [95% ДИ] |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.146 / 0.105 / 0.095 | 0.297 | 0.980 | 0.977 | 0.554 [0.491, 0.611] |
| Qwen2.5-3B | 0.097 / 0.082 / 0.055 | 0.008 | 0.979 | 0.984 | 0.531 [0.480, 0.590] |
| Qwen2.5-7B | 0.090 / 0.049 / 0.018 | 0.008 | 0.977 | 0.977 | 0.551 [0.500, 0.599] |
| Qwen2.5-14B | 0.117 / 0.123 / 0.022 | 0.016 | 0.973 | 0.961 | 0.651 [0.599, 0.703] |
| Llama-3.2-1B | 0.133 / 0.049 / 0.017 | 0.008 | 0.957 | 0.977 | 0.632 [0.585, 0.680] |
| Llama-3.2-3B | 0.087 / 0.053 / 0.007 | 0.047 | 0.967 | 0.992 | 0.649 [0.605, 0.694] |
| Meta-Llama-3-8B | 0.026 / 0.034 / 0.006 | 0.008 | 0.973 | 1.000 | 0.586 [0.542, 0.635] |
| Llama-3.1-8B | 0.033 / 0.013 / 0.011 | 0.000 | 0.982 | 1.000 | 0.652 [0.606, 0.701] |

Устойчивость: остатки DIM доменов (диапазон по O1, O2, O5, O6) и w, построенная только по 20 RDO.

| модель | bypass остатков доменов | induce s=1 остатков доменов | остаток при w только по RDO: доля DIM вдоль w / bypass / induce s=1 / read |
|---|---|---|---|
| Qwen2.5-1.5B | 0.066–0.119 | 0.102–0.211 | 0.258 / 0.453 / 0.359 / 0.994 |
| Qwen2.5-3B | 0.176–0.197 | 0.016–0.039 | 0.298 / 0.264 / 0.078 / 0.995 |
| Qwen2.5-7B | 0.355–0.414 | 0.023–0.031 | 0.245 / 0.467 / 0.039 / 0.996 |
| Qwen2.5-14B | 0.350–0.420 | 0.031–0.039 | 0.289 / 0.482 / 0.047 / 0.998 |
| Llama-3.2-1B | 0.365–0.406 | 0.008–0.023 | 0.274 / 0.447 / 0.031 / 0.996 |
| Llama-3.2-3B | 0.387–0.441 | 0.023–0.031 | 0.323 / 0.453 / 0.039 / 0.998 |
| Meta-Llama-3-8B | 0.178–0.215 | 0.016–0.016 | 0.326 / 0.246 / 0.016 / 1.000 |
| Llama-3.1-8B | 0.352–0.436 | 0.000–0.008 | 0.323 / 0.422 / 0.016 / 0.998 |

**Выводы**

1. Около половины энергии DIM (0.47–0.54) приходится на остаток, который **не снимает отказ и не наводит его** ни на одной из 8 моделей (исключение — слабое наведение у Qwen2.5-1.5B: 0.16 против базы 0.08 при s = 1, 0.30 при s = 2).
2. Остаток **читает вредность так же, как целый DIM**: AUROC 0.993–1.000 против 0.995–1.000.
3. **Аблация остатка делает модель осторожнее**: bypass остатка ниже базы на всех 8 моделях (сильнее всего Qwen2.5-14B 0.38 против 0.55 и Qwen2.5-1.5B 0.10 против 0.24).
4. Часть w вне DIM (w⊥DIM) читает вредность на уровне случайного направления (AUROC 0.53–0.65 против 0.59–0.92), это write-направление без read.
5. С w только по 20 RDO картина та же, кроме Qwen2.5-1.5B: там остаток снимает отказ (0.45 против базы 0.24) — компонента без DIM не забирает всю причинную часть. Основной вариант — w по всем 24 векторам.

Ограничения: прокси без судьи; одна точка на модель (induce на слое DIM, read в точке DIM); RDO семи моделей — один прогон boroday с параметрами `rdo.py` по умолчанию. Исходники: `shared-refusal-component/results/<модель>_salad-o1o2o5o6/{decompose/decompose.json, reports/decompose.md}` (локально и на сервере в `~/f.zakharov/shared-refusal-component/results/`).

## Где лежат исходники

Пути на сервере `peaceful-matsumoto`; «локально» — относительно `~/Desktop/disser/`. Локально (в `shared-refusal-component/results/boroday/`) скопированы только `reports/`, `manifest.json` и конфиги; векторы (`dim/`, `rdo/`, `component/`) и логи остались на сервере.

| модель | сервер | локально | отчёты | конфиг |
|---|---|---|---|---|
| Qwen2.5-1.5B | `~/l.boroday/shared-refusal-component/results/Qwen2.5-1.5B-Instruct` | `shared-refusal-component/results/boroday/Qwen2.5-1.5B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/qwen2.5-1.5b.yaml` |
| Qwen2.5-3B (свой) | `~/f.zakharov/shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6` | `shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6` | `component-all.md`, `component-rdo_only.md`, `analysis-all.md`, `analysis-rdo_only.md`, `ablation.md` | `shared-refusal-component/configs/qwen2.5-3b_salad-o1o2o5o6.yaml` |
| Qwen2.5-7B | `~/l.boroday/shared-refusal-component/results/Qwen2.5-7B-Instruct` | `shared-refusal-component/results/boroday/Qwen2.5-7B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/qwen2.5-7b.yaml` |
| Qwen2.5-14B | `~/l.boroday/shared-refusal-component/results/Qwen2.5-14B-Instruct` | `shared-refusal-component/results/boroday/Qwen2.5-14B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/qwen2.5-14b.yaml` |
| Llama-3.2-1B | `~/l.boroday/shared-refusal-component/results/Llama-3.2-1B-Instruct` | `shared-refusal-component/results/boroday/Llama-3.2-1B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/llama3.2-1b.yaml` |
| Llama-3.2-3B | `~/l.boroday/shared-refusal-component/results/Llama-3.2-3B-Instruct` | `shared-refusal-component/results/boroday/Llama-3.2-3B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/llama3.2-3b.yaml` |
| Meta-Llama-3-8B | `~/l.boroday/shared-refusal-component/results/Meta-Llama-3-8B-Instruct` | `shared-refusal-component/results/boroday/Meta-Llama-3-8B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/llama3-8b.yaml` |
| Llama-3.1-8B | `~/l.boroday/shared-refusal-component/results/Llama-3.1-8B-Instruct` | `shared-refusal-component/results/boroday/Llama-3.1-8B-Instruct` | `component.md`, `component_rdo.md`, `energy.md`, `energy_rdo.md`, `ablation.md`, `ablation_rdo.md`, `read_probe.md` | `shared-refusal-component/results/boroday/configs/llama3.1-8b.yaml` |

Код boroday: `~/l.boroday/shared-refusal-component` (GitHub BorodAILA/shared-refusal-component; ветки `feat/llama-scale`, `feat/llama3-8b`, `feat/qwen2.5-7b`, `feat/qwen2.5-scale`, `feat/read-probe`), логи — `~/l.boroday/shared-refusal-component/logs/`. Исправление шаблона Llama-3 в форке — ветка `fix/llama3-template-quote` в `~/l.boroday/geometry-of-refusal`. В `configs/` boroday есть `qwen2.5-3b.yaml`, но результатов для 3B у boroday нет.

Таблицы 1–3 собраны скриптом [scripts/summarize.py](../scripts/summarize.py), раздел 4 — [scripts/decompose_table.py](../scripts/decompose_table.py) из `decompose.json`; числа — 3 знака.
