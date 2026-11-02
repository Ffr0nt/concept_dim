# Cross Model — общая компонента отказа на других моделях

## Коротко

Сводка пайплайна shared-refusal-component (4 DIM + 20 одномерных RDO, SALAD-Bench O1, O2, O5,
O6) по 8 моделям: Qwen2.5 1.5B / 3B / 7B / 14B и Llama 3.2-1B / 3.2-3B / 3-8B / 3.1-8B.
Qwen2.5-3B — наш запуск `qwen2.5-3b_salad-o1o2o5o6`, остальные 7 посчитаны boroday в форке
BorodAILA/shared-refusal-component (`main`, `d72f0a5`). Сплиты у всех 8 моделей совпадают
побайтно (sha256 в `manifest.json`).

**Общая компонента есть у всех.** λ1 в 8–12 раз выше нуля (p95 ≈ 0.05): у Qwen 0.40–0.43
на всех размерах, у Llama выше и растёт с размером (0.45 → 0.58). Доля энергии DIM вдоль w —
0.44–0.53, по w только из RDO — 0.24–0.32. Аблация w даёт ASR 0.96–0.98 при KL ниже, чем у
DIM. Сиды RDO без общей части (cos(w, RDO) < 0.3) есть только у Qwen, по 2–3 на модель.

## Содержание

| файл | что внутри |
|---|---|
| [reports/results.md](reports/results.md) | сводные таблицы: компонента, аблация, read-probe, **разложение DIM на 8 моделях (раздел 4)**; где лежат исходники |
| [scripts/summarize.py](scripts/summarize.py) | парсер markdown-отчётов в JSON, по которому собраны таблицы |

Модели в этом репозитории не запускаются. Отчёты boroday скопированы локально в
`shared-refusal-component/results/boroday/` (в git не хранятся), векторы и логи — на сервере
в `~/l.boroday/shared-refusal-component/results/<модель>/`.
