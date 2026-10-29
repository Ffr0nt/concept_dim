# Natural Targets — RDO на готовых парах, без DIM в целях

## Коротко

**Статус: код готов, прогонов не было (29.10.2026).**

Цели обучения RDO в базовом протоколе генерируются через DIM (`rdo.py:327-357`): ответ без
отказа — генерация Instruct при аблации DIM, отказ на безобидный запрос — при её прибавлении.
Поэтому сходимость RDO и DIM к общей компоненте $w$ — не независимое свидетельство. Здесь
RDO обучается на целях, где DIM не участвует вовсе: ответ без отказа взят готовым из
BeaverTails (человеческая разметка `is_safe`), слагаемое наведения отказа выключено
(`addition_lambda: 0`), `retain` — собственный ответ модели. Остальное — обычный протокол:
4 DIM + 20 RDO, общая компонента по 24 векторам, судья StrongREJECT.

Главное число: $|\cos|$ новой компоненты с $w$ базового прогона и $|\cos|$ каждого RDO с ним
против LOO-распределения обычных RDO (среднее ≈ 0.58).

Постановка, данные и границы — [idea.md](idea.md).

## Содержание

| файл | что внутри |
|---|---|
| [idea.md](idea.md) | постановка, источник данных и домены, что меряем, исходы, границы |
| [reports/results.md](reports/results.md) | разбор результатов (пока пусто) |

Код — в `shared-refusal-component` (прогон `qwen2.5-3b_beavertails-b1b5b6bv`), а не здесь:
нужен полный протокол, а он уже есть в пайплайне. Что именно добавлено — в
[idea.md](idea.md#реализация). Артефакты прогона — на сервере, в git не хранятся.

## Порядок прогонов

```
cd shared-refusal-component
STEPS=00 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh            # сплиты, CPU
STEPS=01 GPU=0 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh      # цели без DIM, DIM, 20 RDO
STEPS="02 04 05" bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh    # компонента, анализ, сходимость
STEPS=03 GPU=0 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh      # аблация
STEPS=06 GPU=0 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh      # генерация StrongREJECT
STEPS=07 bash launch/qwen2.5-3b_beavertails-b1b5b6bv.sh            # судья, с Mac
```
