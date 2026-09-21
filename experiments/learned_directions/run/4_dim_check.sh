#!/usr/bin/env bash
# Шаг 4 — DIM как шумонезависимый ориентир. CPU, GPU НЕ нужен, модель не грузится:
# читает results/dim/<rung>/<model>/{direction.pt,direction_metadata.json,mean_diffs.pt}.
#
# Отвечает на два вопроса: (а) различает ли DIM концепты вообще, (б) не заражено ли
# сравнение тем, что select_direction выбирает точку (pos, layer) независимо для каждой
# ступени. Печатает в stdout; перенаправь в reports/, если нужен файл.
#
# Использование:
#   bash experiments/learned_directions/run/4_dim_check.sh
#   MODEL=Qwen2.5-3B-Instruct bash experiments/learned_directions/run/4_dim_check.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), MODEL (Qwen2.5-3B-Instruct).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

MODEL="${MODEL:-Qwen2.5-3B-Instruct}"
BASE="$FORK/results"

[ -d "$BASE/dim" ] || { echo "нет каталога DIM: $BASE/dim" >&2; exit 1; }

echo "base=$BASE  model=$MODEL"
cd "$FORK"
uv run python "$EXP/scripts/dim_check.py" --base "$BASE" --model "$MODEL"
