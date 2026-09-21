#!/usr/bin/env bash
# Шаг 5 — реконструкция DIM конусами: матрица концепт x ступень. CPU, GPU НЕ нужен.
# Отвечает: специфичен ли конус своему концепту, или любой конус одинаково хорошо
# восстанавливает любой DIM. Пишет stdout -> reports/dim-reconstruction.md.
#
# Использование:
#   bash experiments/learned_directions/run/5_dim_reconstruction.sh
#   DIMS="5 8" bash experiments/learned_directions/run/5_dim_reconstruction.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), MODEL (Qwen2.5-3B-Instruct),
#      DIMS (пусто = все доступные d), NNULL (1000), SEED (21).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

MODEL="${MODEL:-Qwen2.5-3B-Instruct}"
NNULL="${NNULL:-1000}"
SEED="${SEED:-21}"
BASE="$FORK/results"
OUT="$EXP/reports/dim-reconstruction.md"

[ -d "$BASE/dim" ] || { echo "нет каталога DIM: $BASE/dim" >&2; exit 1; }
[ -d "$BASE/cones" ] || { echo "нет каталога конусов: $BASE/cones" >&2; exit 1; }
mkdir -p "$EXP/reports"

EXTRA=()
[ -n "${DIMS:-}" ] && EXTRA=(--dims $DIMS)

echo "base=$BASE  model=$MODEL  NNULL=$NNULL  DIMS='${DIMS:-все}'  out=$OUT"
cd "$FORK"
uv run python "$EXP/scripts/dim_reconstruction.py" \
  --base "$BASE" --model "$MODEL" --n_null "$NNULL" --seed "$SEED" "${EXTRA[@]}" | tee "$OUT"
