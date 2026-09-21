#!/usr/bin/env bash
# Шаг 1 — спектр среднего проектора по сидам: есть ли общая ось у конусов одной ступени.
# CPU, GPU НЕ нужен, модель не грузится: читает results/cones/<rung>/seed_<s>/dim_<d>.pt
# и results/dim/<rung>/<model>/. scipy не нужен. Пишет reports/common-axis.md.
#
# Использование:
#   bash experiments/common_axis/run/1_common_axis.sh
#   NNULL=2000 DIMS="3 5" bash experiments/common_axis/run/1_common_axis.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), MODEL (Qwen2.5-3B-Instruct),
#      NNULL (500), DIMS (пусто = все k, общие для трёх сидов), SEED (21).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

MODEL="${MODEL:-Qwen2.5-3B-Instruct}"
NNULL="${NNULL:-500}"
SEED="${SEED:-21}"
BASE="$FORK/results"
OUT="$EXP/reports/common-axis.md"

[ -d "$BASE/cones" ] || { echo "нет каталога конусов: $BASE/cones" >&2; exit 1; }
[ -d "$BASE/dim" ] || echo "предупреждение: нет $BASE/dim — секции сравнения с DIM не будет" >&2
mkdir -p "$EXP/reports"

EXTRA=()
[ -n "${DIMS:-}" ] && EXTRA=(--dims $DIMS)

echo "base=$BASE  model=$MODEL  NNULL=$NNULL  DIMS='${DIMS:-все}'  out=$OUT"
cd "$FORK"
uv run python "$EXP/scripts/common_axis.py" \
  --base "$BASE" --model "$MODEL" --n_null "$NNULL" --seed "$SEED" "${EXTRA[@]}" --out "$OUT"
