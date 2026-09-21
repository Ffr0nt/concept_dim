#!/usr/bin/env bash
# Шаг 3 — общая компонента ВСЕХ направлений отказа: выученные реперы + DIM в одной
# популяции. CPU, GPU НЕ нужен. Запускать ПОСЛЕ шагов 1 и 2 (читает конусную ось из
# results/common_axis/global/pooled.pt для сверки; её не перезаписывает).
# Пишет reports/mixed-axis.md и results/common_axis/global/mixed_<точка>_<веса>.pt.
#
# Использование:
#   bash experiments/common_axis/run/3_mixed_axis.sh
#   NNULL=1000 bash experiments/common_axis/run/3_mixed_axis.sh
#
# Env: FORK, MODEL (Qwen2.5-3B-Instruct), NNULL (300), SEED (21), THREADS (8), KMAP.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

MODEL="${MODEL:-Qwen2.5-3B-Instruct}"
NNULL="${NNULL:-300}"
SEED="${SEED:-21}"
KMAP="${KMAP:-theft=5,illegal_activities=5,malicious_use=5,all=7}"
BASE="$FORK/results"
OUT="$EXP/reports/mixed-axis.md"

[ -d "$BASE/cones" ] || { echo "нет каталога конусов: $BASE/cones" >&2; exit 1; }
[ -f "$BASE/common_axis/global/pooled.pt" ] || echo "предупреждение: нет конусной оси — сначала шаг 2" >&2
mkdir -p "$EXP/reports"

echo "base=$BASE  KMAP='$KMAP'  NNULL=$NNULL  out=$OUT"
cd "$FORK"
uv run python "$EXP/scripts/mixed_axis.py" \
  --base "$BASE" --model "$MODEL" --k_map "$KMAP" --n_null "$NNULL" --seed "$SEED" \
  --threads "$THREADS" --out "$OUT"
