#!/usr/bin/env bash
# Шаг 2 — общая ось СРАЗУ ПО ВСЕМ прогонам (ступени × сиды), с контролем общего старта.
# CPU, GPU НЕ нужен, модель не грузится. Читает results/cones/, results/dim/ и оси ступеней
# из results/common_axis/ (их пишет шаг 1 — запускать ПОСЛЕ него).
# Пишет reports/global-axis.md и results/common_axis/global/pooled.pt.
#
# Использование:
#   bash experiments/common_axis/run/2_global_axis.sh
#   KMAP="theft=3,illegal_activities=3,malicious_use=3,all=7" bash .../2_global_axis.sh
#
# Env: FORK, MODEL (Qwen2.5-3B-Instruct), NNULL (500), SEED (21), THREADS (8),
#      KMAP (какую k брать у каждой ступени; у 'all' общих с остальными k нет).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Машина общая и 224-ядерная, матрицы маленькие — ставим ДО импорта torch.
export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

MODEL="${MODEL:-Qwen2.5-3B-Instruct}"
NNULL="${NNULL:-500}"
SEED="${SEED:-21}"
KMAP="${KMAP:-theft=5,illegal_activities=5,malicious_use=5,all=7}"
BASE="$FORK/results"
OUT="$EXP/reports/global-axis.md"

[ -d "$BASE/cones" ] || { echo "нет каталога конусов: $BASE/cones" >&2; exit 1; }
[ -d "$BASE/common_axis" ] || echo "предупреждение: нет $BASE/common_axis — сначала шаг 1" >&2
mkdir -p "$EXP/reports"

echo "base=$BASE  KMAP='$KMAP'  NNULL=$NNULL  out=$OUT"
cd "$FORK"
uv run python "$EXP/scripts/global_axis.py" \
  --base "$BASE" --model "$MODEL" --k_map "$KMAP" --n_null "$NNULL" --seed "$SEED" \
  --threads "$THREADS" --out "$OUT"
