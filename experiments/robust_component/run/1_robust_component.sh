#!/usr/bin/env bash
# Шаг 1 — робастные и взвешенные оценки общей компоненты против обычного PCA (split-half).
# CPU, GPU НЕ нужен, модель не грузится: читает directions.pt запуска shared-refusal-component
# (4 DIM + 20 RDO). Пишет reports/robust-component.md (числа) и
# <fork>/results/robust_component/{axes_<pool>.pt, robust_component.json}.
#
# Использование:
#   bash experiments/robust_component/run/1_robust_component.sh
#   POOLS=rdo NSAMP=300 bash experiments/robust_component/run/1_robust_component.sh
#
# Env: SRC (../shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6/directions/directions.pt
#      относительно concept_dim), FORK (../geometry-of-refusal), POOLS ("all rdo"),
#      NSAMP (1000), SEED (21), THREADS (8).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Машина общая, матрицы крошечные: ограничиваем потоки ДО импорта torch/numpy.
export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"
SRC="${SRC:-$ROOT/../shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6/directions/directions.pt}"
POOLS="${POOLS:-all rdo}"

cd "$ROOT"
read -ra POOL_LIST <<< "$POOLS"
uv run python "$EXP/scripts/robust_component.py" \
  --src "$SRC" --out "$FORK/results/robust_component" --report "$EXP/reports/robust-component.md" \
  --pools "${POOL_LIST[@]}" --n-samples "${NSAMP:-1000}" --seed "${SEED:-21}"
