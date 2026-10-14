#!/usr/bin/env bash
# Шаг 2 — геометрия (CPU): base, no_add, no_abl, invariant — общие компоненты, согласованность
# сидов, пары с base, косинусы между компонентами. Пишет reports/geometry.md,
# <fork>/results/bypass_only/{geometry.json, components.pt}.
#
# Использование:
#   bash experiments/bypass_only/run/2_geometry.sh
#
# Env: NNULL (1000), FORK, SRC, OUT, THREADS (8).
set -euo pipefail
export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
source "$(dirname "$0")/_env.sh"
export CUDA_VISIBLE_DEVICES=""
py geometry.py --out "$OUT" --report "$EXP/reports/geometry.md" --n-null "${NNULL:-1000}"
