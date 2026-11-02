#!/usr/bin/env bash
# Шаг 2 — геометрия вариантов (CPU, модель не грузится): общая компонента каждого варианта,
# согласованность сидов, парное сравнение с base, контроль старта, доля в DIM.
# Берёт варианты, у которых готовы все сиды. Пишет reports/geometry.md,
# <fork>/results/loss_terms/{geometry.json, components.pt} (components.pt читает шаг 3).
#
# Использование:
#   bash experiments/loss_terms/run/2_geometry.sh
#
# Env: NNULL (1000), FORK, SRC, OUT, THREADS (8).
set -euo pipefail
export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
source "$(dirname "$0")/_env.sh"
export CUDA_VISIBLE_DEVICES=""

py geometry.py --out "$OUT" --report "$EXP/reports/geometry.md" --n-null "${NNULL:-1000}"
