#!/usr/bin/env bash
# Гейт-конусы: обучить конусы СТРОГО заданных размерностей С НУЛЯ (независимо, без вложенной
# лестницы 1..d-1) на нескольких сидах, затем eval. Использует FROM_SCRATCH=1 (rdo.py: все оси
# случайны, resume отключён). Пишет cones/<RUNG>/seed_<s>/dim_<d>.pt + eval_test.json.
# Требует готовых data/<RUNG>_splits/ (1_build_splits.sh) и DIM (0_dim.sh). Запускать НА СЕРВЕРЕ (GPU).
#
# Использование (широкая ступень, dims 7/8/9, 3 сида):
#   SEEDS="21 3 7" RUNG="all" DIMS="7 8 9" GPU=3 bash experiments/mvp/run/from_scratch_dims.sh
#
# Env: SEEDS ("21 3 7"), RUNG (all), DIMS ("7 8 9"), SAMPLES (256, MC-сэмплы eval),
#      EVALMAX (9, верх диапазона eval — недостающие dim пропускаются), GPU (3), MODEL.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"

SEEDS="${SEEDS:-21 3 7}"
RUNG="${RUNG:-all}"
DIMS="${DIMS:-7 8 9}"
SAMPLES="${SAMPLES:-256}"
EVALMAX="${EVALMAX:-9}"
GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

echo "RUNG=$RUNG  SEEDS='$SEEDS'  DIMS='$DIMS' (с нуля)  eval@$SAMPLES до d$EVALMAX  GPU=$GPU"
for s in $SEEDS; do
  for d in $DIMS; do
    echo "###### $RUNG seed $s : from-scratch dim $d ######"
    FROM_SCRATCH=1 SEED="$s" CONE_TAG="seed_$s" MINDIM="$d" MAXDIM="$d" GPU="$GPU" \
      RUNGS="$RUNG" MODEL="$MODEL" bash "$HERE/3b_cone_sweep.sh"
  done
  echo "###### $RUNG seed $s : eval dim(s) '$DIMS' @ $SAMPLES ######"
  SEED="$s" CONE_TAG="seed_$s" MAXDIM="$EVALMAX" SAMPLES="$SAMPLES" GPU="$GPU" \
    RUNGS="$RUNG" MODEL="$MODEL" bash "$HERE/4_eval_cones.sh"
done
echo "готово: from-scratch конусы '$DIMS' в cones/$RUNG/seed_<s>/ + eval_test.json (dims $DIMS)"
