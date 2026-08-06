#!/usr/bin/env bash
# Досчёт конуса до большей размерности: resume с dim_{MINDIM-1} и ПЕРЕоценка всех dims.
# Для каждого сида: 3b (MINDIM..MAXDIM, подсев из dim_{MINDIM-1}.pt) → 4 (eval 1..MAXDIM @ SAMPLES).
# Конусы/eval пишутся per-seed в cones/<RUNG>/seed_<s>/. Запускать НА СЕРВЕРЕ (GPU).
#
# Использование (широкий концепт до 8, все сиды, чистые 256 сэмплов):
#   SEEDS="21 3 7" RUNG="malicious_use" MINDIM=6 MAXDIM=8 SAMPLES=256 GPU=3 \
#     bash experiments/mvp/run/extend_dim.sh
#
# Env: SEEDS ("21 3 7"), RUNG (malicious_use), MINDIM (6), MAXDIM (8),
#      SAMPLES (256), GPU (3), MODEL.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"

SEEDS="${SEEDS:-21 3 7}"
RUNG="${RUNG:-malicious_use}"
MINDIM="${MINDIM:-6}"   # >1 => resume: подсев базиса из dim_{MINDIM-1}.pt
MAXDIM="${MAXDIM:-8}"
SAMPLES="${SAMPLES:-256}"
GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

echo "RUNG=$RUNG  SEEDS='$SEEDS'  resume dim $MINDIM..$MAXDIM  eval@$SAMPLES  GPU=$GPU"
for s in $SEEDS; do
  echo "###### $RUNG seed $s : resume dim $MINDIM..$MAXDIM ######"
  SEED="$s" CONE_TAG="seed_$s" MINDIM="$MINDIM" MAXDIM="$MAXDIM" GPU="$GPU" \
    RUNGS="$RUNG" MODEL="$MODEL" bash "$HERE/3b_cone_sweep.sh"
  echo "###### $RUNG seed $s : re-eval dim 1..$MAXDIM @ $SAMPLES ######"
  SEED="$s" CONE_TAG="seed_$s" MAXDIM="$MAXDIM" SAMPLES="$SAMPLES" GPU="$GPU" \
    RUNGS="$RUNG" MODEL="$MODEL" bash "$HERE/4_eval_cones.sh"
done
echo "готово: $RUNG до dim $MAXDIM, eval@$SAMPLES в cones/$RUNG/seed_<s>/"
