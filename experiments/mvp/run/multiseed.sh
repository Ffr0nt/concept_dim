#!/usr/bin/env bash
# Мультисид: для каждого сида обучает конусы (3b) и оценивает (4) в
#   <fork>/results/cones/<rung>/seed_<s>/{dim_*.pt, eval_test.json}
# Конусы сохраняются per-seed => результат воспроизводим. Усреднение — aggregate_seeds.py.
# Запускать НА СЕРВЕРЕ (GPU). ТЯЖЕЛО: N сидов × (обучение конусов + eval).
#
# Использование:
#   SEEDS="21 7 13" GPU=2 bash experiments/mvp/run/multiseed.sh
#   SEEDS="21 7 13 42 100" RUNGS="theft" MAXDIM=8 SAMPLES=256 GPU=2 bash experiments/mvp/run/multiseed.sh
#
# Env: SEEDS ("21 7 13"), GPU (3), RUNGS (все три), MAXDIM (5), SAMPLES (256),
#      NSAMPLE (обуч. сэмплы конуса, 8), MODEL.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"

SEEDS="${SEEDS:-21 3 7}"   # предпочтительные сиды проекта (убыв.): 21 3 7 0
GPU="${GPU:-3}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"
MAXDIM="${MAXDIM:-5}"
SAMPLES="${SAMPLES:-256}"
NSAMPLE="${NSAMPLE:-8}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

echo "СИДЫ=$SEEDS  GPU=$GPU  RUNGS='$RUNGS'  MAXDIM=$MAXDIM  SAMPLES=$SAMPLES  NSAMPLE=$NSAMPLE"
for s in $SEEDS; do
  echo "################## SEED $s ##################"
  SEED="$s" CONE_TAG="seed_$s" BEST_OF_K=1 GPU="$GPU" RUNGS="$RUNGS" MAXDIM="$MAXDIM" \
    NSAMPLE="$NSAMPLE" MODEL="$MODEL" bash "$HERE/3b_cone_sweep.sh"
  SEED="$s" CONE_TAG="seed_$s" GPU="$GPU" RUNGS="$RUNGS" MAXDIM="$MAXDIM" \
    SAMPLES="$SAMPLES" MODEL="$MODEL" bash "$HERE/4_eval_cones.sh"
done
echo "готово: конусы+eval по сидам в results/cones/<rung>/seed_<s>/  →  агрегировать aggregate_seeds.py"
