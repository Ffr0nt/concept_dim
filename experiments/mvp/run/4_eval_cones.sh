#!/usr/bin/env bash
# Этап 4 — оценка конусов на held-out TEST (GPU).
# Требует посчитанных конусов (run/3b_cone_sweep.sh) и DIM (run/0_dim.sh).
# Пишет per-rung результат в <fork>/results/cones/<rung>/eval_test.json
# Запускать НА СЕРВЕРЕ.
#
# Использование:
#   bash experiments/mvp/run/4_eval_cones.sh
#   GPU=2 RUNGS="theft" bash experiments/mvp/run/4_eval_cones.sh
#
# Env: GPU (3), MODEL, RUNGS (все три), MAXDIM (8), FORK
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # concept_dim
PY="$ROOT/.venv/bin/python"                       # venv concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"
MAXDIM="${MAXDIM:-8}"

cd "$FORK"   # data/, results/, .env — относительно cwd
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"
echo "модель=$MODEL  GPU=$GPU  MAXDIM=$MAXDIM  fork=$FORK"
for r in $RUNGS; do
  echo "=== eval cones: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" "$ROOT/experiments/mvp/scripts/eval_cones.py" --model "$MODEL" --max_dim "$MAXDIM"
done
echo "готово: per-rung результаты в $FORK/results/cones/<rung>/eval_test.json"
