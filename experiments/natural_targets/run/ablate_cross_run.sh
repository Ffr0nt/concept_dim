#!/usr/bin/env bash
# Аблация общей части нашей компоненты (без DIM) и базовой — на сервере, на GPU.
#
#   GPU=0 bash run/ablate_cross_run.sh | tee reports/cross_run_ablation.md
#
# Env: GPU (обязателен), HOST, REPO.
set -euo pipefail
: "${GPU:?укажи GPU=<индекс>}"
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ssh "$HOST" "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface} \
  WANDB_MODE=disabled uv run python -" < "$(dirname "$0")/ablate_cross_run.py"
