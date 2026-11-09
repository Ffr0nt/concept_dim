#!/usr/bin/env bash
# Отбелённый DIM (LDA): снимает ли отказ дешевле обычного. На сервере, на GPU.
# GPU нужен всегда, в том числе для --save-only: ковариация считается по активациям.
#
#   GPU=4 bash run/whitened_dim.sh                 # геометрия + аблация
#   GPU=4 bash run/whitened_dim.sh --save-only     # только ковариация и геометрия
#
# Env: GPU (обязателен), HOST, REPO.
set -euo pipefail
: "${GPU:?укажи GPU=<индекс>}"
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ssh "$HOST" "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface} \
  WANDB_MODE=disabled uv run python - ${1:-}" < "$(dirname "$0")/whitened_dim.py"
