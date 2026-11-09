#!/usr/bin/env bash
# Обучение RDO на объединении четырёх доменов (пересечение внутри лосса), на сервере, на GPU.
# Команду rdo.py строит код пайплайна из конфига прогона — гиперпараметры, шаблон чата и
# refusal-токены заведомо те же, что в остальных прогонах.
#
#   GPU=4 bash run/train_pooled.sh                        # сиды 21 3 7 0
#   GPU=4 ARGS="--build-only" bash run/train_pooled.sh    # только собрать данные
#
# Env: GPU (обязателен), ARGS, HOST, REPO.
set -euo pipefail
: "${GPU:?укажи GPU=<индекс>}"
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ssh "$HOST" "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface} WANDB_MODE=disabled \
  uv run python - ${ARGS:-}" < "$(dirname "$0")/train_pooled.py"
