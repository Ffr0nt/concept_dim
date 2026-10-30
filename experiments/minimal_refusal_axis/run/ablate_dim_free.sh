#!/usr/bin/env bash
# Шаги 1-2: снимает ли отказ направление с нулевой долей DIM. На сервере, на GPU.
#
#   bash run/ablate_dim_free.sh --save-only        # только геометрия, CPU
#   GPU=0 bash run/ablate_dim_free.sh | tee reports/dim_free_ablation.md
#
# Env: GPU (для аблации), HOST, REPO.
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
if [[ "${1:-}" == "--save-only" ]]; then
  ssh "$HOST" "cd $REPO && CUDA_VISIBLE_DEVICES= uv run python - --save-only" \
    < "$(dirname "$0")/ablate_dim_free.py"
else
  : "${GPU:?укажи GPU=<индекс> или передай --save-only}"
  ssh "$HOST" "cd $REPO && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface} \
    WANDB_MODE=disabled uv run python -" < "$(dirname "$0")/ablate_dim_free.py"
fi
