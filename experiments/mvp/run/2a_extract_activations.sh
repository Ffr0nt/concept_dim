#!/usr/bin/env bash
# Этап 2a — снятие активаций последнего токена по нескольким слоям (GPU).
# Запускать НА СЕРВЕРЕ. Можно из любой директории — скрипт сам перейдёт в корень concept_dim.
#
# Использование:
#   bash experiments/mvp/run/2a_extract_activations.sh
#   GPU=2 LAYERS="12 18 24" bash experiments/mvp/run/2a_extract_activations.sh
#
# Переменные окружения (все опциональны):
#   GPU     — индекс GPU (по умолчанию 3)
#   LAYERS  — список слоёв через пробел (по умолчанию "6 12 18 24 30")
#   MODEL   — модель (по умолчанию Qwen/Qwen2.5-3B-Instruct)
#   RUNGS   — список ступеней через пробел (по умолчанию все три)
set -euo pipefail

# uv установлен в ~/.cargo/bin (или ~/.local/bin), но интерактивный шелл может их не видеть
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/../../.."   # -> корень concept_dim

GPU="${GPU:-3}"
LAYERS="${LAYERS:-6 12 18 24 30}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"

echo "модель=$MODEL  GPU=$GPU  слои=$LAYERS"
for r in $RUNGS; do
  echo "=== $r ==="
  CUDA_VISIBLE_DEVICES="$GPU" uv run python experiments/mvp/scripts/extract_activations.py \
    --model "$MODEL" --splits "$r" --layers $LAYERS
done
echo "готово: дампы в experiments/mvp/artifacts/acts_<rung>_L<layer>.pt"
