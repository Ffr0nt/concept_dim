#!/usr/bin/env bash
# Этап 2a — снятие активаций последнего токена на слое L (GPU).
# Запускать НА СЕРВЕРЕ. Можно из любой директории — скрипт сам перейдёт в корень concept_dim.
#
# Использование:
#   bash experiments/mvp/run/2a_extract_activations.sh
#   GPU=2 LAYER=18 bash experiments/mvp/run/2a_extract_activations.sh
#
# Переменные окружения (все опциональны):
#   GPU    — индекс GPU (по умолчанию 3)
#   LAYER  — слой L (по умолчанию 18)
#   MODEL  — модель (по умолчанию Qwen/Qwen2.5-3B-Instruct)
#   RUNGS  — список ступеней через пробел (по умолчанию все три)
set -euo pipefail

cd "$(dirname "$0")/../../.."   # -> корень concept_dim

GPU="${GPU:-3}"
LAYER="${LAYER:-18}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"

echo "модель=$MODEL  GPU=$GPU  слой=$LAYER"
for r in $RUNGS; do
  echo "=== $r ==="
  CUDA_VISIBLE_DEVICES="$GPU" uv run python experiments/mvp/scripts/extract_activations.py \
    --model "$MODEL" --splits "$r" --layer "$LAYER"
done
echo "готово: дампы в experiments/mvp/artifacts/acts_<rung>_L${LAYER}.pt"
