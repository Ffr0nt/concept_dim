#!/usr/bin/env bash
# Этап 0 — генерация DIM-направления на КАЖДУЮ ступень (GPU).
# Своё DIM на ступень: пишется в results/dim/<rung>/<model_alias>/.
# Запускать НА СЕРВЕРЕ.
#
# Использование:
#   bash experiments/mvp/run/0_dim.sh
#   GPU=2 RUNGS="theft" bash experiments/mvp/run/0_dim.sh
#
# Env (опционально): GPU (3), MODEL (Qwen/Qwen2.5-3B-Instruct), RUNGS (все три), FORK
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/../../.."   # -> корень concept_dim

GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"
FORK="${FORK:-../geometry-of-refusal}"

# env для пайплайна (run_pipeline.load_dotenv(\"..\") ничего не грузит — задаём явно)
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"

for r in $RUNGS; do
  echo "=== DIM: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python experiments/mvp/scripts/gen_dim.py --model "$MODEL" --fork "$FORK"
done
echo "готово: DIM в <fork>/results/dim/<rung>/$(basename "$MODEL")/{direction.pt, direction_metadata.json, generate_directions/mean_diffs.pt}"
