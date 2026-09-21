#!/usr/bin/env bash
# Шаг 5 (§6б) — остаток DIM после вычитания общей компоненты. GPU.
# Вопрос: сохраняется ли отказ, если из DIM убрать нашу общую ось — то есть вся ли сила
# DIM сидела в ней. Плюс обратный контроль: работает ли ось без своей DIM-овской части.
# Пишет reports/residual-<rung>.md и results/common_axis/residual-<rung>.json.
#
# Использование:
#   GPU=0 bash experiments/common_axis/run/5_dim_residual.sh
#   GPU=0 RUNGS="all theft" ALPHAS="0.5 1.0" bash .../5_dim_residual.sh
#
# Env: GPU (0), MODEL, RUNGS (all), ALPHAS (0.25 0.5 0.75 1.0), BS (16), FORK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-all}"
ALPHAS="${ALPHAS:-0.25 0.5 0.75 1.0}"
BS="${BS:-16}"

cd "$FORK"
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"

[ -f "results/common_axis/global/pooled.pt" ] || {
  echo "нет results/common_axis/global/pooled.pt — сначала CPU-шаги 1-2" >&2; exit 1; }

echo "модель=$MODEL  GPU=$GPU  RUNGS='$RUNGS'  ALPHAS='$ALPHAS'"
for r in $RUNGS; do
  echo "=== §6б остаток DIM: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "$EXP/scripts/ablate_axes.py" --residual \
      --model "$MODEL" --batch_size "$BS" --alphas $ALPHAS \
      --out "$EXP/reports/residual-$r.md" \
      --json_out "$FORK/results/common_axis/residual-$r.json"
done
echo "готово"
