#!/usr/bin/env bash
# Шаг 6 (§7) — наведение отказа: ось это refusal-направление или коррелят harmfulness? GPU.
# Добавляет h <- h + alpha*w на БЕЗОБИДНЫХ промптах в слое пика проекции и меряет
# наведённый отказ; плюс кросс-нейтрализация (наводим одним, аблируем другое).
# Пишет reports/induce-<rung>.md и results/common_axis/induce-<rung>.json.
#
# Использование:
#   GPU=0 bash experiments/common_axis/run/6_induce_axes.sh
#   GPU=0 RUNGS="all theft" SCALES="1 2 4 8" bash .../6_induce_axes.sh
#
# Env: GPU (0), MODEL, RUNGS (all), SCALES (0.5 1 2 4, в кратных ||DIM||), LAYER (пик),
#      BS (16), FORK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-all}"
SCALES="${SCALES:-0.5 1 2 4}"
BS="${BS:-16}"

cd "$FORK"
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"

[ -f "results/common_axis/global/pooled.pt" ] || {
  echo "нет results/common_axis/global/pooled.pt — сначала CPU-шаги 1-2" >&2; exit 1; }

EXTRA=()
[ -n "${LAYER:-}" ] && EXTRA=(--layer "$LAYER")

echo "модель=$MODEL  GPU=$GPU  RUNGS='$RUNGS'  SCALES='$SCALES'"
for r in $RUNGS; do
  echo "=== §7 наведение: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "$EXP/scripts/induce_axes.py" \
      --model "$MODEL" --batch_size "$BS" --scales $SCALES "${EXTRA[@]}" \
      --out "$EXP/reports/induce-$r.md" \
      --json_out "$FORK/results/common_axis/induce-$r.json"
done
echo "готово"
