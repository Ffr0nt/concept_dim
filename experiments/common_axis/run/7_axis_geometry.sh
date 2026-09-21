#!/usr/bin/env bash
# Шаг 8 — геометрия общей оси относительно DIM. Три стадии, задаются STAGE:
#   static  CPU, модель не грузится: масштаб mean-diff на оси, критерий оси q(v) на DIM,
#           энергия DIM в топ-m подпространстве P. Пишет reports/geometry-static.md.
#   cache   GPU, ЗАПУСКАЕТ ПОЛЬЗОВАТЕЛЬ: один форвард-проход без вмешательства, кладёт
#           проекции активаций в results/common_axis/proj-<rung>.pt (~8 МБ на ступень).
#   acts    CPU: распределения проекций, ковариационный тест, удаляемая энергия против
#           измеренного KL_ret. Пишет reports/geometry-<rung>.md.
#   all     static + cache + acts (нужен GPU).
#
# Использование:
#   bash experiments/common_axis/run/7_axis_geometry.sh                      # static
#   GPU=0 STAGE=all bash experiments/common_axis/run/7_axis_geometry.sh      # всё
#   GPU=0 STAGE=cache RUNGS="all theft" bash .../7_axis_geometry.sh
#   STAGE=acts RUNGS=all bash .../7_axis_geometry.sh                         # по готовому кэшу
#
# Env: STAGE (static), GPU (0), MODEL, RUNGS (все четыре), NNULL (2000), SEED (21),
#      THREADS (8), FORK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

export THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

STAGE="${STAGE:-static}"
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_ID="${MODEL##*/}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use all}"
NNULL="${NNULL:-2000}"
SEED="${SEED:-21}"
BASE="$FORK/results"

cd "$FORK"
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-/home/jovyan/.cache/huggingface/hub}"

[ -f "$BASE/common_axis/global/pooled.pt" ] || {
  echo "нет $BASE/common_axis/global/pooled.pt — сначала CPU-шаги 1-2" >&2; exit 1; }
mkdir -p "$EXP/reports"

if [ "$STAGE" = "static" ] || [ "$STAGE" = "all" ]; then
  echo "=== §8 static: геометрия по сохранённым артефактам (CPU) ==="
  uv run python "$EXP/scripts/axis_geometry.py" --stage static \
    --base "$BASE" --model "$MODEL_ID" --n_null "$NNULL" --seed "$SEED" \
    --threads "$THREADS" --out "$EXP/reports/geometry-static.md"
fi

if [ "$STAGE" = "cache" ] || [ "$STAGE" = "all" ]; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed -n "$((GPU+1))p"
  for r in $RUNGS; do
    echo "=== §8 cache: форвард-проход, $r (GPU $GPU) ==="
    REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
      uv run python "$EXP/scripts/cache_projections.py" \
        --model "$MODEL" --seed "$SEED" \
        --out "$BASE/common_axis/proj-$r.pt"
  done
fi

if [ "$STAGE" = "acts" ] || [ "$STAGE" = "all" ]; then
  for r in $RUNGS; do
    [ -f "$BASE/common_axis/proj-$r.pt" ] || { echo "нет кэша proj-$r.pt — пропуск" >&2; continue; }
    echo "=== §8 acts: анализ проекций, $r (CPU) ==="
    uv run python "$EXP/scripts/axis_geometry.py" --stage acts --rung "$r" \
      --base "$BASE" --model "$MODEL_ID" --threads "$THREADS" \
      --out "$EXP/reports/geometry-$r.md"
  done
fi
echo "готово"
