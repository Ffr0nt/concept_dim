#!/usr/bin/env bash
# Шаг 4 (§6) — каузальный тест общих осей: частичная аблация при выравненном KL_ret. GPU.
# ЗАПУСКАЕТ ПОЛЬЗОВАТЕЛЬ. Требует CPU-шагов 1-3 (оси в results/common_axis/) и DIM.
# Пишет reports/ablation-<rung>.md и results/common_axis/ablation-<rung>.json.
#
# Использование (одна карта, ≤2 по правилам проекта):
#   GPU=3 bash experiments/common_axis/run/4_ablate_axes.sh
#   GPU=3 RUNGS="all malicious_use" ALPHAS="0.5 1.0" bash .../4_ablate_axes.sh
#
# Фоном, чтобы не зависеть от ssh-сессии:
#   GPU=3 nohup bash experiments/common_axis/run/4_ablate_axes.sh > /tmp/abl.log 2>&1 &
#
# Env: GPU (3), MODEL, RUNGS (all), ALPHAS (0.25 0.5 0.75 1.0), NRAND (3), MAXM (4),
#      BS (16), FORK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-all}"
ALPHAS="${ALPHAS:-0.25 0.5 0.75 1.0}"
NRAND="${NRAND:-3}"
MAXM="${MAXM:-4}"
BS="${BS:-16}"

cd "$FORK"                                             # data/, results/ — относительно cwd
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"

[ -f "results/common_axis/global/pooled.pt" ] || {
  echo "нет results/common_axis/global/pooled.pt — сначала CPU-шаги 1-2" >&2; exit 1; }

echo "модель=$MODEL  GPU=$GPU  RUNGS='$RUNGS'  ALPHAS='$ALPHAS'  BS=$BS"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed -n "$((GPU+1))p"

for r in $RUNGS; do
  echo "=== §6 аблация: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "$EXP/scripts/ablate_axes.py" \
      --model "$MODEL" --batch_size "$BS" --alphas $ALPHAS \
      --n_rand "$NRAND" --max_m "$MAXM" \
      --out "$EXP/reports/ablation-$r.md" \
      --json_out "$FORK/results/common_axis/ablation-$r.json"
done
echo "готово"
