#!/usr/bin/env bash
# Шаг 7 (§8) — read против write: детектирует ли остаток DIM вредность, будучи каузально
# пустым. GPU, только forward, ничего не обучается (порог подбирается перебором).
# Пишет reports/read-probe-<rung>.md и results/common_axis/read-probe-<rung>.json.
#
# Использование:
#   GPU=0 bash experiments/common_axis/run/7_read_probe.sh
#   GPU=0 RUNGS="all theft" NMAX=500 bash .../7_read_probe.sh
#
# Env: GPU (0), MODEL, RUNGS (all), NMAX (0 = весь пул train+val+test), BS (16), FORK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-all}"
NMAX="${NMAX:-0}"
BS="${BS:-16}"

cd "$FORK"
export SAVE_DIR="./results"
export HUGGINGFACE_CACHE_DIR="/home/jovyan/.cache/huggingface/hub"

[ -f "results/common_axis/global/pooled.pt" ] || {
  echo "нет results/common_axis/global/pooled.pt — сначала CPU-шаги 1-2" >&2; exit 1; }

echo "модель=$MODEL  GPU=$GPU  RUNGS='$RUNGS'  NMAX=$NMAX (0 = весь пул)"
for r in $RUNGS; do
  echo "=== §8 read-пробник: $r ==="
  REFUSAL_SPLITS="$r" DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    uv run python "$EXP/scripts/read_probe.py" \
      --model "$MODEL" --batch_size "$BS" --n_max "$NMAX" \
      --out "$EXP/reports/read-probe-$r.md" \
      --json_out "$FORK/results/common_axis/read-probe-$r.json"
done
echo "готово"
