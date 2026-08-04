#!/usr/bin/env bash
# Этап 3b — перебор размерности конуса (RDO) на каждую ступень (GPU).
# Требует готового per-rung DIM (run/0_dim.sh). Пишет per-d базисы конуса в
#   <fork>/results/cones/<rung>/dim_<d>.pt
# Запускать НА СЕРВЕРЕ.
#
# Использование:
#   bash experiments/mvp/run/3b_cone_sweep.sh
#   RUNGS="theft" MAXDIM=3 bash experiments/mvp/run/3b_cone_sweep.sh   # быстрая проверка
#
# Env: GPU (3), MODEL (Qwen/Qwen2.5-3B-Instruct), RUNGS (все три), MAXDIM (8), FORK
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # concept_dim
PY="$ROOT/.venv/bin/python"                       # venv concept_dim (все зависимости + editable)
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"
MINDIM="${MINDIM:-1}"   # >1 => resume: подсев базиса из dim_{MINDIM-1}.pt
MAXDIM="${MAXDIM:-8}"

cd "$FORK"   # rdo.py читает ./data, ./results, .env относительно cwd
echo "модель=$MODEL  GPU=$GPU  MAXDIM=$MAXDIM  fork=$FORK"
for r in $RUNGS; do
  echo "=== cone sweep: $r (dim 1..$MAXDIM) ==="
  DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" rdo.py --train_cone --model "$MODEL" --splits "$r" \
      --min_cone_dim "$MINDIM" --max_cone_dim "$MAXDIM"
done
echo "готово: per-d конусы в $FORK/results/cones/<rung>/dim_<d>.pt"
