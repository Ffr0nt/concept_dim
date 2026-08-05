#!/usr/bin/env bash
# Этап 3b — перебор размерности конуса (RDO) на каждую ступень (GPU).
# Требует готового per-rung DIM (run/0_dim.sh). Пишет per-d базисы конуса в
#   <fork>/results/cones/<rung>/dim_<d>.pt
# Запускать НА СЕРВЕРЕ.
#
# Использование:
#   bash experiments/mvp/run/3b_cone_sweep.sh
#   RUNGS="theft" MAXDIM=3 bash experiments/mvp/run/3b_cone_sweep.sh   # быстрая проверка
#   BEST_OF_K=3 NSAMPLE=32 bash experiments/mvp/run/3b_cone_sweep.sh   # стабилизация оценки
#
# Env: GPU (3), MODEL, RUNGS (все три), MINDIM (1), MAXDIM (5), FORK,
#      BEST_OF_K (1) — K конусов на размерность, берём лучший по лоссу (убирает «пилу» по d),
#      NSAMPLE (8) — сэмплов внутренности конуса при обучении (L_sample), SEED (21).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # concept_dim
PY="$ROOT/.venv/bin/python"                       # venv concept_dim (все зависимости + editable)
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

GPU="${GPU:-3}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUNGS="${RUNGS:-theft illegal_activities malicious_use}"
MINDIM="${MINDIM:-1}"   # >1 => resume: подсев базиса из dim_{MINDIM-1}.pt
MAXDIM="${MAXDIM:-5}"
export BEST_OF_K="${BEST_OF_K:-1}"   # rdo.py читает из env
export SEED="${SEED:-21}"
export CONE_TAG="${CONE_TAG:-}"      # namespacing конусов: cones/<rung>/<CONE_TAG>/ (мультисид)
NSAMPLE="${NSAMPLE:-8}"

cd "$FORK"   # rdo.py читает ./data, ./results, .env относительно cwd
echo "модель=$MODEL  GPU=$GPU  MAXDIM=$MAXDIM  BEST_OF_K=$BEST_OF_K  NSAMPLE=$NSAMPLE  fork=$FORK"
for r in $RUNGS; do
  echo "=== cone sweep: $r (dim 1..$MAXDIM, best-of-$BEST_OF_K) ==="
  DIM_DIR="dim/$r" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" rdo.py --train_cone --model "$MODEL" --splits "$r" \
      --min_cone_dim "$MINDIM" --max_cone_dim "$MAXDIM" --n_sample "$NSAMPLE"
done
echo "готово: per-d конусы в $FORK/results/cones/<rung>/dim_<d>.pt"
