#!/usr/bin/env bash
# Шаг 3 — каузальные замеры (GPU): аблация (ASR по доменам + KL на harmless) и наведение
# отказа (доля отказов на harmless при добавлении s·‖DIM_union‖·v̂ на слое добавления) для
# компонент вариантов, DIM, случайных и всех векторов вариантов. Нужен шаг 2.
# Готовые направления пропускаются. Пишет <fork>/results/loss_terms/causal.json и
# reports/causal.md.
#
# Использование:
#   GPU=0 bash experiments/loss_terms/run/3_causal.sh
#   GPU=0 COMPONENTS_ONLY=1 bash experiments/loss_terms/run/3_causal.sh   # только w_V, DIM, rand
#   bash experiments/loss_terms/run/3_causal.sh --report-only             # отчёт из json, CPU
#
# Env: GPU, COMPONENTS_ONLY (0), BATCH (16), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

EXTRA=()
if [[ "${1:-}" == "--report-only" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  EXTRA=(--report-only)
else
  need_gpu
  [[ "${COMPONENTS_ONLY:-0}" == 1 ]] && EXTRA=(--components-only)
fi
py causal.py --out "$OUT" --report "$EXP/reports/causal.md" --batch-size "${BATCH:-16}" "${EXTRA[@]}" \
  2>&1 | tee "$OUT/logs/causal-$(date +%Y%m%d-%H%M%S).log"
