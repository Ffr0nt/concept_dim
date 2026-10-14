#!/usr/bin/env bash
# Шаг 0 — калибровка μ для free_norm (GPU, ~3 обучения одномерного RDO).
# Один домен, его первый сид базового запуска, по прогону на каждое μ:
#   <fork>/results/loss_terms/calibrate/mu_<μ>/<домен>/seed_XX.pt
# Затем сводка (CPU) -> reports/calibration.md с μ по правилу из idea.md. Выбранное μ
# вписываем в NORM_LAMBDA (scripts/common.py) и коммитим — только после этого 1_train.sh
# обучит free_norm.
#
# Использование:
#   GPU=0 bash experiments/loss_terms/run/0_calibrate_norm.sh
#   GPU=0 MUS="0.01 0.03" bash experiments/loss_terms/run/0_calibrate_norm.sh   # сдвиг сетки
#   bash experiments/loss_terms/run/0_calibrate_norm.sh --report-only           # только сводка, CPU
#
# Env: GPU, MUS ("0.03 0.1 0.3"), DOMAIN (O1), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

MUS="${MUS:-0.03 0.1 0.3}"
DOMAIN="${DOMAIN:-O1}"

if [[ "${1:-}" != "--report-only" ]]; then
  need_gpu
  read -ra MU_LIST <<< "$MUS"
  echo "калибровка: μ=$MUS  домен=$DOMAIN  GPU=$GPU  out=$OUT"
  py train.py --out "$OUT" --domains "$DOMAIN" --calibrate "${MU_LIST[@]}" \
    2>&1 | tee "$OUT/logs/calibrate-$(date +%Y%m%d-%H%M%S).log"
fi
CUDA_VISIBLE_DEVICES="" py calibration.py --out "$OUT" --report "$EXP/reports/calibration.md"
