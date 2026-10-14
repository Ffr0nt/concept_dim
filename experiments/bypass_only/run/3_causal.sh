#!/usr/bin/env bash
# Шаг 3 — каузальные замеры (GPU): аблация (ASR + KL) и наведение отказа для компонент, DIM,
# случайных и всех векторов; наведение — ещё и со знаком минус (--negative-too), потому что у
# no_add и invariant знак оси обучением не задан. Нужен шаг 2. Готовое пропускается.
# Пишет <fork>/results/bypass_only/causal.json и reports/causal.md.
#
# Использование:
#   GPU=0 bash experiments/bypass_only/run/3_causal.sh
#   bash experiments/bypass_only/run/3_causal.sh --report-only     # отчёт из json, CPU
#
# Env: GPU, BATCH (16), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
EXTRA=(--negative-too)
if [[ "${1:-}" == "--report-only" ]]; then
  export CUDA_VISIBLE_DEVICES=""; EXTRA+=(--report-only)
else
  need_gpu
fi
py causal.py --out "$OUT" --report "$EXP/reports/causal.md" --batch-size "${BATCH:-16}" "${EXTRA[@]}" \
  2>&1 | tee "$OUT/logs/causal-$(date +%Y%m%d-%H%M%S).log"
