#!/usr/bin/env bash
# Шаг 1 — обучение варианта invariant (GPU): одномерный RDO, 3 сида × 4 домена,
#   <fork>/results/bypass_only/invariant/<домен>/seed_XX.pt (+ run.json).
# Лосс: аблация + retain + инвариантность к добавлению ±α·r̂ (rdo.py --addition_mode invariant,
# ветка форка feat/rdo-invariant-addition). base, no_add, no_abl не обучаются — готовы.
# Готовые сиды пропускаются: прерванный прогон просто перезапустить.
# Пик памяти выше, чем у loss_terms (лишний прямой проход на шаг): закладывать ≥ 34 ГБ.
#
# Использование:
#   GPU=0 bash experiments/bypass_only/run/1_train.sh
#   GPU=0 DOMAINS="O1" bash experiments/bypass_only/run/1_train.sh
#
# Env: GPU, DOMAINS (все домены базового запуска), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_gpu
D_ARGS=()
if [[ -n "${DOMAINS:-}" ]]; then read -ra D_LIST <<< "$DOMAINS"; D_ARGS=(--domains "${D_LIST[@]}"); fi
TAG="$(echo "invariant ${DOMAINS:-all}" | tr ' ' '-')"
echo "обучение invariant: домены=${DOMAINS:-все}  GPU=$GPU  out=$OUT"
py train.py --out "$OUT" --variants invariant "${D_ARGS[@]}" \
  2>&1 | tee "$OUT/logs/train-$TAG-$(date +%Y%m%d-%H%M%S).log"
