#!/usr/bin/env bash
# Шаг 1 — обучение вариантов лосса (GPU): одномерный RDO, 3 сида × 4 домена на вариант,
#   <fork>/results/loss_terms/<вариант>/<домен>/seed_XX.pt (+ run.json с гиперпараметрами).
# base не обучается — его сиды берутся из базового запуска shared-refusal-component.
# Готовые сиды пропускаются: прерванный прогон просто перезапустить.
# free_norm требует NORM_LAMBDA в scripts/common.py (шаг 0).
#
# Всё — на одной карте. Два процесса на ней можно разделить по доменам (пишут в разные каталоги):
#   GPU=0 DOMAINS="O1 O2" bash experiments/loss_terms/run/1_train.sh
#   GPU=0 DOMAINS="O5 O6" bash experiments/loss_terms/run/1_train.sh
# Только один вариант:
#   GPU=0 VARIANTS=no_add bash experiments/loss_terms/run/1_train.sh
#
# Env: GPU, VARIANTS ("no_add no_abl free_norm"), DOMAINS (все домены базового запуска),
#      FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_gpu

VARIANTS="${VARIANTS:-no_add no_abl free_norm}"
read -ra V_LIST <<< "$VARIANTS"
D_ARGS=()
if [[ -n "${DOMAINS:-}" ]]; then read -ra D_LIST <<< "$DOMAINS"; D_ARGS=(--domains "${D_LIST[@]}"); fi

TAG="$(echo "${DOMAINS:-all}" | tr ' ' '-')"
echo "обучение: варианты=$VARIANTS  домены=${DOMAINS:-все}  GPU=$GPU  out=$OUT"
py train.py --out "$OUT" --variants "${V_LIST[@]}" "${D_ARGS[@]}" \
  2>&1 | tee "$OUT/logs/train-$TAG-$(date +%Y%m%d-%H%M%S).log"
