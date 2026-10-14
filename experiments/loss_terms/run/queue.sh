#!/usr/bin/env bash
# Очередь шагов эксперимента на одной карте, строго по одному процессу.
#
# Перед каждой задачей ждём, пока на карте GPU будет свободно не меньше MIN_FREE_GB, и
# проверяем это ещё раз через DELAY (соседи могут расти). Чужие процессы на карте не мешают —
# важна только свободная память. Пик одного обучения RDO ≈ 22.5 ГБ (замер 14.10), отсюда
# порог 28 ГБ. Упавшую задачу (например, OOM) повторяем до RETRIES раз: готовые сиды
# раннеры пропускают, так что повтор продолжает с места падения.
#
# Задачи: calibrate — 0_calibrate_norm.sh; <вариант> — 1_train.sh с VARIANTS=<вариант>.
# Логи задач — в <fork>/results/loss_terms/logs/ (пишут сами раннеры).
#
# Запуск в tmux (сессия закроется сама):
#   tmux new-session -d -s fz-loss-terms \
#     "GPU=1 bash experiments/loss_terms/run/queue.sh 2>&1 | tee -a ~/f.zakharov/loss_terms_queue.log"
#
# Env: GPU (обязателен), JOBS ("calibrate no_add no_abl"), MIN_FREE_GB (28), DELAY (60),
#      POLL (60), RETRIES (3).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GPU:?укажи GPU=<индекс>}"
JOBS="${JOBS:-calibrate no_add no_abl}"
MIN_FREE_GB="${MIN_FREE_GB:-28}"; DELAY="${DELAY:-60}"; POLL="${POLL:-60}"; RETRIES="${RETRIES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ts() { date '+%F %T'; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU") / 1024 )); }

wait_free() {
  local f
  while true; do
    f=$(free_gb)
    if (( f >= MIN_FREE_GB )); then
      sleep "$DELAY"; f=$(free_gb)
      (( f >= MIN_FREE_GB )) && { echo "$(ts) свободно $f ГБ — старт"; return; }
    fi
    echo "$(ts) свободно $f ГБ < $MIN_FREE_GB — жду"; sleep "$POLL"
  done
}

cd "$HERE/../../.."   # concept_dim
echo "$(ts) очередь на GPU $GPU: $JOBS (порог $MIN_FREE_GB ГБ, повторов $RETRIES)"
for job in $JOBS; do
  for (( try = 1; try <= RETRIES; try++ )); do
    wait_free
    echo "$(ts) '$job', попытка $try"
    if [[ "$job" == calibrate ]]; then
      GPU="$GPU" bash experiments/loss_terms/run/0_calibrate_norm.sh > /dev/null 2>&1
    else
      GPU="$GPU" VARIANTS="$job" bash experiments/loss_terms/run/1_train.sh > /dev/null 2>&1
    fi
    rc=$?
    echo "$(ts) '$job' завершилась с кодом $rc"
    (( rc == 0 )) && break
  done
  (( rc != 0 )) && echo "$(ts) '$job' не прошла за $RETRIES попыток — перехожу к следующей"
done
echo "$(ts) очередь пройдена"
