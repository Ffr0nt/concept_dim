#!/usr/bin/env bash
# Очередь шагов эксперимента: строго по одному процессу, на любой подходящей карте.
#
# Перед каждой задачей:
#   - с конца предыдущей прошло не меньше COOLDOWN секунд;
#   - выбираем карту из GPUS с наибольшей свободной памятью, если там не меньше MIN_FREE_GB, и
#     проверяем её ещё раз через DELAY (соседи могут расти). Чужие процессы не мешают —
#     важна только свободная память. Пик одного обучения RDO > 22.5 ГБ (OOM 14.10 при 22 ГБ
#     свободных), отсюда порог 28 ГБ.
# После задачи проверяем, что она закончилась корректно: код 0 И на диске все ожидаемые
# файлы (calibrate — по seed_XX.pt на каждое μ, вариант — 12 seed_XX.pt). Иначе повтор до
# RETRIES раз: готовые сиды раннеры пропускают, так что повтор продолжает с места падения.
#
# Задачи: calibrate — 0_calibrate_norm.sh; <вариант> — 1_train.sh с VARIANTS=<вариант>.
# Логи задач — в <fork>/results/loss_terms/logs/ (пишут сами раннеры).
#
# Запуск в tmux (сессия закроется сама):
#   tmux new-session -d -s fz-loss-terms \
#     "bash experiments/loss_terms/run/queue.sh 2>&1 | tee -a ~/f.zakharov/loss_terms_queue.log"
#
# Env: GPUS ("0 1 2 3 4"), JOBS ("calibrate no_add no_abl"), MIN_FREE_GB (28), DELAY (60),
#      POLL (60), COOLDOWN (60), RETRIES (3), MUS ("0.03 0.1 0.3" — как в 0_calibrate_norm.sh),
#      OUT (<fork>/results/loss_terms).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"
OUT="${OUT:-$FORK/results/loss_terms}"
GPUS="${GPUS:-0 1 2 3 4}"
JOBS="${JOBS:-calibrate no_add no_abl}"
MUS="${MUS:-0.03 0.1 0.3}"
MIN_FREE_GB="${MIN_FREE_GB:-28}"; DELAY="${DELAY:-60}"; POLL="${POLL:-60}"
COOLDOWN="${COOLDOWN:-60}"; RETRIES="${RETRIES:-3}"
N_VECTORS=12                                                # 4 домена × 3 сида
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ts() { date '+%F %T'; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1") / 1024 )); }

best_gpu() {   # печатает "<gpu> <свободно ГБ>" карты с наибольшей свободной памятью
  local g f best="" bf=-1
  for g in $GPUS; do f=$(free_gb "$g"); (( f > bf )) && { best=$g; bf=$f; }; done
  echo "$best $bf"
}

pick_gpu() {   # ждёт подходящую карту, печатает её индекс в stdout (сообщения — в stderr)
  local g f
  while true; do
    read -r g f <<< "$(best_gpu)"
    if (( f >= MIN_FREE_GB )); then
      sleep "$DELAY"; f=$(free_gb "$g")
      if (( f >= MIN_FREE_GB )); then echo "$(ts) GPU $g: свободно $f ГБ — беру" >&2; echo "$g"; return; fi
      echo "$(ts) GPU $g: за ${DELAY} с свободное упало до $f ГБ" >&2
    else
      echo "$(ts) больше всего свободно на GPU $g: $f ГБ < $MIN_FREE_GB — жду" >&2
    fi
    sleep "$POLL"
  done
}

complete() {   # все ли результаты задачи на диске
  local job=$1 n mu
  if [[ "$job" == calibrate ]]; then
    for mu in $MUS; do
      n=$(find "$OUT/calibrate/mu_$mu" -name 'seed_*.pt' 2>/dev/null | wc -l)
      (( n >= 1 )) || return 1
    done
    [[ -f "$ROOT/experiments/loss_terms/reports/calibration.md" ]]
  else
    n=$(find "$OUT/$job" -name 'seed_*.pt' 2>/dev/null | wc -l)
    (( n == N_VECTORS ))
  fi
}

cd "$ROOT"
echo "$(ts) очередь: $JOBS; карты: $GPUS; порог $MIN_FREE_GB ГБ; пауза после задачи $COOLDOWN с; повторов $RETRIES"
last_end=0
for job in $JOBS; do
  if complete "$job"; then echo "$(ts) '$job' уже готова — пропуск"; continue; fi
  ok=0
  for (( try = 1; try <= RETRIES; try++ )); do
    wait_left=$(( last_end + COOLDOWN - $(date +%s) ))
    (( wait_left > 0 )) && { echo "$(ts) пауза после предыдущей задачи: $wait_left с"; sleep "$wait_left"; }
    gpu=$(pick_gpu)
    echo "$(ts) СТАРТ '$job' на GPU $gpu, попытка $try"
    if [[ "$job" == calibrate ]]; then
      GPU="$gpu" MUS="$MUS" bash experiments/loss_terms/run/0_calibrate_norm.sh > /dev/null 2>&1
    else
      GPU="$gpu" VARIANTS="$job" bash experiments/loss_terms/run/1_train.sh > /dev/null 2>&1
    fi
    rc=$?
    last_end=$(date +%s)
    if (( rc == 0 )) && complete "$job"; then
      echo "$(ts) ГОТОВО '$job': код 0, все результаты на месте"; ok=1; break
    fi
    oom=$(grep -l "OutOfMemoryError" $(ls -t "$OUT"/logs/*.log 2>/dev/null | head -1) 2>/dev/null | wc -l)
    echo "$(ts) ОШИБКА '$job': код $rc, результаты $(complete "$job" && echo полные || echo неполные)$( (( oom )) && echo ', в логе OOM')"
  done
  (( ok )) || echo "$(ts) '$job' не прошла за $RETRIES попыток — перехожу к следующей"
done
echo "$(ts) очередь пройдена"
