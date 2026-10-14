#!/usr/bin/env bash
# Ждёт освобождения карты и запускает несколько шагов эксперимента параллельно на ней.
#
# Ждём, пока: (1) завершится процесс WAIT_PID (если задан) и (2) на карте GPU не останется
# ни одного процесса. Затем пауза DELAY и проверка, что свободно не меньше MIN_FREE_GB
# (иначе снова ждём). Задачи JOBS стартуют по одной с интервалом STAGGER: следующая — только
# если после предыдущей на карте ещё есть MIN_JOB_GB свободной памяти (защита от OOM).
#
# Задачи: calibrate — 0_calibrate_norm.sh; <вариант> — 1_train.sh с VARIANTS=<вариант>.
# Логи каждой задачи пишут сами раннеры в <fork>/results/loss_terms/logs/.
#
# Запуск в tmux (сессия закроется сама, когда все задачи кончатся):
#   tmux new-session -d -s fz-loss-terms \
#     "GPU=1 WAIT_PID=4145672 bash experiments/loss_terms/run/wait_and_launch.sh 2>&1 | tee -a ~/f.zakharov/loss_terms_launch.log"
#
# Env: GPU (обязателен), WAIT_PID, JOBS ("calibrate no_add no_abl"), DELAY (60), POLL (60),
#      MIN_FREE_GB (60), MIN_JOB_GB (20), STAGGER (90).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GPU:?укажи GPU=<индекс>}"
JOBS="${JOBS:-calibrate no_add no_abl}"
DELAY="${DELAY:-60}"; POLL="${POLL:-60}"; STAGGER="${STAGGER:-90}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"; MIN_JOB_GB="${MIN_JOB_GB:-20}"

ts() { date '+%F %T'; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU") / 1024 )); }
n_procs() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid"; }

echo "$(ts) жду: GPU $GPU ($uuid)${WAIT_PID:+, завершение PID $WAIT_PID}; задачи: $JOBS"
while true; do
  if [[ -n "${WAIT_PID:-}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then sleep "$POLL"; continue; fi
  if (( $(n_procs) > 0 )); then
    echo "$(ts) на GPU $GPU ещё $(n_procs) процесс(ов), свободно $(free_gb) ГБ"; sleep "$POLL"; continue
  fi
  echo "$(ts) GPU $GPU пуста — пауза ${DELAY} с"; sleep "$DELAY"
  if (( $(n_procs) == 0 && $(free_gb) >= MIN_FREE_GB )); then break; fi
  echo "$(ts) за паузу карту заняли (процессов $(n_procs), свободно $(free_gb) ГБ) — жду дальше"
done

cd "$HERE/../../.."   # concept_dim
pids=()
for job in $JOBS; do
  if (( ${#pids[@]} > 0 )); then
    sleep "$STAGGER"
    f=$(free_gb)
    if (( f < MIN_JOB_GB )); then
      echo "$(ts) свободно $f ГБ < $MIN_JOB_GB — '$job' и дальше не запускаю (запусти позже вручную)"; break
    fi
  fi
  echo "$(ts) старт '$job' (свободно $(free_gb) ГБ)"
  if [[ "$job" == calibrate ]]; then
    GPU="$GPU" bash experiments/loss_terms/run/0_calibrate_norm.sh > /dev/null 2>&1 &
  else
    GPU="$GPU" VARIANTS="$job" bash experiments/loss_terms/run/1_train.sh > /dev/null 2>&1 &
  fi
  pids+=($!)
done

for p in "${pids[@]}"; do wait "$p"; echo "$(ts) задача PID $p завершилась с кодом $?"; done
echo "$(ts) все задачи завершены"
