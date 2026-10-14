#!/usr/bin/env bash
# Вторая очередь: доводит эксперимент до конца после первой (queue.sh).
#
# Ждёт, пока закроется tmux-сессия первой очереди (WAIT_SESSION), затем по одной задаче:
#   no_add, no_abl  — дообучение, если первая очередь их не довела (иначе пропуск);
#   calib_more      — калибровка μ из MUS_MORE (сетка 0.03/0.1/0.3 упёрлась в верхнее μ),
#                     пересчёт calibration.md и calibration.json с выбором μ по правилу;
#   free_norm       — обучение с μ из calibration.json;
#   geometry        — шаг 2 (CPU, карта не нужна);
#   causal          — шаг 3 (GPU).
# Правила те же, что у queue.sh: следующая задача — не раньше COOLDOWN секунд после конца
# предыдущей; GPU-задача — на карте из GPUS с наибольшей свободной памятью, если там не меньше
# порога (обучение — MIN_FREE_GB, замеры — MIN_FREE_GB_EVAL), с повторной проверкой через
# DELAY; задача успешна, только если код 0 и все её результаты на диске; иначе повтор до
# RETRIES раз (готовое раннеры пропускают).
#
# Запуск в tmux (сессия закроется сама):
#   tmux new-session -d -s fz-loss-terms-2 \
#     "bash experiments/loss_terms/run/queue_finish.sh 2>&1 | tee -a ~/f.zakharov/loss_terms_queue.log"
#
# Env: WAIT_SESSION (fz-loss-terms), GPUS ("0 1 2 3 4"),
#      JOBS ("no_add no_abl calib_more free_norm geometry causal"), MUS_MORE ("1"),
#      MIN_FREE_GB (28), MIN_FREE_GB_EVAL (20), DELAY (60), POLL (60), COOLDOWN (60),
#      RETRIES (3), OUT (<fork>/results/loss_terms).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"                       # concept_dim
EXP="$ROOT/experiments/loss_terms"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"
OUT="${OUT:-$FORK/results/loss_terms}"
WAIT_SESSION="${WAIT_SESSION:-fz-loss-terms}"
GPUS="${GPUS:-0 1 2 3 4}"
JOBS="${JOBS:-no_add no_abl calib_more free_norm geometry causal}"
MUS_MORE="${MUS_MORE:-1}"            # имя каталога — mu_<μ:g>, поэтому "1", а не "1.0"
MIN_FREE_GB="${MIN_FREE_GB:-28}"; MIN_FREE_GB_EVAL="${MIN_FREE_GB_EVAL:-20}"
DELAY="${DELAY:-60}"; POLL="${POLL:-60}"; COOLDOWN="${COOLDOWN:-60}"; RETRIES="${RETRIES:-3}"
N_VECTORS=12                         # 4 домена × 3 сида
N_DIRECTIONS=60                      # шаг 3: 4 компоненты + 5 DIM + 3 случайных + 48 векторов
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ts() { date '+%F %T'; }
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1") / 1024 )); }

pick_gpu() {   # $1 — порог, ГБ; ждёт подходящую карту, печатает её индекс (сообщения — в stderr)
  local need=$1 g f best bf
  while true; do
    best=""; bf=-1
    for g in $GPUS; do f=$(free_gb "$g"); (( f > bf )) && { best=$g; bf=$f; }; done
    if (( bf >= need )); then
      sleep "$DELAY"; f=$(free_gb "$best")
      if (( f >= need )); then echo "$(ts) GPU $best: свободно $f ГБ — беру" >&2; echo "$best"; return; fi
      echo "$(ts) GPU $best: за ${DELAY} с свободное упало до $f ГБ" >&2
    else
      echo "$(ts) больше всего свободно на GPU $best: $bf ГБ < $need — жду" >&2
    fi
    sleep "$POLL"
  done
}

n_seeds() { find "$1" -name 'seed_*.pt' 2>/dev/null | wc -l; }

complete() {
  case $1 in
    no_add|no_abl|free_norm) (( $(n_seeds "$OUT/$1") == N_VECTORS )) ;;
    calib_more)
      local mu
      for mu in $MUS_MORE; do (( $(n_seeds "$OUT/calibrate/mu_$mu") >= 1 )) || return 1; done
      [[ -f "$OUT/calibration.json" ]] && python3 - "$OUT/calibration.json" $MUS_MORE <<'PY'
import json, sys
mus = json.load(open(sys.argv[1]))["mus"]
sys.exit(0 if all(float(m) in mus for m in sys.argv[2:]) else 1)
PY
      ;;
    geometry)
      [[ -f "$OUT/geometry.json" && -f "$OUT/components.pt" ]] && python3 - "$OUT/geometry.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))["variants"]
sys.exit(0 if {"base", "no_add", "no_abl", "free_norm"} <= set(v) else 1)
PY
      ;;
    causal)
      [[ -f "$OUT/causal.json" && -f "$EXP/reports/causal.md" ]] && python3 - "$OUT/causal.json" $N_DIRECTIONS <<'PY'
import json, sys
d = {r["direction"] for r in json.load(open(sys.argv[1])).get("ablation", [])}
sys.exit(0 if len(d) >= int(sys.argv[2]) else 1)
PY
      ;;
    *) echo "неизвестная задача $1" >&2; return 1 ;;
  esac
}

run_job() {   # $1 — задача, $2 — карта (для geometry не используется)
  case $1 in
    no_add|no_abl|free_norm) GPU="$2" VARIANTS="$1" bash "$EXP/run/1_train.sh" ;;
    calib_more) GPU="$2" MUS="$MUS_MORE" bash "$EXP/run/0_calibrate_norm.sh" ;;
    geometry) bash "$EXP/run/2_geometry.sh" ;;
    causal) GPU="$2" bash "$EXP/run/3_causal.sh" ;;
  esac
}

cd "$ROOT"
echo "$(ts) очередь-2: $JOBS; жду закрытия сессии $WAIT_SESSION"
# "=" — точное имя: без него tmux ищет по префиксу, и fz-loss-terms совпадает с fz-loss-terms-2
while tmux has-session -t "=$WAIT_SESSION" 2>/dev/null; do sleep "$POLL"; done
echo "$(ts) сессия $WAIT_SESSION закрыта"
last_end=$(date +%s)                  # первая очередь только что кончилась — та же пауза
for job in $JOBS; do
  if complete "$job"; then echo "$(ts) '$job' уже готова — пропуск"; continue; fi
  ok=0
  for (( try = 1; try <= RETRIES; try++ )); do
    wait_left=$(( last_end + COOLDOWN - $(date +%s) ))
    (( wait_left > 0 )) && { echo "$(ts) пауза после предыдущей задачи: $wait_left с"; sleep "$wait_left"; }
    case $job in
      geometry) gpu="-" ;;
      causal) gpu=$(pick_gpu "$MIN_FREE_GB_EVAL") ;;
      *) gpu=$(pick_gpu "$MIN_FREE_GB") ;;
    esac
    echo "$(ts) СТАРТ '$job' на GPU $gpu, попытка $try"
    run_job "$job" "$gpu" > /dev/null 2>&1
    rc=$?
    last_end=$(date +%s)
    if (( rc == 0 )) && complete "$job"; then
      echo "$(ts) ГОТОВО '$job': код 0, все результаты на месте"; ok=1; break
    fi
    oom=$(grep -l "OutOfMemoryError" $(ls -t "$OUT"/logs/*.log 2>/dev/null | head -1) 2>/dev/null | wc -l)
    echo "$(ts) ОШИБКА '$job': код $rc, результаты $(complete "$job" && echo полные || echo неполные)$( (( oom )) && echo ', в логе OOM')"
  done
  if (( ! ok )); then
    echo "$(ts) '$job' не прошла за $RETRIES попыток — дальше не иду (следующие шаги зависят от неё)"
    break
  fi
done
echo "$(ts) очередь-2 завершена"
