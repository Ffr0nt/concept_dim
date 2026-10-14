#!/usr/bin/env bash
# Очередь bypass_only: обучение invariant → геометрия → каузальные замеры, строго по одному
# процессу, на любой карте с достаточной памятью.
#
# Ждёт, пока закроются tmux-сессии WAIT_SESSIONS (по умолчанию — очереди loss_terms: им нужны
# те же карты, а их no_add/no_abl нужны здесь). Перед стартом проверяет, что векторы no_add и
# no_abl из loss_terms на месте и что форк на ветке с --addition_mode.
# Правила как у loss_terms/run/queue_finish.sh: следующая задача — не раньше COOLDOWN секунд
# после конца предыдущей; карта — из GPUS с наибольшей свободной памятью, если не меньше порога
# (обучение — MIN_FREE_GB, замеры — MIN_FREE_GB_EVAL), с повторной проверкой через DELAY;
# задача успешна, только если код 0 и все её результаты на диске; иначе повтор до RETRIES раз.
#
# Запуск в tmux (сессия закроется сама):
#   tmux new-session -d -s fz-bypass-only \
#     "bash experiments/bypass_only/run/queue.sh 2>&1 | tee -a ~/f.zakharov/bypass_only_queue.log"
#
# Env: WAIT_SESSIONS ("fz-loss-terms fz-loss-terms-2"), GPUS ("0 1 2 3 4"),
#      JOBS ("invariant geometry causal"), MIN_FREE_GB (34), MIN_FREE_GB_EVAL (20), DELAY (60),
#      POLL (60), COOLDOWN (60), RETRIES (3), OUT (<fork>/results/bypass_only).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"                       # concept_dim
EXP="$ROOT/experiments/bypass_only"
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"
OUT="${OUT:-$FORK/results/bypass_only}"
LT_OUT="$(dirname "$OUT")/loss_terms"
WAIT_SESSIONS="${WAIT_SESSIONS:-fz-loss-terms fz-loss-terms-2}"
GPUS="${GPUS:-0 1 2 3 4}"
JOBS="${JOBS:-invariant geometry causal}"
MIN_FREE_GB="${MIN_FREE_GB:-34}"; MIN_FREE_GB_EVAL="${MIN_FREE_GB_EVAL:-20}"
DELAY="${DELAY:-60}"; POLL="${POLL:-60}"; COOLDOWN="${COOLDOWN:-60}"; RETRIES="${RETRIES:-3}"
N_VECTORS=12                         # 4 домена × 3 сида
N_DIRECTIONS=112                     # шаг 3: 60 (4 компоненты + 5 DIM + 3 случ. + 48 векторов) + 52 со знаком минус
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
    invariant) (( $(n_seeds "$OUT/invariant") == N_VECTORS )) ;;
    geometry)
      [[ -f "$OUT/geometry.json" && -f "$OUT/components.pt" ]] && python3 - "$OUT/geometry.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))["variants"]
sys.exit(0 if {"base", "no_add", "no_abl", "invariant"} <= set(v) else 1)
PY
      ;;
    causal)
      [[ -f "$OUT/causal.json" && -f "$EXP/reports/causal.md" ]] && python3 - "$OUT/causal.json" $N_DIRECTIONS <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
d = {x["direction"] for x in r.get("ablation", []) + r.get("induce", [])}
sys.exit(0 if len(d) >= int(sys.argv[2]) else 1)
PY
      ;;
    *) echo "неизвестная задача $1" >&2; return 1 ;;
  esac
}

run_job() {   # $1 — задача, $2 — карта (для geometry не используется)
  case $1 in
    invariant) GPU="$2" bash "$EXP/run/1_train.sh" ;;
    geometry) bash "$EXP/run/2_geometry.sh" ;;
    causal) GPU="$2" bash "$EXP/run/3_causal.sh" ;;
  esac
}

cd "$ROOT"
echo "$(ts) очередь bypass_only: $JOBS; жду закрытия сессий: $WAIT_SESSIONS"
for sess in $WAIT_SESSIONS; do
  while tmux has-session -t "$sess" 2>/dev/null; do sleep "$POLL"; done
done
echo "$(ts) сессии закрыты"
for v in no_add no_abl; do
  if (( $(n_seeds "$LT_OUT/$v") != N_VECTORS )); then
    echo "$(ts) нет полного набора $v в $LT_OUT — сначала loss_terms; выхожу"; exit 1
  fi
done
if ! grep -q -- "--addition_mode" "$FORK/rdo.py"; then
  echo "$(ts) в $FORK/rdo.py нет --addition_mode — переключи форк на feat/rdo-invariant-addition; выхожу"; exit 1
fi
last_end=$(date +%s)                  # очереди loss_terms только что кончились — та же пауза
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
echo "$(ts) очередь bypass_only завершена"
