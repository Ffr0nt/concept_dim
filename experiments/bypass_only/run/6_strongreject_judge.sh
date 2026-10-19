#!/usr/bin/env bash
# Шаг 6 — StrongREJECT: судья (сеть, запускать на Mac — с сервера OpenRouter недоступен, 403).
# Копирует с сервера ответы шага 5, causal.json (KL) и опорные данные базового запуска (запросы,
# шаблоны рубрики, ответы и оценки none / w:all / DIM), оценивает недостающее рубрикой
# StrongREJECT (gpt-4o-mini через OpenRouter, ≈ $0.0003 за ответ, 31 × 303 ≈ $3) и пишет отчёт
# reports/strongreject.md. Кэш оценок копируется обратно на сервер.
# Ключ — OPENROUTER_API_KEY в окружении или в <shared-refusal-component>/.env (шаблон .env.example).
# shared-refusal-component локально — на ветке feat/strongreject-judge.
#
# Использование (из корня concept_dim на Mac):
#   bash experiments/bypass_only/run/6_strongreject_judge.sh
#   NO_SYNC=1 bash experiments/bypass_only/run/6_strongreject_judge.sh   # без копирования, только судья и отчёт
#
# Env: HOST (peaceful-matsumoto), REMOTE (~/f.zakharov на сервере), RUN (qwen2.5-3b_salad-o1o2o5o6),
#      WORKERS (16), NO_SYNC (0), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
export CUDA_VISIBLE_DEVICES=""
HOST="${HOST:-peaceful-matsumoto}"
REMOTE="${REMOTE:-f.zakharov}"
RUN="${RUN:-qwen2.5-3b_salad-o1o2o5o6}"
BASE_LOCAL="$SRC/results/$RUN"

grep -q "^judge:" "$SRC/configs/$RUN.yaml" || { echo "в $SRC нет секции judge — переключи на feat/strongreject-judge" >&2; exit 1; }

if [[ "${NO_SYNC:-0}" != 1 ]]; then
  echo "копирую с $HOST: ответы bypass_only и опорные данные базового запуска"
  mkdir -p "$OUT" "$BASE_LOCAL"
  ssh "$HOST" "cd $REMOTE/geometry-of-refusal/results/bypass_only && tar czf - strongreject causal.json" | tar xzf - -C "$OUT"
  ssh "$HOST" "cd $REMOTE/shared-refusal-component/results/$RUN && tar czf - manifest.json ablation/ablation.json \
      strongreject/prompts.json strongreject/source \
      strongreject/generations/{none,w_all,dim_O1,dim_O2,dim_O5,dim_O6}.jsonl \
      strongreject/judge/{none,w_all,dim_O1,dim_O2,dim_O5,dim_O6}.jsonl" | tar xzf - -C "$BASE_LOCAL"
fi

py strongreject.py judge --out "$OUT" --report "$EXP/reports/strongreject.md" --workers "${WORKERS:-16}"

if [[ "${NO_SYNC:-0}" != 1 ]]; then
  echo "копирую оценки обратно на $HOST"
  tar czf - -C "$OUT" strongreject/judge strongreject/strongreject.json | \
    ssh "$HOST" "tar xzf - -C $REMOTE/geometry-of-refusal/results/bypass_only"
fi
