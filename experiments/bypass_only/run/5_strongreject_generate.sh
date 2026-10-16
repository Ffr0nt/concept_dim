#!/usr/bin/env bash
# Шаг 5 — StrongREJECT: полные ответы при аблации направлений bypass_only (GPU, сервер).
# 31 направление (7 компонент + 12 векторов invariant_0.03 + 12 векторов no_add) × 303 запроса,
# жадно до 512 токенов — протокол этапа 06 shared-refusal-component, те же запросы и параметры.
# Нужны: шаг 2 (components.pt) и ветка feat/strongreject-judge в shared-refusal-component (там
# секции strongreject/generate/judge и prompts.json базового запуска).
# Пишет <fork>/results/bypass_only/strongreject/{generations/*.jsonl, manifest.json}. Готовое пропускается.
#
# Использование:
#   GPU=1 bash experiments/bypass_only/run/5_strongreject_generate.sh
#
# Env: GPU, FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_gpu
echo "StrongREJECT, генерация: GPU=$GPU  out=$OUT"
py strongreject.py generate --out "$OUT" 2>&1 | tee "$OUT/logs/strongreject-generate-$(date +%Y%m%d-%H%M%S).log"
