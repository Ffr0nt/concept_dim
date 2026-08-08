#!/usr/bin/env bash
# Этап 1 — сборка per-rung сплитов SALAD (CPU). Пишет в форк data/<rung>_splits/
# для ВСЕХ ступеней из build_splits.RUNGS (theft/illegal/malicious/all). Детерминировано
# по сиду => перегенерация существующих ступеней идентична (безопасно перезапускать).
# Тянет SALAD с HF (кэш переиспользуется). Запускать НА СЕРВЕРЕ.
#
# Использование:
#   bash experiments/mvp/run/1_build_splits.sh
#
# Env (опц.): SEED (21), FORK (../geometry-of-refusal), HARMLESS_SRC.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/../../.."   # -> корень concept_dim

SEED="${SEED:-21}"
FORK="${FORK:-../geometry-of-refusal}"
export HF_HOME="${HF_HOME:-/home/jovyan/.cache/huggingface}"

EXTRA=()
[ -n "${HARMLESS_SRC:-}" ] && EXTRA+=(--harmless-src "$HARMLESS_SRC")

uv run --with datasets --with pandas python experiments/mvp/scripts/build_splits.py \
  --seed "$SEED" --fork "$FORK" "${EXTRA[@]}"
echo "готово: сплиты в $(cd "$FORK" && pwd)/data/<rung>_splits/ (в т.ч. all_splits/)"
