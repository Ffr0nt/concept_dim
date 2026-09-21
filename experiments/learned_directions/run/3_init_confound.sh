#!/usr/bin/env bash
# Шаг 3 — конфаундер общей инициализации + бюджет обучения. CPU, GPU НЕ нужен.
# Разводит три режима: A (один концепт / разные сиды), B (разные концепты / один сид,
# заражено общим стартом), C (разные концепты / разные сиды). Печатает в stdout;
# перенаправь в reports/, если нужен файл.
#
# Использование:
#   bash experiments/learned_directions/run/3_init_confound.sh
#   DIMS="2 3 4 5" bash experiments/learned_directions/run/3_init_confound.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), DIMS ("3 5").
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

DIMS="${DIMS:-3 5}"
BASE="$FORK/results/cones"

[ -d "$BASE" ] || { echo "нет каталога конусов: $BASE" >&2; exit 1; }

echo "base=$BASE  DIMS='$DIMS'"
cd "$FORK"
uv run python "$EXP/scripts/init_confound.py" --base "$BASE" --dims $DIMS
