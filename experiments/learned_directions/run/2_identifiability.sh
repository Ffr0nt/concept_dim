#!/usr/bin/env bash
# Шаг 2 — идентифицируемость ОРИЕНТАЦИИ конуса внутри его span-а. CPU, GPU НЕ нужен.
# Отвечает на вопрос: обучена ли ориентация ортанта, или она — шум инициализации.
# Пишет reports/identifiability.json (+ таблицы в stdout).
#
# Отдельный шаг от 1_seed_agreement.sh: тот меряет сходство реперов целиком (span + оси),
# этот изолирует ТОЛЬКО ориентацию — сравнивая наблюдаемое с null из случайных вращений
# репера внутри его собственного span-а (span и проектор при этом не меняются).
#
# Использование:
#   bash experiments/learned_directions/run/2_identifiability.sh
#   NNULL=5000 bash experiments/learned_directions/run/2_identifiability.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), NNULL (2000), SEED (21).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

NNULL="${NNULL:-2000}"
SEED="${SEED:-21}"
BASE="$FORK/results/cones"
OUT="$EXP/reports/identifiability.json"

[ -d "$BASE" ] || { echo "нет каталога конусов: $BASE" >&2; exit 1; }
mkdir -p "$EXP/reports"

echo "base=$BASE  NNULL=$NNULL  SEED=$SEED  out=$OUT"
cd "$FORK"
uv run python "$EXP/scripts/cone_identifiability.py" \
  --base "$BASE" --n_null "$NNULL" --seed "$SEED" --out "$OUT"
