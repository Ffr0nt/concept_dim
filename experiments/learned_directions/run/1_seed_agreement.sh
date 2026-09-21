#!/usr/bin/env bash
# Шаг 1 — попарное сходство выученных реперов между сидами + агрегаты. CPU, GPU НЕ нужен,
# модель не грузится: читает только results/cones/<rung>/seed_<s>/dim_<d>.pt.
# Пишет reports/seed-agreement.md.
#
# scipy нет ни в venv форка, ни в системном python3 — подтягиваем разово через
# `uv run --with scipy`, зависимости репозитория при этом не меняются.
#
# Использование:
#   bash experiments/learned_directions/run/1_seed_agreement.sh
#   NRAND=500 bash experiments/learned_directions/run/1_seed_agreement.sh
#
# Env: FORK (../geometry-of-refusal относительно concept_dim), NRAND (200), SEED (21).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                       # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"

NRAND="${NRAND:-200}"
SEED="${SEED:-21}"
BASE="$FORK/results/cones"
OUT="$EXP/reports/seed-agreement.md"

[ -d "$BASE" ] || { echo "нет каталога конусов: $BASE" >&2; exit 1; }
mkdir -p "$EXP/reports"

echo "base=$BASE  NRAND=$NRAND  SEED=$SEED  out=$OUT"
cd "$FORK"
uv run --with scipy python "$EXP/scripts/seed_agreement.py" \
  --base "$BASE" --n_rand "$NRAND" --seed "$SEED" --out "$OUT"
