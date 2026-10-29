#!/usr/bin/env bash
# Энергия компонент двух прогонов друг в друге и в наборах векторов.
# Как и compare_to_base.sh: векторы на сервере, python едет туда.
# Импортирует compare_to_base.py, поэтому оба файла копируются в temp-каталог рядом.
#
#   bash run/cross_energy.sh > reports/cross-energy.md
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
HERE="$(cd "$(dirname "$0")" && pwd)"
D=".fz_cross_energy"
tar czf - -C "$HERE" compare_to_base.py cross_energy.py \
  | ssh "$HOST" "cd $REPO && mkdir -p $D && tar xzf - -C $D \
    && PYTHONPATH=$D uv run python $D/cross_energy.py; rm -rf $REPO/$D"
