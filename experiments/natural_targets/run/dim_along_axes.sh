#!/usr/bin/env bash
# Какая часть DIM лежит вдоль каждой оси-кандидата, одной мерой.
# Как и compare_to_base.sh: векторы на сервере, python едет туда.
# Импортирует compare_to_base.py, поэтому оба файла копируются в temp-каталог рядом.
#
#   bash run/dim_along_axes.sh > reports/dim-along-axes.md
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
HERE="$(cd "$(dirname "$0")" && pwd)"
D=".fz_dim_along_axes"
tar czf - -C "$HERE" compare_to_base.py dim_along_axes.py \
  | ssh "$HOST" "cd $REPO && mkdir -p $D && tar xzf - -C $D \
    && PYTHONPATH=$D uv run python $D/dim_along_axes.py; rm -rf $REPO/$D"
