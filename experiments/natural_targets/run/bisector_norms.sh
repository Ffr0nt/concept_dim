#!/usr/bin/env bash
# Пересечение двух общих компонент: снимает ли отказ и какая у него норма.
# Как и compare_to_base.sh: векторы на сервере, python едет туда.
# Импортирует compare_to_base.py, поэтому оба файла копируются в temp-каталог рядом.
#
#   bash run/bisector_norms.sh > reports/dim-along-axes.md
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
HERE="$(cd "$(dirname "$0")" && pwd)"
D=".fz_bisector_norms"
tar czf - -C "$HERE" compare_to_base.py bisector_norms.py \
  | ssh "$HOST" "cd $REPO && mkdir -p $D && tar xzf - -C $D \
    && PYTHONPATH=$D uv run python $D/bisector_norms.py; rm -rf $REPO/$D"
