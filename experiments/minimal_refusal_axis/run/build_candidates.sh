#!/usr/bin/env bash
# Геометрия кандидатов для поиска минимальной доли DIM. Векторы лежат на сервере,
# поэтому python едет туда по stdin, как в natural_targets/run/compare_to_base.sh.
#
#   bash run/build_candidates.sh                      # печатает геометрию
#   ARGS="--axis pool --save results/.../x.pt" bash run/build_candidates.sh
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ARGS="${ARGS:---axis pool}"
ssh "$HOST" "cd $REPO && uv run python - $ARGS" < "$(dirname "$0")/build_candidates.py"
