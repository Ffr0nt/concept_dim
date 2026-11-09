#!/usr/bin/env bash
# Пул семейств лосса с равными весами семейству. Векторы лежат на сервере,
# поэтому python едет туда по stdin, как в natural_targets/run/compare_to_base.sh.
#
#   bash run/pool_losses.sh                      # печатает геометрию
#   ARGS="--save results/<прогон>/component/pool_loss.pt" bash run/pool_losses.sh
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ARGS="${ARGS:-}"
ssh "$HOST" "cd $REPO && uv run python - $ARGS" < "$(dirname "$0")/pool_losses.py"
