#!/usr/bin/env bash
# Сводит ablation.json этапа 03 в markdown-таблицы для reports/results.md.
# Как и compare_to_base.sh: данные на сервере, поэтому python едет туда по stdin.
#
#   bash run/summarize_ablation.sh > reports/ablation-summary.md
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ssh "$HOST" "cd $REPO && uv run python -" < "$(dirname "$0")/summarize_ablation.py"
