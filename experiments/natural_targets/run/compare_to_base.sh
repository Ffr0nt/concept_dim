#!/usr/bin/env bash
# Считает геометрию прогона natural_targets против базового прогона на сервере.
# Скрипт лежит здесь (в репозитории диссертации), а векторы — на сервере, поэтому
# python-файл передаётся по ssh на stdin и исполняется в venv пайплайна.
#
#   bash run/compare_to_base.sh > reports/geometry.md
set -euo pipefail
HOST="${HOST:-peaceful-matsumoto}"
REPO="${REPO:-~/f.zakharov/shared-refusal-component}"
ssh "$HOST" "cd $REPO && uv run python -" < "$(dirname "$0")/compare_to_base.py"
