#!/usr/bin/env bash
# Шаг 4 — свип λ_inv (GPU): invariant_0.3, invariant_0.1, invariant_0.03 × 4 домена × 3 сида,
# затем геометрия и каузальные замеры (дозамеряются только новые направления).
# Два процесса обучения на одной карте (по ≈ 34 ГБ), домены поделены пополам; внутри процесса
# варианты идут от большего λ к меньшему, поэтому первым целиком готов invariant_0.3.
# Готовые сиды пропускаются: прерванный прогон просто перезапустить.
#
# Использование:
#   GPU=1 bash experiments/bypass_only/run/4_sweep.sh
#
# Env: GPU, HALF_A ("O1 O2"), HALF_B ("O5 O6"), FORK, SRC, OUT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_gpu
VARS=(invariant_0.3 invariant_0.1 invariant_0.03)
HALF_A="${HALF_A:-O1 O2}"
HALF_B="${HALF_B:-O5 O6}"
STAMP="$(date +%Y%m%d-%H%M%S)"

train_half() {
  local doms="$1" tag; tag="$(echo "$doms" | tr ' ' '-')"
  read -ra D <<< "$doms"
  py train.py --out "$OUT" --variants "${VARS[@]}" --domains "${D[@]}" \
    > "$OUT/logs/train-sweep-$tag-$STAMP.log" 2>&1
}

echo "$(date '+%F %T') свип λ_inv: ${VARS[*]}  GPU=$GPU  половины: [$HALF_A] [$HALF_B]"
train_half "$HALF_A" & PA=$!
sleep 120   # второй процесс — после загрузки модели первым, чтобы пики памяти не совпали
train_half "$HALF_B" & PB=$!
rc=0
wait $PA || { echo "$(date '+%F %T') ОШИБКА обучения [$HALF_A]"; rc=1; }
wait $PB || { echo "$(date '+%F %T') ОШИБКА обучения [$HALF_B]"; rc=1; }
[[ $rc -eq 0 ]] || { echo "обучение не завершено — перезапустить (готовые сиды пропустятся)"; exit 1; }
echo "$(date '+%F %T') обучение готово"

bash "$EXP/run/2_geometry.sh"
echo "$(date '+%F %T') геометрия готова"
bash "$EXP/run/3_causal.sh"
echo "$(date '+%F %T') свип завершён"
