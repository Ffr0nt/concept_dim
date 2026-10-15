"""Варианты эксперимента bypass_only поверх кода loss_terms.

Код шагов (обучение, геометрия, каузальные замеры) общий с loss_terms — обёртки рядом
запускают его скрипты, а этот модуль подменяет для них `common`: свои варианты и откуда брать
векторы. Векторы base — из базового запуска shared-refusal-component, no_add и no_abl — из
результатов loss_terms (<fork>/results/loss_terms/); обучается здесь только invariant.
"""
import importlib.util
import os

_LT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "loss_terms", "scripts")
_spec = importlib.util.spec_from_file_location("lt_common", os.path.join(_LT, "common.py"))
_lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lt)

# Вес слагаемого инвариантности: как у retain (тоже KL на тех же harmless-позициях).
INV_LAMBDA = 1.0
# Свип веса (15.10): при λ_inv = 1 векторы каузально инертны — проверяем, свойство ли это веса.
# 0.01 и 0.003 добавлены после первого свипа: при 0.03 разрыв по ASR с no_add ещё не закрыт.
INV_SWEEP = (0.3, 0.1, 0.03, 0.01, 0.003)

# None — вариант не обучается здесь (векторы готовы).
VARIANTS = {
    "base": None,
    "no_add": None,
    "no_abl": None,
    "invariant": {"addition_mode": "invariant", "addition_lambda": INV_LAMBDA},
    **{f"invariant_{lam:g}": {"addition_mode": "invariant", "addition_lambda": lam} for lam in INV_SWEEP},
}
REUSED_FROM_LOSS_TERMS = ("no_add", "no_abl")


def variant_dir(out_root, variant, domain):
    if variant == "base":
        return _lt.base()[1].rdo(domain)
    if variant in REUSED_FROM_LOSS_TERMS:
        # out_root = <fork>/results/bypass_only -> соседний <fork>/results/loss_terms
        return os.path.join(os.path.dirname(os.path.abspath(out_root)), "loss_terms", variant, domain)
    return os.path.join(out_root, variant, domain)


# функции loss_terms читают VARIANTS и variant_dir из своего модуля — подменяем их там
_lt.VARIANTS = VARIANTS
_lt.variant_dir = variant_dir

BASE_CONFIG, N_SEEDS = _lt.BASE_CONFIG, _lt.N_SEEDS
TRAINED = [v for v, spec in VARIANTS.items() if spec is not None]
base, seeds, load_variant, available = _lt.base, _lt.seeds, _lt.load_variant, _lt.available
dims, point, rdo_only_component = _lt.dims, _lt.point, _lt.rdo_only_component
