"""Общее для шагов эксперимента loss_terms: варианты лосса, сиды, пути, загрузка векторов.

Варианты и сиды заданы только здесь. Данные (сплиты, DIM, кэш целей, гиперпараметры) — из
базового запуска shared-refusal-component; скрипты запускаются в его окружении
(`uv run` из корня shared-refusal-component), поэтому пакет shared_refusal импортируется.
"""
import os

import torch

from shared_refusal.artifacts import REPO, DirectionSet, Run
from shared_refusal.config import load_config
from shared_refusal.directions import rdo

BASE_CONFIG = os.path.join(REPO, "configs", "qwen2.5-3b_salad-o1o2o5o6.yaml")
N_SEEDS = 3   # первые N сидов каждого домена базового запуска — пары с тем же стартом

# Изменения относительно rdo.hparams базового конфига. base не обучается.
# free_norm: norm_lambda — по итогам калибровки. Если NORM_LAMBDA здесь не задан, train.py берёт
# решение правила из <out>/calibration.json (пишет scripts/calibration.py); нет файла — обучение
# free_norm не стартует.
NORM_LAMBDA = None
VARIANTS = {
    "base": None,
    "no_add": {"addition_lambda": 0},
    "no_abl": {"ablation_lambda": 0},
    "free_norm": {"free_norm": True, "norm_lambda": NORM_LAMBDA},
}
TRAINED = [v for v, spec in VARIANTS.items() if spec is not None]


def base():
    """(cfg, run) базового запуска."""
    cfg = load_config(BASE_CONFIG)
    return cfg, Run.from_config(cfg)


def seeds(cfg, domain):
    return list(cfg["rdo"]["seeds"][domain][:N_SEEDS])


def variant_dir(out_root, variant, domain):
    """Каталог seed_XX.pt варианта; у base — каталог RDO базового запуска (только чтение)."""
    if variant == "base":
        return base()[1].rdo(domain)
    return os.path.join(out_root, variant, domain)


def load_variant(out_root, variant, cfg):
    """(DirectionSet векторов варианта, DirectionSet их стартов, {имя: сырой словарь seed_XX.pt}).
    Имена — <домен>_s<сид>: одинаковы во всех вариантах, по ним идёт парное сравнение."""
    names, vecs, meta, inits, raw = [], [], [], [], {}
    for d in cfg["domains"]:
        for s in seeds(cfg, d):
            r = rdo.load_seed(variant_dir(out_root, variant, d), s)
            n = f"{d}_s{s:02d}"
            names.append(n)
            vecs.append(r["vector"])
            inits.append(r["init_vector"])
            meta.append({"method": variant, "domain": d, "seed": s,
                         "norm": float(r["vector"].norm()), "lowest_loss": float(r["lowest_loss"])})
            raw[n] = r
    return (DirectionSet.from_raw(names, vecs, meta),
            DirectionSet.from_raw(names, inits, [dict(m, method="init") for m in meta]), raw)


def available(out_root, cfg):
    """Варианты, у которых есть все сиды всех доменов."""
    out = []
    for v in VARIANTS:
        ok = all(not rdo.pending(variant_dir(out_root, v, d), seeds(cfg, d)) for d in cfg["domains"])
        if ok:
            out.append(v)
    return out


def dims(run, cfg):
    """DIM доменов и объединённой выборки в общей точке, сырые (с нормой)."""
    return {d: torch.load(os.path.join(run.dim(d), "direction.pt"), map_location="cpu", weights_only=True)
            for d in list(cfg["domains"]) + ["union"]}


def point(run):
    m = run.manifest()["dim"]
    return {"pos": m["pos"], "layer": m["layer"]}


def rdo_only_component(run):
    """Общая компонента всех 20 RDO базового запуска (вариант rdo_only этапа 02), если посчитана."""
    p = os.path.join(run.component("rdo_only"), "component.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=True)["w"].double()

