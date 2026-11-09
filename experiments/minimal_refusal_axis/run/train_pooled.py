"""Обучение RDO на объединении четырёх доменов — пересечение внутри лосса.

Обычный протокол обучает вектор на одном домене, а общее берёт усреднением готовых
векторов. Здесь один вектор обучается на объединении всех доменов: доменно-специфичное
решение штрафуется слагаемыми остальных доменов и просто не выучивается.

Лосс не меняется — тот же no_add (L_abl + L_ret, addition_lambda = 0), что и во всём
прогоне. Меняется только источник. Команда rdo.py строится кодом пайплайна из того же
конфига, поэтому шаблон чата, refusal-токены и гиперпараметры заведомо те же.

Запускать на сервере из корня shared-refusal-component: см. run/train_pooled.sh
"""
import argparse
import json
import os

from shared_refusal import models
from shared_refusal.artifacts import Run
from shared_refusal.config import load_config
from shared_refusal.directions import rdo

CONFIG = "configs/qwen2.5-3b_beavertails-b1b5b6bv.yaml"


def load(p):
    with open(p) as f:
        return json.load(f)


def dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"  {os.path.basename(p)}: {len(obj)}")


def build(run, domains, out):
    """Объединённые сплиты и цели.

    rdo.py режет harmless под длину harmful (rdo.py:250) и берёт min после фильтра
    (rdo.py:408). Harmful стал вчетверо длиннее, поэтому harmless и его цели повторяются
    столько же раз — иначе обучение схлопнется обратно к объёму одного домена.
    """
    harmful, targets = [], []
    for d in domains:
        hs = load(os.path.join(run.splits(d), "harmful_train.json"))
        ts = load(run.path("rdo", d, "targets", "harmful_targets.json"))
        assert len(hs) == len(ts), f"{d}: сплит {len(hs)} != цели {len(ts)}"
        assert hs[0]["instruction"] in ts[0]["prompt"], f"{d}: цели не в порядке сплита"
        harmful += hs
        targets += ts
        print(f"{d}: +{len(hs)}")
    hl = load(os.path.join(run.splits(domains[0]), "harmless_train.json"))
    hlt = load(run.path("rdo", domains[0], "targets", "harmless_targets.json"))
    assert len(hl) == len(hlt), f"harmless: сплит {len(hl)} != цели {len(hlt)}"
    k = len(domains)
    print(f"harmless повторяется {k} раза: {len(hl)} -> {len(hl) * k}\n")
    dump(harmful, f"{out}/splits/harmful_train.json")
    dump(hl * k, f"{out}/splits/harmless_train.json")
    dump(targets, f"{out}/targets/harmful_targets.json")
    dump(hlt * k, f"{out}/targets/harmless_targets.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 3, 7, 0])
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(CONFIG)
    run = Run.from_config(cfg)
    fam = models.family(cfg["model"])
    domains = list(cfg["domains"])
    out = run.path("pooled")

    build(run, domains, out)
    if a.build_only:
        return

    hp = dict(cfg["rdo"]["hparams"])
    assert hp["addition_lambda"] == 0, "ожидался лосс no_add"
    cmd = rdo.command(cfg["model"], fam, "pooled", run.path("dim", "union"),
                      f"{out}/splits", f"{out}/targets", f"{out}/rdo", a.seeds, hp)
    print("команда:\n  " + " ".join(cmd) + "\n", flush=True)
    rdo.train(cmd, out)


if __name__ == "__main__":
    main()
