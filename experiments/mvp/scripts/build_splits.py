"""Этап 1 — сборка per-rung сплитов SALAD для MVP concept-dim.

Для каждой ступени вложенной лестницы (Theft ⊂ Illegal Activities ⊂ Malicious Use)
создаёт в форке geometry-of-refusal каталог data/<rung>_splits/ с файлами:
    harmful_{train,val,test}.json   — сэмплы SALAD этой категории
    harmless_{train,val,test}.json  — общий benign-набор (одинаков для всех ступеней)

Зачем train+val+test:
- rdo.py (обучение конуса) читает только *_train.json;
- пайплайн DIM (run_pipeline) читает train + val (+ test для поздней, неблокирующей оценки).

Формат — как ждёт код: [{"instruction": ..., "source": ...}] (читается только instruction).
Категорию фильтруем здесь; сэмпл детерминирован seed'ом (train=seed, val=seed+1, test=seed+2).

⚠️ Для узкого листа (Theft, пул 964) train=900 + val/test пересекаются с train — допустимо
для отбора слоя DIM (не строгий held-out); для крупных ступеней пересечение минимально.

Запуск на сервере (CPU):
    cd ~/f.zakharov/concept_dim
    uv run --with datasets --with pandas python experiments/mvp/scripts/build_splits.py
"""
import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

# ступень -> (колонка таксономии, значение)
RUNGS = {
    "theft":              ("3-category", "O57: Theft"),
    "illegal_activities": ("2-category", "O14: Illegal Activities"),
    "malicious_use":      ("1-category", "O5: Malicious Use"),
}
SIZES = {"train": 900, "val": 128, "test": 128}
SEED_OFFSET = {"train": 0, "val": 1, "test": 2}


def emit(records, path):
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def stratified_sample(pool, n, strat_col, seed):
    """Стратифицированная выборка n строк из pool с сохранением пропорций strat_col
    (по самому мелкому уровню — 3-category). Распределение остатка — по наибольшей
    дробной части (largest-remainder), чтобы сумма была ровно n."""
    counts = pool[strat_col].value_counts()
    exact = counts / counts.sum() * n
    alloc = exact.apply(int)                       # floor
    rem = n - int(alloc.sum())
    fracs = (exact - alloc).sort_values(ascending=False)
    for cat in list(fracs.index)[:rem]:
        alloc[cat] += 1
    parts = []
    for cat, k in alloc.items():
        if k > 0:
            grp = pool[pool[strat_col] == cat]
            parts.append(grp.sample(n=min(int(k), len(grp)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)  # перемешать


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default="../geometry-of-refusal")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--harmless-src", default=None,
                    help="источник benign (по умолчанию data/saladbench_splits/harmless_train.json форка)")
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    data_dir = os.path.join(fork, "data")
    harmless_src = args.harmless_src or os.path.join(
        data_dir, "saladbench_splits", "harmless_train.json")

    # benign: перемешать один раз и нарезать НЕпересекающиеся train/val/test
    harmless_df = pd.DataFrame(json.load(open(harmless_src)))
    need = sum(SIZES.values())
    assert len(harmless_df) >= need, f"benign {len(harmless_df)} < {need}"
    hless = harmless_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    harmless_splits, idx = {}, 0
    for split, n in SIZES.items():
        chunk = hless.iloc[idx:idx + n]
        idx += n
        harmless_splits[split] = [
            {"instruction": r["instruction"], "source": r.get("source", "alpaca")}
            for _, r in chunk.iterrows()]
    print(f"benign source: {harmless_src} ({len(harmless_df)}), "
          f"нарезано train/val/test = {[len(harmless_splits[s]) for s in SIZES]}")

    print("загрузка SALAD base_set ...")
    df = load_dataset("OpenSafetyLab/Salad-Data", name="base_set")["train"].to_pandas()

    for name, (col, val) in RUNGS.items():
        pool = df[df[col] == val]
        out_dir = os.path.join(data_dir, f"{name}_splits")
        os.makedirs(out_dir, exist_ok=True)
        counts = {}
        for split, n in SIZES.items():
            k = min(n, len(pool))
            # стратифицируем по листу (3-category) — сохраняем внутренние пропорции ступени
            s = stratified_sample(pool, k, "3-category", args.seed + SEED_OFFSET[split])
            harmful = [{"instruction": q, "source": f"salad:{name}"}
                       for q in s["question"].tolist()]
            emit(harmful, os.path.join(out_dir, f"harmful_{split}.json"))
            emit(harmless_splits[split], os.path.join(out_dir, f"harmless_{split}.json"))
            counts[split] = len(harmful)
        # для наглядности — распределение train по листьям
        train_s = stratified_sample(pool, min(SIZES["train"], len(pool)), "3-category",
                                    args.seed + SEED_OFFSET["train"])
        leaf_dist = train_s["3-category"].value_counts().to_dict()
        print(f"{name:20s} пул={len(pool):5d} -> harmful {counts}")
        if len(leaf_dist) > 1:
            top = dict(list(train_s["3-category"].value_counts().items())[:4])
            print(f"    train по листьям ({len(leaf_dist)} шт.), топ-4: {top}")

    print("готово.")


if __name__ == "__main__":
    main()
