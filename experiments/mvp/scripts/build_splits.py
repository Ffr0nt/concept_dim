"""Этап 1 — сборка per-rung сплитов SALAD для MVP concept-dim.

Для каждой ступени вложенной лестницы (Theft ⊂ Illegal Activities ⊂ Malicious Use)
создаёт в форке geometry-of-refusal:
    data/<rung>_splits/harmful_train.json   — N сэмплов SALAD этой категории
    data/<rung>_splits/harmless_train.json  — общий benign-набор (копия)

Формат harmful — как ждёт rdo.py: [{"instruction": ..., "source": ...}] (читается только instruction).
Категорию фильтруем здесь (в рантайме rdo.py фильтра нет). Сэмпл детерминирован seed'ом.

Запуск на сервере (CPU):
    cd ~/f.zakharov/concept_dim
    uv run --with datasets --with pandas python experiments/mvp/scripts/build_splits.py
"""
import argparse
import json
import os
import shutil

from datasets import load_dataset

# ступень -> (колонка таксономии, значение)
RUNGS = {
    "theft":              ("3-category", "O57: Theft"),
    "illegal_activities": ("2-category", "O14: Illegal Activities"),
    "malicious_use":      ("1-category", "O5: Malicious Use"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default="../geometry-of-refusal",
                    help="путь к репозиторию geometry-of-refusal (рядом с concept_dim)")
    ap.add_argument("--n", type=int, default=900, help="размер выборки на ступень")
    ap.add_argument("--seed", type=int, default=21, help="сид сэмплирования (один прогон)")
    ap.add_argument("--harmless-src", default=None,
                    help="harmless_train.json для переиспользования "
                         "(по умолчанию data/saladbench_splits/harmless_train.json из форка)")
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    data_dir = os.path.join(fork, "data")
    harmless_src = args.harmless_src or os.path.join(
        data_dir, "saladbench_splits", "harmless_train.json")

    harmless = json.load(open(harmless_src))
    assert len(harmless) >= args.n, f"harmless набор {len(harmless)} < N={args.n}"
    print(f"harmless source: {harmless_src} ({len(harmless)} шт.)")

    print("загрузка SALAD base_set ...")
    df = load_dataset("OpenSafetyLab/Salad-Data", name="base_set")["train"].to_pandas()

    for name, (col, val) in RUNGS.items():
        pool = df[df[col] == val]
        assert len(pool) >= args.n, f"{name}: пул {len(pool)} < N={args.n}"
        sample = pool.sample(n=args.n, random_state=args.seed)
        harmful = [{"instruction": q, "source": f"salad:{name}"}
                   for q in sample["question"].tolist()]

        out_dir = os.path.join(data_dir, f"{name}_splits")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "harmful_train.json"), "w") as f:
            json.dump(harmful, f, ensure_ascii=False, indent=2)
        shutil.copyfile(harmless_src, os.path.join(out_dir, "harmless_train.json"))

        print(f"{name:20s} пул={len(pool):5d} -> harmful={len(harmful)}  ({out_dir})")

    print("готово.")


if __name__ == "__main__":
    main()
