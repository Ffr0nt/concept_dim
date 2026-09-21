"""DIM как шумонезависимый ориентир: что сохранено и различает ли он концепты. CPU.

DIM (diff-in-means) считается детерминированно, без сида — поэтому им можно сравнивать
концепты, не упираясь в run-дисперсию обученных конусов (части 1-4 отчёта).

ГЛАВНАЯ ЛОВУШКА, которую скрипт разводит. `select_direction` выбирает точку (pos, layer)
НЕЗАВИСИМО для каждой ступени. Если сравнивать сохранённые direction.pt «как есть»,
косинусы кластеризуются по точке извлечения, а не по концепту, и таблица меряет место,
а не концепт. Поэтому скрипт печатает три вещи рядом:
  1. наивное сравнение сохранённых direction.pt (с конфаундером);
  2. сравнение в ОБЩЕЙ точке — конфаундер снят;
  3. контраст: один и тот же концепт в двух разных точках.
Если (3) даёт меньший косинус, чем (2), то место извлечения важнее концепта.

Полная сетка кандидатов лежит в mean_diffs.pt [5 позиций, 36 слоёв, 2048]: DIM в любой
точке пересчитывается отсюда, без GPU. Индексация negative-friendly:
mean_diffs[pos, layer] при pos из range(-5, 0) — как в select_direction.py:186.

Слой 0 вырожден (нулевой mean_diff -> nan в косинусе) и исключён из усреднений.

Запуск:
  bash experiments/learned_directions/run/4_dim_check.sh
  python dim_check.py --base <fork>/results --model Qwen2.5-3B-Instruct
"""
import argparse
import itertools
import json
import os

import torch

RUNGS = ["theft", "illegal_activities", "malicious_use", "all"]
PAIRS = list(itertools.combinations(RUNGS, 2))


def cos(a, b):
    return torch.dot(a / a.norm(), b / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/ форка")
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    D, MD, SEL = {}, {}, {}
    print("=== что сохранено ===")
    for r in RUNGS:
        p = os.path.join(args.base, "dim", r, args.model)
        if not os.path.isdir(p):
            print(f"  {r:<20} НЕТ DIM в {p}")
            continue
        D[r] = torch.load(f"{p}/direction.pt", map_location="cpu").double()
        MD[r] = torch.load(f"{p}/generate_directions/mean_diffs.pt", map_location="cpu").double()
        meta = json.load(open(f"{p}/direction_metadata.json"))
        SEL[r] = (meta["pos"], meta["layer"])
        sd = sorted(os.listdir(f"{p}/select_direction"))
        print(f"  {r:<20} direction.pt {tuple(D[r].shape)} {D[r].dtype}  "
              f"pos={meta['pos']} layer={meta['layer']}  ||DIM||={D[r].norm():.4f}")
        print(f"  {'':<20} mean_diffs.pt {tuple(MD[r].shape)}   select_direction/: {sd}")

    have = [r for r in RUNGS if r in D]
    if len(have) < 2:
        return
    print("\n  ||DIM|| используется как alpha при addition (eval_cones.py:61).")

    print("\n=== 1. НАИВНО: сохранённые direction.pt как есть (С КОНФАУНДЕРОМ точки) ===")
    print("| |" + " | ".join(r[:12] for r in have) + " |")
    for a in have:
        row = " | ".join(f"{cos(D[a], D[b]):+.4f}" for b in have)
        print(f"| {a[:18]:<18} | {row} |")
    sites = sorted(set(SEL[r] for r in have))
    print(f"  выбранные точки: " + ", ".join(f"{r}={SEL[r]}" for r in have))
    if len(sites) > 1:
        print("  !! точки РАЗНЫЕ -> кластеры в таблице выше могут отражать место, а не концепт.")

    print("\n=== 2. В ОБЩЕЙ точке (конфаундер снят) ===")
    for p, l in sites:
        print(f"\n  site pos={p} layer={l}:")
        for a, b in itertools.combinations(have, 2):
            print(f"    {a[:12]:<13}/{b[:12]:<13} cos={cos(MD[a][p, l], MD[b][p, l]):+.4f}")

    n_layers = MD[have[0]].shape[1]
    print(f"\n  среднее |cos| по всем 5x{n_layers - 1} точкам (слой 0 вырожден, исключён):")
    for a, b in itertools.combinations(have, 2):
        cs = [abs(cos(MD[a][p, l], MD[b][p, l]))
              for p in range(-MD[a].shape[0], 0) for l in range(1, n_layers)]
        print(f"    {a[:12]:<13}/{b[:12]:<13} mean={sum(cs)/len(cs):.4f}  "
              f"min={min(cs):.4f}  max={max(cs):.4f}")

    if len(sites) > 1:
        print("\n=== 3. КОНТРАСТ: тот же концепт, разные точки ===")
        (p1, l1), (p2, l2) = sites[0], sites[-1]
        for r in have:
            print(f"  {r:<20} cos(DIM@{(p1,l1)}, DIM@{(p2,l2)}) = {cos(MD[r][p1,l1], MD[r][p2,l2]):+.4f}")
        print("\n  Если эти значения НИЖЕ межконцептных из п.2 — точка извлечения влияет")
        print("  сильнее концепта, и её надо фиксировать одну на все ступени.")


if __name__ == "__main__":
    main()
