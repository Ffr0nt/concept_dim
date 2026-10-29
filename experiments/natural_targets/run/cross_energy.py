"""Сколько энергии компонента одного прогона составляет в другом прогоне.

Два разных вопроса, и их нельзя смешивать.

1. Энергия направления в направлении: cos^2. Для единичных векторов симметрична,
   поэтому «друг в друге» — одно число.
2. Энергия направления в НАБОРЕ векторов: E(w, V) = w^T P w при P = (1/N) sum v v^T,
   то есть средний cos^2 направления со всеми векторами набора. Это та же величина,
   максимум которой по w и есть lam_1. Она асимметрична: своя компонента в чужом
   наборе и чужая в своём — разные числа, и сравнивать их надо с lam_1 набора,
   то есть с наилучшим возможным.

Запускать на сервере из корня shared-refusal-component: bash run/cross_energy.sh
"""
import torch

from compare_to_base import BT, SD, VARIANTS, cosabs, spectrum, unit, vectors, w

LABEL = {BT: "наш", SD: "базовый"}


def P(V):
    return V.T @ V / V.shape[0]


def energy(wv, V):
    """Средний cos^2 направления с векторами набора = w^T P w."""
    return float((V @ wv).pow(2).mean())


def subset(vecs, pref):
    names = [n for n in vecs if n.startswith(pref)]
    return torch.stack([vecs[n] for n in names]), names


def main():
    vec = {r: vectors(r) for r in (BT, SD)}
    rdo = {r: subset(vec[r], "rdo_")[0] for r in (BT, SD)}
    alln = {r: torch.stack(list(vec[r].values())) for r in (BT, SD)}
    pooled = spectrum(torch.cat([rdo[BT], rdo[SD]]))[1][:, 0]
    pooled = unit(pooled)

    print("## 1. Энергия компонент друг в друге (cos^2, симметрично)\n")
    print("| наш \\ базовый | " + " | ".join(VARIANTS) + " |")
    print("|---|" + "---|" * len(VARIANTS))
    for v in VARIANTS:
        row = [f"{cosabs(w(BT, v), w(SD, u))**2:.3f}" for u in VARIANTS]
        print(f"| {v} | " + " | ".join(row) + " |")

    print("\n## 2. Энергия направления в наборе 20 RDO (средний cos^2)\n")
    print("Потолок строки — lam_1 этого набора: наилучшее, чего может достичь любое"
          " направление.\n")
    print("| направление | в наших 20 RDO | в базовых 20 RDO |")
    print("|---|---|---|")
    rows = [
        ("lam_1 набора (потолок)", None),
        ("наша компонента rdo_only", w(BT, "rdo_only")),
        ("наша компонента all", w(BT, "all")),
        ("базовая компонента rdo_only", w(SD, "rdo_only")),
        ("базовая компонента all", w(SD, "all")),
        ("общая ось 40 RDO двух прогонов", pooled),
        ("наш DIM union", w(BT, "dim_only")),
        ("базовый DIM union", w(SD, "dim_only")),
    ]
    lam = {r: float(spectrum(rdo[r])[0][0]) for r in (BT, SD)}
    for name, wv in rows:
        if wv is None:
            print(f"| **{name}** | **{lam[BT]:.3f}** | **{lam[SD]:.3f}** |")
        else:
            print(f"| {name} | {energy(wv, rdo[BT]):.3f} | {energy(wv, rdo[SD]):.3f} |")

    print("\n## 3. Доля потолка: насколько чужая компонента хуже своей\n")
    print("| набор | своя компонента | чужая компонента | доля от потолка, своя / чужая |")
    print("|---|---|---|---|")
    for r, o in ((BT, SD), (SD, BT)):
        own, alien = energy(w(r, "rdo_only"), rdo[r]), energy(w(o, "rdo_only"), rdo[r])
        print(f"| {LABEL[r]} 20 RDO | {own:.3f} | {alien:.3f} | "
              f"{own / lam[r]:.3f} / {alien / lam[r]:.3f} |")

    print("\n## 4. Энергия в подпространстве чужих RDO по числу измерений\n")
    print("| k | базовая rdo_only в наших top-k | наша rdo_only в базовых top-k |")
    print("|---|---|---|")
    Wb = spectrum(rdo[BT])[1]
    Ws = spectrum(rdo[SD])[1]
    for k in (1, 2, 3, 5, 10, 20):
        a = float((Wb[:, :k].T @ w(SD, "rdo_only")).pow(2).sum())
        b = float((Ws[:, :k].T @ w(BT, "rdo_only")).pow(2).sum())
        print(f"| {k} | {a:.3f} | {b:.3f} |")
    d = rdo[BT].shape[1]
    print(f"\nСлучайное направление дало бы k/{d}: при k=1 это {1 / d:.4f}, "
          f"при k=20 — {20 / d:.4f}.")

    print("\n## 5. Энергия всех 24 направлений прогона вдоль чужой компоненты\n")
    for r, o in ((BT, SD), (SD, BT)):
        e = energy(w(o, "all"), alln[r])
        l1 = float(spectrum(alln[r])[0][0])
        print(f"- {LABEL[r]} набор 24: чужая компонента держит {e:.3f} "
              f"из потолка {l1:.3f} ({e / l1:.3f} потолка)")


if __name__ == "__main__":
    main()
