"""Какая часть DIM лежит вдоль каждой оси-кандидата — одной мерой.

Мера: энергия набора вдоль направления, E(w, V) = (1/N) sum (v_hat . w)^2, то есть
средний cos^2 оси с векторами набора. Это та же величина, которой мерится lam_1
(lam_1 = max_w E), поэтому у каждого набора есть потолок, и единая сопоставимая
мера — доля потолка E / lam_1 от 0 до 1.

Обратная мера для контроля: сколько самой оси лежит в линейной оболочке 4 DIM,
||Proj_span(DIM) w||^2. Случайное направление дало бы 4/2048 = 0.002.

Запускать на сервере из корня shared-refusal-component: bash run/dim_along_axes.sh
"""
import torch

from compare_to_base import BT, SD, spectrum, unit, vectors


def energy(wv, V):
    return float((V @ wv).pow(2).mean())


def in_span(wv, V):
    """Доля квадрата нормы направления в линейной оболочке строк V."""
    Q = torch.linalg.qr(V.T)[0]
    return float((Q.T @ wv).pow(2).sum())


def main():
    vec = {r: vectors(r) for r in (BT, SD)}
    pick = lambda r, p: torch.stack([v for n, v in vec[r].items() if n.startswith(p)])
    dim = {r: pick(r, "dim_") for r in (BT, SD)}
    rdo = {r: pick(r, "rdo_") for r in (BT, SD)}
    dim8 = torch.cat([dim[BT], dim[SD]])

    def w(run, var):
        from compare_to_base import w as _w
        return _w(run, var)

    pool = unit(spectrum(torch.cat([rdo[BT], rdo[SD]]))[1][:, 0])
    bisect = unit(w(BT, "rdo_only") + torch.sign(w(BT, "rdo_only") @ w(SD, "all")) * w(SD, "all"))
    g = torch.Generator().manual_seed(0)
    rand = unit(torch.randn(dim[BT].shape[1], generator=g, dtype=torch.float64))

    SETS = [
        ("наши 4 DIM", dim[BT]),
        ("базовые 4 DIM", dim[SD]),
        ("все 8 DIM", dim8),
    ]
    AXES = [
        ("наша компонента без DIM", w(BT, "rdo_only")),
        ("наша компонента по всем 24", w(BT, "all")),
        ("базовая компонента без DIM", w(SD, "rdo_only")),
        ("базовая компонента по всем 24", w(SD, "all")),
        ("общая ось 40 RDO двух прогонов", pool),
        ("биссектриса двух компонент", bisect),
        ("наш DIM-компонент (сам набор)", w(BT, "dim_only")),
        ("базовый DIM-компонент", w(SD, "dim_only")),
        ("случайное направление", rand),
    ]
    lam = {name: float(spectrum(V)[0][0]) for name, V in SETS}

    print("## Энергия DIM вдоль оси: E = средний cos^2\n")
    print("| ось | " + " | ".join(n for n, _ in SETS) + " |")
    print("|---|" + "---|" * len(SETS))
    print("| **потолок lam_1 набора** | "
          + " | ".join(f"**{lam[n]:.3f}**" for n, _ in SETS) + " |")
    for an, av in AXES:
        print(f"| {an} | " + " | ".join(f"{energy(av, V):.3f}" for _, V in SETS) + " |")

    print("\n## Единая мера: доля потолка E / lam_1\n")
    print("| ось | " + " | ".join(n for n, _ in SETS) + " |")
    print("|---|" + "---|" * len(SETS))
    for an, av in AXES:
        print(f"| {an} | "
              + " | ".join(f"{energy(av, V) / lam[n]:.3f}" for n, V in SETS) + " |")

    print("\n## Для сравнения: та же мера на наборах RDO\n")
    RSETS = [("наши 20 RDO", rdo[BT]), ("базовые 20 RDO", rdo[SD])]
    rlam = {n: float(spectrum(V)[0][0]) for n, V in RSETS}
    print("| ось | " + " | ".join(n for n, _ in RSETS) + " |")
    print("|---|" + "---|" * len(RSETS))
    for an, av in AXES:
        print(f"| {an} | "
              + " | ".join(f"{energy(av, V) / rlam[n]:.3f}" for n, V in RSETS) + " |")

    print("\n## Обратная мера: сколько оси лежит в оболочке 4 DIM\n")
    print("| ось | в оболочке наших 4 DIM | в оболочке базовых 4 DIM |")
    print("|---|---|---|")
    for an, av in AXES:
        print(f"| {an} | {in_span(av, dim[BT]):.3f} | {in_span(av, dim[SD]):.3f} |")
    print(f"\nСлучайное направление дало бы 4/{dim[BT].shape[1]} = "
          f"{4 / dim[BT].shape[1]:.4f}.")


if __name__ == "__main__":
    main()
