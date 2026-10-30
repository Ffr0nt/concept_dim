"""Строит направления-кандидаты для поиска минимальной доли DIM и печатает их геометрию.

Кандидаты — однопараметрическое семейство на дуге между осью отказа и её же версией,
из которой проекция на DIM вычтена полностью:

    w(t) = normalize((1 - t) * w_perp + t * u),   t от 0 до 1,

где u — ось отказа (общая компонента), w_perp = normalize(u - (u . g_hat) g_hat),
g — вектор DIM union в пространстве активаций. По построению rho(w(0)) = 0 и
rho(w(1)) = |u . g_hat|, и rho(t) монотонна, поэтому порог ищется делением отрезка.

Пишет набор направлений в component/minimal_axis_directions.pt прогона, чтобы этапы
03/06/09 пайплайна могли взять его через ablation.extra / generate.extra.

Запускать на сервере из корня shared-refusal-component: bash run/build_candidates.sh
"""
import argparse
import os

import torch

BT = "results/qwen2.5-3b_beavertails-b1b5b6bv"
SD = "results/qwen2.5-3b_salad-o1o2o5o6"
TS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float64).flatten()
    return v / v.norm()


def comp(run, var):
    p = os.path.join(run, "component", var, "component.pt")
    return unit(torch.load(p, map_location="cpu", weights_only=True)["w"])


def dim_union(run):
    """Вектор DIM union в пространстве активаций, с настоящей нормой."""
    return torch.load(os.path.join(run, "dim/union/direction.pt"),
                      map_location="cpu", weights_only=True).double()


def rdo_vectors(run):
    import json
    with open(os.path.join(run, "manifest.json")) as f:
        names = json.load(f)["directions"]["names"]
    out = []
    for n in names:
        if n.startswith("rdo_"):
            _, dom, seed = n.split("_")
            o = torch.load(os.path.join(run, "rdo", dom, f"seed_{seed[1:]}.pt"),
                           map_location="cpu", weights_only=False)
            out.append(unit(o["vector"]))
    return torch.stack(out)


def spectrum(V):
    lam, U = torch.linalg.eigh(V @ V.T / V.shape[0])
    o = torch.argsort(lam, descending=True)
    W = V.T @ U[:, o]
    return lam[o], W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)


def arc(u, g_hat, t):
    perp = unit(u - (u @ g_hat) * g_hat)
    return unit((1 - t) * perp + t * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="pool",
                    choices=["pool", "bt_rdo_only", "bt_all", "sd_all"])
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    g = dim_union(BT)
    g_hat = unit(g)
    axes = {
        "bt_rdo_only": comp(BT, "rdo_only"),
        "bt_all": comp(BT, "all"),
        "sd_all": comp(SD, "all"),
        "pool": unit(spectrum(torch.cat([rdo_vectors(BT), rdo_vectors(SD)]))[1][:, 0]),
    }
    u = axes[a.axis]

    print(f"## Опорная ось: {a.axis}\n")
    print(f"- ||g|| (DIM union в активациях): {float(g.norm()):.2f}")
    print(f"- rho(u) = |u . g_hat| = {float(abs(u @ g_hat)):.4f}")
    print(f"- |cos| оси с её DIM-свободной версией: "
          f"{float(abs(u @ arc(u, g_hat, 0.0))):.4f} "
          f"(тождество sqrt(1 - rho^2) = {(1 - float(u @ g_hat) ** 2) ** 0.5:.4f})\n")

    print("## Семейство кандидатов\n")
    print("| t | rho = доля DIM | \\|cos\\| с опорной осью | \\|g.w\\| в активациях |")
    print("|---|---|---|---|")
    for t in TS:
        w = arc(u, g_hat, t)
        print(f"| {t:.2f} | {float(abs(w @ g_hat)):.4f} | {float(abs(w @ u)):.4f} "
              f"| {float(abs(g @ w)):.2f} |")

    print("\n## Та же дуга от других осей: |cos| DIM-свободной версии со своей осью\n")
    print("| ось | rho(ось) | \\|cos\\| с DIM-свободной версией |")
    print("|---|---|---|")
    for n, v in axes.items():
        print(f"| {n} | {float(abs(v @ g_hat)):.4f} | "
              f"{float(abs(v @ arc(v, g_hat, 0.0))):.4f} |")

    print("\n## Допуск аблации по углу (известное из прогона natural_targets)\n")
    print("| направление | \\|cos\\| с осью pool | оценка судьи |")
    print("|---|---|---|")
    pool = axes["pool"]
    for n, v, s in (("наша компонента без DIM", axes["bt_rdo_only"], "0.852"),
                    ("базовая компонента", axes["sd_all"], "0.872"),
                    ("сама ось pool", pool, "0.863")):
        print(f"| {n} | {float(abs(v @ pool)):.4f} | {s} |")
    print("| среднее отдельного RDO | ~0.53 (медиана) | 0.413 |")

    if a.save:
        names = [f"w_t{int(round(t * 100)):03d}" for t in TS]
        V = torch.stack([arc(u, g_hat, t) for t in TS]).float()
        torch.save({"V": V, "names": names,
                    "meta": {"axis": a.axis, "ts": TS, "kind": "component"}}, a.save)
        print(f"\nнабор сохранён: {a.save} ({', '.join(names)})")


if __name__ == "__main__":
    main()
