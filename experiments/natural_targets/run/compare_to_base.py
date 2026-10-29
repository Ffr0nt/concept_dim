"""Геометрия: компонента прогона на естественных целях против компоненты базового прогона.

Запускать на сервере из корня shared-refusal-component (векторы лежат в results/):
    bash run/compare_to_base.sh            # см. рядом

Оба прогона извлекают направления в одной точке (pos -1, слой 27), поэтому векторы
сравнимы напрямую. Все косинусы берутся по модулю: знак направления условен.
"""
import json
import os

import torch

BT = "results/qwen2.5-3b_beavertails-b1b5b6bv"
SD = "results/qwen2.5-3b_salad-o1o2o5o6"
VARIANTS = ("all", "rdo_only", "dim_only")


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float64).flatten()
    return v / v.norm()


def spectrum(V):
    """Спектр P = (1/N) sum v v^T: (lam [N], W [hidden, N]) по убыванию."""
    lam, U = torch.linalg.eigh(V @ V.T / V.shape[0])
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order], U[:, order]
    W = V.T @ U
    return lam, W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)


def w(run, var):
    p = os.path.join(run, "component", var, "component.pt")
    return unit(torch.load(p, map_location="cpu", weights_only=True)["w"])


def summary(run, var):
    with open(os.path.join(run, "component", var, "component.json")) as f:
        return json.load(f)


def vectors(run):
    """Все 24 направления прогона в порядке манифеста: {имя: единичный вектор}."""
    with open(os.path.join(run, "manifest.json")) as f:
        names = json.load(f)["directions"]["names"]
    out = {}
    for n in names:
        if n.startswith("dim_"):
            out[n] = unit(torch.load(os.path.join(run, "dim", n[4:], "direction.pt"),
                                     map_location="cpu", weights_only=True))
        else:
            _, dom, seed = n.split("_")
            o = torch.load(os.path.join(run, "rdo", dom, f"seed_{seed[1:]}.pt"),
                           map_location="cpu", weights_only=False)
            out[n] = unit(o["vector"])
    return out


def cosabs(a, b):
    return abs(float(a @ b))


def stats(xs):
    t = torch.tensor(sorted(xs))
    return f"медиана {t.median():.3f}  от {t[0]:.3f} до {t[-1]:.3f}"


def main():
    print("## lam_1 общей компоненты (доля дисперсии в одном направлении)\n")
    print("| вариант | N | наш lam_1 | базовый lam_1 | нулевой уровень p95 |")
    print("|---|---|---|---|---|")
    for v in VARIANTS:
        a, b = summary(BT, v), summary(SD, v)
        print(f"| {v} | {a['N']} | {a['lam'][0]:.3f} | {b['lam'][0]:.3f} | {a['null']['p95']:.3f} |")

    print("\n## |cos| между компонентами прогонов\n")
    print("| наш \\ базовый | " + " | ".join(VARIANTS) + " |")
    print("|---|" + "---|" * len(VARIANTS))
    for v in VARIANTS:
        row = [f"{cosabs(w(BT, v), w(SD, u)):.3f}" for u in VARIANTS]
        print(f"| {v} | " + " | ".join(row) + " |")

    VBT, VSD = vectors(BT), vectors(SD)
    rdo_bt = [v for n, v in VBT.items() if n.startswith("rdo_")]
    rdo_sd = [v for n, v in VSD.items() if n.startswith("rdo_")]
    dim_bt = [v for n, v in VBT.items() if n.startswith("dim_")]

    print("\n## отдельные направления против чужой компоненты\n")
    for name, vs, ref in [
            ("наши 20 RDO против базовой w:all", rdo_bt, w(SD, "all")),
            ("наши 20 RDO против базовой w:rdo_only", rdo_bt, w(SD, "rdo_only")),
            ("наши 4 DIM против базовой w:all", dim_bt, w(SD, "all")),
            ("базовые 20 RDO против нашей w:rdo_only", rdo_sd, w(BT, "rdo_only")),
    ]:
        print(f"- {name}: {stats([cosabs(v, ref) for v in vs])}")

    print("\n## опора: LOO внутри каждого прогона (сид против компоненты остальных 19)\n")
    for tag, vs in [("наш", rdo_bt), ("базовый", rdo_sd)]:
        loo = []
        V = torch.stack(vs)
        for i in range(len(vs)):
            rest = torch.cat([V[:i], V[i + 1:]])
            lam, W = spectrum(rest)
            loo.append(cosabs(V[i], unit(W[:, 0])))
        print(f"- {tag} прогон: {stats(loo)}")

    print("\n## сколько базовой компоненты лежит в подпространстве наших RDO\n")
    lam, W = spectrum(torch.stack(rdo_bt))
    print("| k | доля ||w_base||^2 в top-k собственных векторах наших RDO |")
    print("|---|---|")
    for k in (1, 2, 3, 5, 10):
        share = float((W[:, :k].T @ w(SD, "all")).pow(2).sum())
        print(f"| {k} | {share:.3f} |")
    print(f"\nДля сравнения: та же доля для случайного направления ≈ k/2048 "
          f"(k=1 → {1/2048:.4f}).")


if __name__ == "__main__":
    main()
