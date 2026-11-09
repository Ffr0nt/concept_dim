"""Пересечение по-разному обученных направлений: пул семейств лосса с равными весами.

Все семейства обучены на одних данных (SALAD, прогон qwen2.5-3b_salad-o1o2o5o6), тем же
протоколом и с тех же стартов, различается только лосс. Поэтому их векторы сравнимы
напрямую — но складывать их в один пул с равными весами НА ВЕКТОР нельзя: у семейств
разное N, и сплочённое семейство перетягивает ковариацию. Поэтому

    P = (1/K) sum_семейств P_семейства,   P_семейства = (1/N) sum v_hat v_hat^T,

то есть равный вес семейству. Для сравнения считается и наивный пул с равными весами на
вектор — чтобы видно было, насколько это важно.

В пул идут только семейства, снимающие отказ по отдельности: no_abl оценивает ось наведения
(ASR 0.40 при 0.31 у случайного), invariant при lambda >= 0.1 уходит в направления, на
которые модель не реагирует. Списки — в ADMIT/REJECT, обоснование в idea.md.

Запускать на сервере из корня shared-refusal-component: bash run/pool_losses.sh
"""
import argparse
import glob
import json
import os

import torch

FORK = "../geometry-of-refusal/results"
SALAD = "results/qwen2.5-3b_salad-o1o2o5o6"
BT = "results/qwen2.5-3b_beavertails-b1b5b6bv"

# Семейства, снимающие отказ по отдельности (ASR при a=1 из loss_terms и bypass_only).
ADMIT = [
    ("base", f"{SALAD}/rdo", 0.91),
    ("no_add", f"{FORK}/loss_terms/no_add", 0.90),
    ("free_norm", f"{FORK}/loss_terms/free_norm", 0.91),
    ("inv_0.003", f"{FORK}/bypass_only/invariant_0.003", 0.88),
    ("inv_0.01", f"{FORK}/bypass_only/invariant_0.01", 0.85),
    ("inv_0.03", f"{FORK}/bypass_only/invariant_0.03", 0.87),
]
# Отвергнутые: не оценивают ось снятия отказа.
REJECT = [
    ("no_abl", f"{FORK}/loss_terms/no_abl", 0.40),
    ("inv_0.1", f"{FORK}/bypass_only/invariant_0.1", 0.75),
    ("inv_0.3", f"{FORK}/bypass_only/invariant_0.3", 0.58),
    ("inv_1", f"{FORK}/bypass_only/invariant", 0.34),
]


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float64).flatten()
    return v / v.norm()


def family(d):
    """Единичные векторы семейства и их слои извлечения."""
    V, layers = [], set()
    for p in sorted(glob.glob(os.path.join(d, "*", "seed_*.pt"))):
        o = torch.load(p, map_location="cpu", weights_only=False)
        V.append(unit(o["vector"]))
        layers.add(int(o["add_layer"]))
    return torch.stack(V), layers


def spectrum(P):
    lam, U = torch.linalg.eigh(P)
    o = torch.argsort(lam, descending=True)
    return lam[o], U[:, o]


def cov(V):
    return V.T @ V / V.shape[0]


def comp(run, var):
    p = os.path.join(run, "component", var, "component.pt")
    return unit(torch.load(p, map_location="cpu", weights_only=True)["w"])


def null_lam1(n, d, n_null, seed):
    """p95 lambda_1 для n случайных единичных векторов в d измерениях."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_null):
        V = torch.randn(n, d, generator=g, dtype=torch.float64)
        V = V / V.norm(dim=1, keepdim=True)
        out.append(float(spectrum(cov(V))[0][0]))
    return torch.tensor(out).quantile(0.95).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    fams, rejected = {}, {}
    print("## Инвентарь семейств\n")
    print("| семейство | N | домены | слой | \\|v\\| | ASR одиночных | lam_1 семейства | в пул |")
    print("|---|---|---|---|---|---|---|---|")
    for name, d, asr in ADMIT + REJECT:
        if not glob.glob(os.path.join(d, "*", "seed_*.pt")):
            print(f"| {name} | — | НЕ НАЙДЕНО: {d} | | | | | — |")
            continue
        V, layers = family(d)
        doms = sorted({os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(d, "*", "seed_*.pt"))})
        raw = [float(torch.as_tensor(torch.load(p, map_location="cpu",
                                                weights_only=False)["vector"]).norm())
               for p in sorted(glob.glob(os.path.join(d, "*", "seed_*.pt")))]
        l1 = float(spectrum(cov(V))[0][0])
        adm = any(name == n for n, _, _ in ADMIT)
        (fams if adm else rejected)[name] = V
        print(f"| {name} | {len(V)} | {','.join(doms)} | {sorted(layers)} | "
              f"{min(raw):.1f}–{max(raw):.1f} | {asr:.2f} | {l1:.3f} | "
              f"{'да' if adm else '**нет**'} |")

    layers_all = set()
    for name, d, _ in ADMIT:
        if glob.glob(os.path.join(d, "*", "seed_*.pt")):
            layers_all |= family(d)[1]
    print(f"\nСлои извлечения всех допущенных семейств: {sorted(layers_all)}. "
          f"{'Совпадают — векторы сравнимы напрямую.' if len(layers_all) == 1 else '**РАЗНЫЕ: пул некорректен, разбираться до продолжения.**'}")

    d = next(iter(fams.values())).shape[1]
    K = len(fams)
    P_fam = sum(cov(V) for V in fams.values()) / K
    lam_fam, U_fam = spectrum(P_fam)
    w_fam = unit(U_fam[:, 0])

    V_all = torch.cat(list(fams.values()))
    lam_vec, U_vec = spectrum(cov(V_all))
    w_vec = unit(U_vec[:, 0])

    print(f"\n## Пул {K} семейств, {len(V_all)} векторов\n")
    n_eff = int(round(len(V_all) / K)) * K
    null = null_lam1(min(n_eff, 80), d, a.n_null, a.null_seed)
    print(f"| пул | lam_1 | lam_2 | нулевой уровень p95 |")
    print("|---|---|---|---|")
    print(f"| равный вес семейству | **{float(lam_fam[0]):.3f}** | {float(lam_fam[1]):.3f} | {null:.3f} |")
    print(f"| равный вес вектору (наивный) | {float(lam_vec[0]):.3f} | {float(lam_vec[1]):.3f} | {null:.3f} |")
    print(f"\n|cos| между двумя пуловыми осями: {float(abs(w_fam @ w_vec)):.4f}")

    print("\n## Пуловая ось против компоненты каждого семейства\n")
    print("| семейство | lam_1 своя | \\|cos\\| своей компоненты с пулом | cos^2 | "
          "энергия пула в семействе | доля потолка |")
    print("|---|---|---|---|---|---|")
    for name, V in fams.items():
        lam, U = spectrum(cov(V))
        own = unit(U[:, 0])
        c = float(abs(own @ w_fam))
        e = float((V @ w_fam).pow(2).mean())
        print(f"| {name} | {float(lam[0]):.3f} | {c:.4f} | {c**2:.4f} | {e:.3f} | "
              f"{e / float(lam[0]):.3f} |")

    print("\n## Отвергнутые семейства: насколько они в стороне\n")
    print("| семейство | \\|cos\\| своей компоненты с пулом | энергия пула в нём |")
    print("|---|---|---|")
    for name, V in rejected.items():
        own = unit(spectrum(cov(V))[1][:, 0])
        print(f"| {name} | {float(abs(own @ w_fam)):.4f} | "
              f"{float((V @ w_fam).pow(2).mean()):.3f} |")

    print("\n## Пуловая ось против уже известных осей\n")
    known = {
        "компонента SALAD all (базовая)": comp(SALAD, "all"),
        "компонента SALAD rdo_only": comp(SALAD, "rdo_only"),
        "компонента BeaverTails rdo_only (наша)": comp(BT, "rdo_only"),
        "DIM union SALAD": unit(torch.load(f"{SALAD}/dim/union/direction.pt",
                                           map_location="cpu", weights_only=True).double()),
        "DIM union BeaverTails": unit(torch.load(f"{BT}/dim/union/direction.pt",
                                                 map_location="cpu", weights_only=True).double()),
    }
    print("| ось | \\|cos\\| с пулом семейств |")
    print("|---|---|")
    for n, v in known.items():
        print(f"| {n} | {float(abs(v @ w_fam)):.4f} |")

    print("\n## Confound: не унаследовано ли согласие от общей инициализации\n")
    print("Семейства обучались с парных стартов. Если согласие идёт от старта, векторы с")
    print("ОДНИМ сидом будут согласны сильнее, чем с РАЗНЫМИ. Сравниваем эти два числа.\n")
    keyed, inits = {}, {}
    for name, fdir, _ in ADMIT:          # не d: d — размерность, затенять нельзя
        ps = sorted(glob.glob(os.path.join(fdir, "*", "seed_*.pt")))
        if not ps:
            continue
        keyed[name], inits[name] = {}, {}
        for pth in ps:
            o = torch.load(pth, map_location="cpu", weights_only=False)
            k = (os.path.basename(os.path.dirname(pth)), int(o["seed"]))
            keyed[name][k] = unit(o["vector"])
            iv = o.get("init_vector")
            if iv is None:
                continue
            iv = torch.as_tensor(iv)
            if iv.numel() == 0:
                continue
            if iv.dim() > 1:              # старт конуса — матрица рёбер, берём первое
                iv = iv[0]
            if iv.numel() == d:
                inits[name][k] = unit(iv)
    print("| пара семейств | \\|cos\\| тот же сид | \\|cos\\| другие сиды | общих сидов |")
    print("|---|---|---|---|")
    names = list(keyed)
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            shared = sorted(set(keyed[A]) & set(keyed[B]))
            if not shared:
                continue
            same = [abs(float(keyed[A][k] @ keyed[B][k])) for k in shared]
            diff = [abs(float(keyed[A][k] @ keyed[B][m]))
                    for k in shared for m in shared if k != m]
            print(f"| {A} ↔ {B} | {torch.tensor(same).median():.3f} | "
                  f"{torch.tensor(diff).median():.3f} | {len(shared)} |")
    print("\n| семейство | \\|cos\\| вектора со своим стартом | медиана |")
    print("|---|---|---|")
    for name in names:
        if not inits[name]:
            print(f"| {name} | старт не сохранён | — |")
            continue
        cs = torch.tensor([abs(float(keyed[name][k] @ inits[name][k]))
                           for k in inits[name]])
        print(f"| {name} | {cs.min():.3f}–{cs.max():.3f} | {cs.median():.3f} |")

    if a.save:
        names = ["w_pool_loss", "w_pool_loss_naive"]
        torch.save({"V": torch.stack([w_fam, w_vec]).float(), "names": names,
                    "meta": {"families": list(fams), "kind": "component"}}, a.save)
        print(f"\nнабор сохранён: {a.save} ({', '.join(names)})")


if __name__ == "__main__":
    main()
