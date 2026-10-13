"""Робастные и взвешенные оценки общей компоненты против обычного собственного вектора.

Пул — единичные направления (DIM и RDO) из directions.pt запуска shared-refusal-component.
Для каждой оценки (estimator) считаем split-half ρ²_N = |cos(w_A, w_B)| на одних и тех же
парах непересекающихся половин A, B по N векторов и подгоняем 1/ρ² − 1 = k/N: меньше k —
чище компонента при том же числе векторов. Все веса и выбросы оцениваются ВНУТРИ половины,
иначе split-half перестаёт быть честным.

Оценки (все работают с осью: знак вектора не важен):
  pca   — первый собственный вектор (1/N) sum v vᵀ, равные веса (текущий метод);
  snr   — тот же собственный вектор с весами α_i = 1/(1 − c_i²); c_i — общая часть вектора
          из однофакторной модели |cos(v_i, v_j)| ≈ c_i c_j (без опоры на w);
  mf    — взвешенное среднее выровненных по знаку векторов с весами c_i/(1 − c_i²)
          (согласованный фильтр);
  fms   — Fast Median Subspace (Lerman, Maunu): минимум суммы расстояний до оси, а не их
          квадратов; IRLS с весами 1/max(dist, δ);
  trim  — обрезанный PCA: 5 раз выкидываем долю q векторов, дальше всего от оси, и пересчитываем.

Всё считается в пространстве коэффициентов по векторам пула через матрицу Грама G = V Vᵀ:
ось w = Vᵀβ, так что hidden = 2048 нигде не нужен, кроме финального сохранения осей.
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch

ESTIMATORS = ("pca", "snr", "mf", "fms", "trim")


# ---------- оценки оси по подмножеству S (индексы пула); возвращают коэффициенты β, w = V_Sᵀβ ----------

def _top(M):
    lam, U = np.linalg.eigh(M)
    return lam[-1], U[:, -1]


def _norm(G, beta):
    return math.sqrt(max(beta @ G @ beta, 1e-300))


def pca(G, alpha=None):
    a = np.ones(len(G)) if alpha is None else alpha
    s = np.sqrt(a)
    _, u = _top(s[:, None] * G * s[None, :])
    beta = s * u
    return beta / _norm(G, beta)


def factor_c(G, iters=200):
    """Однофакторная модель |G_ij| ≈ c_i c_j (i ≠ j): principal axis factoring. N ≥ 3."""
    A = np.abs(G)
    n = len(A)
    c = np.sqrt(np.clip((A.sum(1) - 1) / (n - 1), 1e-6, 1))
    for _ in range(iters):
        M = A.copy()
        np.fill_diagonal(M, c ** 2)
        lam, u = _top(M)
        c_new = np.clip(np.sqrt(max(lam, 0)) * np.abs(u), 0, 0.999)
        if np.max(np.abs(c_new - c)) < 1e-10:
            c = c_new
            break
        c = c_new
    return c


def snr(G):
    if len(G) < 3:
        return pca(G)
    c = factor_c(G)
    return pca(G, 1 / (1 - c ** 2))


def mf(G):
    if len(G) < 3:
        return pca(G)
    c = factor_c(G)
    sign = np.sign(G @ pca(G))
    sign[sign == 0] = 1
    beta = sign * c / (1 - c ** 2)
    return beta / _norm(G, beta)


def _dist(G, beta):
    cos = (G @ beta) / _norm(G, beta)          # cos(v_i, w), строки V единичные
    return np.sqrt(np.clip(1 - cos ** 2, 0, 1))


def fms(G, delta=1e-3, iters=200):
    beta = pca(G)
    for _ in range(iters):
        new = pca(G, 1 / np.maximum(_dist(G, beta), delta))
        if abs(abs(new @ G @ beta) - 1) < 1e-12:
            return new
        beta = new
    return beta


def trim(G, q=0.2, rounds=5):
    n = len(G)
    drop = int(q * n)
    keep = np.arange(n)
    beta = pca(G)
    for _ in range(rounds):
        if drop == 0:
            break
        keep = np.sort(np.argsort(_dist(G, beta))[: n - drop])
        sub = pca(G[np.ix_(keep, keep)])
        beta = np.zeros(n)
        beta[keep] = sub
    return beta


FN = {"pca": pca, "snr": snr, "mf": mf, "fms": fms, "trim": trim}


def axis(G_pool, S, est):
    """Ось оценки est по подмножеству S: коэффициенты длины n_pool (нули вне S)."""
    S = np.asarray(S)
    beta = np.zeros(len(G_pool))
    beta[S] = FN[est](G_pool[np.ix_(S, S)])
    return beta


def cos_axes(G_pool, a, b):
    return abs(a @ G_pool @ b) / (_norm(G_pool, a) * _norm(G_pool, b))


# ---------- выборка пар половин (как в shared-refusal-component/convergence.py) ----------

def disjoint_pairs(n, k, n_samples, rng):
    total = math.comb(n, k) * math.comb(n - k, k) // 2
    if total <= n_samples:
        out = set()
        for a in itertools.combinations(range(n), k):
            rest = [i for i in range(n) if i not in a]
            for b in itertools.combinations(rest, k):
                out.add(tuple(sorted((a, b))))
        return sorted(out)
    seen = set()
    while len(seen) < n_samples:
        p = rng.permutation(n)
        seen.add(tuple(sorted((tuple(sorted(p[:k])), tuple(sorted(p[k:2 * k]))))))
    return sorted(seen)


def fit_k(ns, rho2):
    x = np.array([1 / n for n in ns])
    y = np.array([1 / r - 1 for r in rho2])
    return float((x * y).sum() / (x * x).sum())


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


# ---------- эксперимент ----------

def run_pool(names, meta, V, n_samples, seed):
    G = V @ V.T
    n = len(names)
    rng = np.random.default_rng(seed)
    split = {}
    for k in range(1, n // 2 + 1):
        pairs = disjoint_pairs(n, k, n_samples, rng)
        r = {e: np.array([cos_axes(G, axis(G, a, e), axis(G, b, e)) for a, b in pairs]) for e in ESTIMATORS}
        split[k] = {e: {"mean": float(r[e].mean()), "p05": q(r[e], 0.05), "p95": q(r[e], 0.95),
                        "diff_vs_pca": float((r[e] - r["pca"]).mean()),
                        "diff_se": float((r[e] - r["pca"]).std() / math.sqrt(len(pairs))),
                        "win_vs_pca": float((r[e] > r["pca"] + 1e-12).mean())} for e in ESTIMATORS}
        split[k]["n_pairs"] = len(pairs)
        print(f"  N={k:2d} ({len(pairs):4d} пар)  " +
              "  ".join(f"{e}={split[k][e]['mean']:.3f}" for e in ESTIMATORS), flush=True)
    ns = sorted(split)
    k_fit = {e: fit_k(ns, [split[k][e]["mean"] for k in ns]) for e in ESTIMATORS}

    full = {e: axis(G, np.arange(n), e) for e in ESTIMATORS}
    c = factor_c(G)
    d_fms = _dist(G, full["fms"])
    d_trim = _dist(G, full["trim"])
    kept_trim = set(np.argsort(d_trim)[: n - int(0.2 * n)].tolist())
    per_vector = [{"name": names[i], "method": meta[i].get("method"), "c": float(c[i]),
                   "w_snr": float(1 / (1 - c[i] ** 2)), "w_mf": float(c[i] / (1 - c[i] ** 2)),
                   "w_fms": float(1 / max(d_fms[i], 1e-3)), "trim_kept": i in kept_trim,
                   "cos2_pca": float((G[i] @ full["pca"]) ** 2 / (full["pca"] @ G @ full["pca"])),
                   } for i in range(n)]
    agree = {e: cos_axes(G, full[e], full["pca"]) ** 2 for e in ESTIMATORS}
    return {"names": names, "n": n, "split_half": {str(k): v for k, v in split.items()}, "k": k_fit,
            "full_cos2_vs_pca": agree, "per_vector": per_vector, "d_trim": d_trim.tolist()}, full


def report(results, src, seed, n_samples):
    L = ["# Робастные и взвешенные оценки общей компоненты", "",
         f"Источник: `{src}`. Split-half на одних и тех же парах для всех оценок, не больше "
         f"{n_samples} пар на N, сид {seed}. Веса и выбросы оцениваются внутри каждой половины.", ""]
    for pool, R in results.items():
        L += [f"## Пул `{pool}` — {R['n']} векторов", "",
              "k из 1/ρ² − 1 = k/N (меньше — чище при том же N) и что это даёт:", "",
              "| оценка | k | ρ² при N = 12 по модели | N для ρ² ≥ 0.9 | cos² с pca на всём пуле |",
              "|---|---|---|---|---|"]
        for e in ESTIMATORS:
            k = R["k"][e]
            L.append(f"| {e} | **{k:.3f}** | {1 / (1 + k / 12):.3f} | {math.ceil(9 * k)} | "
                     f"{R['full_cos2_vs_pca'][e]:.4f} |")
        L += ["", "ρ² по N (split-half) и разница с pca на тех же парах (± SE; доля пар, где лучше):", "",
              "| N | пар | " + " | ".join(ESTIMATORS) + " | " +
              " | ".join(f"{e} − pca" for e in ESTIMATORS if e != "pca") + " |",
              "|---|---|" + "---|" * (2 * len(ESTIMATORS) - 1)]
        for k, s in R["split_half"].items():
            L.append(f"| {k} | {s['n_pairs']} | " + " | ".join(f"{s[e]['mean']:.3f}" for e in ESTIMATORS) + " | " +
                     " | ".join(f"{s[e]['diff_vs_pca']:+.3f} ± {s[e]['diff_se']:.3f} ({s[e]['win_vs_pca']:.0%})"
                                for e in ESTIMATORS if e != "pca") + " |")
        L += ["", "Веса на всём пуле (c — общая часть из однофакторной модели; веса нормированы на среднее 1):", "",
              "| вектор | c | cos²(v, w_pca) | вес snr | вес mf | вес fms | trim оставил |", "|---|---|---|---|---|---|---|"]
        pv = R["per_vector"]
        mean = {key: sum(abs(p[key]) for p in pv) / len(pv) for key in ("w_snr", "w_mf", "w_fms")}
        for p in pv:
            L.append(f"| {p['name']} | {p['c']:.3f} | {p['cos2_pca']:.3f} | {p['w_snr'] / mean['w_snr']:.2f} | "
                     f"{p['w_mf'] / mean['w_mf']:.2f} | {p['w_fms'] / mean['w_fms']:.2f} | "
                     f"{'да' if p['trim_kept'] else '**нет**'} |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directions.pt запуска shared-refusal-component")
    ap.add_argument("--out", required=True, help="каталог артефактов (оси, JSON)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--pools", nargs="+", default=["all", "rdo"])
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    d = torch.load(args.src, map_location="cpu", weights_only=True)
    names, meta = d["names"], d["meta"]
    V_all = d["V"].double().numpy()
    results = {}
    os.makedirs(args.out, exist_ok=True)
    for pool in args.pools:
        idx = [i for i, m in enumerate(meta) if pool == "all" or m.get("method") == pool]
        print(f"пул {pool}: {len(idx)} векторов", flush=True)
        V = V_all[idx]
        R, full = run_pool([names[i] for i in idx], [meta[i] for i in idx], V, args.n_samples, args.seed)
        results[pool] = R
        axes = {e: torch.from_numpy(V.T @ b / np.linalg.norm(V.T @ b)) for e, b in full.items()}
        torch.save({"axes": axes, "pool": R["names"]}, os.path.join(args.out, f"axes_{pool}.pt"))
        print("  k: " + "  ".join(f"{e}={R['k'][e]:.3f}" for e in ESTIMATORS), flush=True)
    with open(os.path.join(args.out, "robust_component.json"), "w") as f:
        json.dump({"src": args.src, "seed": args.seed, "n_samples": args.n_samples, "pools": results}, f,
                  ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(report(results, args.src, args.seed, args.n_samples)) + "\n")
    print(f"[отчёт: {args.report}]")


if __name__ == "__main__":
    main()
