"""Доп. анализ (CPU): split-half k общей компоненты каждого варианта, как в robust_component (pca):
ρ²_N = |cos(w_A, w_B)| по непересекающимся половинам, 1/ρ² − 1 = k/N. Разбор — reports/results.md.

Запуск из корня shared-refusal-component: uv run python <путь>/split_half.py
"""
import os, sys, math, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import base, seeds
from shared_refusal.directions import rdo
cfg, run = base()
root = os.path.expanduser("~/f.zakharov/geometry-of-refusal/results")
def unit(x): x = x.double().flatten().numpy(); return x / np.linalg.norm(x)
def pairs(n, k, ns, rng):
    tot = math.comb(n, k) * math.comb(n - k, k) // 2
    if tot <= ns:
        out = set()
        for a in itertools.combinations(range(n), k):
            rest = [i for i in range(n) if i not in a]
            for b in itertools.combinations(rest, k): out.add(tuple(sorted((a, b))))
        return sorted(out)
    s = set()
    while len(s) < ns:
        p = rng.permutation(n); s.add(tuple(sorted((tuple(sorted(p[:k])), tuple(sorted(p[k:2*k]))))))
    return sorted(s)
def fit_k(ns, r):
    x = np.array([1/n for n in ns]); y = np.array([1/v - 1 for v in r]); return float((x*y).sum()/(x*x).sum())
def run_set(name, V, rng):
    n = len(V); ns = list(range(2, n//2 + 1)); rho = []
    for k in ns:
        ps = pairs(n, k, 3000, rng)
        G = V @ V.T
        def ax(S):
            S = list(S); u = np.linalg.eigh(G[np.ix_(S, S)])[1][:, -1]; b = np.zeros(n); b[S] = u
            return b / math.sqrt(b @ G @ b)
        rho.append(np.mean([abs(ax(a) @ G @ ax(b)) for a, b in ps]))
    K = fit_k(ns, rho)
    print(f"{name:22s} N={n:2d}  k={K:5.2f}  ρ² при N=12: {1/(1+K/12):.3f}  N для ρ²≥0.9: {math.ceil(9*K):3d}   ρ²_N: " +
          " ".join(f"{k}:{r:.3f}" for k, r in zip(ns, rho)))
rng = np.random.default_rng(0)
B20 = np.stack([unit(rdo.load_seed(run.rdo(d), s)["vector"]) for d in cfg["domains"] for s in cfg["rdo"]["seeds"][d]])
run_set("base, все 20 RDO", B20, rng)
for v, sub in [("base", None), ("no_add","loss_terms"), ("no_abl","loss_terms"), ("free_norm","loss_terms"), ("invariant","bypass_only")]:
    V = np.stack([unit(rdo.load_seed(run.rdo(d) if sub is None else os.path.join(root, sub, v, d), s)["vector"])
                  for d in cfg["domains"] for s in seeds(cfg, d)])
    run_set(f"{v} (12)", V, rng)
