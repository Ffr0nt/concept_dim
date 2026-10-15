"""Доп. анализ (CPU): |cos| одиночных векторов каждого варианта с общей компонентой w20
(20 RDO базового запуска); для base — leave-one-out. Печатает сводку, разбор — reports/results.md.

Запуск из корня shared-refusal-component: uv run python <путь>/single_vs_w20.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import base, seeds, rdo_only_component
from shared_refusal.directions import rdo
cfg, run = base()
w20 = rdo_only_component(run)
def unit(x): x = x.double().flatten(); return x / x.norm()
def top(V):
    P = sum(torch.outer(v, v) for v in V) / len(V)
    e, U = torch.linalg.eigh(P); return U[:, -1]
B = {}
for d in cfg["domains"]:
    for s in cfg["rdo"]["seeds"][d]:
        B[f"{d}_s{s:02d}"] = unit(rdo.load_seed(run.rdo(d), s)["vector"])
names = list(B)
def stats(c):
    a = torch.tensor(c).abs(); return f"mean|cos|={a.mean():.3f}  mean cos²={(a**2).mean():.3f}  median|cos|={a.median():.3f}  min={a.min():.2f} max={a.max():.2f}"
ins = [float(B[n] @ w20) for n in names]
loo = []
for n in names:
    w = top([B[m] for m in names if m != n]); w = w * torch.sign(w @ w20)
    loo.append(float(B[n] @ w))
print(f"N базовых RDO = {len(names)}, |w20|-check sign")
print("base RDO (20), в выборке  :", stats(ins))
print("base RDO (20), leave-one-out:", stats(loo))
sel = [f"{d}_s{s:02d}" for d in cfg["domains"] for s in seeds(cfg, d)]
print("base (12 из loss_terms), LOO:", stats([loo[names.index(n)] for n in sel]))
root = os.path.expanduser("~/f.zakharov/geometry-of-refusal/results")
for v, sub in [("no_add","loss_terms"),("no_abl","loss_terms"),("free_norm","loss_terms"),("invariant","bypass_only")]:
    c = []
    for d in cfg["domains"]:
        for s in seeds(cfg, d):
            c.append(float(unit(rdo.load_seed(os.path.join(root, sub, v, d), s)["vector"]) @ w20))
    print(f"{v:10s} (12) с w20         :", stats(c), " знаки:", "".join("+" if x>0 else "-" for x in c))
