"""Сводка по моделям из отчётов shared-refusal-component (boroday + свой Qwen2.5-3B)."""
import json, re, sys
from pathlib import Path

DISSER = Path("/Users/f.zakharov/Desktop/disser")
BOR = DISSER / "shared-refusal-component/results/boroday"
OWN = DISSER / "shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6"
DOMS = ["O1", "O2", "O5", "O6"]


def rows(text):
    """Строки markdown-таблиц как списки ячеек."""
    out = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            out.append([c.strip().strip("*") for c in line.strip("|").split("|")])
    return out


def num(s):
    m = re.match(r"^[+-]?\d+(\.\d+)?", s.replace("**", ""))
    return float(m.group()) if m else None


def component(path):
    t = path.read_text()
    hidden = int(re.search(r"hidden = (\d+)", t).group(1))
    r = rows(t)
    head = next(x for x in r if num(x[1]) is not None and len(x) == 7)
    cos = {x[0]: num(x[1]) for x in r if x[0].startswith(("rdo_", "dim_"))}
    rdo = {k: v for k, v in cos.items() if k.startswith("rdo_")}
    return dict(hidden=hidden, lam1=num(head[1]), lam2=num(head[2]), p95=num(head[5]),
                bad=sorted(k.split("_")[-1] for k, v in rdo.items() if v < 0.3))


def energy(path):
    t = path.read_text()
    e = [num(x[1]) for x in rows(t) if x[0] in DOMS and len(x) == 4]
    dimcos = [num(x[j]) for x in rows(t) if x[0] in DOMS and len(x) == 5 for j in range(1, 5)]
    dimcos = [c for c in dimcos if c < 0.9999]
    return dict(energy=sum(e) / len(e), dim_cos_min=min(dimcos))


def ablation(path, a="1.00"):
    t = path.read_text()
    base = [float(x) for x in re.findall(r"O\d (\d\.\d+)", t.split("Без аблации:")[1].splitlines()[0])]
    r = [x for x in rows(t) if len(x) == 8 and x[1] == a]
    def grp(pref):
        g = [x for x in r if x[0].startswith(pref)]
        return sum(num(x[7]) for x in g) / len(g), sum(num(x[2]) for x in g) / len(g)
    return dict(base=sum(base) / len(base), w=grp("w"), dim=grp("dim_"), rdo=grp("rdo_"), rand=grp("rand"))


def read_probe(path, layer):
    if not path.exists():
        return None
    r = rows(path.read_text())
    hdr = next(x for x in r if x[0] == "слой")
    row = next(x for x in r if x[0] == str(layer))
    d = dict(zip(hdr, row))
    n_layers = max(int(x[0]) for x in r if x[0].isdigit()) + 1
    return dict(n_layers=n_layers, dim=num(d["dim"]), w=num(d["w"]), w_perp=num(d["w_perp_dim"]),
                rand=num(d["rand"]))


models = []
for d in sorted(p.parent for p in BOR.glob("*/manifest.json")):
    m = json.loads((d / "manifest.json").read_text())
    rp = d / "reports"
    models.append(dict(
        name=d.name, src="boroday", path=f"~/l.boroday/shared-refusal-component/results/{d.name}",
        local=f"shared-refusal-component/results/boroday/{d.name}",
        point=(m["dim"]["pos"], m["dim"]["layer"]),
        c_all=component(rp / "component.md"), c_rdo=component(rp / "component_rdo.md"),
        e_all=energy(rp / "energy.md"), e_rdo=energy(rp / "energy_rdo.md"),
        abl=ablation(rp / "ablation.md"), abl_rdo=ablation(rp / "ablation_rdo.md"),
        rp=read_probe(rp / "read_probe.md", m["dim"]["layer"]),
    ))
m = json.loads((OWN / "manifest.json").read_text())
rp = OWN / "reports"
models.append(dict(
    name="Qwen2.5-3B-Instruct", src="свой", path="~/f.zakharov/shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6",
    local="shared-refusal-component/results/qwen2.5-3b_salad-o1o2o5o6",
    point=(m["dim"]["pos"], m["dim"]["layer"]),
    c_all=component(rp / "component-all.md"), c_rdo=component(rp / "component-rdo_only.md"),
    e_all=energy(rp / "analysis-all.md"), e_rdo=energy(rp / "analysis-rdo_only.md"),
    abl=ablation(rp / "ablation.md"), abl_rdo=None, rp=None,
))

order = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
         "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3-8B-Instruct", "Llama-3.1-8B-Instruct"]
models.sort(key=lambda x: order.index(x["name"]))
json.dump(models, sys.stdout, ensure_ascii=False, indent=1, default=list)
