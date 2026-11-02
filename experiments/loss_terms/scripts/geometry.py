"""Шаг 2 — геометрия вариантов (CPU, модель не грузится).

На каждый вариант (12 векторов: 4 домена × 3 сида): общая компонента w_V и её λ против нуля,
согласованность сидов внутри домена и между доменами, парное сравнение с base (тот же старт),
контроль старта, доля w_V в DIM, у free_norm — выученная норма.

Пишет <out>/geometry.json, <out>/components.pt (w_V всех вариантов — их аблирует шаг 3) и
отчёт --report.
"""
import argparse
import itertools
import os

import torch

from common import available, base, dims, load_variant, rdo_only_component
from shared_refusal.artifacts import write_json
from shared_refusal.geometry import null_lam1, shared_component, unit
from shared_refusal.report import table


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def seed_agreement(ds):
    """Средний |cos| пар сидов внутри домена (по доменам и в целом) и пар из разных доменов."""
    within, cross = {}, []
    for i, j in itertools.combinations(range(len(ds)), 2):
        c = abs(float(ds.V[i] @ ds.V[j]))
        di, dj = ds.meta[i]["domain"], ds.meta[j]["domain"]
        if di == dj:
            within.setdefault(di, []).append(c)
        else:
            cross.append(c)
    return {"within": {d: mean(x) for d, x in within.items()},
            "within_mean": mean(c for x in within.values() for c in x), "cross_mean": mean(cross)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--null-seed", type=int, default=0)
    args = ap.parse_args()

    cfg, run = base()
    domains = list(cfg["domains"])
    variants = available(args.out, cfg)
    assert "base" in variants, "нет сидов base в базовом запуске"
    print(f"варианты с полным набором сидов: {variants}")

    D = {k: unit(v.double()) for k, v in dims(run, cfg).items()}
    loaded = {v: load_variant(args.out, v, cfg) for v in variants}
    names = loaded["base"][0].names
    for v, (ds, init, _) in loaded.items():
        assert ds.names == names, f"{v}: другой состав сидов"
        assert torch.allclose(init.V, loaded["base"][1].V), f"{v}: старты не совпадают с base — пары нет"

    N, hidden = len(names), loaded["base"][0].hidden
    null = null_lam1(N, hidden, args.n_null, args.null_seed)
    null_band = {"mean": float(null.mean()), "p95": float(null.quantile(0.95)), "n": args.n_null}
    w_rdo = rdo_only_component(run)

    comps, res = {}, {}
    for v, (ds, init, raw) in loaded.items():
        w, lam, cos = shared_component(ds.V, D["union"])
        comps[v] = w
        vb = loaded["base"][0]
        res[v] = {
            "lam": [float(x) for x in lam[:4]],
            "cos_w": dict(zip(ds.names, map(float, cos))),
            "agreement": seed_agreement(ds),
            "pair_cos_base": {n: abs(float(ds[n] @ vb[n])) for n in names},
            "cos_init": {n: float(ds[n] @ init[n]) for n in names},
            "cos2_w_init": mean(float((w @ init.V[i]) ** 2) for i in range(len(init))),
            "dim_energy": {d: float((w @ D[d]) ** 2) for d in D},
            "norm": {n: m["norm"] for n, m in zip(ds.names, ds.meta)},
            "lowest_loss": {n: m["lowest_loss"] for n, m in zip(ds.names, ds.meta)},
            "steps": {n: len(raw[n]["train_losses"]) for n in names},
        }
    for v in variants:
        res[v]["cos_w_base"] = float(comps[v] @ comps["base"])
        res[v]["cos_w_rdo_only"] = None if w_rdo is None else float(comps[v] @ w_rdo)

    torch.save({"variants": variants, "W": torch.stack([comps[v] for v in variants]).float()},
               os.path.join(args.out, "components.pt"))
    write_json({"names": names, "null": null_band, "variants": res}, os.path.join(args.out, "geometry.json"))

    # ---------- отчёт ----------
    L = ["# Геометрия вариантов", "",
         f"Базовый запуск `{os.path.basename(run.root)}`, N = {N} векторов на вариант "
         f"({len(domains)} домена × {N // len(domains)} сида). Нуль λ1 для N = {N}: "
         f"среднее {null_band['mean']:.3f}, p95 {null_band['p95']:.3f} ({args.n_null} повторов).", "",
         "## Общая компонента и согласованность сидов", ""]
    L += table(["вариант", "λ1", "λ2", "cos(w, w_base)", "cos(w, w_rdo_only)", "|cos| сидов в домене",
                "|cos| между доменами", "|cos| с парой base", "cos со стартом", "cos²(w, старт)"],
               [[v, f"{r['lam'][0]:.3f}", f"{r['lam'][1]:.3f}", f"{r['cos_w_base']:+.3f}",
                 "—" if r["cos_w_rdo_only"] is None else f"{r['cos_w_rdo_only']:+.3f}",
                 f"{r['agreement']['within_mean']:.3f}", f"{r['agreement']['cross_mean']:.3f}",
                 f"{mean(r['pair_cos_base'].values()):.3f}", f"{mean(r['cos_init'].values()):+.3f}",
                 f"{r['cos2_w_init']:.4f}"] for v, r in res.items()])
    L += ["", "## |cos| сидов внутри домена", ""]
    L += table(["вариант"] + domains,
               [[v] + [f"{r['agreement']['within'][d]:.3f}" for d in domains] for v, r in res.items()])
    L += ["", "## Доля w в DIM: cos²(w, DIM)", ""]
    L += table(["вариант"] + domains + ["union"],
               [[v] + [f"{r['dim_energy'][d]:.3f}" for d in domains + ["union"]] for v, r in res.items()])
    L += ["", "## По векторам", "",
          "cos(v, w_V) — с компонентой своего варианта; пара — тот же сид в base; "
          "‖v‖ отличается от 1 только у free_norm.", ""]
    hdr = ["вектор"] + [f"{v}: cos(v,w) / пара / ‖v‖ / loss / шагов" for v in variants]
    L += table(hdr, [[n] + [f"{res[v]['cos_w'][n]:+.2f} / {res[v]['pair_cos_base'][n]:.2f} / "
                            f"{res[v]['norm'][n]:.2f} / {res[v]['lowest_loss'][n]:.2f} / {res[v]['steps'][n]}"
                            for v in variants] for n in names])
    L += ["", "## Косинусы между компонентами вариантов", ""]
    L += table([""] + variants, [[a] + [f"{float(comps[a] @ comps[b]):+.3f}" for b in variants] for a in variants])
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
