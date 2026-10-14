"""Шаг 3 — каузальные замеры (GPU): снимает ли направление отказ и наводит ли его.

Направления: общие компоненты вариантов w_V (из шага 2), все векторы вариантов, DIM доменов
и union, n_rand случайных.

  снятие  — частичная аблация во всех слоях h <- h - a (h·v̂) v̂ (models.ablate, как этап 03
            shared-refusal-component): ASR на harmful_test каждого домена (refusal score < 0),
            KL на harmless_test;
  наведение — h <- h + s·α0·v̂ на входе слоя добавления (все позиции, как rdo.py),
            α0 = ‖DIM_union‖: доля отказов (refusal score > 0) и средний score на harmless_test.
            У векторов free_norm дополнительно s = α_train‖z‖/α0 — выученная сила.

Test-сплиты не фильтруются. Готовые направления пропускаются (<out>/causal.json пишется после
каждого), так что прерванный прогон можно перезапустить. --report-only — только отчёт из json.
"""
import argparse
import functools
import os

import torch

from common import available, base, dims, load_variant, point
from shared_refusal import models, provenance
from shared_refusal.ablation import asr
from shared_refusal.artifacts import read_json, write_json
from shared_refusal.geometry import kl, random_directions, unit
from shared_refusal.report import table


def induce_logits(model, prompts, v, scale, layer, batch_size):
    """Логиты последней позиции при добавлении scale·v̂ на вход блока layer."""
    v = unit(v.double()).to(model.dtype).to(model.device)
    out = []
    for i in range(0, len(prompts), batch_size):
        with model.trace(prompts[i:i + batch_size]):
            models.layers(model)[layer].input += scale * v
            logits = model.lm_head.output[:, -1].save()
        out.append(logits.value.detach().float().cpu())
        torch.cuda.empty_cache()
    return torch.cat(out)


def directions(args, cfg, run):
    """[(имя, вектор, метаданные)] в порядке замера: сначала дешёвое и главное."""
    D = dims(run, cfg)
    comp = torch.load(os.path.join(args.out, "components.pt"), map_location="cpu", weights_only=True)
    out = [(f"w:{v}", w.double(), {"kind": "component", "variant": v}) for v, w in zip(comp["variants"], comp["W"])]
    out += [(f"dim:{d}", D[d].double(), {"kind": "dim", "domain": d}) for d in D]
    out += [(f"rand_{i}", r.double(), {"kind": "random"})
            for i, r in enumerate(random_directions(D["union"].shape[0], args.n_rand, seed=args.rand_seed))]
    if not args.components_only:
        for v in available(args.out, cfg):
            ds, _, raw = load_variant(args.out, v, cfg)
            for n, vec, m in zip(ds.names, ds.V, ds.meta):
                out.append((f"{v}:{n}", vec, {"kind": "vector", "variant": v, "domain": m["domain"], "seed": m["seed"],
                                               "native_scale": float(raw[n]["alpha"]) * m["norm"]
                                               if v == "free_norm" else None}))
    if args.negative_too:
        # наведение вектором со знаком минус: аблация от знака не зависит, поэтому эти строки —
        # только наведение (нужно, когда знак оси обучением не задан: no_add, invariant)
        out += [(f"{n}(-)", -v, {**m, "sign": -1, "native_scale": None})
                for n, v, m in out if m["kind"] in ("component", "vector")]
    return out, float(D["union"].norm())


def measure(args):
    cfg, run = base()
    domains = list(cfg["domains"])
    fam = models.family(cfg["model"])
    layer = point(run)["layer"]

    def test_prompts(domain, harmtype):
        rows = read_json(os.path.join(run.splits(domain), f"{harmtype}_test.json"))
        return models.prompts(fam, [r["instruction"] for r in rows])

    harmful = {d: test_prompts(d, "harmful") for d in domains}
    harmless = test_prompts(domains[0], "harmless")
    todo, alpha0 = directions(args, cfg, run)

    path = os.path.join(args.out, "causal.json")
    res = read_json(path) if os.path.exists(path) else {}
    done = {r["direction"] for r in res.get("ablation", []) + res.get("induce", [])}
    if all(n in done for n, _, _ in todo):
        print("все направления уже замерены")
        return

    provenance.check_revision(cfg["model"], cfg["model_revision"])
    model = models.load(cfg["model"], cfg["model_revision"])
    fwd = functools.partial(models.last_token_logits, model, batch_size=args.batch_size)
    ind = functools.partial(induce_logits, model, layer=layer, batch_size=args.batch_size)
    rtoks = fam["refusal_toks"]

    if "base" not in res:
        base_harmless = fwd(harmless, None, 0)
        sc = models.refusal_score(base_harmless, rtoks)
        res["base"] = {"asr": {d: asr(fwd(p, None, 0), rtoks) for d, p in harmful.items()},
                       "harmless_refusal_rate": float((sc > 0).float().mean()),
                       "harmless_score": float(sc.mean())}
        res.update({"alpha0": alpha0, "layer": layer, "alphas": args.alphas, "scales": args.scales,
                    "n_test": {d: len(p) for d, p in harmful.items()}, "n_harmless": len(harmless),
                    "ablation": [], "induce": []})
        write_json(res, path)
    print(f"без вмешательства: ASR " + "  ".join(f"{d}={x:.3f}" for d, x in res["base"]["asr"].items()) +
          f"; harmless refusal {res['base']['harmless_refusal_rate']:.3f}; α0={alpha0:.2f}, слой {layer}", flush=True)
    base_harmless = fwd(harmless, None, 0)

    for name, v, meta in todo:
        if name in done:
            continue
        for a in ([] if meta.get("sign") == -1 else args.alphas):
            by_d = {d: asr(fwd(p, v, a), rtoks) for d, p in harmful.items()}
            row = {"direction": name, **meta, "alpha": a, "kl": kl(base_harmless, fwd(harmless, v, a)),
                   "asr": by_d, "asr_mean": sum(by_d.values()) / len(by_d)}
            res["ablation"].append(row)
            print(f"[abl] {name:22s} a={a:.2f} KL={row['kl']:.4f} ASR={row['asr_mean']:.3f}", flush=True)
        scales = list(args.scales) + ([meta["native_scale"] / alpha0] if meta.get("native_scale") else [])
        for s in scales:
            sc = models.refusal_score(ind(harmless, v, s * alpha0), rtoks)
            row = {"direction": name, **meta, "scale": s, "native": s not in args.scales,
                   "refusal_rate": float((sc > 0).float().mean()), "score": float(sc.mean())}
            res["induce"].append(row)
            print(f"[ind] {name:22s} s={s:.2f} refusal={row['refusal_rate']:.3f} score={row['score']:+.2f}", flush=True)
        write_json(res, path)


def report(args):
    res = read_json(os.path.join(args.out, "causal.json"))
    abl, ind = res["ablation"], res["induce"]

    def group_key(r):
        if r["kind"] == "vector":
            return f"{r['variant']}{'(-)' if r.get('sign') == -1 else ''} (среднее по сидам)"
        if r["kind"] == "random":
            return "rand (среднее)"
        return r["direction"]

    def agg(rows, keys):
        out = {}
        for r in rows:
            out.setdefault(group_key(r), []).append(r)
        return {g: {k: sum(r[k] for r in rs) / len(rs) for k in keys} | {"n": len(rs)} for g, rs in out.items()}

    L = ["# Каузальные замеры", "",
         f"Без вмешательства: ASR " + ", ".join(f"{d} {x:.3f}" for d, x in res["base"]["asr"].items()) +
         f"; отказ на harmless {res['base']['harmless_refusal_rate']:.3f}. "
         f"Наведение — слой {res['layer']}, α0 = ‖DIM_union‖ = {res['alpha0']:.2f}.", "",
         "**Сравнивать направления только при близком KL.**", "", "## Снятие отказа (аблация)", ""]
    for a in res["alphas"]:
        g = agg([r for r in abl if r["alpha"] == a], ["kl", "asr_mean"])
        L += [f"a = {a}", ""] + table(["направление", "n", "KL", "ASR (среднее по доменам)"],
                                      [[k, x["n"], f"{x['kl']:.4f}", f"{x['asr_mean']:.3f}"]
                                       for k, x in sorted(g.items(), key=lambda t: t[1]["kl"])]) + [""]
    L += ["## Наведение отказа (harmless_test)", ""]
    g_by_s = {s: agg([r for r in ind if r["scale"] == s and not r["native"]], ["refusal_rate", "score"])
              for s in res["scales"]}
    groups = list(next(iter(g_by_s.values())))
    L += table(["направление"] + [f"s = {s}" for s in res["scales"]],
               [[k] + [f"{g_by_s[s][k]['refusal_rate']:.3f} ({g_by_s[s][k]['score']:+.1f})" for s in res["scales"]]
                for k in groups])
    native = [r for r in ind if r["native"]]
    if native:
        L += ["", "free_norm с выученной силой α_train‖z‖:", ""]
        L += table(["вектор", "s = α_train‖z‖/α0", "отказ", "score"],
                   [[r["direction"], f"{r['scale']:.2f}", f"{r['refusal_rate']:.3f}", f"{r['score']:+.2f}"] for r in native])
    L += ["", "## По векторам: ASR при a = 1 и наведение при s = 1", ""]
    vec = [r for r in abl if r["kind"] == "vector" and r["alpha"] == 1.0]
    ind1 = {r["direction"]: r for r in ind if r["kind"] == "vector" and r["scale"] == 1.0 and not r["native"]}
    rate = lambda k: f"{ind1[k]['refusal_rate']:.3f}" if k in ind1 else "—"
    L += table(["вектор", "KL", "ASR", "отказ при s=1", "отказ при s=1, −v"],
               [[r["direction"], f"{r['kl']:.4f}", f"{r['asr_mean']:.3f}", rate(r["direction"]),
                 rate(r["direction"] + "(-)")] for r in vec])
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--rand-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--components-only", action="store_true", help="только w_V, DIM и случайные (быстро)")
    ap.add_argument("--negative-too", action="store_true",
                    help="наведение ещё и векторами со знаком минус (компоненты и векторы вариантов)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        measure(args)
    report(args)


if __name__ == "__main__":
    main()
