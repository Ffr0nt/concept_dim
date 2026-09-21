"""§9 (кэш) — один форвард-проход: проекции активаций на ось, DIM и их остатки. GPU.

Зачем отдельный скрипт. В форке сохраняются только mean_diffs — активации, УЖЕ усреднённые
по промптам. Поэтому распределение проекции по промптам (d', E|p|, дисперсия) из артефактов
не достаётся: нужен один прогон без вмешательства и без генерации. Он дешёвый — столько же
форвардов, сколько одна точка свипа из шага 4, — но отделён от анализа, чтобы анализ
(axis_geometry.py --stage acts) считался на CPU сколько угодно раз.

ЧТО СНИМАЕТСЯ. Ровно те тензоры, в которые бьёт аблация шагов 4-5: вход блока (`resid`),
выход self_attn (`attn`), выход mlp (`mlp`) — во всех слоях. На каждый тензор:
  проекции на все направления по последним n_eoi позициям (нужны распределения по промптам);
  суммы p, p^2, h и ||h||^2 по ВСЕМ позициям промпта;
  сумма ОТНОШЕНИЙ p^2/||h||^2 по токенам и сумма h на последней позиции.

ЗАЧЕМ ОТНОШЕНИЯ, А НЕ ТОЛЬКО СУММЫ. У Qwen норма residual stream на sink-токене (первая
позиция) на порядки больше, чем в точке чтения: замер на ступени `all`, слой 10 — E||h||^2 по
всем токенам 3.7e5 против 3.3e3 по пяти последним позициям, отношение 113x. Поэтому
энергетически взвешенная «удалённая доля» sum p^2 / sum ||h||^2 меряет в основном sink-токен
и для случайного направления даёт 3e-5 вместо 1/2048; как мера ущерба она непригодна.
Робастная величина — среднее по токенам отношение p^2/||h||^2 (каждый токен весит одинаково),
она и накапливается отдельно в sum_rel.
Батч фиксирован в 1: при батчировании в тензор попадают паддинг-позиции, и суммы по
позициям перестают быть суммами по реальным токенам. Полные активации не сохраняются —
всё сворачивается внутри trace, на диск уходит ~8 МБ на ступень.

НАПРАВЛЕНИЯ — те же имена, что в ablation-<rung>.json и residual-<rung>.json
(w_cone, w_mixed, dim, dim_perp_cone, cone_perp_dim, rand_0), чтобы анализ мог сверить
удаляемую энергию с измеренным KL_ret без ручного сопоставления.

Env: SAVE_DIR, DIM_DIR, REFUSAL_SPLITS, HUGGINGFACE_CACHE_DIR. Запускает пользователь
(GPU) — см. run/8_axis_geometry.sh. Проверка формы без GPU: --device cpu --limit 2.
"""
import argparse
import json
import os

import torch

SITES = ("resid", "attn", "mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_eoi", type=int, default=5, help="сколько последних позиций хранить")
    ap.add_argument("--limit", type=int, default=None, help="обрезать промпты (для проверки)")
    ap.add_argument("--device", default=None, help="cpu — структурная проверка без GPU")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from nnsight import LanguageModel

    for v in ("SAVE_DIR", "DIM_DIR", "REFUSAL_SPLITS"):
        assert os.getenv(v), f"env {v} не задан"
    rung, save_dir = os.getenv("REFUSAL_SPLITS"), os.getenv("SAVE_DIR")
    model_id = args.model.split("/")[-1]

    QWEN25 = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
              "helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n"
              "<|im_start|>assistant\n")

    def load_prompts(h):
        rows = json.load(open(f"data/{rung}_splits/{h}_test.json"))
        if args.limit:
            rows = rows[:args.limit]
        return [QWEN25.format(instruction=r["instruction"]) for r in rows]

    # --- направления ---------------------------------------------------------
    unit = lambda v: v / v.norm()
    ax = os.path.join(save_dir, "common_axis")
    dim_path = f"{save_dir}/{os.getenv('DIM_DIR')}/{model_id}"
    dim_raw = torch.load(f"{dim_path}/direction.pt", map_location="cpu").float()
    meta = json.load(open(f"{dim_path}/direction_metadata.json"))
    native = (int(meta["pos"]), int(meta["layer"]))
    d = unit(dim_raw)
    W_cone = torch.load(os.path.join(ax, "global", "pooled.pt"),
                        map_location="cpu")["W"].float()
    mp = os.path.join(ax, "global", "mixed_native_balanced.pt")
    W_mixed = torch.load(mp, map_location="cpu")["W"].float() if os.path.exists(mp) else None
    w1 = W_cone[0]
    g = torch.Generator().manual_seed(args.seed)

    DIRS = {"w_cone": w1, "dim": d,
            "dim_perp_cone": unit(d - (d @ w1) * w1),
            "cone_perp_dim": unit(w1 - (w1 @ d) * d),
            "rand_0": unit(torch.randn(W_cone.shape[1], generator=g))}
    if W_mixed is not None:
        DIRS["w_mixed"] = W_mixed[0]
    names = list(DIRS)
    V = torch.stack([DIRS[n] for n in names])            # [D, hidden]

    # --- модель --------------------------------------------------------------
    kw = dict(cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"))
    if args.device == "cpu":
        model = LanguageModel(args.model, torch_dtype=torch.float32, **kw)
    else:
        model = LanguageModel(args.model, device_map="auto",
                              torch_dtype=torch.bfloat16, **kw)
    model.requires_grad_(False)
    with model.trace("Hello"):
        pass
    layers = model.model.layers
    L, hidden = len(layers), V.shape[1]
    dev = next(model.model.parameters()).device
    Vt = V.T.to(dev).float()
    print(f"rung={rung}  слоёв={L}  hidden={hidden}  направления={names}  "
          f"точка DIM={native}  device={dev}")

    def collect(prompts):
        n = len(prompts)
        acc = {
            "n_prompts": n,
            "eoi": {s: torch.zeros(n, args.n_eoi, L, len(names)) for s in SITES},
            "eoi_n": {s: torch.zeros(n, args.n_eoi, L) for s in SITES},
            "sum_p": {s: torch.zeros(L, len(names), dtype=torch.float64) for s in SITES},
            "sum_p2": {s: torch.zeros(L, len(names), dtype=torch.float64) for s in SITES},
            "sum_rel": {s: torch.zeros(L, len(names), dtype=torch.float64) for s in SITES},
            "sum_h": {s: torch.zeros(L, hidden, dtype=torch.float64) for s in SITES},
            "sum_h_last": {s: torch.zeros(L, hidden, dtype=torch.float64) for s in SITES},
            "sum_h2": {s: torch.zeros(L, dtype=torch.float64) for s in SITES},
            "n_tok": {s: 0 for s in SITES},
        }
        for i, pr in enumerate(prompts):
            with model.trace(pr):
                saved = []
                for layer in layers:
                    tens = (layer.input, layer.self_attn.output[0], layer.mlp.output)
                    for t in tens:
                        h = t[0].float()                  # [seq, hidden], батч = 1
                        saved.append((
                            (h @ Vt).save(),              # [seq, D]
                            h.pow(2).sum(-1).save(),      # [seq]
                            h.sum(0).save(),              # [hidden]
                            h[-1].save(),                 # [hidden], последняя позиция
                        ))
            for li in range(L):
                for si, s in enumerate(SITES):
                    p, n2, sh, hl = (x.value.detach().float().cpu()
                                     for x in saved[li * len(SITES) + si])
                    k = min(args.n_eoi, p.shape[0])
                    acc["eoi"][s][i, -k:, li] = p[-k:]
                    acc["eoi_n"][s][i, -k:, li] = n2[-k:].sqrt()
                    acc["sum_p"][s][li] += p.double().sum(0)
                    acc["sum_p2"][s][li] += p.double().pow(2).sum(0)
                    acc["sum_rel"][s][li] += (p.double().pow(2)
                                              / n2.double().clamp(min=1e-12).unsqueeze(1)).sum(0)
                    acc["sum_h"][s][li] += sh.double()
                    acc["sum_h_last"][s][li] += hl.double()
                    acc["sum_h2"][s][li] += float(n2.double().sum())
                    if li == 0:
                        acc["n_tok"][s] += p.shape[0]
            del saved
            if args.device != "cpu":
                torch.cuda.empty_cache()
            if (i + 1) % 16 == 0 or i + 1 == n:
                print(f"  {i+1}/{n}", flush=True)
        return acc

    sets = {}
    for name in ("harmful", "harmless"):
        prompts = load_prompts(name)
        print(f"=== {name}: {len(prompts)} промптов ===")
        sets[name] = collect(prompts)

    out = {"rung": rung, "model": model_id, "dirs": names, "V": V,
           "native": native, "n_eoi": args.n_eoi, "n_layers": L, "hidden": hidden,
           "sites": SITES, "dim_norm": float(dim_raw.norm()), "sets": sets}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"[записано: {args.out}]  "
          f"{os.path.getsize(args.out) / 2**20:.1f} МБ")


if __name__ == "__main__":
    main()
