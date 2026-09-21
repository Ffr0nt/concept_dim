"""§8 — read против write: детектирует ли остаток DIM вредность, будучи каузально пустым. GPU.

Шаги 4-7 показали: вся КАУЗАЛЬНАЯ сила DIM (снимать отказ аблацией, наводить его добавлением)
сидит в малой общей компоненте — 18-30% энергии. Остаток инертен с обеих сторон. Но DIM
по построению — разность средних, то есть прежде всего ДЕТЕКТОР: направление, по проекции
на которое harmful отличается от harmless. Вопрос: где живёт эта read-роль?

Гипотеза: в остатке. Тогда DIM = маленькая write-компонента (медиирует отказ) + большая
read-компонента (коррелирует с вредностью, но причинно не участвует), и это объясняет,
почему метод разности средних вообще даёт рабочее направление.

Метод. Для каждого слоя берётся активация последнего токена, проецируется на направление —
получается один скаляр на промпт. Качество детектора меряется двумя способами:
  AUROC   ранговая мера на TEST (128+128), порог не нужен; знак направления произволен,
          поэтому берётся max(auc, 1-auc);
  acc     точность на TEST с порогом, подобранным на TRAIN (256+256) — честный held-out.
Плюс d Коэна на train как мера разделимости.

Направления: w_cone, w_mixed, dim, dim_perp_cone (остаток), cone_perp_dim, rand и
ПОТОЛОК — разность средних, посчитанная НА ЭТОМ ЖЕ СЛОЕ по train (`meandiff_layer`).
Потолок нужен, чтобы отличить «направление плохое» от «слой неинформативный».

Env: SAVE_DIR, DIM_DIR, REFUSAL_SPLITS, HUGGINGFACE_CACHE_DIR. Только forward, без обучения.
"""
import argparse
import json
import os

import torch


def auroc(pos, neg):
    """Ранговый AUROC; знак направления произволен -> max(auc, 1-auc)."""
    x = torch.cat([pos, neg])
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
    a = ((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()
    return max(a, 1 - a)


def best_threshold(pos, neg):
    """Порог с максимальной точностью на train (перебор по серединам между значениями)."""
    v = torch.cat([pos, neg]).sort().values
    cuts = (v[:-1] + v[1:]) / 2
    best, bt, bs = -1.0, 0.0, 1
    for t in cuts:
        for sgn in (1, -1):
            acc = (((pos - t) * sgn > 0).float().sum() + ((neg - t) * sgn <= 0).float().sum()) / (len(pos) + len(neg))
            if acc.item() > best:
                best, bt, bs = acc.item(), float(t), sgn
    return bt, bs, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=256, help="промптов каждого класса на train")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from nnsight import LanguageModel

    for v in ("SAVE_DIR", "DIM_DIR", "REFUSAL_SPLITS"):
        assert os.getenv(v), f"env {v} не задан"
    rung, save_dir = os.getenv("REFUSAL_SPLITS"), os.getenv("SAVE_DIR")
    model_id = args.model.split("/")[-1]

    model = LanguageModel(args.model, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
                          device_map="auto", torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    with model.trace("Hello"):
        pass

    QWEN25 = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
              "helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n"
              "<|im_start|>assistant\n")

    def load_prompts(h, split, n=None):
        rows = json.load(open(f"data/{rung}_splits/{h}_{split}.json"))
        if n is not None:
            rows = rows[:n]
        return [QWEN25.format(instruction=r["instruction"]) for r in rows]

    tr_pos = load_prompts("harmful", "train", args.n_train)
    tr_neg = load_prompts("harmless", "train", args.n_train)
    te_pos, te_neg = load_prompts("harmful", "test"), load_prompts("harmless", "test")
    bs = args.batch_size

    # --- направления ---
    ax = os.path.join(save_dir, "common_axis")
    unit = lambda v: v / v.norm()
    d = unit(torch.load(f"{save_dir}/{os.getenv('DIM_DIR')}/{model_id}/direction.pt",
                        map_location="cpu").float())
    W_cone = torch.load(os.path.join(ax, "global", "pooled.pt"), map_location="cpu")["W"].float()
    mp = os.path.join(ax, "global", "mixed_native_balanced.pt")
    W_mixed = torch.load(mp, map_location="cpu")["W"].float() if os.path.exists(mp) else None
    g = torch.Generator().manual_seed(args.seed)
    DIRS = {"w_cone": W_cone[0], "dim": d,
            "dim_perp_cone": unit(d - (d @ W_cone[0]) * W_cone[0]),
            "cone_perp_dim": unit(W_cone[0] - (W_cone[0] @ d) * d),
            "rand": unit(torch.randn(W_cone.shape[1], generator=g))}
    if W_mixed is not None:
        DIRS["w_mixed"] = W_mixed[0]

    # --- активации последнего токена по всем слоям ---
    def acts(prompts):
        out = []
        for i in range(0, len(prompts), bs):
            with model.trace(prompts[i:i + bs]):
                per = [lay.output[0][:, -1].save() for lay in model.model.layers]
            out.append(torch.stack([p.value.detach().float().cpu() for p in per], dim=1))
            torch.cuda.empty_cache()
        return torch.cat(out, 0)                      # [n, n_layers, hidden]

    print(f"rung={rung}  train={len(tr_pos)}+{len(tr_neg)}  test={len(te_pos)}+{len(te_neg)}")
    A = {k: acts(v) for k, v in (("tr_pos", tr_pos), ("tr_neg", tr_neg),
                                 ("te_pos", te_pos), ("te_neg", te_neg))}
    n_layers = A["tr_pos"].shape[1]
    print(f"активации сняты: {n_layers} слоёв")

    rows = []
    for L in range(n_layers):
        # потолок слоя: разность средних, посчитанная на train ЭТОГО слоя
        md = unit(A["tr_pos"][:, L].mean(0) - A["tr_neg"][:, L].mean(0))
        for name, v in list(DIRS.items()) + [("meandiff_layer", md)]:
            trp, trn = A["tr_pos"][:, L] @ v, A["tr_neg"][:, L] @ v
            tep, ten = A["te_pos"][:, L] @ v, A["te_neg"][:, L] @ v
            t, sgn, tr_acc = best_threshold(trp, trn)
            acc = ((((tep - t) * sgn > 0).float().sum() + ((ten - t) * sgn <= 0).float().sum())
                   / (len(tep) + len(ten))).item()
            pooled = ((trp.var() + trn.var()) / 2).sqrt()
            rows.append({"layer": L, "direction": name, "auroc": auroc(tep, ten),
                         "acc_test": acc, "acc_train": tr_acc,
                         "cohen_d": float((trp.mean() - trn.mean()).abs() / pooled.clamp(min=1e-9))})

    names = list(DIRS) + ["meandiff_layer"]
    best_layer = {n: max((r for r in rows if r["direction"] == n), key=lambda r: r["auroc"])
                  for n in names}
    for n in names:
        b = best_layer[n]
        print(f"{n:16s} лучший слой {b['layer']:2d}: AUROC={b['auroc']:.3f} acc={b['acc_test']:.3f} d={b['cohen_d']:.2f}")

    L = [f"# §8. Read против write: детектирует ли остаток DIM вредность? Ступень `{rung}`", "",
         f"Модель {model_id}. Проекция активации последнего токена на направление; "
         f"порог подобран на train ({len(tr_pos)}+{len(tr_neg)}), точность и AUROC — "
         f"на test ({len(te_pos)}+{len(te_neg)}).",
         "Знак направления произволен, поэтому AUROC берётся как max(auc, 1−auc).",
         "`meandiff_layer` — разность средних, посчитанная на train ЭТОГО слоя: потолок слоя.", "",
         "## Лучший слой каждого направления", "",
         "| направление | слой | AUROC | acc (test) | d Коэна |", "|---|---|---|---|---|"]
    for n in names:
        b = best_layer[n]
        L.append(f"| {n} | {b['layer']} | **{b['auroc']:.3f}** | {b['acc_test']:.3f} | {b['cohen_d']:.2f} |")
    L += ["", "## По слоям (AUROC на test)", "",
          "| слой | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for l in range(n_layers):
        cells = [f"{[r for r in rows if r['layer'] == l and r['direction'] == n][0]['auroc']:.3f}"
                 for n in names]
        L.append(f"| {l} | " + " | ".join(cells) + " |")
    md_txt = "\n".join(L) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(md_txt)
        print(f"[записано: {args.out}]")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump({"rung": rung, "rows": rows}, open(args.json_out, "w"), indent=1)
        print(f"[записано: {args.json_out}]")


if __name__ == "__main__":
    main()
