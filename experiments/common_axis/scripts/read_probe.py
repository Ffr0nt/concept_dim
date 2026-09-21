"""§8 — read против write: детектирует ли остаток DIM вредность, будучи каузально пустым. GPU.

Шаги 4-7 показали: вся КАУЗАЛЬНАЯ сила DIM (снимать отказ аблацией, наводить его добавлением)
сидит в малой общей компоненте — 18-30% энергии. Остаток инертен с обеих сторон. Но DIM
по построению — разность средних, то есть прежде всего ДЕТЕКТОР: направление, по проекции
на которое harmful отличается от harmless. Вопрос: где живёт эта read-роль?

Гипотеза: в остатке. Тогда DIM = маленькая write-компонента (медиирует отказ) + большая
read-компонента (коррелирует с вредностью, но причинно не участвует), и это объясняет,
почему метод разности средних вообще даёт рабочее направление.

Метод. Для каждого слоя берётся активация последнего токена, проецируется на направление —
получается один скаляр на промпт.

ВЫБОРКА. Берётся ВЕСЬ доступный пул: train + val + test (900+128+128 на класс), с дедупом
(в сплитах ступени `all` пересечений 8/7/1). Это ~1150 промптов на класс против 128 в первой
версии. Такое возможно, потому что AUROC — ранговая мера и порога не требует: обучать нечего,
отдельный held-out нужен только для точности.

  AUROC   на всём пуле; знак направления произволен, поэтому берётся max(auc, 1-auc).
          95% ДИ — бутстрэп по промптам (стратифицированный, на фиксированных слоях).
  acc     5-фолдовая кросс-валидация: порог подбирается на четырёх фолдах, точность меряется
          на пятом, результат усредняется. Порог ищется векторно по cumsum, а не перебором.
  d Коэна на всём пуле как мера разделимости.

Репортится на ФИКСИРОВАННЫХ слоях (начальный, средний, add_layer ступени, последний): выбор
лучшего из 36 слоёв завышает оценку отбором, особенно у слабых направлений (случайное так
берёт 0.866 вместо медианных ~0.64).

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
    """Порог максимальной точности, векторно через cumsum: O(n log n) вместо перебора.

    Скоры сортируются; для разреза после i-го элемента правило «класс 1, если score > t»
    даёт TP = P - (положительных слева), TN = (отрицательных слева). Вторая ориентация
    (знак направления произволен) считается тем же cumsum-ом с другой стороны.
    """
    v = torch.cat([pos, neg])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    order = v.argsort()
    vs, ys = v[order], y[order]
    cp = torch.cumsum(ys, 0)                       # положительных в префиксе
    cn = torch.cumsum(1 - ys, 0)                   # отрицательных в префиксе
    P, N = len(pos), len(neg)
    acc_pos = ((P - cp) + cn) / (P + N)            # «больше порога -> класс 1»
    acc_neg = (cp + (N - cn)) / (P + N)            # обратная ориентация
    i_p, i_n = int(acc_pos.argmax()), int(acc_neg.argmax())
    mid = lambda i: float(vs[i] if i == len(vs) - 1 else (vs[i] + vs[i + 1]) / 2)
    if acc_pos[i_p] >= acc_neg[i_n]:
        return mid(i_p), 1, float(acc_pos[i_p])
    return mid(i_n), -1, float(acc_neg[i_n])


def cv_accuracy(pos, neg, k=5, seed=21):
    """Точность с порогом: k-фолд, порог на k-1 фолдах, замер на отложенном."""
    g = torch.Generator().manual_seed(seed)
    ip, in_ = torch.randperm(len(pos), generator=g), torch.randperm(len(neg), generator=g)
    accs = []
    for f in range(k):
        hp, hn = ip[f::k], in_[f::k]
        mp = torch.ones(len(pos), dtype=torch.bool); mp[hp] = False
        mn = torch.ones(len(neg), dtype=torch.bool); mn[hn] = False
        t, sgn, _ = best_threshold(pos[mp], neg[mn])
        tp, tn = pos[hp], neg[hn]
        accs.append(float((((tp - t) * sgn > 0).sum() + ((tn - t) * sgn <= 0).sum())
                          / (len(tp) + len(tn))))
    return sum(accs) / len(accs)


def auroc_ci(pos, neg, n_boot=1000, seed=21):
    """95% ДИ AUROC бутстрэпом по промптам, стратифицированно по классам."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        p = pos[torch.randint(len(pos), (len(pos),), generator=g)]
        n = neg[torch.randint(len(neg), (len(neg),), generator=g)]
        vals.append(auroc(p, n))
    v = torch.tensor(vals).sort().values
    return float(v[int(0.025 * n_boot)]), float(v[int(0.975 * n_boot) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_max", type=int, default=0,
                    help="ограничить пул каждого класса (0 = весь train+val+test)")
    ap.add_argument("--folds", type=int, default=5, help="фолдов для точности")
    ap.add_argument("--n_boot", type=int, default=1000, help="бутстрэп-повторов для ДИ")
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

    def load_pool(h):
        """Весь пул класса: train + val + test с дедупом по тексту инструкции."""
        seen, out = set(), []
        for split in ("train", "val", "test"):
            for r in json.load(open(f"data/{rung}_splits/{h}_{split}.json")):
                if r["instruction"] not in seen:
                    seen.add(r["instruction"])
                    out.append(QWEN25.format(instruction=r["instruction"]))
        return out[:args.n_max] if args.n_max else out

    pool_pos, pool_neg = load_pool("harmful"), load_pool("harmless")
    bs = args.batch_size

    # --- направления ---
    ax = os.path.join(save_dir, "common_axis")
    unit = lambda v: v / v.norm()
    dim_path = f"{save_dir}/{os.getenv('DIM_DIR')}/{model_id}"
    d = unit(torch.load(f"{dim_path}/direction.pt", map_location="cpu").float())
    add_layer = int(json.load(open(f"{dim_path}/direction_metadata.json"))["layer"])
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
            # float16: пул ~2300 промптов x 36 слоёв x 2048 в float32 — это 0.7 ГБ,
            # для проекций половинной точности активаций достаточно
            out.append(torch.stack([p.value.detach().half().cpu() for p in per], dim=1))
            torch.cuda.empty_cache()
        return torch.cat(out, 0)                      # [n, n_layers, hidden]

    print(f"rung={rung}  пул: {len(pool_pos)} harmful + {len(pool_neg)} harmless "
          f"(train+val+test, дедуп)")
    A = {"pos": acts(pool_pos), "neg": acts(pool_neg)}
    n_layers = A["pos"].shape[1]
    print(f"активации сняты: {n_layers} слоёв")

    fixed_layers = {0, n_layers // 2, add_layer, n_layers - 1}
    rows = []
    for L in range(n_layers):
        # потолок слоя: разность средних ЭТОГО слоя. Считается на том же пуле, что и оценка,
        # поэтому это именно ПОТОЛОК (оптимистичный ориентир), а не честный held-out.
        HP, HN = A["pos"][:, L].float(), A["neg"][:, L].float()
        md = unit(HP.mean(0) - HN.mean(0))
        for name, v in list(DIRS.items()) + [("meandiff_layer", md)]:
            sp, sn = HP @ v, HN @ v
            pooled = ((sp.var() + sn.var()) / 2).sqrt()
            r = {"layer": L, "direction": name, "auroc": auroc(sp, sn),
                 "acc_cv": cv_accuracy(sp, sn, args.folds, args.seed),
                 "cohen_d": float((sp.mean() - sn.mean()).abs() / pooled.clamp(min=1e-9))}
            if L in fixed_layers:            # ДИ считаем только там, где репортим
                lo, hi = auroc_ci(sp, sn, args.n_boot, args.seed)
                r["auroc_lo"], r["auroc_hi"] = lo, hi
            rows.append(r)

    names = list(DIRS) + ["meandiff_layer"]
    best_layer = {n: max((r for r in rows if r["direction"] == n), key=lambda r: r["auroc"])
                  for n in names}
    for n in names:
        b = best_layer[n]
        print(f"{n:16s} лучший слой {b['layer']:2d}: AUROC={b['auroc']:.3f} "
              f"acc_cv={b['acc_cv']:.3f} d={b['cohen_d']:.2f}")

    fixed = [("начальный", 0), ("средний", n_layers // 2),
             ("add_layer", add_layer), ("последний", n_layers - 1)]
    fixed = [(t, l) for t, l in fixed if 0 <= l < n_layers]
    cell = lambda l, n, k: [r for r in rows if r["layer"] == l and r["direction"] == n][0][k]

    L = [f"# §8. Read против write: детектирует ли остаток DIM вредность? Ступень `{rung}`", "",
         f"Модель {model_id}. Проекция активации последнего токена на направление. "
         f"Пул: **{len(pool_pos)} harmful + {len(pool_neg)} harmless** (train+val+test, "
         f"дедуп). AUROC порога не требует и считается на всём пуле, 95% ДИ — бутстрэп "
         f"({args.n_boot}); точность — {args.folds}-фолдовая кросс-валидация.",
         "Знак направления произволен, поэтому AUROC берётся как max(auc, 1−auc).",
         "`meandiff_layer` — разность средних, посчитанная на train ЭТОГО слоя: потолок слоя.", "",
         "## Фиксированные слои (основная таблица)", "",
         "Слои зафиксированы заранее, без отбора: выбор лучшего из "
         f"{n_layers} завышает оценку у слабых направлений.", "",
         "AUROC на test:", "",
         "| направление | " + " | ".join(f"{t} ({l})" for t, l in fixed) + " |",
         "|" + "---|" * (len(fixed) + 1)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(
            f"{cell(l, n, 'auroc'):.3f} <sub>[{cell(l, n, 'auroc_lo'):.3f}, "
            f"{cell(l, n, 'auroc_hi'):.3f}]</sub>" for _, l in fixed) + " |")
    L += ["", f"В квадратных скобках — 95% ДИ бутстрэпом по промптам.", "",
          f"Точность, {args.folds}-фолдовая кросс-валидация:", "",
          "| направление | " + " | ".join(f"{t} ({l})" for t, l in fixed) + " |",
          "|" + "---|" * (len(fixed) + 1)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{cell(l, n, 'acc_cv'):.3f}" for _, l in fixed) + " |")
    L += ["",
         "## Лучший слой каждого направления (для справки; оценка завышена отбором)", "",
         "| направление | слой | AUROC | acc (CV) | d Коэна |", "|---|---|---|---|---|"]
    for n in names:
        b = best_layer[n]
        L.append(f"| {n} | {b['layer']} | **{b['auroc']:.3f}** | {b['acc_cv']:.3f} | {b['cohen_d']:.2f} |")
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
