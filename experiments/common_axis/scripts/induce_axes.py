"""§7 — наведение отказа общей осью: ось это refusal или harmfulness? GPU.

Аблация (шаги 4-5) отвечает на вопрос «нужно ли направление для отказа». Здесь обратный,
write-вопрос: ДОСТАТОЧНО ли его. Добавляем h <- h + alpha*w на БЕЗОБИДНЫХ промптах в одном
слое и смотрим, начинает ли модель отказывать. Если да — ось кодирует отказ; если нет, а
аблация при этом работала, — она коррелят «вредности» промпта, а не медиатор отказа.

СЛОЙ. По умолчанию — add_layer своей ступени (та же точка, что у DIM в
direction_metadata.json и в eval_cones.py), чтобы наведение было сопоставимо с протоколом
статьи. Профиль проекции по слоям всё равно считается и печатается, НО в нормированном
виде: cos(h_last, w) = <h,w>/||h||. Сырая |<h,w>| для выбора слоя непригодна — она растёт
с глубиной вместе с нормой активаций, и её argmax всегда упирается в последние слои
(на ступени `all` это давало слой 35 из 36, где вмешательство уже ни на что не влияет).
Переопределяется через --layer.

МАСШТАБ alpha. Единица — норма DIM-вектора ступени ||r||, как в eval_cones.py (там
add_layer-добавление идёт с alpha = ||DIM||). Свип задаётся в кратных этой единицы, так что
alpha=1 сопоставим с протоколом статьи.

МЕТРИКИ (всё на harmless_test):
  refusal_rate  доля промптов с refusal_metric > 0 — наведённый отказ. Базовый уровень
                считается без вмешательства.
  KL_ret        цена вмешательства для безобидного поведения; сравнивать направления
                можно только при близком KL.

КРОСС-НЕЙТРАЛИЗАЦИЯ. Отдельный блок: наводим отказ одним направлением и ОДНОВРЕМЕННО
аблируем другое во всех слоях. Масштаб берётся НЕ максимальный из свипа, а тот, при котором
наведение этим направлением реально работает (argmax refusal_rate по свипу): отклик на alpha
немонотонен — при перестреле (2-8 x ||DIM||) отказ снова падает до нуля вместе с осмысленным
выходом, и кросс-тест в этом режиме меряет поломку, а не нейтрализацию. Если аблация w снимает отказ, наведённый DIM, — они делят
один канал; если нет — каналы разные. Контроль: та же пара со случайным направлением.

Env: SAVE_DIR, DIM_DIR, REFUSAL_SPLITS, HUGGINGFACE_CACHE_DIR. Запускает пользователь.
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], help="alpha в кратных ||DIM||")
    ap.add_argument("--cross_scale", type=float, default=None,
                    help="масштаб для кросс-теста; по умолчанию argmax refusal_rate по свипу")
    ap.add_argument("--layer", type=int, default=None,
                    help="слой вмешательства (по умолчанию add_layer ступени из метаданных DIM)")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from nnsight import LanguageModel
    from scoring import projection_einops, refusal_metric

    for v in ("SAVE_DIR", "DIM_DIR", "REFUSAL_SPLITS"):
        assert os.getenv(v), f"env {v} не задан"
    rung, save_dir = os.getenv("REFUSAL_SPLITS"), os.getenv("SAVE_DIR")
    model_id = args.model.split("/")[-1]

    model = LanguageModel(args.model, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
                          device_map="auto", torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    with model.trace("Hello"):
        pass

    refusal_toks = [40, 2121] if "qwen2.5" in args.model.lower() else [40]
    QWEN25 = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
              "helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n"
              "<|im_start|>assistant\n")

    def load_prompts(h):
        return [QWEN25.format(instruction=r["instruction"])
                for r in json.load(open(f"data/{rung}_splits/{h}_test.json"))]

    harmful, harmless = load_prompts("harmful"), load_prompts("harmless")
    bs = args.batch_size

    # --- направления ---
    ax = os.path.join(save_dir, "common_axis")
    dim_path = f"{save_dir}/{os.getenv('DIM_DIR')}/{model_id}"
    dim_raw = torch.load(f"{dim_path}/direction.pt", map_location="cpu").float()
    add_layer = int(json.load(open(f"{dim_path}/direction_metadata.json"))["layer"])
    unit = lambda v: v / v.norm()
    dim_norm = float(dim_raw.norm())
    d = unit(dim_raw)
    W_cone = torch.load(os.path.join(ax, "global", "pooled.pt"), map_location="cpu")["W"].float()
    mp = os.path.join(ax, "global", "mixed_native_balanced.pt")
    W_mixed = torch.load(mp, map_location="cpu")["W"].float() if os.path.exists(mp) else None
    g = torch.Generator().manual_seed(args.seed)
    rnd = unit(torch.randn(W_cone.shape[1], generator=g))

    DIRS = {"w_cone": W_cone[0], "dim": d,
            "dim_perp_cone": unit(d - (d @ W_cone[0]) * W_cone[0]),
            "cone_perp_dim": unit(W_cone[0] - (W_cone[0] @ d) * d),
            "rand": rnd}
    if W_mixed is not None:
        DIRS["w_mixed"] = W_mixed[0]
    print(f"rung={rung}  ||DIM||={dim_norm:.3f}  направления: {list(DIRS)}")

    # --- профиль проекции по слоям: где ось «живёт» ---
    def proj_profile(prompts, v):
        """Нормированная проекция cos(h_last, w) по слоям — без нормировки argmax всегда
        уходит в последние слои вслед за ростом ||h||."""
        vv = v.to(model.dtype).to(model.device)
        acc = None
        for i in range(0, len(prompts), bs):
            with model.trace(prompts[i:i + bs]):
                per = [(layer.output[0][:, -1] @ vv / layer.output[0][:, -1].norm(dim=-1)).save()
                       for layer in model.model.layers]
            vals = torch.tensor([float(p.value.float().mean()) for p in per])
            acc = vals if acc is None else acc + vals
            torch.cuda.empty_cache()
        return acc / max(1, (len(prompts) + bs - 1) // bs)

    w1 = DIRS["w_cone"]
    pf_harm, pf_less = proj_profile(harmful, w1), proj_profile(harmless, w1)
    gap = pf_harm - pf_less
    layer = args.layer if args.layer is not None else add_layer
    print(f"add_layer ступени = {add_layer}; пик cos-разрыва = слой {int(gap.argmax())} "
          f"({float(gap.max()):+.3f}); вмешательство в слой {layer}")

    # --- прогоны ---
    def run(prompts, add=None, abl=None, want="metric"):
        av = None if add is None else (add[0].to(model.dtype).to(model.device), add[1])
        bv = None if abl is None else abl.to(model.dtype).to(model.device)
        out = []
        for i in range(0, len(prompts), bs):
            with model.trace(prompts[i:i + bs]):
                if bv is not None:
                    for lay in model.model.layers:
                        lay.input -= projection_einops(lay.input, bv)
                        a = lay.self_attn.output[0][:]
                        lay.self_attn.output[0][:] -= projection_einops(a, bv)
                        m = lay.mlp.output[:]
                        lay.mlp.output[:] -= projection_einops(m, bv)
                if av is not None:
                    model.model.layers[layer].input += av[1] * av[0]
                s = (refusal_metric(model.lm_head.output[:, -1], refusal_toks)
                     if want == "metric" else model.lm_head.output[:, -1]).save()
            out.append(s.value.detach().float().cpu())
            torch.cuda.empty_cache()
        return torch.cat(out, 0)

    def kl(base, new):
        return float(torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(base.double(), -1),
            torch.nn.functional.softmax(new.double(), -1), reduction="batchmean"))

    base_m = run(harmless, want="metric")
    base_l = run(harmless, want="logits")
    base_rr = float((base_m > 0).float().mean())
    print(f"harmless без вмешательства: refusal_rate={base_rr:.3f}  "
          f"metric_mean={float(base_m.mean()):+.3f}")

    rows = []
    for name, v in DIRS.items():
        for sc in args.scales:
            a = sc * dim_norm
            m = run(harmless, add=(v, a), want="metric")
            lg = run(harmless, add=(v, a), want="logits")
            r = {"block": "induce", "direction": name, "scale": sc, "alpha": a,
                 "refusal_rate": float((m > 0).float().mean()),
                 "metric_mean": float(m.mean()), "kl_ret": kl(base_l, lg)}
            rows.append(r)
            print(f"[induce] {name:14s} x{sc:<4} rr={r['refusal_rate']:.3f} "
                  f"metric={r['metric_mean']:+.3f} KL={r['kl_ret']:.4f}", flush=True)

    # --- кросс-нейтрализация в РАБОЧЕЙ точке наведения ---
    best = {}
    for name in DIRS:
        cand = [r for r in rows if r["block"] == "induce" and r["direction"] == name]
        best[name] = (args.cross_scale if args.cross_scale is not None
                      else max(cand, key=lambda r: r["refusal_rate"])["scale"])
        rr = max(cand, key=lambda r: r["refusal_rate"])["refusal_rate"]
        print(f"рабочая точка {name}: x{best[name]} (rr={rr:.3f})")
    pairs = [("dim", "w_cone"), ("dim", "rand"), ("dim", "dim"),
             ("w_cone", "dim"), ("w_cone", "rand"), ("w_cone", "w_cone")]
    if "w_mixed" in DIRS:
        pairs += [("w_mixed", "dim"), ("w_mixed", "rand"), ("w_mixed", "w_mixed")]
    for ind, abl in pairs:
        sc_i = best[ind]
        m = run(harmless, add=(DIRS[ind], sc_i * dim_norm), abl=DIRS[abl], want="metric")
        base_rr_ind = [r for r in rows if r["block"] == "induce"
                       and r["direction"] == ind and r["scale"] == sc_i][0]["refusal_rate"]
        r = {"block": "cross", "induce": ind, "ablate": abl, "scale": sc_i,
             "induce_only_rr": base_rr_ind,
             "refusal_rate": float((m > 0).float().mean()), "metric_mean": float(m.mean())}
        rows.append(r)
        print(f"[cross] наводим {ind:8s} x{sc_i} (одно даёт {base_rr_ind:.3f}) + аблируем "
              f"{abl:8s} -> rr={r['refusal_rate']:.3f} metric={r['metric_mean']:+.3f}",
              flush=True)

    ind_rows = [r for r in rows if r["block"] == "induce"]
    cr_rows = [r for r in rows if r["block"] == "cross"]
    L = [f"# §7. Наведение отказа: ось — refusal или harmfulness? Ступень `{rung}`", "",
         f"Модель {model_id}; harmless_test = {len(harmless)}. Добавление h ← h + α·w на вход "
         f"блока слоя **{layer}** (add_layer ступени, как в eval_cones.py; "
         f"пик cos-разрыва harmful−harmless — слой {int(gap.argmax())}).",
         f"Единица α = ‖DIM‖ = {dim_norm:.3f}, как в eval_cones.py.",
         f"Без вмешательства: refusal_rate = {base_rr:.3f}, metric_mean = "
         f"{float(base_m.mean()):+.3f}.", "",
         "**Сравнивать направления можно только при близком KL_ret.**", "",
         "| направление | α (×‖DIM‖) | refusal_rate | metric | KL_ret |", "|---|---|---|---|---|"]
    for r in ind_rows:
        L.append(f"| {r['direction']} | {r['scale']} | **{r['refusal_rate']:.3f}** | "
                 f"{r['metric_mean']:+.3f} | {r['kl_ret']:.4f} |")
    L += ["", "## Кросс-нейтрализация (наведение в рабочей точке + аблация во всех слоях)", "",
          "Масштаб выбран по argmax refusal_rate свипа: отклик немонотонен, при перестреле "
          "отказ падает вместе с осмысленным выходом.", "",
          "| наводим | α (×‖DIM‖) | одно наведение | + аблируем | refusal_rate | metric |",
          "|---|---|---|---|---|---|"]
    for r in cr_rows:
        L.append(f"| {r['induce']} | {r['scale']} | {r['induce_only_rr']:.3f} | "
                 f"{r['ablate']} | **{r['refusal_rate']:.3f}** | {r['metric_mean']:+.3f} |")
    L += ["", "## Профиль нормированной проекции cos(h, w_cone) по слоям", "",
          "| слой | harmful | harmless | разрыв |", "|---|---|---|---|"]
    for i in range(len(pf_harm)):
        L.append(f"| {i} | {float(pf_harm[i]):.3f} | {float(pf_less[i]):.3f} | "
                 f"{float(gap[i]):+.3f} |")
    md = "\n".join(L) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(md)
        print(f"[записано: {args.out}]")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump({"rung": rung, "layer": layer, "dim_norm": dim_norm,
                   "base_refusal_rate": base_rr, "rows": rows,
                   "profile_harmful": pf_harm.tolist(), "profile_harmless": pf_less.tolist()},
                  open(args.json_out, "w"), indent=1)
        print(f"[записано: {args.json_out}]")


if __name__ == "__main__":
    main()
