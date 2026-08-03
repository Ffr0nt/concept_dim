"""Этап 2a — снятие активаций последнего токена на слое(ях) L (вход для метрик разброса).

Прогоняет N промптов ступени через модель и сохраняет матрицу [N, d_model] активаций
residual stream на ВХОДЕ блока L (позиция последнего токена) — по каждому запрошенному слою.
Несколько слоёв снимаются за ОДИН проход (хуки на все блоки сразу). Зеркалит паттерн
get_mean_activations (refusal_direction/pipeline/submodules/generate_directions.py),
но хранит пер-промптные векторы вместо среднего.

GPU-операция — запускает ПОЛЬЗОВАТЕЛЬ (свободный GPU через CUDA_VISIBLE_DEVICES):
    cd ~/f.zakharov/concept_dim
    CUDA_VISIBLE_DEVICES=<N> uv run python experiments/mvp/scripts/extract_activations.py \
        --model Qwen/Qwen2.5-3B-Instruct --splits theft --layers 6 12 18 24 30
"""
import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--splits", required=True,
                    help="имя ступени: theft / illegal_activities / malicious_use")
    ap.add_argument("--layers", type=int, nargs="+", default=[18],
                    help="индексы блоков (можно несколько за один проход)")
    ap.add_argument("--fork", default="../geometry-of-refusal")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="каталог для дампов (по умолчанию experiments/mvp/artifacts)")
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    # refusal_direction/ — отдельный source-root (импорты вида `from pipeline...`)
    sys.path.insert(0, os.path.join(fork, "refusal_direction"))
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.utils.hook_utils import add_hooks

    harmful_path = os.path.join(fork, "data", f"{args.splits}_splits", "harmful_train.json")
    instructions = [d["instruction"] for d in json.load(open(harmful_path))]
    print(f"{args.splits}: {len(instructions)} промптов из {harmful_path}; слои {args.layers}")

    model_base = construct_model_base(args.model)
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    for L in args.layers:
        assert 0 <= L < n_layers, f"layer {L} вне [0, {n_layers})"

    stores = {L: [] for L in args.layers}

    def make_hook(L):
        def hook_fn(module, inp):
            # inp[0]: [batch, seq, d]; левый паддинг → индекс -1 = реальный последний токен
            stores[L].append(inp[0][:, -1, :].detach().float().cpu())
        return hook_fn

    pre_hooks = [(model_base.model_block_modules[L], make_hook(L)) for L in args.layers]

    for i in range(0, len(instructions), args.batch_size):
        batch = instructions[i:i + args.batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(input_ids=inputs.input_ids.to(model.device),
                      attention_mask=inputs.attention_mask.to(model.device))

    out_dir = os.path.abspath(args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "artifacts"))
    os.makedirs(out_dir, exist_ok=True)
    for L in args.layers:
        acts = torch.cat(stores[L], dim=0)  # [N, d_model]
        out = os.path.join(out_dir, f"acts_{args.splits}_L{L}.pt")
        torch.save({"acts": acts, "model": args.model, "splits": args.splits,
                    "layer": L, "n_layers": n_layers}, out)
        print(f"  L={L}: сохранено {out}  shape={tuple(acts.shape)}")


if __name__ == "__main__":
    main()
