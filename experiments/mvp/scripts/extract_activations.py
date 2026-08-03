"""Этап 2a — снятие активаций последнего токена на слое L (вход для 1−cos).

Прогоняет N промптов ступени через модель и сохраняет матрицу [N, d_model] активаций
residual stream на ВХОДЕ блока L, позиция последнего токена. Зеркалит паттерн
get_mean_activations (refusal_direction/pipeline/submodules/generate_directions.py),
но хранит пер-промптные векторы вместо среднего.

GPU-операция — запускает ПОЛЬЗОВАТЕЛЬ (выбрать свободный GPU через CUDA_VISIBLE_DEVICES):
    cd ~/f.zakharov/concept_dim
    CUDA_VISIBLE_DEVICES=<N> uv run python experiments/mvp/scripts/extract_activations.py \
        --model Qwen/Qwen2.5-3B-Instruct --splits theft --layer 18
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
    ap.add_argument("--layer", type=int, required=True, help="слой L (индекс блока)")
    ap.add_argument("--fork", default="../geometry-of-refusal")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="путь для дампа .pt "
                         "(по умолчанию experiments/mvp/artifacts/acts_<splits>_L<layer>.pt)")
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    # refusal_direction/ — отдельный source-root (импорты вида `from pipeline...`)
    sys.path.insert(0, os.path.join(fork, "refusal_direction"))
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.utils.hook_utils import add_hooks

    harmful_path = os.path.join(fork, "data", f"{args.splits}_splits", "harmful_train.json")
    instructions = [d["instruction"] for d in json.load(open(harmful_path))]
    print(f"{args.splits}: {len(instructions)} промптов из {harmful_path}")

    model_base = construct_model_base(args.model)
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    assert 0 <= args.layer < n_layers, f"layer {args.layer} вне [0, {n_layers})"
    block = model_base.model_block_modules[args.layer]

    store = []

    def hook_fn(module, inp):
        # inp[0]: [batch, seq, d]; левый паддинг → индекс -1 = реальный последний токен
        store.append(inp[0][:, -1, :].detach().float().cpu())

    for i in range(0, len(instructions), args.batch_size):
        batch = instructions[i:i + args.batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=[(block, hook_fn)], module_forward_hooks=[]):
            with torch.no_grad():
                model(input_ids=inputs.input_ids.to(model.device),
                      attention_mask=inputs.attention_mask.to(model.device))

    acts = torch.cat(store, dim=0)  # [N, d_model]

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "artifacts", f"acts_{args.splits}_L{args.layer}.pt")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"acts": acts, "model": args.model, "splits": args.splits,
                "layer": args.layer, "n_layers": n_layers}, out)
    print(f"сохранено: {out}  shape={tuple(acts.shape)}")


if __name__ == "__main__":
    main()
