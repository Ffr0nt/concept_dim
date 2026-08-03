"""Этап 0 — генерация DIM-направления (только шаги 1-2 пайплайна, без eval).

Повторяет шаги run_pipeline до select_and_save_direction включительно, но НЕ импортирует
сам run_pipeline (тот тянет evaluate_jailbreak -> strong_reject, которого нет в окружении)
и НЕ запускает тяжёлые eval-шаги (генерация completions), которые нам не нужны и могут
падать на Qwen2.5. Сохраняет в results/<DIM_DIR>/<model_alias>/:
    direction.pt, direction_metadata.json, generate_directions/mean_diffs.pt

GPU-операция — запускает пользователь через run/0_dim.sh.
Env (задаёт раннер): SAVE_DIR, DIM_DIR, REFUSAL_SPLITS, HUGGINGFACE_CACHE_DIR.
"""
import argparse
import json
import os
import random
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--fork", default="../geometry-of-refusal")
    args = ap.parse_args()

    fork = os.path.abspath(args.fork)
    sys.path.insert(0, os.path.join(fork, "refusal_direction"))
    # только leaf-модули (без strong_reject / evaluate_jailbreak)
    from pipeline.config import Config
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.generate_directions import generate_directions
    from pipeline.submodules.select_direction import select_direction, get_refusal_scores
    from dataset.load_dataset import load_dataset_split

    for v in ("SAVE_DIR", "DIM_DIR"):
        assert os.getenv(v), f"env {v} не задан"
    print(f"splits={os.getenv('REFUSAL_SPLITS')}  DIM_DIR={os.getenv('DIM_DIR')}")

    cfg = Config(model_alias=os.path.basename(args.model), model_path=args.model)
    model_base = construct_model_base(cfg.model_path)

    # --- load_and_sample_datasets (как в run_pipeline) ---
    random.seed(42)
    harmful_train = load_dataset_split(harmtype="harmful", split="train", instructions_only=True)
    harmless_train = load_dataset_split(harmtype="harmless", split="train", instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype="harmful", split="val", instructions_only=True)
    harmless_val = load_dataset_split(harmtype="harmless", split="val", instructions_only=True)[:len(harmful_val)]

    # --- filter_data (по refusal-скорам) ---
    def keep(dataset, scores, cmp):
        return [x for x, s in zip(dataset, scores.tolist()) if cmp(s)]

    if cfg.filter_train:
        hs = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        ls = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = keep(harmful_train, hs, lambda s: s > 0)
        harmless_train = keep(harmless_train, ls, lambda s: s < 0)[:len(harmful_train)]
        print(f"filter train -> harmful={len(harmful_train)} harmless={len(harmless_train)}")
    if cfg.filter_val:
        hs = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        ls = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = keep(harmful_val, hs, lambda s: s > 0)
        harmless_val = keep(harmless_val, ls, lambda s: s < 0)

    # --- шаг 1: кандидатные направления (mean diffs) ---
    art = cfg.artifact_path()
    os.makedirs(os.path.join(art, "generate_directions"), exist_ok=True)
    mean_diffs = generate_directions(model_base, harmful_train, harmless_train,
                                     artifact_dir=os.path.join(art, "generate_directions"))
    torch.save(mean_diffs, os.path.join(art, "generate_directions/mean_diffs.pt"))

    # --- шаг 2: выбор лучшего направления ---
    os.makedirs(os.path.join(art, "select_direction"), exist_ok=True)
    pos, layer, direction = select_direction(model_base, harmful_val, harmless_val, mean_diffs,
                                             artifact_dir=os.path.join(art, "select_direction"))
    with open(os.path.join(art, "direction_metadata.json"), "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)
    torch.save(direction, os.path.join(art, "direction.pt"))

    print(f"DIM сохранён: pos={pos} layer={layer} -> {art}")


if __name__ == "__main__":
    main()
