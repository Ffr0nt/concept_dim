"""Общая компонента ВСЕХ направлений, наводящих отказ: выученные реперы + DIM. CPU.

Шаг 1 искал ось внутри ступени, шаг 2 — по всем выученным реперам сразу. Здесь популяция
смешанная: каждый конус входит как репер B_j [k, hidden], каждый DIM — как одномерный репер
[1, hidden]. Вопрос: есть ли направление, общее для всего, что наводит отказ, независимо от
способа получения (обучение конуса или разность средних).

Конусная компонента сама по себе уже сохранена отдельно
(results/common_axis/global/pooled.pt, шаг 2) и здесь НЕ пересчитывается — она служит
точкой отсчёта: cos(w_mixed, w_cones) показывает, увела ли DIM общую ось в сторону.

ВЗВЕШИВАНИЕ. P = sum_j c_j B_j^T B_j, sum c_j = 1. Каждый объект — проектор с собственным
значением 1 на своём span-е, поэтому вес объекта не зависит от его k: конус с k=5 и DIM с
k=1 входят на равных. Два режима:
  pooled    c_j = 1/N по всем N объектам (конусов 12, DIM 4 — конусы численно доминируют);
  balanced  половина веса группе конусов, половина группе DIM.
Пол lam_1 = max_j c_j (взаимно ортогональные span-ы): 1/16 = 0.062 для pooled и
1/8 = 0.125 для balanced. Потолок 1.0.

Спектр считается через малую матрицу: ненулевые с.з. P = M^T C M совпадают с с.з.
C^{1/2} M M^T C^{1/2}, где M — стек всех реперов, C = diag(c_j, повторённый k_j раз).

ТОЧКА ИЗВЛЕЧЕНИЯ DIM. DIM кластеризуются по ней, а не по концепту (theft~illegal 0.994,
malicious~all 0.985, между группами 0.38 — см. reports/results.md). Поэтому варианты
считаются отдельно: native (как выбрал пайплайн) и каждая общая точка.

Запуск:
  bash experiments/common_axis/run/3_mixed_axis.sh
  python mixed_axis.py --base <fork>/results --out reports/mixed-axis.md
"""
import argparse
import os

import torch

from common_axis import RUNGS, SEEDS, load_dim, load_frame, rand_frames, unit


def parse_kmap(s):
    return {p.split("=")[0].strip(): int(p.split("=")[1]) for p in s.split(",")}


def weighted_spectrum(frames, weights):
    """Спектр P = sum_j c_j B_j^T B_j через малую матрицу. frames: [k_j, hidden]."""
    M = torch.cat(frames, dim=0)
    c = torch.cat([torch.full((f.shape[0],), w, dtype=torch.float64)
                   for f, w in zip(frames, weights)])
    S = c.sqrt().unsqueeze(1) * M                      # C^{1/2} M
    lam, U = torch.linalg.eigh(S @ S.T)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order], U[:, order]
    W = S.T @ U
    W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)
    return lam, W


def null_band(shapes, weights, hidden, n_null, gen, n_lam=3):
    acc = []
    for _ in range(n_null):
        lam, _ = weighted_spectrum(rand_frames(shapes, hidden, gen), weights)
        acc.append(lam[:n_lam])
    A = torch.stack(acc)
    return A.mean(0), A.quantile(0.95, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--k_map", default="theft=5,illegal_activities=5,malicious_use=5,all=7")
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--threads", type=int, default=int(os.getenv("THREADS", "8")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    gen = torch.Generator().manual_seed(args.seed)
    KMAP = parse_kmap(args.k_map)
    save_dir = os.path.join(args.base, "common_axis", "global") if args.save_dir is None \
        else args.save_dir
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # --- популяция ---
    cones, cone_names = [], []
    for r in RUNGS:
        for s in SEEDS:
            B = load_frame(args.base, r, s, KMAP[r])
            if B is not None:
                cones.append(B)
                cone_names.append(f"{r}/seed_{s} (k={KMAP[r]})")
    D, MD, SEL = load_dim(args.base, args.model)
    concepts = [r for r in RUNGS if r in D]
    hidden = cones[0].shape[1]
    sites = sorted(set(SEL[r] for r in concepts))

    # конусная ось из шага 2 — точка отсчёта, не пересчитываем
    p_cone = os.path.join(args.base, "common_axis", "global", "pooled.pt")
    w_cone = torch.load(p_cone, map_location="cpu")["W"][0].double() \
        if os.path.exists(p_cone) else None

    emit("# Общая компонента всех направлений отказа: выученные реперы + DIM")
    emit()
    emit(f"Популяция: {len(cones)} выученных реперов ("
         + ", ".join(f"{r} k={KMAP[r]}" for r in RUNGS if r in KMAP)
         + f", сиды {SEEDS}) + {len(concepts)} DIM как одномерные реперы. hidden = {hidden}.")
    emit(f"Нуль — {args.n_null} наборов случайных ортонормированных реперов тех же форм и весов.")
    emit()
    emit("Веса: **pooled** — 1/N по всем объектам; **balanced** — половина веса конусам, "
         "половина DIM. Пол lam_1 = max_j c_j, потолок 1.0.")
    emit()
    if w_cone is None:
        emit("⚠️ не найден results/common_axis/global/pooled.pt — сначала шаг 2.")
    else:
        emit("Конусная общая компонента (шаг 2) лежит отдельно в "
             "`results/common_axis/global/pooled.pt` и здесь используется только для сверки.")
    emit()

    variants = [("native", None)] + [(f"pos{p}_L{l}", (p, l)) for p, l in sites]
    saved = {}

    emit("## Спектр смешанной популяции")
    emit()
    emit("| DIM в точке | веса | N | пол | lam_1 | нуль (среднее / p95) | lam_2 | "
         "cos(w_1, w_1 конусов) |")
    emit("|---|---|---|---|---|---|---|---|")

    for vname, site in variants:
        dims = [(unit(D[r]) if site is None else unit(MD[r][site[0], site[1]])).unsqueeze(0)
                for r in concepts]
        frames = cones + dims
        shapes = [f.shape[0] for f in frames]
        for mode in ("pooled", "balanced"):
            if mode == "pooled":
                wts = [1.0 / len(frames)] * len(frames)
            else:
                wts = ([0.5 / len(cones)] * len(cones)) + ([0.5 / len(dims)] * len(dims))
            lam, W = weighted_spectrum(frames, wts)
            w1 = W[:, 0]
            if sum(float((B @ w1).sum()) for B in frames) < 0:
                w1 = -w1
            nm, np95 = null_band(shapes, wts, hidden, args.n_null, gen)
            cc = f"{abs(float(w_cone @ w1)):.3f}" if w_cone is not None else "—"
            emit(f"| {vname} | {mode} | {len(frames)} | {max(wts):.3f} | "
                 f"**{lam[0]:.3f}** | {nm[0]:.3f} / {np95[0]:.3f} | {lam[1]:.3f} | {cc} |")
            saved[(vname, mode)] = (lam, W, w1, frames, wts)
    emit()

    # --- alpha по объектам для balanced в каждой точке ---
    for vname, _ in variants:
        lam, W, w1, frames, wts = saved[(vname, "balanced")]
        emit(f"## Кто несёт ось: DIM в точке {vname}, веса balanced")
        emit()
        alphas = [float((B @ w1).norm()) for B in frames]
        names = cone_names + [f"DIM {r}" for r in concepts]
        cone_a = alphas[:len(cones)]
        dim_a = alphas[len(cones):]
        emit(f"Конусы: alpha от {min(cone_a):.3f} до {max(cone_a):.3f} "
             f"(среднее {sum(cone_a)/len(cone_a):.3f}). "
             f"DIM: " + ", ".join(f"{r} {a:.3f}" for r, a in zip(concepts, dim_a)) + ".")
        emit()
        emit("| объект | alpha = \\|B w_1\\| |")
        emit("|---|---|")
        for n, a in zip(names, alphas):
            mark = "**" if n.startswith("DIM") else ""
            emit(f"| {mark}{n}{mark} | {mark}{a:.3f}{mark} |")
        emit()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for (vname, mode), (lam, W, w1, frames, wts) in saved.items():
            m = min(W.shape[1], 16)
            Wm = W[:, :m].T.clone()
            Wm[0] = w1
            torch.save({"variant": f"mixed_{vname}_{mode}", "k_map": KMAP,
                        "n_cones": len(cones), "n_dim": len(concepts),
                        "cone_names": cone_names, "dim_site": vname, "weights": mode,
                        "W": Wm.float(), "lam": lam[:m].float(),
                        "floor": max(wts)},
                       os.path.join(save_dir, f"mixed_{vname}_{mode}.pt"))
        emit(f"Смешанные оси сохранены: `{save_dir}/mixed_<точка>_<веса>.pt`. "
             f"Конусная ось шага 2 (`pooled.pt`) не перезаписывается.")
        emit()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"\n[записано: {args.out}]")


if __name__ == "__main__":
    main()
