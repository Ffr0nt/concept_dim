"""Общая ось СРАЗУ ПО ВСЕМ прогонам: ступени x сиды. CPU, GPU не нужен.

Шаг 1 ([common_axis.py](common_axis.py)) усреднял по сидам ВНУТРИ ступени и нашёл ось с
lam_1 = 0.67-0.87 при нуле 0.37-0.41. Здесь вопрос шире: есть ли направление, общее для всех
конусов вообще, независимо от концепта.

КОНФАУНДЕР, из-за которого нельзя просто усреднить всё. Сид старта в rdo.py:1126 ставится
как base_seed*1000 + i*10 + k, БЕЗ участия ступени: при одном сиде все ступени стартуют из
ОДНОГО случайного репера, а обучение короткое (16-54 шага на 2048 измерений). Поэтому у
любой межступенчатой величины есть тривиальное объяснение «общий старт». Часть 4
learned_directions показала это на rho_span: один сид даёт в 1.5-2.5 раза больше, чем разные.

Поэтому считаются ЧЕТЫРЕ варианта, а не один:

  same_seed  B. все ступени при ОДНОМ сиде — общий старт присутствует.
  mixed      C. каждой ступени СВОЙ сид, усреднение по всем таким расстановкам —
                общего старта нет ни у одной пары. ЧИСТЫЙ вариант.
  pooled     всё сразу: ступени x сиды (N = 4*3 = 12). Максимум данных, но внутри него
                есть и пары с общим стартом — читать вместе с B/C.
  within     ось ступени из шага 1 (N = 3 сида), для сравнения.

B и C считаются при РАВНОМ N — иначе lam_1 несравнимы: у него пол 1/N (взаимно
ортогональные span-ы), то есть при разном N разные шкалы. Варианты с 3 узкими ступенями
(N=3, все шесть расстановок сидов чисты) и с 4 ступенями (N=4, при трёх сидах одна пара
неизбежно делит сид — помечено в отчёте).

Разные k у ступеней: `all` обучалась только на 7..12, остальные на 1..5 (или 1..8), общего k
нет. Состав задаётся --k_map, по умолчанию theft=5, illegal_activities=5, malicious_use=5,
all=7 — минимальная доступная у `all` и середина у остальных.

Дополнительно: cos(w_1^global, w_1^rung) — насколько глобальная ось совпадает с осью каждой
ступени из шага 1 (читает сохранённые results/common_axis/<rung>/dim_<k>.pt), и cos с DIM.

Запуск:
  bash experiments/common_axis/run/2_global_axis.sh
  python global_axis.py --base <fork>/results --out reports/global-axis.md
"""
import argparse
import itertools
import os

import torch

from common_axis import (RUNGS, SEEDS, load_dim, load_frame, null_band, nu_of, spectrum,
                         unit)

NARROW = ["theft", "illegal_activities", "malicious_use"]


def parse_kmap(s):
    out = {}
    for part in s.split(","):
        r, v = part.split("=")
        out[r.strip()] = int(v)
    return out


_NULL_CACHE = {}


def variant_stats(frames, n_null, gen, null_cache=_NULL_CACHE):
    lam, W, M = spectrum(frames)
    w1 = W[:, 0]
    if sum(float((B @ w1).sum()) for B in frames) < 0:
        w1 = -w1
    alphas = [(B @ w1).norm().item() for B in frames]
    nus = [nu_of(B, w1)[0] for B in frames]
    # нуль зависит только от НАБОРА ФОРМ — у C-расстановок он один и тот же, считаем раз
    shapes = tuple(f.shape[0] for f in frames)
    key = (shapes, frames[0].shape[1], n_null)
    if key not in null_cache:
        null_cache[key] = null_band(list(shapes), frames[0].shape[1], n_null, gen)
    nm, np95 = null_cache[key]
    return {"lam": lam, "W": W, "M": M, "w1": w1, "alphas": alphas, "nus": nus,
            "null_mean": nm, "null_p95": np95, "N": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--k_map", default="theft=5,illegal_activities=5,malicious_use=5,all=7")
    ap.add_argument("--n_null", type=int, default=500)
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

    F = {}   # (rung, seed) -> репер [k, hidden]
    for r in RUNGS:
        for s in SEEDS:
            B = load_frame(args.base, r, s, KMAP[r])
            if B is not None:
                F[(r, s)] = B
    if not F:
        raise SystemExit("не найдено ни одного репера — проверь --base и --k_map")
    hidden = next(iter(F.values())).shape[1]
    have = [r for r in RUNGS if all((r, s) in F for s in SEEDS)]

    emit("# Общая ось по всем прогонам: ступени × сиды")
    emit()
    emit(f"Состав: " + ", ".join(f"{r} k={KMAP[r]}" for r in have)
         + f"; сиды {SEEDS}; hidden = {hidden}.")
    emit(f"Нуль — {args.n_null} наборов независимых случайных ортонормированных реперов той же формы.")
    emit()
    emit("**Пол** lam_1 = 1/N (взаимно ортогональные span-ы), поэтому варианты сравнимы "
         "только при РАВНОМ N. **B** = общий старт (один сид на все ступени), "
         "**C** = чистый (у каждой ступени свой сид).")
    emit()

    res = {}

    # --- B vs C при N = 3 (узкие ступени) и N = 4 (все) ---
    for tag, rungs in (("3 узкие ступени", [r for r in NARROW if r in have]),
                       ("4 ступени", have)):
        if len(rungs) < 2:
            continue
        emit(f"## {tag} (N = {len(rungs)})")
        emit()
        emit("| вариант | состав сидов | lam_1 | нуль (среднее / p95) | lam_2 | lam_3 | "
             "alpha (min/среднее/max) |")
        emit("|---|---|---|---|---|---|---|")

        rows = []
        for s in SEEDS:                                     # B: общий старт
            st = variant_stats([F[(r, s)] for r in rungs], args.n_null, gen)
            rows.append((f"B. один сид {s}", f"все {s}", st))
        # C: все расстановки сидов, где ни одна пара не делит сид (при 4 ступенях и 3 сидах
        # это невозможно — берём расстановки с минимумом совпадений)
        # критерий чистоты — число ПАР, делящих сид (а не максимальная кратность):
        # при 4 ступенях и 3 сидах расстановка (2,1,1) даёт одну такую пару, (2,2) — две,
        # поэтому берём только расстановки с минимумом пар.
        def shared_pairs(combo):
            return sum(combo.count(s) * (combo.count(s) - 1) // 2 for s in set(combo))
        combos = list(itertools.product(SEEDS, repeat=len(rungs)))
        best_rep = min(shared_pairs(c) for c in combos)
        clean = [c for c in combos if shared_pairs(c) == best_rep]
        acc = []
        for combo in clean:
            st = variant_stats([F[(r, s)] for r, s in zip(rungs, combo)], args.n_null, gen)
            acc.append(st)
        mean_lam = torch.stack([a["lam"][:3] for a in acc]).mean(0)
        note = ("все разные" if best_rep == 0
                else f"{best_rep} пара делит сид ({len(SEEDS)} сида на {len(rungs)} ступени)")
        rows.append((f"C. разные сиды (среднее по {len(clean)} расстановкам)", note,
                     {"lam": mean_lam, "null_mean": acc[0]["null_mean"],
                      "null_p95": acc[0]["null_p95"],
                      "alphas": [a for st in acc for a in st["alphas"]], "N": len(rungs)}))
        for name, comp, st in rows:
            al = st["alphas"]
            emit(f"| {name} | {comp} | **{st['lam'][0]:.3f}** | "
                 f"{st['null_mean'][0]:.3f} / {st['null_p95'][0]:.3f} | {st['lam'][1]:.3f} | "
                 f"{st['lam'][2]:.3f} | "
                 f"{min(al):.3f} / {sum(al)/len(al):.3f} / {max(al):.3f} |")
        emit()
        b_mean = sum(r[2]["lam"][0].item() for r in rows[:-1]) / (len(rows) - 1)
        c_val = rows[-1][2]["lam"][0].item()
        emit(f"B/C = **{b_mean / c_val:.2f}×** ({b_mean:.3f} против {c_val:.3f}). "
             f"Чем ближе к 1, тем меньше общий старт объясняет общую ось.")
        emit()
        res[tag] = rows

    # --- pooled: всё сразу ---
    pooled_keys = [(r, s) for r in have for s in SEEDS]
    st = variant_stats([F[k] for k in pooled_keys], args.n_null, gen)
    emit(f"## pooled: все прогоны сразу (N = {st['N']})")
    emit()
    emit(f"lam_1 = **{st['lam'][0]:.3f}** при нуле {st['null_mean'][0]:.3f} / "
         f"p95 {st['null_p95'][0]:.3f} (пол 1/N = {1/st['N']:.3f}); "
         f"lam_2 = {st['lam'][1]:.3f}, lam_3 = {st['lam'][2]:.3f}.")
    emit()
    emit("| ступень | сид | alpha | nu(+w) |")
    emit("|---|---|---|---|")
    for (r, s), a, nu in zip(pooled_keys, st["alphas"], st["nus"]):
        emit(f"| {r} | {s} | {a:.3f} | {nu:.3f} |")
    emit()

    # --- совпадение с осями ступеней из шага 1 ---
    emit("## Глобальная ось против осей ступеней (шаг 1)")
    emit()
    D, MD, SEL = load_dim(args.base, args.model)
    emit("| ступень | cos(w_1^global, w_1^rung) | cos(w_1^global, DIM ступени) | "
         "cos(w_1^rung, DIM ступени) |")
    emit("|---|---|---|---|")
    for r in have:
        p = os.path.join(args.base, "common_axis", r, f"dim_{KMAP[r]}.pt")
        wr = torch.load(p, map_location="cpu")["W"][0].double() if os.path.exists(p) else None
        g = unit(MD[r][SEL[r][0], SEL[r][1]]) if r in MD else None
        c1 = f"{abs(float(wr @ st['w1'])):.3f}" if wr is not None else "—"
        c2 = f"{abs(float(g @ st['w1'])):.3f}" if g is not None else "—"
        c3 = f"{abs(float(g @ wr)):.3f}" if (g is not None and wr is not None) else "—"
        emit(f"| {r} | {c1} | {c2} | {c3} |")
    emit()
    emit("DIM берётся в native-точке ступени; колонка 2 сравнима между ступенями только "
         "там, где точка совпадает (см. SEL в отчёте шага 1).")
    emit()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        m = min(st["W"].shape[1], 16)
        Wm = st["W"][:, :m].T.clone()
        Wm[0] = st["w1"]
        torch.save({"variant": "pooled", "members": [list(k) for k in pooled_keys],
                    "k_map": KMAP, "W": Wm.float(), "lam": st["lam"][:m].float(),
                    "lam_null_mean": st["null_mean"].float(),
                    "lam_null_p95": st["null_p95"].float(), "n_null": args.n_null,
                    "floor": 1.0 / st["N"], "alphas": torch.tensor(st["alphas"]),
                    "nu_plus": torch.tensor(st["nus"])},
                   os.path.join(save_dir, "pooled.pt"))
        emit(f"Глобальная ось сохранена: `{save_dir}/pooled.pt` (ключ `W`, [m, hidden]).")
        emit()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"\n[записано: {args.out}]")


if __name__ == "__main__":
    main()
