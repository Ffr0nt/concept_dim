"""Насколько хорошо DIM каждого концепта восстанавливается конусом каждой ступени. CPU.

Вопрос: если взять DIM концепта t (детерминирован, без сида) и конус ступени r, какая доля
DIM лежит в span-е конуса, и какая — внутри самого конуса (с ограничением w >= 0)?
Матрица 4x4 (концепт x ступень) отвечает, специфичен ли конус своему концепту.

Величины. Для единичного g = DIM/||DIM|| и репера B ([d, hidden], строки — рёбра конуса),
c = B g:
  R_span = sum_i c_i^2               доля энергии g, восстановимая из SPAN-а конуса, [0,1]
  R_cone = sum_i max(c_i, 0)^2       то же при w >= 0, т.е. из САМОГО конуса; <= R_span
  Delta  = R_span - R_cone = sum_i min(c_i,0)^2
  nu     = Delta / R_span            доля in-span энергии в минус-ортанте

R_cone — это замкнутая форма NNLS: при ортонормированном B задача min_{w>=0} ||Bw - g||
распадается покоординатно, w_i = max(c_i, 0), и ||r_cone||^2 = 1 - R_cone.
Солвер не нужен.

Нулевые уровни (без них числа не читаются при hidden=2048):
  rand  — случайный ортонормированный репер той же формы: E[R_span] = d/hidden;
  null  — B -> BQ, Q ~ Haar(d): span и R_span инвариантны, меняется только ориентация
          ортанта. Изолирует nu. E[nu] ~ 0.5.

ЛОВУШКА ТОЧКИ ИЗВЛЕЧЕНИЯ (см. часть 5 отчёта): select_direction выбирает (pos, layer)
независимо для каждой ступени, и смена точки меняет DIM сильнее, чем смена концепта.
Поэтому скрипт считает ВСЕ варианты: native (сохранённый direction.pt каждого концепта)
и каждую из общих точек, пересчитанную из mean_diffs.pt. Сравнивать матрицы между
концептами честно только в общей точке.

Запуск:
  bash experiments/learned_directions/run/5_dim_reconstruction.sh
  python dim_reconstruction.py --base <fork>/results --dims 5
"""
import argparse
import json
import os
import statistics as st

import torch

RUNGS = ["theft", "illegal_activities", "malicious_use", "all"]
SEEDS = [21, 3, 7]


def load_frame(base, rung, seed, dim):
    p = os.path.join(base, "cones", rung, f"seed_{seed}", f"dim_{dim}.pt")
    if not os.path.exists(p):
        return None
    v = torch.load(p, map_location="cpu")["vectors"]
    return (v if torch.is_tensor(v) else torch.stack(list(v))).double()


def dims_of(base, rung, seed):
    p = os.path.join(base, "cones", rung, f"seed_{seed}")
    if not os.path.isdir(p):
        return set()
    return {int(f[4:-3]) for f in os.listdir(p) if f.startswith("dim_") and f.endswith(".pt")}


def stats(g, B, n_null, gen):
    """R_span, R_cone, Delta, nu, cos_max, argmax-ось + null по вращениям ортанта."""
    c = B @ g                                   # [d]
    r_span = (c ** 2).sum().item()
    r_cone = (c.clamp(min=0.0) ** 2).sum().item()
    delta = r_span - r_cone
    nu = delta / r_span if r_span > 0 else float("nan")
    d = B.shape[0]
    nulls = []
    for _ in range(n_null):
        Q, R = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
        cq = (Q * torch.sign(torch.diagonal(R)).unsqueeze(0)).T @ c   # вращение ортанта
        nulls.append(((cq.clamp(max=0.0) ** 2).sum() / r_span).item())
    nulls = torch.tensor(nulls)
    return {
        "R_span": r_span, "R_cone": r_cone, "delta": delta, "nu": nu,
        "cos_max": c.abs().max().item(), "argmax": int(c.abs().argmax()),
        "nu_null": nulls.mean().item(), "nu_pct": (nulls <= nu).double().mean().item(),
    }


def rand_R_span(d, hidden, n, gen):
    """Случайный уровень R_span: единичный g против случайного ортонормированного репера."""
    acc = []
    for _ in range(n):
        Q, _ = torch.linalg.qr(torch.randn(hidden, d, generator=gen, dtype=torch.float64))
        g = torch.randn(hidden, generator=gen, dtype=torch.float64)
        g = g / g.norm()
        acc.append(((Q.T @ g) ** 2).sum().item())
    return st.mean(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/ форка")
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="какие d показывать; по умолчанию все доступные у ступени")
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_rand", type=int, default=100)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--scan_only", action="store_true",
                    help="только скан по (pos, layer): где DIM ближе всего к конусу")
    ap.add_argument("--aggregate", action="store_true",
                    help="только сводка: ОДНО число на ступень (среднее по сидам и концептам)")
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(args.seed)

    # --- DIM во всех вариантах точки ---
    D, MD, SEL = {}, {}, {}
    for r in RUNGS:
        p = os.path.join(args.base, "dim", r, args.model)
        if not os.path.isdir(p):
            continue
        D[r] = torch.load(f"{p}/direction.pt", map_location="cpu").double()
        MD[r] = torch.load(f"{p}/generate_directions/mean_diffs.pt", map_location="cpu").double()
        m = json.load(open(f"{p}/direction_metadata.json"))
        SEL[r] = (m["pos"], m["layer"])
    concepts = [r for r in RUNGS if r in D]
    hidden = D[concepts[0]].shape[0]
    sites = sorted(set(SEL.values()))

    variants = [("native", None)] + [(f"pos{p}_L{l}", (p, l)) for p, l in sites]
    print(f"концепты с DIM: {concepts}")
    print(f"выбранные пайплайном точки: " + ", ".join(f"{r}={SEL[r]}" for r in concepts))
    print(f"варианты точки DIM: {[v[0] for v in variants]}")
    print("ВНИМАНИЕ: сравнивать матрицу МЕЖДУ концептами честно только в общей точке —")
    print("в native у концептов разные (pos, layer), см. часть 5 отчёта.\n")

    rand_cache = {}

    # --- сводка: ОДНО число на ступень (среднее по сидам и по концептам) ---
    # Точность = cos(g, Bw) при w = max(B g, 0). При ортонормированном B это ровно
    # sqrt(R_cone): ||Bw||^2 = R_cone и <g, Bw> = R_cone. Относительная невязка
    # ||Bw - g|| = sqrt(1 - R_cone), т.к. ||g||=1.
    #
    # Две колонки, потому что «точность» зависит от того, ЧТО считать целевым DIM:
    #   native    — сохранённые direction.pt как есть. Но у концептов РАЗНЫЕ (pos, layer),
    #               и конус заякорен на слое своего DIM (add_layer, rdo.py:187-195),
    #               поэтому ступени тут в неравных условиях.
    #   best-site — каждый целевой DIM берётся в точке, максимизирующей R_span для этого
    #               конуса. Снимает фору по слою, сравнение ступеней честное.
    if args.aggregate:
        n_pos, n_layers = MD[concepts[0]].shape[0], MD[concepts[0]].shape[1]
        print("Точность = cos(DIM, лучшая неотрицательная реконструкция из конуса) = sqrt(R_cone).")
        print("Усреднение: по 3 сидам и по всем 4 целевым концептам. Невязка = sqrt(1 - R_cone).\n")
        print("| ступень | d | точность native | невязка | точность best-site | невязка | rand |")
        print("|---|---|---|---|---|---|---|")
        summary, by_dim = {}, {}
        for rung in RUNGS:
            avail = sorted(set.intersection(*(dims_of(args.base, rung, s) for s in SEEDS))
                           ) if all(dims_of(args.base, rung, s) for s in SEEDS) else []
            dims = [d for d in avail if args.dims is None or d in args.dims]
            if not dims:
                continue
            per_d = []
            for d in dims:
                frames = [load_frame(args.base, rung, s, d) for s in SEEDS]
                if (d, hidden) not in rand_cache:
                    rand_cache[(d, hidden)] = rand_R_span(d, hidden, args.n_rand, gen)
                nat, bst = [], []
                for t in concepts:
                    gn = D[t] / D[t].norm()
                    nat += [(( B @ gn).clamp(min=0) ** 2).sum().item() for B in frames]
                    best = -1.0
                    for p in range(-n_pos, 0):
                        for l in range(1, n_layers):
                            g = MD[t][p, l]
                            g = g / g.norm()
                            v = st.mean(((B @ g) ** 2).sum().item() for B in frames)
                            if v > best:
                                best, bs = v, (p, l)
                    gb = MD[t][bs[0], bs[1]]
                    gb = gb / gb.norm()
                    bst += [((B @ gb).clamp(min=0) ** 2).sum().item() for B in frames]
                mn, mb = st.mean(nat), st.mean(bst)
                per_d.append((d, mn, mb))
                by_dim.setdefault(d, []).append((rung, mn, mb))
                print(f"| {rung} | {d} | {mn ** 0.5:.4f} | {(1-mn) ** 0.5:.4f} | "
                      f"{mb ** 0.5:.4f} | {(1-mb) ** 0.5:.4f} | {rand_cache[(d,hidden)] ** 0.5:.4f} |")
            summary[rung] = (dims, st.mean(x[1] for x in per_d), st.mean(x[2] for x in per_d))

        # агрегация ПО РАЗМЕРНОСТИ: концепты схлопнуты, ось — d.
        # Усредняем R_cone (не косинус!) с равным весом на ступень, потом корень.
        print("\n### ПО РАЗМЕРНОСТИ (ступени усреднены)\n")
        print("| d | ступени (n) | точность native | невязка | точность best-site | невязка | rand |")
        print("|---|---|---|---|---|---|---|")
        for d in sorted(by_dim):
            rs = by_dim[d]
            mn = st.mean(x[1] for x in rs)
            mb = st.mean(x[2] for x in rs)
            names = ", ".join(x[0][:12] for x in rs)
            print(f"| {d} | {names} ({len(rs)}) | {mn ** 0.5:.4f} | {(1-mn) ** 0.5:.4f} | "
                  f"{mb ** 0.5:.4f} | {(1-mb) ** 0.5:.4f} | {rand_cache[(d,hidden)] ** 0.5:.4f} |")
        print("\nВНИМАНИЕ: состав ступеней меняется с d (d<=5 — три узкие, d>=9 — только 'all'),")
        print("поэтому тренд вдоль таблицы смешан с изменением состава. Срез постоянного")
        print("состава — d=1..5 (три узкие ступени) и malicious_use отдельно (единственная")
        print("ступень с непрерывным d=1..8).")

        print("\n### ОДНО ЧИСЛО НА СТУПЕНЬ (ещё и среднее по всем d ступени)\n")
        print("| ступень | d | точность native | невязка | точность best-site | невязка |")
        print("|---|---|---|---|---|---|")
        for r, (dims, mn, mb) in summary.items():
            print(f"| {r} | {min(dims)}–{max(dims)} | {mn ** 0.5:.4f} | {(1-mn) ** 0.5:.4f} | "
                  f"{mb ** 0.5:.4f} | {(1-mb) ** 0.5:.4f} |")
        print("\nВНИМАНИЕ: диапазоны d у ступеней РАЗНЫЕ (all начинается с 7), поэтому")
        print("последняя таблица усредняет по разным наборам d — сравнивать ступени между")
        print("собой по ней нельзя, см. таблицу по d выше.")
        return

    # --- скан по всем (pos, layer): в какой точке DIM ближе всего к конусу? ---
    # Низкий cos_max в таблицах ниже можно было бы списать на то, что DIM привязан к
    # ОДНОМУ слою, а конус аблируется во всех. Скан это проверяет: если и в лучшей точке
    # R_span остаётся низким, дело не в выборе слоя.
    n_pos, n_layers = MD[concepts[0]].shape[0], MD[concepts[0]].shape[1]
    print("=" * 78)
    print(f"### Скан по всем {n_pos}x{n_layers - 1} точкам (слой 0 вырожден, пропущен):")
    print("### в какой точке DIM лучше всего ложится в span конуса")
    for rung in RUNGS:
        avail = sorted(set.intersection(*(dims_of(args.base, rung, s) for s in SEEDS))
                       ) if all(dims_of(args.base, rung, s) for s in SEEDS) else []
        dims = [d for d in avail if args.dims is None or d in args.dims]
        if not dims:
            continue
        print(f"\n конус ступени: {rung}")
        print("  | d | целевой DIM | лучшая точка | R_span@лучшей | R_span@выбранной | rand |")
        print("  |---|---|---|---|---|---|")
        for d in dims:
            frames = [load_frame(args.base, rung, s, d) for s in SEEDS]
            if (d, hidden) not in rand_cache:
                rand_cache[(d, hidden)] = rand_R_span(d, hidden, args.n_rand, gen)
            for t in concepts:
                best, best_site = -1.0, None
                for p in range(-n_pos, 0):
                    for l in range(1, n_layers):
                        g = MD[t][p, l]
                        g = g / g.norm()
                        v = st.mean(((B @ g) ** 2).sum().item() for B in frames)
                        if v > best:
                            best, best_site = v, (p, l)
                sp, sl = SEL[t]
                gs = MD[t][sp, sl]
                gs = gs / gs.norm()
                at_sel = st.mean(((B @ gs) ** 2).sum().item() for B in frames)
                print(f"  | {d} | {t[:18]} | pos{best_site[0]} L{best_site[1]} | {best:.4f} | "
                      f"{at_sel:.4f} (pos{sp} L{sl}) | {rand_cache[(d,hidden)]:.4f} |")
    if args.scan_only:
        return

    for vname, site in variants:  # noqa: одна переменная кэша на скан и на таблицы
        print("=" * 78)
        print(f"### Вариант DIM: {vname}")
        for rung in RUNGS:
            avail = sorted(set.intersection(*(dims_of(args.base, rung, s) for s in SEEDS))
                           ) if all(dims_of(args.base, rung, s) for s in SEEDS) else []
            dims = [d for d in avail if args.dims is None or d in args.dims]
            if not dims:
                continue
            print(f"\n конус ступени: {rung}")
            print("  | d | целевой DIM | R_span | R_cone | Δ | nu | nu_null | pct | cos_max | ось | rand R_span |")
            print("  |---|---|---|---|---|---|---|---|---|---|---|")
            for d in dims:
                if (d, hidden) not in rand_cache:
                    rand_cache[(d, hidden)] = rand_R_span(d, hidden, args.n_rand, gen)
                for t in concepts:
                    g = D[t] if site is None else MD[t][site[0], site[1]]
                    g = g / g.norm()
                    rows = [stats(g, load_frame(args.base, rung, s, d), args.n_null, gen)
                            for s in SEEDS]
                    m = {k: st.mean(r[k] for r in rows) for k in
                         ("R_span", "R_cone", "delta", "nu", "cos_max", "nu_null", "nu_pct")}
                    sd = st.stdev([r["R_span"] for r in rows]) if len(rows) > 1 else 0.0
                    ax = "/".join(str(r["argmax"]) for r in rows)
                    mark = " <-- свой" if t == rung else ""
                    print(f"  | {d} | {t[:18]} | {m['R_span']:.4f}±{sd:.4f} | {m['R_cone']:.4f} | "
                          f"{m['delta']:.4f} | {m['nu']:.4f} | {m['nu_null']:.4f} | {m['nu_pct']:.3f} | "
                          f"{m['cos_max']:.4f} | {ax} | {rand_cache[(d,hidden)]:.4f} |{mark}")


if __name__ == "__main__":
    main()
