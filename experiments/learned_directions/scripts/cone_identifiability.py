"""Идентифицируемость ориентации конуса + предпросмотр Δ (конус vs span). CPU, без GPU.

Зачем. В RDO (rdo.py) базис конуса ортонормирован ВСЕГДА: orthogonalize() применяется
in-place после каждого optimizer.step(), градиент проецируется на касательную к сфере.
Поэтому в чекпоинте лежит ортонормированный репер — но это не выход QR над каким-то
набором генераторов, а сам обученный параметр: лосс видит только λ>=0 (sample_*.abs()),
т.е. столбцы репера И ЕСТЬ рёбра симплициального конуса C = {Σ λ_i b_i : λ_i >= 0}.

Открытый вопрос не «стёрлась ли структура», а «идентифицирована ли ОРИЕНТАЦИЯ репера
внутри span-а». Если лосс почти плоский по вращениям span-а, ортант — шум инициализации,
и любая конусная величина (в т.ч. Δ) меряет шум, а не отношение концептов.

Статистика. Для генератора g (||g||=1) и репера B (столбцы — рёбра), c = B^T g:
    nu(g,B) = Σ_i min(c_i,0)^2 / Σ_i c_i^2
Доля in-span энергии, лежащей в ОТРИЦАТЕЛЬНЫХ координатах. nu=0 — g внутри конуса B;
nu~0.5 — ориентация случайна. Знаменатель Σ c_i^2 = ||P_B g||^2 отделяет ориентацию
от совпадения span-ов (его отдельно репортим как in_span).

Связь с NNLS. min_{w>=0} ||Bw - g||^2 при ортонормированном B распадается покоординатно:
w_i = max(c_i,0), residual^2 = ||g||^2 - Σ max(c_i,0)^2. Отсюда
    Δ = ||r_cone||^2 - ||r_span||^2 = Σ_i min(c_i,0)^2 = nu * in_span.
Солвер не нужен, Δ — замкнутая форма.

Null. B -> B Q, Q ~ Haar(d_b): span и проектор B B^T инвариантны, меняется только
ориентация ортанта. В координатах это C -> C Q^T, поэтому null считается мгновенно
и изолирует ровно ориентацию (in_span под Q не меняется).

Тесты:
  self    — A=B, sanity: nu ровно 0, in_span ровно 1.
  nest    — A=dim_{d-1}, B=dim_d того же rung/сида. Осмысленно ТОЛЬКО для warm-start
            прогонов (multiseed.sh, resume): там nu должен быть низким. Но in_span<1
            там ожидаем — resume лишь ИНИЦИАЛИЗИРУЕТ старые оси, не фиксирует их
            (fixed_basis_vectors — отдельный механизм), поэтому они дрейфуют.
            Для ступени 'all' секция бессмысленна: её d обучались с FROM_SCRATCH=1
            (from_scratch_dims.sh), вложенности между d там нет по построению.
  seed    — ГЛАВНЫЙ: A и B — одна ступень и одна d, разные сиды. Если nu_obs не отличается
            от null, ориентация НЕ идентифицирована => Δ не спасти ничем.
  ladder  — предпросмотр Δ: A — генераторы узкой ступени, B — репер широкой (тот же сид).
            Осмысленно ТОЛЬКО если seed-тест прошёл.

Запуск:
  python cone_identifiability.py --base <fork>/results/cones
"""
import argparse
import json
import os

import torch

# ступени по вложенности: лист -> задача -> домен -> всё
LADDER = ["theft", "illegal_activities", "malicious_use", "all"]
SEEDS = [21, 3, 7]


def load_frame(base, rung, seed, dim):
    """[d, hidden] репер конуса, строки — рёбра (единичные, взаимно ортогональные)."""
    path = os.path.join(base, rung, f"seed_{seed}", f"dim_{dim}.pt")
    if not os.path.exists(path):
        return None
    V = torch.load(path, map_location="cpu")["vectors"]
    V = V if torch.is_tensor(V) else torch.stack(list(V))
    return V.double()


def available_dims(base, rung, seed):
    d = os.path.join(base, rung, f"seed_{seed}")
    if not os.path.isdir(d):
        return []
    dims = [int(f[4:-3]) for f in os.listdir(d) if f.startswith("dim_") and f.endswith(".pt")]
    return sorted(dims)


def haar(d, g):
    """Haar-равномерная ортогональная матрица d x d (QR от гауссовой + фикс знаков R)."""
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def stat(C):
    """(nu_mean, in_span_mean) по строкам C = A @ B.T ([d_a, d_b])."""
    in_span = (C ** 2).sum(dim=1)
    neg = (C.clamp(max=0.0) ** 2).sum(dim=1)
    return (neg / in_span.clamp(min=1e-30)).mean().item(), in_span.mean().item()


def compare(A, B, n_null, g):
    """nu/in_span + null-распределение по случайным вращениям репера B внутри его span-а."""
    C = A @ B.T
    nu_obs, in_span = stat(C)
    nulls = torch.tensor([stat(C @ haar(B.shape[0], g).T)[0] for _ in range(n_null)])
    mu, sd = nulls.mean().item(), nulls.std().item()
    return {
        "d_src": A.shape[0], "d_tgt": B.shape[0],
        "in_span": in_span,                       # = rho_span на генератор, доля [0,1]
        "nu": nu_obs,                             # доля in-span энергии в минус-ортанте
        "delta": nu_obs * in_span,                # Δ = ||r_cone||^2 - ||r_span||^2
        "null_mean": mu, "null_sd": sd,
        "z": (nu_obs - mu) / sd if sd > 0 else float("nan"),
        "pct_le": (nulls <= nu_obs).double().mean().item(),  # доля null не выше наблюдения
    }


def row(tag, r):
    return (f"  {tag:<34} nu={r['nu']:.4f}  null={r['null_mean']:.4f}±{r['null_sd']:.4f}  "
            f"z={r['z']:+7.2f}  pct={r['pct_le']:.4f}  in_span={r['in_span']:.4f}  "
            f"Δ={r['delta']:.4f}  ({r['d_src']}->{r['d_tgt']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/cones форка")
    ap.add_argument("--n_null", type=int, default=2000, help="вращений в null-распределении")
    ap.add_argument("--seed", type=int, default=21, help="сид ГСЧ для null")
    ap.add_argument("--out", default=None, help="куда писать JSON (по умолчанию <base>/identifiability.json)")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    out = {"n_null": args.n_null, "rng_seed": args.seed, "self": [], "nest": [], "seed": [], "ladder": []}

    inv = {r: {s: available_dims(args.base, r, s) for s in SEEDS} for r in LADDER}
    print("=== артефакты (rung: сид -> dims) ===")
    for r in LADDER:
        for s in SEEDS:
            if inv[r][s]:
                print(f"  {r:<20} seed_{s:<3} dims={inv[r][s]}")

    # --- sanity 1: репер против самого себя. nu обязан быть ровно 0, in_span ровно 1 ---
    print("\n=== self (sanity: ожидаем nu=0, in_span=1) ===")
    for r in LADDER:
        dims = inv[r][SEEDS[0]]
        if not dims:
            continue
        A = load_frame(args.base, r, SEEDS[0], dims[-1])
        res = compare(A, A, args.n_null, g)
        res.update(rung=r, seed=SEEDS[0], dim=dims[-1])
        out["self"].append(res)
        print(row(f"{r} s{SEEDS[0]} d{dims[-1]}", res))

    # --- sanity 2: вложенность warm-start. dim_{d-1} внутри dim_d => nu~0, in_span~1 ---
    print("\n=== nest (sanity warm-start: dim_{d-1} -> dim_d, ожидаем nu~0) ===")
    for r in LADDER:
        for s in SEEDS:
            dims = inv[r][s]
            for d in dims:
                if d - 1 not in dims:
                    continue
                A, B = load_frame(args.base, r, s, d - 1), load_frame(args.base, r, s, d)
                res = compare(A, B, args.n_null, g)
                res.update(rung=r, seed=s, dim_src=d - 1, dim_tgt=d)
                out["nest"].append(res)
            if out["nest"] and out["nest"][-1].get("rung") == r and out["nest"][-1].get("seed") == s:
                print(row(f"{r} s{s} d{out['nest'][-1]['dim_src']}->d{out['nest'][-1]['dim_tgt']}", out["nest"][-1]))

    # --- ГЛАВНЫЙ ТЕСТ: та же ступень и та же d, разные сиды ---
    print("\n=== seed (идентифицируемость ориентации; nu<<null => ортант воспроизводим) ===")
    for r in LADDER:
        for i, sa in enumerate(SEEDS):
            for sb in SEEDS[i + 1:]:
                common = sorted(set(inv[r][sa]) & set(inv[r][sb]))
                for d in common:
                    A, B = load_frame(args.base, r, sa, d), load_frame(args.base, r, sb, d)
                    res = compare(A, B, args.n_null, g)
                    res.update(rung=r, seed_src=sa, seed_tgt=sb, dim=d)
                    out["seed"].append(res)
                    print(row(f"{r} d{d} s{sa}->s{sb}", res))

    # --- предпросмотр Δ: узкая ступень -> широкая, тот же сид (валидно только если seed прошёл) ---
    print("\n=== ladder (Δ: генераторы узкой ступени против конуса широкой, тот же сид) ===")
    for i, narrow in enumerate(LADDER):
        for wide in LADDER[i + 1:]:
            for s in SEEDS:
                dn, dw = inv[narrow][s], inv[wide][s]
                if not dn or not dw:
                    continue
                A, B = load_frame(args.base, narrow, s, dn[-1]), load_frame(args.base, wide, s, dw[-1])
                res = compare(A, B, args.n_null, g)
                res.update(narrow=narrow, wide=wide, seed=s, dim_narrow=dn[-1], dim_wide=dw[-1])
                out["ladder"].append(res)
                print(row(f"{narrow} d{dn[-1]} -> {wide} d{dw[-1]} s{s}", res))

    path = args.out or os.path.join(args.base, "identifiability.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[JSON -> {path}]")

    # --- вердикт по главному тесту ---
    if out["seed"]:
        rows = out["seed"]
        nu = sum(r["nu"] for r in rows) / len(rows)
        nl = sum(r["null_mean"] for r in rows) / len(rows)
        hi = [r for r in rows if r["dim"] >= 4]     # d<=3 вырождены: при d=1 nu ∈ {0,1}
        ok = sum(r["pct_le"] <= 0.05 for r in rows)
        ok_hi = sum(r["pct_le"] <= 0.05 for r in hi)
        floor = min((r["z"] for r in out["self"]), default=float("nan"))
        print(f"\nВЕРДИКТ seed-теста ({len(rows)} пар):")
        print(f"  nu={nu:.4f} против null={nl:.4f}")
        if hi:
            nu_hi = sum(r['nu'] for r in hi) / len(hi)
            print(f"  на d>=4 ({len(hi)} пар): nu={nu_hi:.4f}, "
                  f"путь от случайного к идеальному = {(nl-nu_hi)/nl:.0%}")
        print(f"  строк с percentile<=0.05: {ok}/{len(rows)} (на d>=4: {ok_hi}/{len(hi)})")
        print(f"  ПОЛ шкалы z (self, идеальное совпадение) = {floor:+.2f} — z НЕ гауссов "
              f"z-score, читать percentile.")
        print("  Трактовка: percentile<=0.05 на большинстве строк d>=4 => ориентация")
        print("  идентифицирована. Иначе ортант неотличим от случайного вращения span-а.")
        print("  ВАЖНО: nu условна — нормирована на in_span. Проверь in_span: если он мал,")
        print("  ориентация согласована лишь внутри малой общей доли, и Δ всё равно шумна.")


if __name__ == "__main__":
    main()
