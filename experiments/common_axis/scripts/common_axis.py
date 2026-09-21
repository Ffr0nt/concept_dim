"""Общая ось конусов отказа: спектр среднего проектора по сидам. CPU, GPU не нужен.

Вопрос. Конусы, обученные на одной ступени с разными сидами, различаются (часть 1
learned_directions: cos_match 0.25-0.39). Есть ли внутри них ОДНО общее направление,
а различия — случайный остаток?

Средний проектор по N реперам одной ступени и одной размерности:
    P = (1/N) sum_j B_j^T B_j,     B_j: [k, hidden], строки ортонормированы
P зависит только от span-а каждого конуса, не от выбора базиса внутри него.
w_1 (верхний собственный вектор) — кандидат в общую ось; lam_1 = mean_j ||B_j w_1||^2.

ПОЛ СПЕКТРА, без которого числа не читаются. Если span-ы ВЗАИМНО ОРТОГОНАЛЬНЫ, у P ровно
N*k ненулевых с.з., каждое = 1/N. То есть lam_1 >= 1/N ВСЕГДА (при N=3 это 0.333), а
lam_1 = 1 означает ось, целиком лежащую во всех конусах. Величина k/hidden = 0.0024 —
это СРЕДНЕЕ с.з. по всему пространству, а не уровень lam_1; сравнивать lam_1 с ней нельзя.
Поэтому нуль считается эмпирически на случайных ортонормированных реперах той же формы.

Вычисление через малую матрицу. Ненулевой спектр P = (1/N) M^T M совпадает со спектром
(1/N) M M^T, где M = [B_1; ...; B_N] — [N*k, hidden]. Это матрица N*k x N*k (15x15 при
N=3, k=5) вместо 2048x2048, поэтому и наблюдение, и 500+ повторов нуля считаются мгновенно.
w_k = M^T u_k / ||M^T u_k||, где u_k — собственный вектор малой матрицы.

Что считается на каждую (ступень, k):
  lam_1..lam_4 + нулевая полоса (среднее и p95 по n_null случайным наборам);
  alpha_j = ||B_j w_1|| — сколько общей оси лежит в конусе каждого сида (lam_1 = mean alpha_j^2);
  nu(+w_1), nu(-w_1) — тест вхождения оси В КОНУС, а не только в span. Для ортонормированного
    B задача min_{w>=0} ||B^T w - g|| распадается покоординатно, солвер не нужен:
    c = B w_1, nu = sum min(c,0)^2 / sum c^2. Знак важен: span не видит его, конус видит.
    Тождество nu(-w) = 1 - nu(w) точное, поэтому репортим оба и берём ближайший ортант;
    нулевой уровень nu ~ 0.5 (вращение ортанта Q ~ Haar).
  cos(w_1, DIM) — native-точка каждой ступени, каждая из ОБЩИХ точек (pos, layer) и лучшая
    точка по скану mean_diffs. Сравнивать между ступенями честно только в общей точке:
    select_direction выбирает (pos, layer) независимо, и смена точки двигает DIM сильнее
    смены концепта (часть 5 learned_directions).
  cos(w_1, u_1^DIM) — u_1^DIM: первый сингулярный вектор стека DIM всех ступеней.
  power-iteration из DIM: v <- P v со старта v_0 = DIM. Остаётся (cos(v_inf, v_0) ~ 1)
    или уходит к w_1 — read-инвариант против write-инварианта.

АРТЕФАКТЫ. Сами оси w_k сохраняются в <base>/common_axis/<rung>/dim_<k>.pt (ключ "W",
[m, hidden] float32, m = min(N*k, 16), строки — оси по убыванию lam) вместе со спектром и
per-seed величинами. Без этого §6 нечего аблировать: отчёт содержит только числа. Каталог
results/ форка в git не хранится (.gitignore:254), как cones/ и dim/.

Конфаундер общей инициализации СЮДА НЕ ВХОДИТ по построению: сид старта в rdo.py:1126
не зависит от ступени, поэтому общий старт есть у разных СТУПЕНЕЙ при одном сиде, а здесь
усреднение идёт ровно по сидам. Для P между ступенями (§4 заметки) нужен отдельный нуль
на стартовых реперах.

ПОТОКИ. Матрицы тут маленькие (QR 2048xk, eigh N*k x N*k), а машина общая и 224-ядерная:
по умолчанию torch поднимает ~175 потоков и жжёт ~25 ядер на арифметику, которой хватило бы
одного. Поэтому число потоков ограничено явно (--threads, по умолчанию 8); раннер дополнительно
выставляет OMP_NUM_THREADS до импорта torch.

Запуск:
  bash experiments/common_axis/run/1_common_axis.sh
  python common_axis.py --base <fork>/results --out reports/common-axis.md
"""
import argparse
import json
import os

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


def spectrum(frames):
    """Спектр P = (1/N) sum B_j^T B_j через малую матрицу M M^T. Возвращает (lam, W, M)."""
    M = torch.cat(frames, dim=0)                     # [N*k, hidden]
    N = len(frames)
    G = (M @ M.T) / N                                # [N*k, N*k]
    lam, U = torch.linalg.eigh(G)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order], U[:, order]
    W = M.T @ U                                      # [hidden, N*k]
    W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)
    return lam, W, M


def rand_frames(shapes, hidden, gen):
    out = []
    for k in shapes:
        Q, _ = torch.linalg.qr(torch.randn(hidden, k, generator=gen, dtype=torch.float64))
        out.append(Q.T)
    return out


def null_band(shapes, hidden, n_null, gen, n_lam=4):
    """Эмпирический нуль: независимые случайные ортонормированные реперы той же формы."""
    acc = []
    for _ in range(n_null):
        lam, _, _ = spectrum(rand_frames(shapes, hidden, gen))
        acc.append(lam[:n_lam])
    A = torch.stack(acc)                             # [n_null, n_lam]
    return A.mean(0), A.quantile(0.95, dim=0)


def nu_of(B, w):
    """Доля in-span энергии w в ОТРИЦАТЕЛЬНЫХ координатах конуса B (замкнутый NNLS)."""
    c = B @ w
    s = (c ** 2).sum()
    if s <= 0:
        return float("nan"), 0.0
    return ((c.clamp(max=0.0) ** 2).sum() / s).item(), s.sqrt().item()


def power_from(M, v0, iters=200):
    """v <- P v, P = (1/N) M^T M. Старт v0 (единичный)."""
    N_ = M.shape[0]
    v = v0 / v0.norm()
    for _ in range(iters):
        v = M.T @ (M @ v)
        n = v.norm()
        if n <= 0:
            return v0, 0.0
        v = v / n
    return v, (v @ v0 / v0.norm()).abs().item()


def load_dim(base, model):
    """DIM всех ступеней: native-вектор, mean_diffs (все точки), выбранная точка."""
    D, MD, SEL = {}, {}, {}
    for r in RUNGS:
        p = os.path.join(base, "dim", r, model)
        if not os.path.isdir(p):
            continue
        D[r] = torch.load(f"{p}/direction.pt", map_location="cpu").double()
        MD[r] = torch.load(f"{p}/generate_directions/mean_diffs.pt", map_location="cpu").double()
        m = json.load(open(f"{p}/direction_metadata.json"))
        SEL[r] = (int(m["pos"]), int(m["layer"]))
    return D, MD, SEL


def unit(v):
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/ форка")
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--dims", type=int, nargs="+", default=None, help="какие k считать")
    ap.add_argument("--n_null", type=int, default=500)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_dir", default=None,
                    help="куда класть оси w_k (по умолчанию <base>/common_axis); --save_dir '' отключает")
    ap.add_argument("--threads", type=int, default=int(os.getenv("THREADS", "8")),
                    help="потоков torch; машина общая, матрицы маленькие — см. докстринг")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    save_dir = os.path.join(args.base, "common_axis") if args.save_dir is None else args.save_dir

    gen = torch.Generator().manual_seed(args.seed)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    D, MD, SEL = load_dim(args.base, args.model)
    concepts = [r for r in RUNGS if r in D]
    sites = sorted(set(SEL[r] for r in concepts))
    N = len(SEEDS)

    emit("# Общая ось конусов: средний проектор по сидам")
    emit()
    emit(f"P = (1/N) Σ_j B_jᵀB_j по N = {N} сидам {SEEDS} внутри ступени, отдельно на каждую k.")
    emit(f"Нуль — {args.n_null} наборов независимых случайных ортонормированных реперов той же формы.")
    emit()
    emit("**Как читать lam_1.** Пол = 1/N = "
         f"{1.0/N:.3f} (взаимно ортогональные span-ы), потолок = 1.0 (ось целиком во всех конусах).")
    emit("Величина k/hidden ≈ 0.002 — среднее с.з. по всему пространству, к lam_1 отношения не имеет.")
    emit()
    if concepts:
        emit(f"DIM найден для: {concepts}; выбранные пайплайном точки: "
             + ", ".join(f"{r}={SEL[r]}" for r in concepts))
    else:
        emit("DIM не найден — часть про сравнение с DIM пропущена.")
    emit()

    # стек DIM в каждой общей точке -> u1^DIM
    u1_dim = {}
    for site in sites:
        S = torch.stack([unit(MD[r][site[0], site[1]]) for r in concepts])
        u1_dim[site] = torch.linalg.svd(S, full_matrices=False)[2][0]

    for rung in RUNGS:
        avail = [dims_of(args.base, rung, s) for s in SEEDS]
        if not all(avail):
            continue
        ks = sorted(set.intersection(*avail))
        if args.dims is not None:
            ks = [k for k in ks if k in args.dims]
        if not ks:
            continue

        emit(f"## {rung}")
        emit()
        emit("| k | lam_1 | нуль (среднее / p95) | lam_2 | lam_3 | "
             + " | ".join(f"alpha_{s}" for s in SEEDS) + " | nu(+w) | nu(−w) |")
        emit("|" + "---|" * (7 + len(SEEDS)))

        keep = {}
        for k in ks:
            frames = [load_frame(args.base, rung, s, k) for s in SEEDS]
            if any(f is None for f in frames):
                continue
            hidden = frames[0].shape[1]
            lam, W, M = spectrum(frames)
            w1 = W[:, 0]
            # знак общей оси: span его не видит, конус видит — берём ортант с большей массой
            if sum(float((B @ w1).sum()) for B in frames) < 0:
                w1 = -w1
            alphas = [(B @ w1).norm().item() for B in frames]
            nus = [nu_of(B, w1)[0] for B in frames]
            nus_m = [nu_of(B, -w1)[0] for B in frames]
            nm, np95 = null_band([f.shape[0] for f in frames], hidden, args.n_null, gen)
            emit(f"| {k} | **{lam[0]:.3f}** | {nm[0]:.3f} / {np95[0]:.3f} | {lam[1]:.3f} | "
                 f"{lam[2]:.3f} | " + " | ".join(f"{a:.3f}" for a in alphas)
                 + f" | {sum(nus)/len(nus):.3f} | {sum(nus_m)/len(nus_m):.3f} |")
            keep[k] = (lam, w1, M, frames)

            if save_dir:
                m = min(W.shape[1], 16)
                Wm = W[:, :m].T.clone()               # [m, hidden], строки — оси
                Wm[0] = w1                            # знак w_1 уже зафиксирован по ортанту
                d = os.path.join(save_dir, rung)
                os.makedirs(d, exist_ok=True)
                torch.save({
                    "rung": rung, "k": k, "seeds": list(SEEDS), "hidden": hidden,
                    "W": Wm.float(), "lam": lam[:m].float(),
                    "lam_null_mean": nm.float(), "lam_null_p95": np95.float(),
                    "n_null": args.n_null, "floor": 1.0 / len(frames),
                    "alphas": torch.tensor(alphas), "nu_plus": torch.tensor(nus),
                    "cone_files": [os.path.join("cones", rung, f"seed_{s}", f"dim_{k}.pt")
                                   for s in SEEDS],
                }, os.path.join(d, f"dim_{k}.pt"))

        emit()
        emit(f"alpha_j = ||B_j w_1||, lam_1 = mean_j alpha_j². nu — средняя по сидам доля "
             f"in-span энергии w_1 в минус-ортанте (0 = ось внутри конуса, ~0.5 = ортант случаен; "
             f"nu(−w) = 1 − nu(w) тождественно).")
        emit()

        if concepts and keep:
            emit(f"### {rung}: общая ось против DIM")
            emit()
            emit("| k | native own | " + " | ".join(f"pos{p}_L{l}" for p, l in sites)
                 + " | лучшая точка | cos(w_1, u_1^DIM) | power из DIM: cos(v_∞,v_0) | cos(v_∞,w_1) |")
            emit("|" + "---|" * (6 + len(sites)))
            n_pos, n_layers = MD[concepts[0]].shape[0], MD[concepts[0]].shape[1]
            for k, (lam, w1, M, frames) in keep.items():
                cells = []
                if rung in D:
                    cells.append(f"{abs(float(unit(D[rung]) @ w1)):.3f}")
                else:
                    cells.append("—")
                for p, l in sites:
                    g = unit(MD[rung][p, l]) if rung in MD else None
                    cells.append(f"{abs(float(g @ w1)):.3f}" if g is not None else "—")
                best, bsite = -1.0, None
                if rung in MD:
                    for p in range(-n_pos, 0):
                        for l in range(1, n_layers):
                            c = abs(float(unit(MD[rung][p, l]) @ w1))
                            if c > best:
                                best, bsite = c, (p, l)
                cells.append(f"{best:.3f} @ {bsite}" if bsite else "—")
                site0 = SEL.get(rung, sites[0])
                cells.append(f"{abs(float(u1_dim[site0] @ w1)):.3f}")
                g0 = unit(MD[rung][site0[0], site0[1]]) if rung in MD else w1
                v_inf, stay = power_from(M, g0)
                cells.append(f"{stay:.3f}")
                cells.append(f"{abs(float(v_inf @ w1)):.3f}")
                emit(f"| {k} | " + " | ".join(cells) + " |")
            emit()
            emit(f"Точка сравнения с u_1^DIM и старт power-iteration — {SEL.get(rung)} "
                 f"(native-точка ступени). Между ступенями сравнимы только колонки общих точек.")
            emit()

    if save_dir:
        emit(f"Оси сохранены: `{save_dir}/<ступень>/dim_<k>.pt` "
             f"(ключ `W`, [m, hidden], строки — оси по убыванию lam; m = min(N·k, 16)).")
        emit()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[записано: {args.out}]")


if __name__ == "__main__":
    main()
