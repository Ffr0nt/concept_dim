"""§9 — геометрия общей оси относительно DIM: масштаб сигнала, дисперсия, что стирается. CPU.

Шаги 4-5 показали ФУНКЦИОНАЛЬНУЮ асимметрию (ось снимает отказ дешевле DIM, остаток DIM
инертен). Здесь — геометрическая причина: чем ось отличается от DIM как вектор в residual
stream. Нормы самих направлений сравнивать нечего: w_1 выходит из eigh единичным, DIM
нормируется перед аблацией. Информативны три другие величины.

1. МАСШТАБ СИГНАЛА. ||r|| = ||h_harmful - h_harmless|| до нормировки — размер эффекта в
   единицах residual stream (43.2 у theft/illegal в точке (-1,27), 21.7-23.2 у malicious/all
   в (-4,22): точка извлечения меняет его вдвое). У оси такого аналога нет — она не разность
   средних, — но его можно сконструировать: <r, w_1> — сколько mean-diff приходится на ось.
   Считается по всем 5x36 точкам mean_diffs, а не только в native: где ось собирает
   максимум сырого сигнала, и совпадает ли это с точкой, выбранной пайплайном.

2. КВАДРАТИЧНАЯ ФОРМА КОНУСОВ НА DIM. q(v) = v^T P v = (1/N) sum_j ||B_j v||^2 — ровно тот
   функционал, максимум которого и есть ось (q(w_1) = lam_1 = 0.628). Прогнать через него
   DIM — значит измерить DIM критерием оси, а не косинусом: q(r) говорит, насколько DIM
   похож на общую компоненту конусов «в среднем по всем 12 прогонам». Рядом:
   ||P r|| (насколько проектор сжимает DIM против lam_1 для оси), alpha_j = ||B_j r|| по
   каждому из 12 конусов (сравнимо с коридором 0.708-0.855 у оси), энергия DIM в span-е
   всех 12 конусов (нуль = rank/hidden = 66/2048 = 0.032) и накопленная энергия DIM в
   топ-m собственном подпространстве P — та самая «содержательная замена» пустой колонке
   power-итерации из шага 1.

3. ПРОЕКЦИИ АКТИВАЦИЙ (--stage acts, нужен кэш от cache_projections.py, GPU).
   p_v(x) = <h(x), v> по промптам:
   - d' = (mu_harmful - mu_harmless)/sigma — разделяющая способность (read-сторона);
   - E|p| и E|p|/E||h|| — абсолютный уровень проекции: сколько вектора реально удаляет
     аблация h <- h - a (v^T h) v;
   - Var(p)/(tr Sigma / hidden) — доля дисперсии на направлении против изотропного нуля:
     сидит ли DIM на высокодисперсной оси (ковариационный тест). Считается ПО ПРОМПТАМ на
     последней позиции: по всем токенам и дисперсию, и удаляемую энергию съедает sink-токен
     (у Qwen E||h||^2 на нём на два порядка больше, чем в точке чтения), поэтому доля нормы
     усредняется по токенам как отношение, а не как отношение сумм.
   Read-сторона пересекается с §8 (read_probe.py, AUROC/точность детектора): там она
   мерится как качество классификатора, здесь — как d' и уровень проекции, потому что
   нужна та же шкала, в которой считается удаляемая энергия.

   Сумма E[p^2] по ВСЕМ слоям и всем трём точкам вмешательства (вход блока, выход attn,
   выход mlp) — прямая мера «сколько сигнала стирается», и её сверяем с измеренным KL_ret
   из ablation-<rung>.json: превращает «ось дешевле по KL» из наблюдения в механизм.

Что здесь НЕ так, как в исходном плане: пер-промптных активаций в форке нет, сохраняются
только mean_diffs (усреднённые по промптам) — поэтому п. 3 требует одного форвард-прохода
(GPU, без генерации), а не «считается на уже сохранённых активациях».

Запуск:
  bash experiments/common_axis/run/8_axis_geometry.sh              # static, CPU
  STAGE=all GPU=0 bash .../8_axis_geometry.sh                     # + кэш и acts
"""
import argparse
import json
import os

import torch

from common_axis import RUNGS, SEEDS, load_dim, load_frame, unit

SITES = ("resid", "attn", "mlp")


# ---------------------------------------------------------------- популяция конусов

def load_population(base, pooled):
    """Те же 12 конусов, из которых собрана pooled-ось шага 2 (список в самом файле)."""
    kmap = pooled["k_map"]
    frames, names = [], []
    for rung, seed in pooled["members"]:
        B = load_frame(base, rung, int(seed), kmap[rung])
        assert B is not None, f"нет конуса {rung}/seed_{seed}/dim_{kmap[rung]}"
        frames.append(B)
        names.append(f"{rung}/{seed}")
    return frames, names


def q_form(frames, v):
    """q(v) = v^T P v = (1/N) sum_j ||B_j v||^2. Максимум q и есть общая ось; q(w_1) = lam_1."""
    return float(sum((B @ v).pow(2).sum() for B in frames) / len(frames))


def p_apply(frames, v):
    """P v без построения 2048x2048."""
    acc = torch.zeros_like(v)
    for B in frames:
        acc += B.T @ (B @ v)
    return acc / len(frames)


def alphas_of(frames, v):
    return [float((B @ v).norm()) for B in frames]


def q_null(frames, hidden, n, gen):
    vals = []
    for _ in range(n):
        g = unit(torch.randn(hidden, generator=gen, dtype=torch.float64))
        vals.append(q_form(frames, g))
    t = torch.tensor(vals)
    return float(t.mean()), float(t.quantile(0.95))


# ---------------------------------------------------------------- stage: static

def stage_static(args, emit):
    base = args.base
    pooled = torch.load(os.path.join(base, "common_axis", "global", "pooled.pt"),
                        map_location="cpu")
    W_cone = pooled["W"].double()
    lam = pooled["lam"].double()
    mp = os.path.join(base, "common_axis", "global", "mixed_native_balanced.pt")
    W_mixed = torch.load(mp, map_location="cpu")["W"].double() if os.path.exists(mp) else None
    frames, fnames = load_population(base, pooled)
    hidden = frames[0].shape[1]
    N = len(frames)
    D, MD, SEL = load_dim(base, args.model)
    concepts = [r for r in RUNGS if r in D]
    sites = sorted(set(SEL[r] for r in concepts))
    n_pos = MD[concepts[0]].shape[0]
    gen = torch.Generator().manual_seed(args.seed)

    w1 = W_cone[0]
    wm = W_mixed[0] if W_mixed is not None else None

    emit("# §9. Геометрия общей оси относительно DIM (статическая часть)")
    emit()
    emit(f"Популяция — те же {N} конусов, что у pooled-оси шага 2 "
         f"(k_map {dict(pooled['k_map'])}, сиды {SEEDS}); hidden = {hidden}, "
         f"lam_1 = {float(lam[0]):.3f}.")
    emit("Направления единичны по построению, поэтому сравниваются не их нормы, а масштаб "
         "сигнала, квадратичная форма конусов и (в части `acts`) проекции активаций.")
    emit()

    # --- 1. масштаб сигнала -------------------------------------------------
    emit("## 1. Масштаб mean-diff и сколько его лежит на оси")
    emit()
    emit("$\\|r\\|$ — норма СЫРОЙ разности средних в выбранной точке (это и есть "
         "`direction.pt`), $\\langle r, w\\rangle$ — сколько её приходится на ось, "
         "в тех же единицах residual stream.")
    emit()
    emit("| ступень | точка | $\\|r\\|$ | $\\langle r, w_{cone}\\rangle$ | $\\cos$ | "
         "$\\langle r, w_{mix}\\rangle$ | $\\cos$ |")
    emit("|---|---|---|---|---|---|---|")
    for r in concepts:
        pos, lay = SEL[r]
        raw = MD[r][pos + n_pos, lay]
        assert float(unit(raw) @ unit(D[r])) > 0.999, f"native mean_diff != direction.pt у {r}"
        nr = float(raw.norm())
        c1, cm = float(raw @ w1), (float(raw @ wm) if wm is not None else float("nan"))
        emit(f"| {r} | ({pos}, {lay}) | {nr:.2f} | {c1:+.2f} | {c1/nr:+.3f} | "
             f"{cm:+.2f} | {cm/nr:+.3f} |")
    emit()

    emit("Где по сетке (pos, layer) ось собирает больше всего сырого сигнала mean-diff "
         "(против того, что выбрал `select_direction`):")
    emit()
    emit("| ступень | native: $\\langle r, w_{cone}\\rangle$ | argmax по сетке | там "
         "$\\langle r, w_{cone}\\rangle$ | там $\\|r\\|$ | там $\\cos$ |")
    emit("|---|---|---|---|---|---|")
    for r in concepts:
        pos, lay = SEL[r]
        M = MD[r]                                        # [n_pos, n_layer, hidden]
        proj = (M @ w1).abs()                            # [n_pos, n_layer]
        nrm = M.norm(dim=-1)
        flat = int(proj.argmax())
        bp, bl = flat // proj.shape[1], flat % proj.shape[1]
        emit(f"| {r} | {float((M[pos + n_pos, lay] @ w1)):+.2f} | "
             f"({bp - n_pos}, {bl}) | {float(M[bp, bl] @ w1):+.2f} | "
             f"{float(nrm[bp, bl]):.2f} | {float(M[bp, bl] @ w1) / float(nrm[bp, bl]):+.3f} |")
    emit()

    # --- 2. квадратичная форма ---------------------------------------------
    emit("## 2. Критерий оси, применённый к DIM")
    emit()
    emit("$q(v) = v^\\top P v = \\frac1N\\sum_j\\|B_j v\\|^2$ — функционал, максимум "
         "которого и есть ось ($q(w_1) = \\lambda_1$). $\\|Pv\\|$ — насколько проектор "
         "конусов сжимает вектор.")
    emit()
    nm, np95 = q_null(frames, hidden, args.n_null, gen)
    rows = [("w_cone ($=w_1$)", w1), ("w_cone $w_2$", W_cone[1]), ("w_cone $w_3$", W_cone[2])]
    if wm is not None:
        rows.append(("w_mixed", wm))
    for r in concepts:
        rows.append((f"DIM {r} (native)", unit(D[r])))
    for r in concepts:
        d = unit(D[r])
        rows.append((f"dim_perp_cone {r}", unit(d - (d @ w1) * w1)))
    rows.append(("cone_perp_dim (all)", unit(w1 - (w1 @ unit(D["all"])) * unit(D["all"]))))

    # span всех 12 конусов: энергия вектора внутри объединённого span-а
    M_all = torch.cat(frames, dim=0)
    Q, _ = torch.linalg.qr(M_all.T)                      # [hidden, rank]
    rank = Q.shape[1]

    emit("| направление | $q(v)$ | $q/\\lambda_1$ | $\\|Pv\\|$ | $\\|Pv\\|/\\lambda_1$ | "
         "энергия в span всех конусов |")
    emit("|---|---|---|---|---|---|")
    for name, v in rows:
        qv = q_form(frames, v)
        pv = float(p_apply(frames, v).norm())
        sp = float((Q.T @ v).pow(2).sum())
        emit(f"| {name} | **{qv:.3f}** | {qv/float(lam[0]):.2f} | {pv:.3f} | "
             f"{pv/float(lam[0]):.2f} | {sp:.3f} |")
    emit(f"| случайное (нуль, {args.n_null} шт.) | {nm:.4f} / p95 {np95:.4f} | "
         f"{nm/float(lam[0]):.3f} | — | — | {rank}/{hidden} = {rank/hidden:.3f} |")
    emit()
    emit(f"Аналитический нуль $q$: $\\bar k/\\text{{hidden}} = "
         f"{sum(B.shape[0] for B in frames)/N:.1f}/{hidden} = "
         f"{sum(B.shape[0] for B in frames)/N/hidden:.4f}$; ранг объединённого span-а "
         f"= {rank}.")
    emit()

    emit("$\\alpha_j = \\|B_j v\\|$ по каждому из 12 конусов — в тех же единицах, что "
         "коридор оси:")
    emit()
    emit("| вектор | min | медиана | max |")
    emit("|---|---|---|---|")
    ax_a = pooled["alphas"].double()
    emit(f"| w_cone (из pooled.pt) | {float(ax_a.min()):.3f} | "
         f"{float(ax_a.median()):.3f} | {float(ax_a.max()):.3f} |")
    for name, v in rows:
        if not name.startswith("DIM"):
            continue
        a = torch.tensor(alphas_of(frames, v))
        emit(f"| {name} | {float(a.min()):.3f} | {float(a.median()):.3f} | "
             f"{float(a.max()):.3f} |")
    emit()

    # --- 3. накопленная энергия DIM в топ-m -------------------------------
    ms = [m for m in (1, 2, 3, 5, 8, 12, 16) if m <= W_cone.shape[0]]
    emit("## 3. Накопленная энергия DIM в топ-$m$ собственном подпространстве $P$")
    emit()
    emit("$\\sum_{k \\le m}\\langle w_k, \\hat r\\rangle^2$ — сколько общих осей нужно, "
         "чтобы объяснить DIM (в файле сохранены только первые "
         f"{W_cone.shape[0]} осей из {rank} ненулевых).")
    emit()
    emit("| ступень | " + " | ".join(f"m={m}" for m in ms) + " |")
    emit("|---|" + "---|" * len(ms))
    for r in concepts:
        c = (W_cone @ unit(D[r])) ** 2
        emit(f"| {r} | " + " | ".join(f"{float(c[:m].sum()):.3f}" for m in ms) + " |")
    if wm is not None:
        emit()
        emit("Для сверки, та же величина для смешанной оси: "
             + ", ".join(f"m={m}: {float(((W_cone @ wm) ** 2)[:m].sum()):.3f}" for m in ms)
             + ".")
    emit()

    emit("Спектр $P$ (первые 8): " + ", ".join(f"{float(x):.3f}" for x in lam[:8]) + ".")
    emit()


# ---------------------------------------------------------------- stage: acts

def load_kl(base, rung):
    """Измеренный KL_ret и ASR из шагов 4-5 — для сверки с удаляемой энергией."""
    out = {}
    for f in (f"ablation-{rung}.json", f"residual-{rung}.json"):
        p = os.path.join(base, "common_axis", f)
        if not os.path.exists(p):
            continue
        for row in json.load(open(p))["rows"]:
            if int(row["rank"]) != 1:
                continue
            out.setdefault(row["direction"], {})[float(row["alpha"])] = (
                float(row["kl_ret"]), float(row["asr"]))
    return out


def stage_acts(args, emit):
    cp = os.path.join(args.base, "common_axis", f"proj-{args.rung}.pt")
    assert os.path.exists(cp), f"нет кэша {cp} — сначала STAGE=cache (GPU)"
    C = torch.load(cp, map_location="cpu")
    dirs = C["dirs"]
    pos, lay = C["native"]
    n_pos = C["n_eoi"]
    L = C["n_layers"]
    hidden = C["hidden"]
    S = C["sites"]
    kl = load_kl(args.base, args.rung)

    def st(setname):
        return C["sets"][setname]

    emit(f"# §9 (acts). Проекции активаций: ось против DIM — ступень `{args.rung}`")
    emit()
    emit(f"Модель {C['model']}, harmful_test = {st('harmful')['n_prompts']}, "
         f"harmless_test = {st('harmless')['n_prompts']}, слоёв {L}, "
         f"точка DIM ({pos}, {lay}).")
    emit("Проекции снимаются в тех же трёх точках, куда бьёт аблация: вход блока "
         "(`resid`), выход attn, выход mlp. Активации не центрированы — это сырой "
         "residual stream.")
    emit()

    # --- A. распределения в native-точке ---------------------------------
    emit("## A. Распределение проекции в точке DIM (вход блока, токен $-1$)")
    emit()
    emit("$d' = (\\mu_h - \\mu_b)/\\sigma$ по объединённой дисперсии; $E|p|$ — средний "
         "модуль проекции, то есть длина вектора, который удаляет полная аблация.")
    emit()
    emit("| направление | $\\mu_h$ | $\\mu_b$ | $\\sigma$ | $d'$ | $E|p|$ harmful | "
         "$E|p|$ harmless | $E|p|/E\\|h\\|$ harmless |")
    emit("|---|---|---|---|---|---|---|---|")
    ph = st("harmful")["eoi"]["resid"][:, -1, lay, :]     # [n, D]
    pb = st("harmless")["eoi"]["resid"][:, -1, lay, :]
    nh = st("harmful")["eoi_n"]["resid"][:, -1, lay]
    nb = st("harmless")["eoi_n"]["resid"][:, -1, lay]
    for i, nm in enumerate(dirs):
        mh, mb = float(ph[:, i].mean()), float(pb[:, i].mean())
        sd = float(torch.cat([ph[:, i] - mh, pb[:, i] - mb]).pow(2).mean().sqrt())
        emit(f"| {nm} | {mh:+.2f} | {mb:+.2f} | {sd:.2f} | **{(mh-mb)/sd:+.2f}** | "
             f"{float(ph[:, i].abs().mean()):.2f} | {float(pb[:, i].abs().mean()):.2f} | "
             f"{float(pb[:, i].abs().mean())/float(nb.mean()):.4f} |")
    emit()
    emit(f"Для масштаба: $E\\|h\\|$ = {float(nh.mean()):.1f} (harmful) / "
         f"{float(nb.mean()):.1f} (harmless) в этой точке.")
    emit()

    # --- B. доля дисперсии (ковариационный тест) -------------------------
    emit("## B. Доля дисперсии на направлении (ковариационный тест)")
    emit()
    emit("$\\mathrm{Var}(p_v)$ ПО ПРОМПТАМ на последней позиции против изотропного уровня "
         "$\\operatorname{tr}\\Sigma/d$ той же популяции: лежит ли направление на "
         "высокодисперсной оси активаций. По всем токенам эту величину считать нельзя — "
         "её съедает sink-токен (см. секцию C).")
    emit()
    s_ = st("harmless")
    n_p = s_["n_prompts"]
    emit("| направление | " + " | ".join(f"Var/изотроп, {x}" for x in S) + " |")
    emit("|---|" + "---|" * len(S))
    iso = {}
    for site in S:
        h2 = float((s_["eoi_n"][site][:, -1, lay] ** 2).mean())
        mh = s_["sum_h_last"][site][lay] / n_p
        iso[site] = (h2 - float(mh.pow(2).sum())) / hidden
    for i, nm in enumerate(dirs):
        cells = []
        for site in S:
            pv = s_["eoi"][site][:, -1, lay, i].double()
            cells.append(f"{float(pv.var(unbiased=False)) / iso[site]:.1f}")
        emit(f"| {nm} | " + " | ".join(cells) + " |")
    emit()
    emit("Единица — дисперсия случайного направления в среднем (изотропный нуль); "
         "значение 10 означает «вдесятеро дисперснее случайного».")
    emit()

    # --- C. что стирается при аблации и цена по KL ------------------------
    emit("## C. Сколько сигнала стирает аблация и чем это оборачивается в $KL_{ret}$")
    emit()
    emit("Полная аблация удаляет из каждого тензора вектор длиной $|p_v|$ во всех слоях и "
         "всех трёх точках вмешательства. Основная мера — **средняя по токенам доля нормы** "
         "$\\overline{p^2/\\|h\\|^2}$ (каждый токен весит одинаково, изотропный нуль = "
         f"1/hidden = {1/hidden:.5f}). Энергетически взвешенная версия "
         "$\\sum p^2/\\sum\\|h\\|^2$ дана рядом, но читать её нельзя как ущерб: у Qwen "
         "норма residual stream на sink-токене на два порядка выше, чем в точке чтения, и "
         "она эту сумму и определяет.")
    emit()
    emit("| направление | доля нормы (harmless) | доля нормы (harmful) | энергетически "
         "взвешенная | $KL_{ret}$ ($a{=}1$) | ASR ($a{=}1$) |")
    emit("|---|---|---|---|---|---|")
    rel, wgt = {}, {}
    for setname in ("harmless", "harmful"):
        ss = st(setname)
        num = torch.zeros(len(dirs), dtype=torch.float64)
        eng_num, eng_den, cnt = torch.zeros(len(dirs), dtype=torch.float64), 0.0, 0
        for site in S:
            num += ss["sum_rel"][site].sum(0).double() / float(ss["n_tok"][site])
            cnt += ss["sum_rel"][site].shape[0]
            eng_num += ss["sum_p2"][site].sum(0).double()
            eng_den += float(ss["sum_h2"][site].sum())
        rel[setname] = num / cnt
        wgt[setname] = eng_num / eng_den
    for i, nm in enumerate(dirs):
        k = kl.get(nm, {}).get(1.0)
        ks = f"{k[0]:.4f}" if k else "—"
        asr = f"{k[1]:.3f}" if k else "—"
        emit(f"| {nm} | **{float(rel['harmless'][i]):.5f}** | "
             f"{float(rel['harmful'][i]):.5f} | {float(wgt['harmless'][i]):.5f} | "
             f"{ks} | {asr} |")
    emit()
    emit("Сравнивать направления по $KL$ при разной удаляемой доле — и есть проверка "
         "механизма: если ось дешевле DIM просто потому, что стирает меньше, отношение "
         "$KL$ к доле нормы у них совпадёт.")
    emit()
    for nm in dirs:
        k = kl.get(nm, {}).get(1.0)
        if k:
            i = dirs.index(nm)
            emit(f"- {nm}: $KL$/доля = {k[0] / float(rel['harmless'][i]):.1f}")
    emit()

    # --- D. профиль по слоям ---------------------------------------------
    step = max(1, L // 9)
    show = list(range(0, L, step))
    emit("## D. Профиль по слоям (вход блока, harmless/harmful)")
    emit()
    emit("$d'$ — разделяющая способность на токене $-1$; $E|p|/E\\|h\\|$ — доля нормы, "
         "которую снимает аблация (harmless).")
    emit()
    emit("| слой | " + " | ".join(f"{nm}: $d'$ / rel" for nm in dirs) + " |")
    emit("|---|" + "---|" * len(dirs))
    for l in show:
        cells = []
        for i in range(len(dirs)):
            a = st("harmful")["eoi"]["resid"][:, -1, l, i]
            b = st("harmless")["eoi"]["resid"][:, -1, l, i]
            sd = float(torch.cat([a - a.mean(), b - b.mean()]).pow(2).mean().sqrt())
            d_ = (float(a.mean()) - float(b.mean())) / max(sd, 1e-9)
            nrm = float(st("harmless")["eoi_n"]["resid"][:, -1, l].mean())
            cells.append(f"{d_:+.2f} / {float(b.abs().mean())/nrm:.3f}")
        emit(f"| {l}{' ←DIM' if l == lay else ''} | " + " | ".join(cells) + " |")
    emit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="каталог results/ форка")
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--stage", choices=("static", "acts"), default="static")
    ap.add_argument("--rung", default="all", help="ступень для stage=acts")
    ap.add_argument("--n_null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--threads", type=int, default=int(os.getenv("THREADS", "8")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    (stage_static if args.stage == "static" else stage_acts)(args, emit)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"[записано: {args.out}]")


if __name__ == "__main__":
    main()
