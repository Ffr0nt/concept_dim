"""Пересечение двух общих компонент: снимает ли отказ и какая у него норма.

«Пересечение» — в том же смысле, в каком строится общая компонента: первый собственный
вектор P = (1/N) sum v v^T по набору из двух компонент. Для N = 2 это тождественно
биссектриса, поэтому вопрос о норме имеет точный ответ.

Три разных «нормы», и путать их нельзя:
  1. норма самой оси — единичная по построению, и аблация к масштабу инвариантна
     (вычитается ww^T/||w||^2), поэтому на снятие отказа она не влияет вовсе;
  2. длина проекции каждой компоненты на ось — сколько от компоненты уцелело;
  3. длина проекции НАСТОЯЩЕГО сдвига активаций на ось — сколько аблация реально
     вычитает из активации. Сдвиг берётся готовым: это mean_diffs этапа DIM, разность
     средних harmful и harmless в рабочей точке, с настоящей нормой в пространстве
     активаций.

Запускать на сервере из корня shared-refusal-component: bash run/bisector_norms.sh
"""
import json
import os

import torch

from compare_to_base import BT, SD, spectrum, unit, vectors, w


def mean_diff(run):
    """Настоящий сдвиг активаций в рабочей точке: тот же вектор, что DIM union."""
    md = torch.load(os.path.join(run, "dim/union/generate_directions/mean_diffs.pt"),
                    map_location="cpu", weights_only=True).double()
    d = torch.load(os.path.join(run, "dim/union/direction.pt"),
                   map_location="cpu", weights_only=True).double()
    for i in range(md.shape[0]):
        for j in range(md.shape[1]):
            if torch.allclose(md[i, j], d, atol=1e-4):
                return d, (i, j)
    return d, None


def main():
    wb, ws = w(BT, "rdo_only"), w(SD, "all")
    c = float(wb @ ws)
    ws = ws if c >= 0 else -ws          # знак направления условен
    c = abs(c)

    lam, W = spectrum(torch.stack([wb, ws]))
    bis = unit(W[:, 0])
    pool = unit(spectrum(torch.cat([
        torch.stack([v for n, v in vectors(r).items() if n.startswith("rdo_")])
        for r in (BT, SD)]))[1][:, 0])

    print("## Что судья видел\n")
    names = torch.load(f"{BT}/component/cross_run_directions.pt",
                       map_location="cpu", weights_only=False)["names"]
    print(f"Направления, поданные на этапы 06-07: {', '.join(names)}.")
    print("Биссектрисы среди них нет: у неё только прокси-замер этапа 03.\n")

    print("## Пересечение двух компонент\n")
    print(f"- |cos| между компонентами: {c:.4f}")
    print(f"- lam_1 пары: {float(lam[0]):.4f}, тождество (1+|cos|)/2 = {(1 + c) / 2:.4f}")
    print(f"- биссектриса совпадает с нормированной суммой: "
          f"{float(abs(bis @ unit(wb + ws))):.6f}")
    print(f"- |cos| биссектрисы с общей осью 40 RDO: {float(abs(bis @ pool)):.4f}")

    print("\n## Норма 1: сама ось\n")
    print(f"- ||w_bt|| = {float(wb.norm()):.4f}, ||w_sd|| = {float(ws.norm()):.4f}, "
          f"||биссектриса|| = {float(bis.norm()):.4f}")
    print(f"- ||w_bt + w_sd|| = {float((wb + ws).norm()):.4f} "
          f"(при полном совпадении было бы 2.0000, тождество sqrt(2+2cos) = "
          f"{(2 + 2 * c) ** 0.5:.4f})")
    print("- аблация инвариантна к масштабу: вычитается ww^T/||w||^2, поэтому норма оси "
          "на снятие отказа не влияет.")

    print("\n## Норма 2: сколько от каждой компоненты уцелело на оси\n")
    print("| компонента | длина проекции на биссектрису | доля энергии |")
    print("|---|---|---|")
    for n, v in (("наша без DIM", wb), ("базовая по всем 24", ws)):
        p = float(abs(v @ bis))
        print(f"| {n} | {p:.4f} | {p ** 2:.4f} |")
    print(f"\nДоля энергии {float(abs(wb @ bis)) ** 2:.4f} совпадает с lam_1 пары "
          f"{float(lam[0]):.4f} — это одно и то же число.")

    print("\n## Норма 3: сколько ось вычитает из настоящего сдвига активаций\n")
    g, where = mean_diff(BT)
    print(f"Сдвиг активаций в рабочей точке (mean_diffs{where}): ||g|| = "
          f"{float(g.norm()):.2f} в пространстве активаций.\n")
    AX = [("наша компонента без DIM", wb), ("базовая компонента по всем 24", ws),
          ("биссектриса двух компонент", bis), ("общая ось 40 RDO", pool),
          ("наш DIM union (сам сдвиг)", unit(g))]
    print("| ось | |g·w| | доля ||g|| |")
    print("|---|---|---|")
    for n, v in AX:
        p = float(abs(g @ v))
        print(f"| {n} | {p:.2f} | {p / float(g.norm()):.4f} |")

    print("\n## Эмпирическая цена правки: KL при alpha = 1.0\n")
    rows = json.load(open(f"{BT}/component/cross_run_ablation.json"))["rows"]
    print("| направление | KL | прокси-ASR |")
    print("|---|---|---|")
    for r in rows:
        if abs(r["alpha"] - 1.0) < 1e-6 and r["direction"] in (
                "w_bisect", "w_pool_rdo", "w_bt_rdo", "w_sd_all", "rand_x"):
            print(f"| {r['direction']} | {r['kl']:.4f} | {r['asr_mean']:.3f} |")


if __name__ == "__main__":
    main()
