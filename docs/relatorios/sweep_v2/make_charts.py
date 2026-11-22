"""Gráficos do relatório do sweep v2 (lê os CSVs desta pasta)."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = "integration_height_object.pearson_r"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def end_labels(ax, items):
    """Rótulos diretos no fim de cada linha, afastados para não colidir."""
    items = sorted(items, key=lambda t: t[1])
    ylo, yhi = ax.get_ylim()
    gap = (yhi - ylo) * 0.045
    placed = []
    for x, y, text in items:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)


def chart_radius():
    rows = list(csv.DictReader(open(HERE / "A1.csv")))
    combos = [("1", "1", "smooth + mediana"), ("1", "0", "só smooth"), ("0", "1", "só mediana"), ("0", "0", "sem denoise")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, ds, title in zip(axes, ("d1", "d2"), ("d1: melon14, sem brilho", "d2: melon14, brilho 0,5")):
        labels = []
        for i, (sm, med, lab) in enumerate(combos):
            pts = []
            for r in rows:
                m = re.fullmatch(rf"fourier_r([\d.]+)_sm{sm}_med{med}_zb0", r["name"])
                if r["dataset"] == ds and m and r[R]:
                    pts.append((float(m.group(1)), float(r[R])))
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=PAL[i], lw=2, marker=MARK[i], ms=6, mec="white", mew=1.2, label=lab)
            labels.append((xs[-1], ys[-1], lab))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("raio do filtro fourier", fontsize=9, color=MUTED)
        style(ax)
        ax.set_xlim(0.015, 0.135)
        end_labels(ax, labels)
    axes[0].set_ylabel("r da altura (Pearson)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    fig.suptitle("A1: fourier, r da altura × raio por combinação de denoise (zero_border off)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "chart_A1_radius.png", dpi=150)


def _hints_points(rows, ds):
    wmap = {"uh0_initzero": 0.0, "_center:lap9_ss3.5": 0.1}
    pts = []
    for r in rows:
        if r["dataset"] != ds or not r[R]:
            continue
        w = wmap.get(r["name"])
        m = re.fullmatch(r"uh1_hw([\d.]+)", r["name"])
        if m:
            w = float(m.group(1))
        if w is not None:
            pts.append((max(w, 0.01), float(r[R])))  # 0 (sem hints) desenhado em 0,01 no eixo log
    return sorted(pts)


def chart_hints():
    rows = list(csv.DictReader(open(HERE / "Vhints.csv")))
    names = {"d1": "d1 melon14", "d2": "d2 melon14 brilho", "d3": "d3 melon24 (teste)", "d4": "d4 sem textura (teste)"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, ds in enumerate(("d1", "d2", "d3", "d4")):
        ax = b if ds == "d4" else a
        xs, ys = zip(*_hints_points(rows, ds))
        ax.plot(xs, ys, color=PAL[i], lw=2, marker=MARK[i], ms=6, mec="white", mew=1.2, label=names[ds])
    for ax in (a, b):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 5])
        ax.set_xticklabels(["0\n(sem)", "0,03", "0,1\n(atual)", "0,3", "1\n(recom.)", "5"])
        ax.set_xlim(0.008, 7)
        ax.set_xlabel("hints_weight (escala log)", fontsize=9, color=MUTED)
        style(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    a.set_ylim(0.78, 1.0)
    a.set_ylabel("r da altura (Pearson)", fontsize=9, color=MUTED)
    a.set_title("cenas com textura: o pico fica em ~1", fontsize=10, color=INK, loc="left")
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_title("sem textura: hints só atrapalham", fontsize=10, color=INK, loc="left")
    b.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    fig.suptitle("Vhints: laplaciano k9 + σ 3,5, r da altura × peso dos hints",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "chart_Vhints.png", dpi=150)


if __name__ == "__main__":
    chart_radius()
    chart_hints()
