"""Shared journal style for the figure scripts (single column 3.4 in, double column 7.1 in)."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from presage.utils.common import ROOT

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#4a3aa7", red="#e34948", magenta="#e87ba4", yellow="#c98500")
INK, MUTED, GRID = "#1a1a1a", "#6b6b66", "#e4e3de"
W1, W2 = 3.4, 7.1
OUT = ROOT / "reports/figures/publication"; OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig); print("saved", name)


def panel(ax, s): ax.text(-0.02, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def tag(fig, t="Synthetic data (sim_v1)"): fig.text(0.995, 1.0, t, fontsize=6.5, color=MUTED, style="italic", ha="right", va="bottom")
