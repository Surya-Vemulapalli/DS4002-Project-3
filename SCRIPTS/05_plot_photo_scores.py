"""
05_plot_photo_scores.py

PURPOSE
    Descriptive figures of the three photo-quality scores (no relationship with
    price, no tests; those belong to the later analysis step):
      1. photo_score_distributions.png  histogram of each raw score (all listings),
                                        with the median marked
      2. photo_scores_by_borough.png    box plot of each raw score by borough group

INPUTS
    DATA/final/analysis_data.csv

OUTPUTS
    OUTPUT/figures/photo_score_distributions.png
    OUTPUT/figures/photo_scores_by_borough.png

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/05_plot_photo_scores.py

APPROXIMATE RUNTIME
    A few seconds.
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_CSV = Path("DATA/final/analysis_data.csv")
FIG_DIR = Path("OUTPUT/figures")

# Score column -> (panel title, scale note)
SCORES = {
    "musiq_koniq": ("MUSIQ-KonIQ", "technical quality, 0-100"),
    "musiq_ava": ("MUSIQ-AVA", "aesthetic quality, 1-10"),
    "clipiqa": ("CLIP-IQA", "overall quality, 0-1"),
}
BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx/Staten Island"]

# Colours: one series hue plus recessive ink/grid (light mode).
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def style_axes(ax) -> None:
    """Recessive axes: no top/right spines, light horizontal grid behind the data."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {DATA_CSV}: {len(df):,} rows")
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

    # -----------------------------------------------------------------------
    # Figure 1: histograms of each raw score with the median marked.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, (col, (title, note)) in zip(axes, SCORES.items()):
        style_axes(ax)
        x = df[col]
        ax.hist(x, bins=40, color=SERIES, edgecolor=SURFACE, linewidth=1)
        med = x.median()
        ax.axvline(med, color=INK, linewidth=1.2, linestyle="--")
        # Label sits just above the plot area so it never collides with the bars.
        ax.text(med, 1.01, f"median {med:.2f}", color=INK, fontsize=9, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_xlabel(note, color=INK_2, fontsize=9)
        ax.set_ylabel("Listings" if col == "musiq_koniq" else "", color=INK_2, fontsize=9)
    fig.suptitle(f"Distribution of photo-quality scores (n = {len(df):,} listing photos)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out1 = FIG_DIR / "photo_score_distributions.png"
    fig.savefig(out1, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: box plots of each raw score by borough group.
    # -----------------------------------------------------------------------
    counts = df["borough_group"].value_counts()
    labels = [f"{b}\n(n={counts[b]})" for b in BOROUGH_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, (col, (title, note)) in zip(axes, SCORES.items()):
        style_axes(ax)
        data = [df.loc[df["borough_group"] == b, col] for b in BOROUGH_ORDER]
        ax.boxplot(data, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=SERIES, edgecolor=SERIES, alpha=0.85),
                   medianprops=dict(color=SURFACE, linewidth=2),
                   whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_2,
                                   markeredgecolor="none", alpha=0.5))
        ax.set_xticks(range(1, len(BOROUGH_ORDER) + 1), labels, fontsize=8)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_ylabel(note, color=INK_2, fontsize=9)
    fig.suptitle("Photo-quality scores by borough group (box = middle 50%, line = median)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = FIG_DIR / "photo_scores_by_borough.png"
    fig.savefig(out2, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
