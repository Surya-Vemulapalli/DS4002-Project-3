"""
05_plot_explanatory.py

PURPOSE
    Descriptive (explanatory) figures of the data; no models or tests.
    Price figures (the MI2 exploratory plots), using every listing eligible for
    sampling: Entire home/apt or Private room with a price, bedrooms,
    bathrooms_text and minimum_nights (the same filters as script 01, before
    one-per-host and sampling; 18,767 listings):
      1. price_by_borough.png           median nightly price by borough
      2. price_by_bedrooms_room_type.png median nightly price by bedrooms and room type
    Photo-score figures, using the final analysis sample (1,972 listings):
      3. photo_score_distributions.png  histogram of each raw score, median marked
      4. photo_scores_by_borough.png    box plot of each raw score by borough group

INPUTS
    DATA/raw/listings.csv
    DATA/final/analysis_data.csv

OUTPUTS
    OUTPUT/figures/price_by_borough.png
    OUTPUT/figures/price_by_bedrooms_room_type.png
    OUTPUT/figures/photo_score_distributions.png
    OUTPUT/figures/photo_scores_by_borough.png

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/05_plot_explanatory.py

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

RAW_CSV = Path("DATA/raw/listings.csv")
DATA_CSV = Path("DATA/final/analysis_data.csv")
FIG_DIR = Path("OUTPUT/figures")

# Same eligibility filters as script 01 (before one-per-host and sampling).
KEEP_ROOM_TYPES = ["Entire home/apt", "Private room"]
REQUIRED_NONMISSING = ["price", "bedrooms", "bathrooms_text", "minimum_nights"]
BOROUGHS_5 = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
ROOM_COLORS = {"Entire home/apt": "#2a78d6", "Private room": "#eb6834"}  # categorical slots 1, 2

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


def plot_price_figures() -> None:
    """Median nightly price by borough, and by bedrooms x room type (eligible listings)."""
    # -----------------------------------------------------------------------
    # Load the raw listings and apply script 01's eligibility filters.
    # -----------------------------------------------------------------------
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"Loaded {RAW_CSV}: {len(raw):,} rows")
    raw = raw[raw["room_type"].isin(KEEP_ROOM_TYPES)]
    print(f"After keeping room_type in {KEEP_ROOM_TYPES}: {len(raw):,} rows")
    raw = raw.dropna(subset=REQUIRED_NONMISSING).copy()
    print(f"After dropping rows missing any of {REQUIRED_NONMISSING}: {len(raw):,} rows")
    raw["price_usd"] = pd.to_numeric(raw["price"].astype(str).str.replace(r"[$,]", "", regex=True))

    # -----------------------------------------------------------------------
    # Figure: median price by borough (one series, value and n labels).
    # -----------------------------------------------------------------------
    by_b = raw.groupby("neighbourhood_group_cleansed")["price_usd"].agg(["median", "size"]).loc[BOROUGHS_5]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(by_b))
    ax.bar(x, by_b["median"], width=0.6, color=SERIES)
    for xi, v in zip(x, by_b["median"]):
        ax.text(xi, v + 4, f"${v:,.0f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, [f"{b}\n(n={n:,})" for b, n in zip(by_b.index, by_b["size"])], fontsize=9, color=INK)
    ax.set_ylabel("Median nightly price (USD)", color=INK_2, fontsize=9)
    ax.set_title(f"Median nightly price by borough (n = {len(raw):,} eligible listings)",
                 loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    out = FIG_DIR / "price_by_borough.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")

    # -----------------------------------------------------------------------
    # Figure: median price by bedrooms (0-3, 4+) and room type, grouped bars.
    # -----------------------------------------------------------------------
    raw["bed_group"] = raw["bedrooms"].clip(upper=4).astype(int)
    med = raw.pivot_table(index="bed_group", columns="room_type", values="price_usd", aggfunc="median")
    cnt = raw.pivot_table(index="bed_group", columns="room_type", values="price_usd", aggfunc="size")
    labels = ["0", "1", "2", "3", "4+"]
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(med))
    w = 0.38
    for j, rt in enumerate(KEEP_ROOM_TYPES):
        xs = x + (j - 0.5) * w
        ax.bar(xs, med[rt], width=w - 0.03, color=ROOM_COLORS[rt], label=rt)
        for xi, v, n in zip(xs, med[rt], cnt[rt]):
            ax.text(xi, v + 6, f"${v:,.0f}\nn={n:,}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, labels, fontsize=10, color=INK)
    ax.set_xlabel("Bedrooms", color=INK_2, fontsize=9)
    ax.set_ylabel("Median nightly price (USD)", color=INK_2, fontsize=9)
    ax.set_ylim(0, med.max().max() * 1.18)
    ax.set_title("Median nightly price by bedrooms and room type", loc="left", fontsize=12,
                 fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "price_by_bedrooms_room_type.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    plot_price_figures()

    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {DATA_CSV}: {len(df):,} rows")

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
