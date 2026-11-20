"""
07_evaluate_models.py

PURPOSE
    Evaluate the fitted models on the held-out TEST set and summarize the
    borough-specific photo effects from M3:
      1. Test-set prediction error for M1, M2 and M3. Predictions of log price
         are converted back to dollars with exp(prediction) (no smearing
         correction), then:
           test_MAE_usd       mean absolute error in dollars
           test_MedAE_usd     median absolute error in dollars
           test_R2_log_price  R-squared on the log-price scale
           MAE_change_vs_M1   model MAE minus M1 MAE (negative = more accurate)
           boot_CI_low/high   95% bootstrap CI for MAE_change_vs_M1
                              (1,000 resamples of the test set, seed 42; the same
                              resamples are used for M2 and M3)
      2. The same MAE for each borough group.
      3. From M3, the effect of a +1 SD photo score on log price in each borough
         group: Manhattan = the score's main effect; other groups = main effect +
         interaction. 95% CIs and p-values use the model's HC3 robust covariance.
         pct_price_change_per_SD = 100 * (exp(coef) - 1).

INPUTS
    DATA/final/analysis_data.csv
    OUTPUT/models/m1.pickle, m2.pickle, m3.pickle   (from 04_fit_models.py)

OUTPUTS
    OUTPUT/tables/mae_model_comparison.csv
    OUTPUT/tables/mae_by_borough.csv
    OUTPUT/tables/photo_effects_by_borough.csv
    OUTPUT/figures/mae_model_comparison.png
    OUTPUT/figures/mae_by_borough.png
    OUTPUT/figures/photo_effects_by_borough.png

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/07_evaluate_models.py

APPROXIMATE RUNTIME
    About 10 seconds.
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

# ---------------------------------------------------------------------------
# Configuration: relative paths, seed 42.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_CSV = Path("DATA/final/analysis_data.csv")
MODEL_DIR = Path("OUTPUT/models")
TABLE_DIR = Path("OUTPUT/tables")
FIG_DIR = Path("OUTPUT/figures")

N_BOOT = 1000
ALPHA = 0.05

MODELS = {"m1": "M1 baseline", "m2": "M2 + photo scores", "m3": "M3 + photo x borough"}
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx/Staten Island"]
# Borough group -> its dummy in the model (Manhattan is the baseline, no dummy)
BOROUGH_DUMMY = {"Manhattan": None, "Brooklyn": "brooklyn", "Queens": "queens", "Bronx/Staten Island": "bronx_si"}
SCORES = {"z_musiq_koniq": "MUSIQ-KonIQ (technical)", "z_musiq_ava": "MUSIQ-AVA (aesthetic)", "z_clipiqa": "CLIP-IQA"}

# Colours: neutral grey for the baseline, then categorical slots 1 and 2.
COLORS = {"m1": "#52514e", "m2": "#2a78d6", "m3": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style_axes(ax, grid_axis: str = "y") -> None:
    """Recessive axes: no top/right spines, light grid behind the data."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """R-squared of predictions yhat for observed y."""
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load the data, the TEST rows and the three saved models.
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {DATA_CSV}: {len(df):,} rows")
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"After keeping test rows only: {len(test):,} rows")
    models = {k: load_pickle(str(MODEL_DIR / f"{k}.pickle")) for k in MODELS}

    # Predictions: log scale, and dollars via exp(); absolute errors in dollars.
    price = test["price"].to_numpy()
    log_price = test["log_price"].to_numpy()
    pred_log = {k: m.predict(test).to_numpy() for k, m in models.items()}
    abs_err = {k: np.abs(price - np.exp(p)) for k, p in pred_log.items()}

    # -----------------------------------------------------------------------
    # 1. Overall test-set error, with a bootstrap CI for each model's MAE
    #    change vs M1. One generator and the same resampled rows are used
    #    for every model so the comparisons are paired.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    n = len(test)
    boot = {"m2": np.empty(N_BOOT), "m3": np.empty(N_BOOT)}
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        base = abs_err["m1"][i].mean()
        for k in boot:
            boot[k][b] = abs_err[k][i].mean() - base

    rows = []
    for k, label in MODELS.items():
        row = {"model": label, "n_test": n,
               "test_MAE_usd": abs_err[k].mean(),
               "test_MedAE_usd": np.median(abs_err[k]),
               "test_R2_log_price": r2(log_price, pred_log[k]),
               "MAE_change_vs_M1": np.nan, "boot_CI_low": np.nan, "boot_CI_high": np.nan}
        if k != "m1":
            row["MAE_change_vs_M1"] = abs_err[k].mean() - abs_err["m1"].mean()
            row["boot_CI_low"], row["boot_CI_high"] = np.percentile(boot[k], [2.5, 97.5])
        rows.append(row)
    mae_models = pd.DataFrame(rows).round(4)
    mae_models.to_csv(TABLE_DIR / "mae_model_comparison.csv", index=False)
    print("Test-set error by model:")
    print(mae_models.to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. Test-set MAE within each borough group.
    # -----------------------------------------------------------------------
    rows = []
    for bname in BOROUGHS:
        mask = (test["borough_group"] == bname).to_numpy()
        row = {"borough": bname, "n_test": int(mask.sum())}
        for k in MODELS:
            row[f"MAE_{k}"] = abs_err[k][mask].mean()
        row["M2_minus_M1"] = row["MAE_m2"] - row["MAE_m1"]
        row["M3_minus_M1"] = row["MAE_m3"] - row["MAE_m1"]
        rows.append(row)
    mae_borough = pd.DataFrame(rows).round(2)
    mae_borough.to_csv(TABLE_DIR / "mae_by_borough.csv", index=False)
    print("Test-set MAE by borough group:")
    print(mae_borough.to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. Borough-specific photo effects from M3 (training-set fit, HC3).
    #    Each effect is a linear combination of coefficients tested with
    #    t_test, which uses the model's robust covariance matrix.
    # -----------------------------------------------------------------------
    m3 = models["m3"]
    rows = []
    for z, zlabel in SCORES.items():
        for bname in BOROUGHS:
            d = BOROUGH_DUMMY[bname]
            expr = f"{z} = 0" if d is None else f"{z} + {z}:{d} = 0"
            t = m3.t_test(expr)
            coef = float(np.squeeze(t.effect))
            lo, hi = np.squeeze(t.conf_int(alpha=ALPHA))
            rows.append({"score": zlabel, "borough": bname, "coef_log_price_per_SD": coef,
                         "ci_low": lo, "ci_high": hi,
                         "pct_price_change_per_SD": 100 * (np.exp(coef) - 1),
                         "p_value": float(np.squeeze(t.pvalue))})
    effects = pd.DataFrame(rows).round(4)
    effects.to_csv(TABLE_DIR / "photo_effects_by_borough.csv", index=False)
    print("Photo-score effects by borough (M3):")
    print(effects.to_string(index=False))

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

    # -----------------------------------------------------------------------
    # Figure: overall test-set MAE by model, value labels on each bar.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(MODELS))
    bars = ax.bar(x, mae_models["test_MAE_usd"], width=0.6, color=[COLORS[k] for k in MODELS])
    for bar, v in zip(bars, mae_models["test_MAE_usd"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"${v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, list(MODELS.values()), fontsize=9.5, color=INK)
    ax.set_ylabel("Test-set MAE (USD)", color=INK_2)
    ax.set_title(f"Test-set MAE by model (n = {n})", loc="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_model_comparison.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure: test-set MAE by borough group, grouped bars per model.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(BOROUGHS))
    w = 0.26
    for j, (k, label) in enumerate(MODELS.items()):
        ax.bar(x + (j - 1) * w, mae_borough[f"MAE_{k}"], width=w - 0.02, color=COLORS[k], label=label)
    ax.set_xticks(x, [f"{b}\n(n={c})" for b, c in zip(mae_borough["borough"], mae_borough["n_test"])],
                  fontsize=9.5, color=INK)
    ax.set_ylabel("Test-set MAE (USD)", color=INK_2)
    ax.set_title("Test-set MAE by borough group", loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_by_borough.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure: M3 photo effects with 95% CIs, one panel per score.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor=SURFACE)
    ypos = np.arange(len(BOROUGHS))[::-1]  # Manhattan at the top
    for ax, zlabel in zip(axes, SCORES.values()):
        style_axes(ax, grid_axis="x")
        e = effects[effects["score"] == zlabel].set_index("borough").loc[BOROUGHS]
        ax.errorbar(e["coef_log_price_per_SD"], ypos,
                    xerr=[e["coef_log_price_per_SD"] - e["ci_low"], e["ci_high"] - e["coef_log_price_per_SD"]],
                    fmt="o", color=COLORS["m2"], ecolor=COLORS["m2"], capsize=4, markersize=8, linewidth=1.5)
        ax.axvline(0, color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(zlabel, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("Change in log price per +1 SD (95% CI)", color=INK_2)
    axes[0].set_yticks(ypos, BOROUGHS, fontsize=10, color=INK)
    fig.suptitle("Photo-score effect on log price by borough (M3, HC3 robust CIs)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "photo_effects_by_borough.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved 3 tables to {TABLE_DIR} and 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
