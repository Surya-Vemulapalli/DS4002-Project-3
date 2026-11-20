"""
09_photo_review_robustness.py

PURPOSE
    Summarize the manual photo-authenticity review (from 08_make_photo_review.py)
    and run the robustness check planned in MI2: refit M1 and M2 without the
    listings whose photos were flagged, and compare the partial F-test and
    test-set delta R^2 with the main results from 06_statistical_tests.ipynb.

INPUT
    DATA/final/analysis_data.csv
    OUTPUT/tables/photo_review_results.csv   (labels saved from the review page)

OUTPUT
    OUTPUT/tables/photo_review_summary.csv     count and % of each review label
    OUTPUT/tables/photo_review_robustness.csv  F-test and delta R^2 with and
                                               without flagged listings

NOTES
    - Model formulas, HC3 robust SEs and the train/test split match
      OUTPUT/models/model_specs.md and notebook 06.
    - Run from the top-level repository folder:
          python SCRIPTS/09_photo_review_robustness.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "SCRIPTS" else Path(".")
TABLES = ROOT / "OUTPUT" / "tables"
FLAG_LABELS = ["rendering", "virtual_staging", "heavy_edit", "not_space"]   # labels that count as flagged

# ---- 1. Summarize the review labels ----
review = pd.read_csv(TABLES / "photo_review_results.csv", dtype={"id": str})
counts = review["label"].fillna("unlabeled").value_counts()
summary = pd.DataFrame({"label": counts.index, "n": counts.values})
summary["pct"] = (100 * summary["n"] / len(review)).round(1)
summary.to_csv(TABLES / "photo_review_summary.csv", index=False)
flagged_ids = set(review.loc[review["label"].isin(FLAG_LABELS), "id"])
print(summary.to_string(index=False))
print(f"Flagged: {len(flagged_ids)} of {len(review)} reviewed photos")

# ---- 2. Load the analysis data and define the models (same as notebook 06) ----
df = pd.read_csv(ROOT / "DATA" / "final" / "analysis_data.csv", dtype={"id": str}, low_memory=False)
BASE = ("log_price ~ private_room + bedrooms + bathrooms + shared_bath + accommodates"
        " + brooklyn + queens + bronx_si + min30 + private_room:bedrooms")
PHOTO = ["z_musiq_koniq", "z_musiq_ava", "z_clipiqa"]


def r2(y, yhat):
    """Out-of-sample R^2: 1 - SSE / SST."""
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def run(data, label):
    """Fit M1 and M2 on the training rows and return the key test results."""
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    m1 = smf.ols(BASE, data=train).fit(cov_type="HC3")
    m2 = smf.ols(BASE + " + " + " + ".join(PHOTO), data=train).fit(cov_type="HC3")
    # Robust Wald F-test that all three photo coefficients are zero
    w = m2.wald_test(" = 0, ".join(PHOTO) + " = 0", use_f=True, scalar=True)
    y = test["log_price"].values
    r1, r2_ = r2(y, m1.predict(test).values), r2(y, m2.predict(test).values)
    return {"sample": label, "n_train": len(train), "n_test": len(test),
            "F": round(float(w.statistic), 3), "F_p": round(float(w.pvalue), 5),
            "test_R2_M1": round(r1, 4), "test_R2_M2": round(r2_, 4), "delta_R2": round(r2_ - r1, 4),
            **{f"coef_{p}": round(m2.params[p], 4) for p in PHOTO},
            **{f"p_{p}": round(m2.pvalues[p], 4) for p in PHOTO}}


# ---- 3. Compare all listings vs. without flagged listings ----
results = pd.DataFrame([
    run(df, "All listings"),
    run(df[~df["id"].isin(flagged_ids)], f"Excluding {len(flagged_ids)} flagged listings"),
])
results.to_csv(TABLES / "photo_review_robustness.csv", index=False)
print(results.T.to_string(header=False))
