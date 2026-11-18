"""
05_fit_models.py

PURPOSE
    Fit three nested OLS models of log nightly price on the TRAINING rows only,
    with HC3 heteroskedasticity-robust standard errors (statsmodels formula API):
        M1 baseline:     room type + bedrooms + bathrooms + shared_bath + accommodates
                         + borough + min30 + room type x bedrooms
        M2 photo model:  M1 + the three standardized photo scores
        M3 interaction:  M2 + each standardized photo score x borough
    Dummy coding (see 04_build_analysis_data.py):
        private_room                (baseline Entire home/apt)
        brooklyn, queens, bronx_si  (baseline Manhattan)
    This script only FITS and SAVES the models. It runs no hypothesis tests,
    test-set evaluation or interpretation; that is a later step.

INPUTS
    DATA/final/analysis_data.csv

OUTPUTS
    OUTPUT/models/m1.pickle, m2.pickle, m3.pickle   fitted statsmodels results (load with
                                                     statsmodels.iolib.smpickle.load_pickle)
    OUTPUT/models/model_specs.md                     formulas, number of parameters, training N

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/05_fit_models.py

APPROXIMATE RUNTIME
    A few seconds.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle

# ---------------------------------------------------------------------------
# Configuration: relative paths, seed 42 (OLS is deterministic, set for consistency).
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_CSV = Path("DATA/final/analysis_data.csv")
MODEL_DIR = Path("OUTPUT/models")

# ---------------------------------------------------------------------------
# Model formulas, built up so that the nesting M1 within M2 within M3 is explicit.
# ---------------------------------------------------------------------------
BOROUGH = ["brooklyn", "queens", "bronx_si"]
PHOTO_Z = ["z_musiq_koniq", "z_musiq_ava", "z_clipiqa"]

M1_RHS = ("private_room + bedrooms + bathrooms + shared_bath + accommodates"
          " + brooklyn + queens + bronx_si + min30 + private_room:bedrooms")
M2_RHS = M1_RHS + " + " + " + ".join(PHOTO_Z)
M3_RHS = M2_RHS + " + " + " + ".join(f"{z}:{b}" for z in PHOTO_Z for b in BOROUGH)

FORMULAS = {
    "m1": "log_price ~ " + M1_RHS,
    "m2": "log_price ~ " + M2_RHS,
    "m3": "log_price ~ " + M3_RHS,
}
LABELS = {"m1": "M1 baseline", "m2": "M2 photo model", "m3": "M3 interaction model"}

# Every model variable must be present and non-missing in the training rows.
MODEL_VARS = ["log_price", "private_room", "bedrooms", "bathrooms", "shared_bath",
              "accommodates", "min30"] + BOROUGH + PHOTO_Z


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data and keep TRAINING rows only.
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {DATA_CSV}: {len(df):,} rows")
    train = df[df["split"] == "train"].copy()
    print(f"After keeping training rows only: {len(train):,} rows")
    n_missing = train[MODEL_VARS].isna().sum()
    if n_missing.any():
        raise SystemExit(f"Missing values in model variables:\n{n_missing[n_missing > 0]}")

    # -----------------------------------------------------------------------
    # Fit each model with HC3 robust SEs, watching for warnings and checking
    # that the design matrix has full column rank (no perfect collinearity).
    # -----------------------------------------------------------------------
    spec_rows = []
    for key, formula in FORMULAS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = smf.ols(formula, data=train).fit(cov_type="HC3")
        X = res.model.exog
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            raise SystemExit(f"{key}: design matrix is rank deficient ({rank} < {X.shape[1]})")
        for w in caught:
            print(f"  WARNING while fitting {key}: {w.category.__name__}: {w.message}")
        if len(res.params) != X.shape[1] or int(res.nobs) != len(train):
            raise SystemExit(f"{key}: unexpected parameter count or N")

        # Save, then load back to confirm the pickle works.
        path = MODEL_DIR / f"{key}.pickle"
        res.save(str(path))
        reloaded = load_pickle(str(path))
        assert np.allclose(reloaded.params.values, res.params.values), f"{key}: reload mismatch"

        print(f"{LABELS[key]}: {len(res.params)} parameters, N = {int(res.nobs)}, "
              f"condition number = {np.linalg.cond(X):,.0f}, warnings = {len(caught)}; saved {path}")
        spec_rows.append({"key": key, "label": LABELS[key], "formula": formula,
                          "n_params": len(res.params), "nobs": int(res.nobs),
                          "terms": list(res.params.index)})

    # -----------------------------------------------------------------------
    # Write model_specs.md describing each model.
    # -----------------------------------------------------------------------
    lines = ["# Model specifications", "",
             "OLS (statsmodels formula API), HC3 heteroskedasticity-robust standard errors,",
             f"fit on the training rows of `{DATA_CSV.as_posix()}` only (`split == \"train\"`).", "",
             "Baselines: room type = Entire home/apt; borough = Manhattan "
             "(`bronx_si` = Bronx + Staten Island combined). Photo scores are standardized "
             "with the training-set mean and SD.", ""]
    for r in spec_rows:
        lines += [f"## {r['label']} (`{r['key']}.pickle`)", "",
                  "```", r["formula"], "```", "",
                  f"- Number of parameters (incl. intercept): {r['n_params']}",
                  f"- Training rows used: {r['nobs']}",
                  f"- Terms: {', '.join(f'`{t}`' for t in r['terms'])}", ""]
    (MODEL_DIR / "model_specs.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {MODEL_DIR / 'model_specs.md'}")


if __name__ == "__main__":
    main()
