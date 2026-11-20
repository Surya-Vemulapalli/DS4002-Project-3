"""
03_build_analysis_data.py

PURPOSE
    Build the modelling dataset:
      1. Merge the sample with the photo scores; drop listings missing any score.
      2. Drop listings missing bedrooms or bathrooms_text; parse bathrooms_text
         into a numeric `bathrooms` and a `shared_bath` flag.
      3. Convert price text ("$1,234.00") to a number and create log_price.
      4. Create dummy variables:
           private_room            (baseline: Entire home/apt)
           brooklyn, queens, bronx_si  (baseline: Manhattan; bronx_si = Bronx + Staten Island)
           min30                   minimum_nights >= 30
      5. Random 80/20 train/test split (column `split`).
      6. Standardize the three photo scores using TRAINING-set mean and SD only
         (z_musiq_koniq, z_musiq_ava, z_clipiqa); raw scores are kept.

INPUTS
    DATA/intermediate/sample_listings.csv
    DATA/intermediate/photo_scores.csv

OUTPUTS
    DATA/final/analysis_data.csv
    OUTPUT/tables/standardization_params.csv   training mean/SD used for each score

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/03_build_analysis_data.py

APPROXIMATE RUNTIME
    A few seconds.
"""

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration: relative paths, seed 42.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SAMPLE_CSV = Path("DATA/intermediate/sample_listings.csv")
SCORES_CSV = Path("DATA/intermediate/photo_scores.csv")
OUT_CSV = Path("DATA/final/analysis_data.csv")
STD_CSV = Path("OUTPUT/tables/standardization_params.csv")

SCORES = ["musiq_koniq", "musiq_ava", "clipiqa"]
TRAIN_FRAC = 0.8
HALF_BATH_LABELS = {"half-bath", "shared half-bath", "private half-bath"}


def parse_bathrooms(text: str) -> float:
    """'1.5 shared baths' -> 1.5; any half-bath label -> 0.5. Raises on anything unexpected."""
    t = str(text).strip().lower()
    if t in HALF_BATH_LABELS:
        return 0.5
    m = re.match(r"^(\d+(?:\.\d+)?)\s+(?:shared\s+|private\s+)?baths?$", t)
    if not m:
        raise ValueError(f"Unrecognised bathrooms_text: {text!r}")
    return float(m.group(1))


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    STD_CSV.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Merge sample with photo scores; drop listings missing any score.
    # -----------------------------------------------------------------------
    sample = pd.read_csv(SAMPLE_CSV, low_memory=False)
    scores = pd.read_csv(SCORES_CSV).drop_duplicates("id", keep="last")
    print(f"Sample: {len(sample):,} rows; photo score rows: {len(scores):,}")
    df = sample.merge(scores, on="id", how="left", validate="one_to_one")
    print(f"After merging sample with photo scores: {len(df):,} rows")
    df = df.dropna(subset=SCORES)
    print(f"After dropping listings with any missing photo score: {len(df):,} rows")

    # -----------------------------------------------------------------------
    # 2. Bedrooms / bathrooms: drop missing, parse bathrooms_text.
    # -----------------------------------------------------------------------
    df = df.dropna(subset=["bedrooms", "bathrooms_text"])
    print(f"After dropping listings missing bedrooms or bathrooms_text: {len(df):,} rows")
    df["bathrooms"] = df["bathrooms_text"].map(parse_bathrooms)
    df["shared_bath"] = df["bathrooms_text"].str.lower().str.contains("shared").astype(int)

    # -----------------------------------------------------------------------
    # 3. Price: text -> number, then natural log.
    # -----------------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(df["price"]):
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[$,]", "", regex=True))
    df = df[df["price"] > 0]
    print(f"After requiring price > 0 (needed for the log): {len(df):,} rows")
    df["log_price"] = np.log(df["price"])

    # -----------------------------------------------------------------------
    # 4. Dummy variables.
    # -----------------------------------------------------------------------
    df["private_room"] = (df["room_type"] == "Private room").astype(int)          # baseline Entire home/apt
    b = df["neighbourhood_group_cleansed"]
    df["borough_group"] = np.where(b.isin(["Bronx", "Staten Island"]), "Bronx/Staten Island", b)
    df["brooklyn"] = (df["borough_group"] == "Brooklyn").astype(int)              # baseline Manhattan
    df["queens"] = (df["borough_group"] == "Queens").astype(int)
    df["bronx_si"] = (df["borough_group"] == "Bronx/Staten Island").astype(int)
    df["min30"] = (df["minimum_nights"] >= 30).astype(int)

    # -----------------------------------------------------------------------
    # 5. Random 80/20 train/test split (seeded permutation of rows).
    # -----------------------------------------------------------------------
    df = df.sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    n_train = int(round(TRAIN_FRAC * len(df)))
    train_idx = rng.permutation(len(df))[:n_train]
    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"
    print(f"Split: {(df['split'] == 'train').sum():,} train / {(df['split'] == 'test').sum():,} test")

    # -----------------------------------------------------------------------
    # 6. Standardize photo scores with TRAINING mean/SD only (SD with ddof=1),
    #    and apply the same transform to the test rows.
    # -----------------------------------------------------------------------
    train = df["split"] == "train"
    params = []
    for s in SCORES:
        mu, sd = df.loc[train, s].mean(), df.loc[train, s].std(ddof=1)
        df[f"z_{s}"] = (df[s] - mu) / sd
        params.append({"score": s, "train_mean": mu, "train_sd": sd})
    pd.DataFrame(params).to_csv(STD_CSV, index=False)

    # -----------------------------------------------------------------------
    # 7. Save. Key analysis columns first, then the remaining listing columns.
    # -----------------------------------------------------------------------
    key = ["id", "host_id", "split", "price", "log_price", "room_type", "private_room",
           "bedrooms", "bathrooms_text", "bathrooms", "shared_bath", "accommodates",
           "neighbourhood_group_cleansed", "borough_group", "brooklyn", "queens", "bronx_si",
           "minimum_nights", "min30"] + SCORES + [f"z_{s}" for s in SCORES]
    df = df[key + [c for c in df.columns if c not in key]]
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} ({len(df):,} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
