"""
01_sample_and_download_photos.py

PURPOSE
    Draw the analysis sample of NYC Airbnb listings and download each listing's
    main photo.
      1. Keep only "Entire home/apt" and "Private room" listings.
      2. Keep only listings with a price, bedrooms, bathrooms_text and
         minimum_nights (filtered BEFORE sampling so the final N stays near 2,000).
      3. Keep one randomly chosen listing per host_id.
      4. Stratified random sample of 2,000 listings: 500 from each of the four
         borough groups Manhattan, Brooklyn, Queens and Bronx/Staten Island
         (Staten Island is too small to stand alone).
      5. Download each main photo (resized 720px-wide version, falling back to
         the original URL), drop listings whose photo fails, and record a
         comparison of dropped vs kept listings.

INPUTS
    DATA/raw/listings.csv                       Inside Airbnb NYC, Sept 2026 scrape (see DATA/README.md)

OUTPUTS
    DATA/photos/<id>.jpg                        downloaded photos (git-ignored; copyrighted)
    DATA/intermediate/sample_listings.csv       final sample, personal host fields removed
    OUTPUT/tables/dropped_photos.csv            ids whose photo failed + reason
    OUTPUT/tables/dropped_photo_check.csv       dropped vs kept comparison
    OUTPUT/tables/runtime_log.csv               (appended) runtime of this step

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/01_sample_and_download_photos.py

APPROXIMATE RUNTIME
    ~3-6 minutes for the first run (network-bound); seconds on rerun because
    already-downloaded photos are skipped.
"""

import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration: paths are relative to the repo root; seed 42 everywhere.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT_CSV = Path("DATA/raw/listings.csv")
PHOTO_DIR = Path("DATA/photos")
SAMPLE_CSV = Path("DATA/intermediate/sample_listings.csv")
TABLE_DIR = Path("OUTPUT/tables")
DROPPED_CSV = TABLE_DIR / "dropped_photos.csv"
DROPPED_CHECK_CSV = TABLE_DIR / "dropped_photo_check.csv"
RUNTIME_LOG = TABLE_DIR / "runtime_log.csv"

KEEP_ROOM_TYPES = ["Entire home/apt", "Private room"]
REQUIRED_NONMISSING = ["price", "bedrooms", "bathrooms_text", "minimum_nights"]
PER_GROUP = 500  # 4 borough groups x 500 = 2,000 listings

N_WORKERS = 8
TIMEOUT_S = 20
N_RETRIES = 2  # retries after the first attempt, per URL
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research image fetcher)"}

# Personal host fields that must never be written to disk.
# host_profile_id is also removed because it identifies the host's profile page;
# host_id is kept as required.
PERSONAL_HOST_FIELDS = [
    "host_name", "host_about",
    "host_thumbnail_url", "host_picture_url",
    "host_url", "host_profile_url", "host_profile_id",
]


def borough_group(b: str) -> str:
    """Map a borough to the 4-level sampling/model group (Bronx + Staten Island combined)."""
    return "Bronx/Staten Island" if b in ("Bronx", "Staten Island") else b


def resized_url(url: str) -> str:
    """Convert .../pictures/... to the resized .../im/pictures/...?im_w=720 form."""
    base = url.split("?")[0]
    if "/im/pictures/" not in base:
        base = base.replace("/pictures/", "/im/pictures/", 1)
    return base + "?im_w=720"


def fetch(url: str) -> bytes:
    """GET a URL with up to N_RETRIES retries; returns bytes that PIL can open, else raises."""
    last_err = None
    for attempt in range(N_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
            r.raise_for_status()
            # Make sure the payload really is a readable image before accepting it.
            with Image.open(io.BytesIO(r.content)) as im:
                im.load()
            return r.content
        except Exception as e:  # network error, HTTP error or undecodable image
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def image_ok(path: Path) -> bool:
    """True if the file exists and PIL can fully decode it."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def download_one(listing_id: int, url: str) -> tuple[int, str]:
    """Download one listing photo. Returns (id, status), where status is 'ok', 'skipped' or an error message."""
    out = PHOTO_DIR / f"{listing_id}.jpg"
    if image_ok(out):
        return listing_id, "skipped"  # already downloaded on a previous run
    errors = []
    # Try the resized URL first, then fall back to the original URL.
    for u in (resized_url(url), url):
        try:
            content = fetch(u)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(content)
            tmp.replace(out)  # atomic rename so an interrupted write never leaves a half file
            return listing_id, "ok"
        except Exception as e:
            errors.append(str(e)[:200])
    return listing_id, "failed: " + " | ".join(errors)


def main() -> None:
    t0 = time.time()
    for d in (PHOTO_DIR, SAMPLE_CSV.parent, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load the listings and apply the row filters, printing counts each step.
    # -----------------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {INPUT_CSV}: {len(df):,} rows")

    df = df[df["room_type"].isin(KEEP_ROOM_TYPES)]
    print(f"After keeping room_type in {KEEP_ROOM_TYPES}: {len(df):,} rows")

    df = df.dropna(subset=REQUIRED_NONMISSING)
    print(f"After dropping rows missing any of {REQUIRED_NONMISSING}: {len(df):,} rows")

    # -----------------------------------------------------------------------
    # One random listing per host: shuffle with the seed, then keep the first
    # row for each host_id.
    # -----------------------------------------------------------------------
    df = df.sample(frac=1, random_state=SEED).drop_duplicates("host_id", keep="first")
    print(f"After keeping one random listing per host_id: {len(df):,} rows")

    # -----------------------------------------------------------------------
    # Stratified sample: 500 listings from each of the four borough groups.
    # -----------------------------------------------------------------------
    df["borough_group"] = df["neighbourhood_group_cleansed"].map(borough_group)
    print("Eligible listings per borough group:")
    print(df["borough_group"].value_counts().to_string())
    # Groups are processed in sorted order so the draw is reproducible.
    sample = (
        pd.concat([df[df["borough_group"] == g].sample(n=PER_GROUP, random_state=SEED)
                   for g in sorted(df["borough_group"].unique())])
        .sort_values("id")
        .reset_index(drop=True)
    )
    print(f"After stratified sampling ({PER_GROUP} per borough group): {len(sample):,} rows")
    print(sample["neighbourhood_group_cleansed"].value_counts().to_string())

    # Remove personal host fields right away so they never reach disk.
    sample = sample.drop(columns=[c for c in PERSONAL_HOST_FIELDS if c in sample.columns])

    # -----------------------------------------------------------------------
    # Download photos in parallel (8 threads), skipping ones already on disk.
    # -----------------------------------------------------------------------
    t_dl = time.time()
    statuses = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(download_one, int(i), u) for i, u in zip(sample["id"], sample["picture_url"])]
        for k, fut in enumerate(as_completed(futures), 1):
            lid, status = fut.result()
            statuses[lid] = status
            if k % 200 == 0 or k == len(futures):
                print(f"  photos processed: {k}/{len(futures)}")
    dl_seconds = time.time() - t_dl
    n_ok = sum(s == "ok" for s in statuses.values())
    n_skip = sum(s == "skipped" for s in statuses.values())
    print(f"Download finished in {dl_seconds:.1f}s: {n_ok} downloaded, {n_skip} already present, "
          f"{len(statuses) - n_ok - n_skip} failed")

    # -----------------------------------------------------------------------
    # Drop listings whose photo failed or cannot be opened, and record them.
    # -----------------------------------------------------------------------
    sample["photo_status"] = sample["id"].map(statuses)
    sample["photo_ok"] = [image_ok(PHOTO_DIR / f"{i}.jpg") for i in sample["id"]]
    dropped = sample[~sample["photo_ok"]]
    kept = sample[sample["photo_ok"]]
    pd.DataFrame({"id": dropped["id"], "reason": dropped["photo_status"]}).to_csv(DROPPED_CSV, index=False)
    print(f"After dropping listings whose photo failed: {len(kept):,} rows ({len(dropped)} dropped)")

    # Compare dropped vs kept on log price, borough shares and room type shares.
    def summarize(d: pd.DataFrame, label: str) -> dict:
        lp = np.log(pd.to_numeric(d["price"].astype(str).str.replace(r"[$,]", "", regex=True)))
        row = {"group": label, "n": len(d),
               "mean_log_price": lp.mean() if len(d) else np.nan,
               "median_log_price": lp.median() if len(d) else np.nan}
        for b in sorted(sample["neighbourhood_group_cleansed"].unique()):
            row[f"share_{b}"] = (d["neighbourhood_group_cleansed"] == b).mean() if len(d) else np.nan
        for r in KEEP_ROOM_TYPES:
            row[f"share_{r}"] = (d["room_type"] == r).mean() if len(d) else np.nan
        return row

    check = pd.DataFrame([summarize(dropped, "dropped"), summarize(kept, "kept")])
    check.to_csv(DROPPED_CHECK_CSV, index=False)
    print("Dropped vs kept check:")
    print(check.T.to_string())

    # -----------------------------------------------------------------------
    # Save the sample (personal host fields already removed; host_id kept).
    # -----------------------------------------------------------------------
    kept = kept.drop(columns=["photo_status", "photo_ok"])
    kept.to_csv(SAMPLE_CSV, index=False)
    print(f"Saved {SAMPLE_CSV} ({len(kept):,} rows)")

    # Append runtime to a shared log so the pipeline timing can be reported.
    log = pd.DataFrame([{"step": "01_download", "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                         "device": "n/a", "n_items": len(sample), "n_new": n_ok,
                         "download_seconds": round(dl_seconds, 1), "total_seconds": round(time.time() - t0, 1),
                         "scoring_seconds": np.nan}])
    log.to_csv(RUNTIME_LOG, mode="a", header=not RUNTIME_LOG.exists(), index=False)


if __name__ == "__main__":
    main()
