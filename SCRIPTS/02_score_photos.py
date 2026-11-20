"""
02_score_photos.py

PURPOSE
    Score every sampled listing photo with three no-reference image-quality
    models from the pyiqa library:
        musiq_koniq  pyiqa.create_metric("musiq")      technical quality (KonIQ-10k)
        musiq_ava    pyiqa.create_metric("musiq-ava")  aesthetic quality (AVA)
        clipiqa      pyiqa.create_metric("clipiqa")    CLIP-based quality
    Images are passed at their downloaded size and aspect ratio; this script
    never resizes, stretches or crops them.

    Scores are appended to the output CSV one image at a time, so an
    interrupted run resumes where it stopped. Images that fail are logged.

DEVICE
    --device auto  (default) Intel iGPU ("xpu") if available, else CUDA if available, else CPU
    --device xpu   force the Intel integrated GPU (errors if unavailable)
    --device cpu   force the CPU
    If a model fails on the chosen device the script stops with an error
    instead of silently switching devices, so every score has a known device.

INPUTS
    DATA/intermediate/sample_listings.csv
    DATA/photos/<id>.jpg

OUTPUTS
    DATA/intermediate/photo_scores.csv     id, musiq_koniq, musiq_ava, clipiqa (appended incrementally)
    OUTPUT/tables/score_failures.csv       ids that failed to score + error (appended)
    OUTPUT/tables/runtime_log.csv          (appended) device and runtime of this step

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/02_score_photos.py [--device auto|xpu|cpu]

APPROXIMATE RUNTIME
    First run also downloads model weights (~0.5 GB, one time). Scoring ~2,000
    images takes roughly 5-15 min on the Intel Arc iGPU, or 30-90 min on CPU.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyiqa
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration: relative paths, seed 42.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMPLE_CSV = Path("DATA/intermediate/sample_listings.csv")
PHOTO_DIR = Path("DATA/photos")
SCORES_CSV = Path("DATA/intermediate/photo_scores.csv")
FAIL_CSV = Path("OUTPUT/tables/score_failures.csv")
RUNTIME_LOG = Path("OUTPUT/tables/runtime_log.csv")

# Output column name -> pyiqa metric name
METRICS = {"musiq_koniq": "musiq", "musiq_ava": "musiq-ava", "clipiqa": "clipiqa"}
SCORE_COLS = ["id"] + list(METRICS)


def pick_device(choice: str) -> torch.device:
    """Resolve the --device option to a torch.device."""
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    if choice == "xpu":
        if not xpu_ok:
            raise SystemExit("--device xpu requested but torch.xpu is not available "
                             "(is the PyTorch XPU build installed in .venv?)")
        return torch.device("xpu")
    if choice == "cpu":
        return torch.device("cpu")
    # auto: prefer the Intel iGPU, then CUDA, then CPU
    if xpu_ok:
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """Load an image as a 1x3xHxW float tensor in [0,1] at its native size (no resizing)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return TF.to_tensor(im).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "xpu", "cpu"], default="auto")
    args = parser.parse_args()

    t0 = time.time()
    device = pick_device(args.device)
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else str(device)
    print(f"Scoring device: {device} ({dev_name})")
    FAIL_CSV.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Work out which ids still need scores (resume support).
    # -----------------------------------------------------------------------
    ids = pd.read_csv(SAMPLE_CSV, usecols=["id"])["id"].astype("int64").tolist()
    print(f"Listings in sample: {len(ids):,}")
    done = set()
    if SCORES_CSV.exists():
        done = set(pd.read_csv(SCORES_CSV, usecols=["id"])["id"].astype("int64"))
    todo = [i for i in ids if i not in done]
    print(f"Already scored: {len(done):,}; remaining to score: {len(todo):,}")
    if not todo:
        return

    # -----------------------------------------------------------------------
    # Build the three pyiqa metrics on the chosen device (weights download
    # automatically on first use and are cached by pyiqa).
    # -----------------------------------------------------------------------
    models = {}
    for col, name in METRICS.items():
        try:
            models[col] = pyiqa.create_metric(name, device=device, as_loss=False)
            models[col].eval()
        except Exception as e:
            raise SystemExit(f"Could not create metric '{name}' on {device}: {e}")

    # -----------------------------------------------------------------------
    # Score each image and append its row immediately so progress is saved.
    # -----------------------------------------------------------------------
    new_file = not SCORES_CSV.exists()
    fail_new = not FAIL_CSV.exists()
    n_ok = n_fail = 0
    t_score = time.time()
    with open(SCORES_CSV, "a", newline="") as fs, open(FAIL_CSV, "a", newline="") as ff:
        ws, wf = csv.writer(fs), csv.writer(ff)
        if new_file:
            ws.writerow(SCORE_COLS)
        if fail_new:
            wf.writerow(["id", "device", "error"])
        for k, lid in enumerate(todo, 1):
            try:
                x = load_image(PHOTO_DIR / f"{lid}.jpg", device)
                with torch.no_grad():
                    row = [lid] + [float(models[c](x).item()) for c in METRICS]
                if not all(np.isfinite(row[1:])):
                    raise ValueError(f"non-finite score {row[1:]}")
                ws.writerow(row)
                fs.flush()
                n_ok += 1
            except Exception as e:
                # On the very first image a failure most likely means the device
                # cannot run a model; stop rather than log 2,000 identical errors.
                if n_ok == 0 and k == 1:
                    raise SystemExit(f"First image failed on {device}: {type(e).__name__}: {e}\n"
                                     f"Try --device cpu.")
                wf.writerow([lid, str(device), f"{type(e).__name__}: {e}"[:500]])
                ff.flush()
                n_fail += 1
                print(f"  FAILED id={lid}: {type(e).__name__}: {e}")
            if k % 100 == 0 or k == len(todo):
                rate = (time.time() - t_score) / k
                print(f"  scored {k}/{len(todo)}  ({rate:.2f}s/image, ~{rate * (len(todo) - k) / 60:.1f} min left)")

    score_seconds = time.time() - t_score
    print(f"Scoring finished in {score_seconds:.1f}s: {n_ok} scored, {n_fail} failed on {device}")

    # Append this run's timing to the shared runtime log.
    log = pd.DataFrame([{"step": "02_score", "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                         "device": f"{device} ({dev_name})", "n_items": len(todo), "n_new": n_ok,
                         "download_seconds": np.nan, "total_seconds": round(time.time() - t0, 1),
                         "scoring_seconds": round(score_seconds, 1)}])
    log.to_csv(RUNTIME_LOG, mode="a", header=not RUNTIME_LOG.exists(), index=False)


if __name__ == "__main__":
    main()
