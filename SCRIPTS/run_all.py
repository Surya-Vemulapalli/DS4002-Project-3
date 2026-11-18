"""
run_all.py

PURPOSE
    Run the preprocessing and model-fitting pipeline in order:
        02_sample_and_download_photos.py -> 03_score_photos.py
        -> 04_build_analysis_data.py -> 05_fit_models.py
    Stops at the first script that fails.

OPTIONS
    --skip-photos   Skip 02-04 and fit models from the existing
                    DATA/final/analysis_data.csv (runs 05 only).
    --device        auto | xpu | cpu, passed to 03_score_photos.py (default auto).

INPUTS / OUTPUTS
    See the header of each script.

HOW TO RUN (from the repo root)
    .venv\\Scripts\\python.exe SCRIPTS/run_all.py [--skip-photos] [--device auto|xpu|cpu]

APPROXIMATE RUNTIME
    Full run ~10-25 min on the Intel iGPU (30-90+ min on CPU); --skip-photos takes seconds.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path("SCRIPTS")


def run(script: str, extra: list[str] | None = None) -> None:
    """Run one pipeline script with the current interpreter; exit if it fails."""
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra or [])
    print(f"\n=== Running {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"{script} failed with exit code {result.returncode}")
    print(f"=== {script} finished in {time.time() - t0:.1f}s ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline scripts 02-05 in order.")
    parser.add_argument("--skip-photos", action="store_true",
                        help="skip 02-04 and fit models from the existing DATA/final/analysis_data.csv")
    parser.add_argument("--device", choices=["auto", "xpu", "cpu"], default="auto",
                        help="device for photo scoring in 03 (default auto)")
    args = parser.parse_args()

    if args.skip_photos:
        if not Path("DATA/final/analysis_data.csv").exists():
            sys.exit("--skip-photos needs DATA/final/analysis_data.csv; run the full pipeline first.")
    else:
        run("02_sample_and_download_photos.py")
        run("03_score_photos.py", ["--device", args.device])
        run("04_build_analysis_data.py")
    run("05_fit_models.py")


if __name__ == "__main__":
    main()
