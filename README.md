# DS4002 Project 3: Airbnb Photo Quality and Nightly Price in NYC

This project tests whether the quality of an Airbnb listing's main photo is related to its nightly price in New York City. A random sample of about 2,000 listings from the Inside Airbnb NYC scrape (September 2026) is used. Each listing's main photo is scored by three image-quality models (MUSIQ-KonIQ, MUSIQ-AVA and CLIP-IQA), and nested regression models predict log nightly price from listing details with and without the photo scores.

## Contents of this repository

| Folder / file | What it holds |
|---|---|
| `listings.csv` | Raw input data: Inside Airbnb NYC listings, September 2026 scrape (29,918 listings). |
| `SCRIPTS/` | All code, numbered in the order it runs (`01_` to `07_`), plus `run_all.py`, which runs every script except the notebook (06) in one go. |
| `DATA/` | Data created by the scripts: the listing sample, the photo scores and the final analysis dataset. The downloaded photos also go here, but they are **not** in the repository (see below). |
| `OUTPUT/` | Results: fitted models, tables and figures. |
| `requirements.txt` | Python packages, with the exact versions used. |
| `LICENSE` | License for this repository. |

**Photos are not included.** Listing photos are copyrighted by their owners, so `DATA/photos/` is listed in `.gitignore`. Script 01 downloads them again from Airbnb's servers.

## 1. Software and platform

**Software**
- **Python 3.11.** Python 3.14 is *not* supported, because the PyTorch build used does not support it yet.
- **Google Colab**, for the statistical-tests notebook (`06_statistical_tests.ipynb`). Any Jupyter environment with pandas, numpy and statsmodels also works.

**Add-on Python packages** (exact versions are in `requirements.txt`)
| Package | Used for |
|---|---|
| `pandas`, `numpy` | Data handling |
| `requests`, `pillow` | Downloading and opening photos |
| `torch`, `torchvision` | Deep-learning backend for the image-quality models |
| `pyiqa` | Image-quality models: MUSIQ-KonIQ, MUSIQ-AVA, CLIP-IQA |
| `statsmodels`, `patsy`, `scipy` | Regression models, robust standard errors, statistical tests |
| `matplotlib` | Figures |

**Platform**
- Windows 11, on a laptop with an Intel Core Ultra 9 288V CPU and Intel Arc 140V integrated GPU.
- The photo scoring ran on the integrated GPU through PyTorch's Intel GPU (`xpu`) build. Scoring also runs on CPU only, just more slowly, and works on macOS and Linux (see step 4 below).

## 2. Map of documentation

```
DS4002-Project-3/
├── README.md                          This file
├── LICENSE
├── .gitignore                         Files git should not track (photos, virtual environment)
├── requirements.txt                   Pinned Python package versions
├── listings.csv                       Raw Inside Airbnb NYC data (input)
│
├── SCRIPTS/
│   ├── 01_sample_and_download_photos.py   Filter listings, draw the sample, download photos
│   ├── 02_score_photos.py                 Score each photo with MUSIQ-KonIQ, MUSIQ-AVA, CLIP-IQA
│   ├── 03_build_analysis_data.py          Clean variables, train/test split, standardize scores
│   ├── 04_fit_models.py                   Fit regression models M1, M2, M3 on the training set
│   ├── 05_plot_photo_scores.py            Figures of the photo-score distributions
│   ├── 06_statistical_tests.ipynb         VIF, partial F-test, t-tests, test-set ΔR², bootstrap CI
│   ├── 07_evaluate_models.py              Test-set error (MAE) by model and borough; M3 photo effects by borough
│   └── run_all.py                         Runs scripts 01–05 and 07 in order
│
├── DATA/
│   ├── photos/                        Downloaded listing photos (NOT in the repo; created by script 01)
│   ├── intermediate/
│   │   ├── sample_listings.csv        The sampled listings (personal host fields removed)
│   │   └── photo_scores.csv           Raw photo scores: id, musiq_koniq, musiq_ava, clipiqa
│   └── final/
│       └── analysis_data.csv          Final dataset used for modelling (1,972 listings)
│
└── OUTPUT/
    ├── models/
    │   ├── m1.pickle                  M1 baseline model (listing details only)
    │   ├── m2.pickle                  M2 photo model (M1 + three photo scores)
    │   ├── m3.pickle                  M3 interaction model (M2 + photo score × borough)
    │   └── model_specs.md             Exact formula, number of parameters and training N for each model
    ├── tables/
    │   ├── dropped_photos.csv         Listings dropped because their photo could not be downloaded
    │   ├── dropped_photo_check.csv    Dropped vs kept listings: price, borough and room type
    │   ├── score_failures.csv         Photos that failed to score (none)
    │   ├── standardization_params.csv Training-set mean and SD used to standardize each score
    │   ├── runtime_log.csv            How long downloading and scoring took, and on which device
    │   ├── vif.csv                    Variance inflation factors (notebook 06)
    │   ├── f_test.csv                 Partial F-test, M2 vs M1 (notebook 06)
    │   ├── t_tests.csv                t-tests for each photo score in M2 (notebook 06)
    │   ├── hypothesis_test_summary.csv Summary of all test criteria (notebook 06)
    │   ├── bootstrap_samples.csv      1,000 bootstrap ΔR² values (notebook 06)
    │   ├── mae_model_comparison.csv   Test-set MAE, median error and R² by model, with bootstrap CIs (script 07)
    │   ├── mae_by_borough.csv         Test-set MAE for each model within each borough group (script 07)
    │   └── photo_effects_by_borough.csv Effect of each photo score on price in each borough, from M3 (script 07)
    └── figures/
        ├── photo_score_distributions.png  Histograms of the three photo scores (script 05)
        ├── photo_scores_by_borough.png    Photo scores by borough group (script 05)
        ├── mae_model_comparison.png       Bar chart of test-set MAE by model (script 07)
        ├── mae_by_borough.png             Test-set MAE by model and borough group (script 07)
        └── photo_effects_by_borough.png   M3 photo-score effects with 95% CIs, by borough (script 07)
```

## 3. Instructions for reproducing the results

There are two ways to reproduce the results:

- **Option A (fast, exact; about 5 minutes):** start from the committed `DATA/final/analysis_data.csv`. You skip the photo download and scoring and get exactly the same models and test results.
- **Option B (full rebuild; about 30–60 minutes):** start from `listings.csv`, download the photos again and score them yourself.

Option B may not match the committed results exactly. Airbnb hosts delete or replace photos over time, so a later download can lose a few more listings or get a different photo. Scores computed on CPU also differ from the GPU scores in about the 4th decimal place.

All commands are run in a terminal **from the top-level repository folder** (`DS4002-Project-3/`). All random steps use seed 42.

### Step 1: Get the repository

```bash
git clone https://github.com/Surya-Vemulapalli/DS4002-Project-3.git
cd DS4002-Project-3
```

### Step 2: Install Python 3.11

Download Python 3.11 from https://www.python.org/downloads/ and install it.
- On Windows, check that `py -3.11 --version` prints `Python 3.11.x`.
- On macOS/Linux, check that `python3.11 --version` does.

### Step 3: Create a virtual environment and install the packages

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux, or any computer without an Intel GPU.** The `+xpu` packages in `requirements.txt` only work with Intel GPUs, so install the standard CPU builds of PyTorch instead:
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.15.0 torchvision==0.29.2
.venv/bin/python -m pip install pyiqa==0.1.16 pandas==3.1.1 numpy==2.4.6 statsmodels==0.15.0 \
    patsy==1.0.3 scipy==1.17.1 requests==2.34.2 pillow==13.0.0 matplotlib==3.11.2
```

In the commands below, `PY` means your environment's Python:
- Windows: `.venv\Scripts\python.exe`
- macOS/Linux: `.venv/bin/python`

### Step 4: Run the pipeline

**Option A (from the committed analysis data):**
```bash
PY SCRIPTS/run_all.py --skip-photos
```
This takes under a minute and runs:
- script 04, which fits M1, M2 and M3 and saves them to `OUTPUT/models/`;
- script 05, which saves the two photo-score figures;
- script 07, which saves the test-set error and borough-effect tables and figures.

**Option B (full rebuild from `listings.csv`):**
```bash
PY SCRIPTS/run_all.py --device auto
```
Use `--device cpu` if you have no Intel GPU, or if the GPU option gives an error. This runs, in order:

1. **`01_sample_and_download_photos.py`** (about 2–5 minutes; needs internet)
   - Keeps "Entire home/apt" and "Private room" listings that have a price, bedrooms, bathrooms and minimum nights.
   - Keeps one random listing per host.
   - Draws 500 listings from each of four borough groups: Manhattan, Brooklyn, Queens, and Bronx + Staten Island combined.
   - Downloads each listing's main photo to `DATA/photos/`.
   - Drops listings whose photo can't be downloaded.
2. **`02_score_photos.py`** (about 20 minutes on the Intel GPU, about 50 minutes on CPU)
   - Scores each photo with the three models. On the first run, pyiqa also downloads the model weights (about 0.5 GB).
   - If it is interrupted, run it again: it resumes where it stopped.
3. **`03_build_analysis_data.py`** (seconds)
   - Builds the model variables and makes the 80/20 train/test split.
   - Standardizes the photo scores using the training set.
4. **`04_fit_models.py`** (seconds): fits M1, M2 and M3 on the training rows with HC3 robust standard errors.
5. **`05_plot_photo_scores.py`** (seconds): saves the photo-score figures.
6. **`07_evaluate_models.py`** (about 10 seconds)
   - Predicts prices for the 394 test listings with each model and measures the error in dollars (MAE).
   - Computes a 95% bootstrap CI for each model's improvement over M1.
   - Calculates how much each photo score moves price in each borough, using M3.

Each script prints its row count after every filtering step. Compare these with the expected values below.

### Step 5: Run the statistical tests (notebook 06)

1. Open https://colab.research.google.com and choose **File → Upload notebook**, then select `SCRIPTS/06_statistical_tests.ipynb`.
2. Click the folder icon on the left and upload `DATA/final/analysis_data.csv`.
3. Choose **Runtime → Run all**. This takes about a minute.
4. In the same Colab file panel, download the five CSV files the notebook creates: `vif.csv`, `f_test.csv`, `t_tests.csv`, `hypothesis_test_summary.csv` and `bootstrap_samples.csv`. Move them into `OUTPUT/tables/`.

To run the notebook locally instead, install Jupyter (`PY -m pip install jupyter`), copy `DATA/final/analysis_data.csv` into the `SCRIPTS/` folder, and run all cells. The CSVs are saved next to the notebook.

### Step 6: Check your results against these values

| Check | Expected value |
|---|---|
| Listings after filtering (script 01) | 18,767 |
| Listings after one per host (script 01) | 9,518 |
| Sampled listings (script 01) | 2,000 (500 per borough group) |
| Listings with a usable photo | 1,972 (28 dropped)*. Option B may differ slightly |
| Train / test rows | 1,578 / 394* |
| Parameters in M1 / M2 / M3 | 11 / 14 / 23 |
| Partial F-test, M2 vs M1 (notebook 06) | F(3, 1564) = 5.69, p ≈ 0.0007* |
| Test-set ΔR², M2 − M1 (notebook 06) | 0.0097* |
| Test-set MAE, M1 / M2 / M3 (script 07) | $93.11 / $91.63 / $92.62* |

\* Exact for Option A. For Option B these can shift a little if more photos have disappeared since the original download.
