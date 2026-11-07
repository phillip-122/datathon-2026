# House Edge — Main Round Submission

## Entry

- Team ID: Jonathan Kalsky
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv
- Forecast SHA-256: 2ae6e79b6a1834100fb2f1ac99c699321611436c769f2d68d43e07759f330756
- Code version: v1.0-hgb-production
- Environment: Python 3.11+ / 3.13 (`image: python`)

## Setup

- Python/runtime version: Python 3.11+ (tested on Python 3.13)
- OS tested: macOS / Linux (standard CPU)
- Dependencies: `numpy>=2.0.0`, `scikit-learn>=1.5.0`, `joblib>=1.4.0` (listed in `requirements.txt`)
- Required hardware: 2 CPU cores, 4 GiB RAM (well within 16 GiB limit, runs in < 2 seconds offline, no GPU required)
- Input data path: `/data` (expects `queries.csv` and `history.csv`, under `/data/test` or `/data`)

The runner mounts the current round's data at `/data`, code at `/submission` (read-only), and output at `/output`. Prediction runs strictly offline without network access.

## Recreate the forecast

Run from code directory:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Model files: Pre-fitted and saved in `./artifacts/`:
  - `artifacts/base_forecaster.json`: Context and pooled recency priors
  - `artifacts/hgb_model.joblib`: HistGradientBoostingClassifier trained on 21,200 multi-cutoff historical slices
  - `artifacts/meta.json`: Model hyperparameters and metadata
- Features: Cutoff-safe machine type/signal interactions, rolling win-rate windows (1-48 ticks), recency decay frequencies, 8-tick lag sequence interactions, win/loss streak dynamics, peer machine cross-gameplay features, and hierarchical priors.
- Runtime: ~1.5 seconds on CPU.
- Validation Score: 35.0743 / 60 Technical Points (Mean Loss Ratio: 0.7107)
  - stable: 0.6684
  - short_history: 0.7826
  - drift: 0.7588
  - shared_change: 0.6328

## Local check

```sh
python check_submission.py submission.csv
```
Result: VALID FORMAT: 1536 queries | main | test.
