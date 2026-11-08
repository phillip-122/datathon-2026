# Phillip Lab — House Edge

All commands below run from the repository root (`datathon-2026`). Use Python 3.11+.

## Active submission

Upload these two matching files from `compiled_versions/active_champion_submission/`:

- `submission.csv` — all test-query probabilities.
- `submission_entry.zip` — frozen code, fitted models, dependencies and reproduction command.

Current model: Jonathan V3 + changing-lag specialist (25% specialist), followed by regularized probability calibration. Full validation: **37.481400/60**, audit: **37.476425/60**. The full score partly includes calibration fitting data; it is not entirely held out. Audit episodes have been repeatedly inspected. Test score remains hidden. Calibration penalty selection uses episode-cross-fitted tuning predictions.

## Layout

| Folder | Purpose |
|---|---|
| `src/` | Importable model families and tests; package names kept stable |
| `research/tools/` | Comparison and historical blend-building tools |
| `runs/` | Local experiment outputs and working copies; ignored by Git |
| `artifacts/neural_xgboost/` | Earlier neural/XGBoost model used by ensemble training |
| `compiled_versions/` | Frozen submission milestones, reports and SHA-256 manifests |
| `archive/starter_runs/` | Original starter baselines and predictions |
| `side_projects/casino/` | Separate manual casino experiment GUI |
| `lab.py` | Stable entry point that locates source code and shared organizer library |

The shared organizer code remains in root `slot_forecasting/` and data in root `public/`. Jonathan's folders are unchanged. Generated environments and Python caches are ignored.

## Setup and tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r Phillip_Lab/requirements.txt
.\.venv\Scripts\python.exe Phillip_Lab/lab.py test
```

## Reproduce the active submission without training

```powershell
.\.venv\Scripts\python.exe Phillip_Lab/lab.py predict --data public --out Phillip_Lab/runs/reproduced.csv
.\.venv\Scripts\python.exe check_submission.py Phillip_Lab/runs/reproduced.csv
```

The launcher extracts the immutable submission ZIP to a temporary folder and runs its declared command. It does not depend on local research runs.

## Train experiments

```powershell
.\.venv\Scripts\python.exe Phillip_Lab/lab.py train-calibration --data public --out Phillip_Lab/runs/calibration-new
.\.venv\Scripts\python.exe Phillip_Lab/lab.py train-lags --data public --out Phillip_Lab/runs/lags-new
.\.venv\Scripts\python.exe Phillip_Lab/lab.py train-ensemble --data public --out Phillip_Lab/runs/ensemble-new --epochs 100
.\.venv\Scripts\python.exe Phillip_Lab/lab.py train-xgboost --data public --out Phillip_Lab/runs/xgb-new --depths 4 6 8 10 12 --max-rounds 1500 --checkpoint-every 100
```

Other experiments: `train-hgb`, `train-win`, `train-structure`. An omitted `--out` generates a timestamped directory under `runs/`. Calibration defaults to the committed changing-lag milestone, so it works on a fresh clone. Every run requires a fresh directory. Individual package READMEs explain the methods; historical commands there may use the earlier layout. The commands above are the supported current interface.

## Compare validation CSVs

```powershell
.\.venv\Scripts\python.exe Phillip_Lab/lab.py compare --a Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv --b Phillip_Lab/compiled_versions/v3_calibrated/validation_predictions.csv
```

Results default to `Phillip_Lab/runs/comparisons/comparison.json`. Frozen package contents retain their original module names for offline reproduction. Training code uses module locations rather than assuming packages live at the repository root.
