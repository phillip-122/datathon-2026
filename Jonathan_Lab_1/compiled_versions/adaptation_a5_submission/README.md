# Adaptation A5 — recommended Part 2 files

Upload these two matching files to the adaptation form:

- `submission.csv`
- `submission_entry.zip`

Continues Jonathan's A4 adaptation champion from repository commit `957f082`.
Full validation: **12.360656 / 20**, compared with A4 **11.764303 / 20**.
Audit episodes: **12.314882**, compared with A4 **11.708280**.
Tuning: **12.433979**. The hidden test score is unknown.

The update blends A4's win probability 50/50 with a machine-local Bayesian
temporal estimate. The estimator averages binary single-lag and two-lag
conditional models for delays 1–8, using marginal-likelihood evidence,
64-tick recency weighting, strength 4, and temperature 2. The probabilities
of positive payout sizes conditional on winning remain those from A4.
Every history input is strictly before the prediction cutoff; unavailable
ticks remain missing. Other episodes never contribute local evidence.

Selection compares candidates on tuning episodes. Residual correction and
structural projection candidates use three condition-balanced episode folds.
Local history estimators have no fitted validation coefficients and are
selected directly on tuning episodes. Audit episodes do not select this
update. Jonathan's underlying A4 was previously tuned on public validation,
so these audit results are not an entirely untouched estimate of test quality.

The ZIP opens directly to `README.md`, `reproduce.json`, `predict.py`, pinned
`requirements.txt`, model configuration and frozen A4 source/weights. It does
not contain forecast lookup tables or target files. Install requirements
before offline inference, then run from the extracted directory:

```powershell
python predict.py --data /absolute/path/to/adaptation/public --out /absolute/path/to/submission.csv
```

The official adaptation checker validates all 1,536 test queries. Inference
from a fresh ZIP extraction exactly matches the submitted CSV; hashes and
comparison results are recorded in `MANIFEST.json`. Two regression tests
cover future-history exclusion, episode isolation, missing observations and
probability normalization.

Training/selection from repository root:

```powershell
.venv/Scripts/python.exe Jonathan_Lab_1/research/adaptation_a5/predict.py --train --data Jonathan_Lab_1/adaptation-participants/public --out Jonathan_Lab_1/compiled_versions/adaptation_a5_new_run --archive Jonathan_Lab_1/compiled_versions/adaptation_active_submission/submission_entry.zip
```

Use a fresh output folder. `report.json` preserves every comparison and fold.
The original A4 package remains available. No entry has been uploaded.
