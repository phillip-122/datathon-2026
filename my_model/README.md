# House Edge XGBoost model

Python 3.11 or newer. Install `requirements.txt` before an offline run.
From the original kit root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r my_model/requirements.txt
.\.venv\Scripts\python.exe -m my_model.train --data public --out house-edge-output
```

Training reads released training histories and labels only. Every fourth observed
historical tick from tick 32 onward also becomes a training example, using strictly
earlier observations from the target and neighboring machines. Machine type and
signal are treated as fixed query context. Missing observations remain NaN.
Each machine contributes equal total training weight. IDs and condition labels
are excluded from model features. Features are lag 1â€“8 payout indicators, rolling
payout frequencies and counts, and earlier neighboring-machine frequencies.

Three tree depths (2, 3, 4), each trained through 1500 rounds with checkpoints every 100 rounds, are compared with fixed blend
weights against pooled_recent on the organizer's tuning episodes. A zero tree
weight permits selection of the baseline if XGBoost does not help. Audit episodes
are scored only after selection; repeating experiments after viewing audit results
means this audit is no longer an untouched check. Test outcomes are never used.
Settings, package versions, hashes and scores are in training-report.json.

Prediction from the original kit:

```powershell
.\.venv\Scripts\python.exe -m my_model.predict --model-dir house-edge-output --data public --out reproduced.csv
```

The generated entry.zip contains code, pinned dependencies and saved models.
In the extracted submission root, after dependencies have been prepared, run:

```text
python -m my_model.predict --model-dir . --data /data --out /output/submission.csv
```

`/data` must contain the test/queries.csv and test/history.csv files.
Upload submission.csv separately from entry.zip. Local scripts do not submit.
Prediction uses two CPU threads and requires no network. Saved-model reload and
test CSV differences are verified to be at most 1e-8 before packaging.

Control the search with --max-rounds 1500 --checkpoint-every 100. Each trajectory is trained once; exact tree prefixes are scored on tuning episodes. The selected prefix, not the final trajectory, is saved. Checkpoint scores are in training-report.json.

Depth search defaults to 2 through 10. Override with --depths 2 3 4. The best depth, round checkpoint and blend are selected jointly on tuning episodes. Deeper trees can use substantially more time and memory.

Transition features: lags 1–8 match 0, 1, or 2+ payout categories within the same machine. For trailing 32 ticks, 128 ticks, and all earlier history, features contain match counts and five outcome probabilities, smoothed with strength 8 toward the window payout distribution (itself smoothed with 0.5 per class). Missing current lag yields zero support and unknown probabilities. All transition outcome ticks precede the example cutoff. This adds 144 features and requires retraining; old saved models must use their packaged feature code.
