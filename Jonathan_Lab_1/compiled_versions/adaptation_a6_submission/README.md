# Adaptation A6 — Jonathan A4 plus Bayesian temporal residual correction

Upload `submission.csv` and `submission_entry.zip` together for Part 2.

Full public validation: **16.634796 / 20** (includes correction-fitting episodes).
Audit episodes: **15.637562 / 20**. Cross-fitted tuning selection: **15.050522 / 20**.
Previous A5: 12.360656 full / 12.314882 audit. Original Jonathan A4: 11.764303 full.
Public validation has been inspected repeatedly, including while developing A4.
The audit score is not an untouched estimate of test quality. Hidden test score
is unknown; the 16.63 figure must not be described as fully held out.

The predictor first runs Jonathan's frozen A4 adaptation model. A bank of 36
Bayesian local temporal experts uses binary and three-category payout contexts
at single and paired lags 1–8, half-lives 24/64/infinite, prior strengths 1/4/16
and evidence temperatures 1/2. Missing ticks remain unobserved. All inputs are
strictly earlier than each query cutoff. History summaries include changes in
local and peer payout rates and lag polarity. Other episodes are isolated.

A HistGradientBoostingRegressor learns A4's p_0 residual from those features:
150 iterations, 7 leaves, minimum leaf size 25, L2 10, learning rate .04,
early stopping disabled, random seed 42. It is fitted only on tuning episodes.
Three condition-balanced episode folds select the correction; condition-normalized
reference errors weight its training loss. Winning probabilities are clipped to
[.001,.999]; conditional positive payout proportions are retained from A4.
Audit targets never fit the new correction. No target files, condition labels,
training process, forecast lookup table, internet or API is needed at inference.

Install pinned requirements before the offline run. From the extracted ZIP:

```
python predict.py --data /absolute/path/to/adaptation/public --out /absolute/path/to/submission.csv
```

`--data` also accepts the direct folder with queries.csv and history.csv.
The ZIP has root README.md/reproduce.json and contains all fitted model weights.
No data files or secrets are packaged. Use the current adaptation kit's checker;
the old main kit's checker rejects the new 80/20 metadata.

Training and model selection from the repository root (use a fresh output folder):

```
.venv/Scripts/python.exe Jonathan_Lab_1/research/adaptation_a6/run.py --data adaptation/kit/public --out Phillip_Lab/runs/adaptation_a6_new
```

The organizer adaptation data is also at Jonathan_Lab_1/adaptation-participants/public.
Model comparisons and limitations are in report.json. Source and artifact hashes,
format checks and clean-extraction reproduction results are in MANIFEST.json
beside the upload files. Nothing is submitted automatically.
