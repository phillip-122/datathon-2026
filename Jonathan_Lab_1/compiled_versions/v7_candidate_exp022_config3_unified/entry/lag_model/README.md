> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# Changing lag relationship experiment

Run from repository root:

`.venv/Scripts/python.exe -m lag_model.run train --data public --out house-edge-output-changing-lags-02`

Reuses the frozen Jonathan V3 feature code and adds 128 features. For lags 1–8, condition on the category currently observed at that lag (0, 1, 2+). Compare the last 32 or 64 ticks with the older portion of the trailing 192 ticks. Each comparison includes recent/older match counts, smoothed conditional win rates, conditional-minus-marginal win rates (lifts), lift differences and lift products. A negative product indicates a reversal in the estimated association. Estimates shrink toward each window's marginal rate with strength 8. Missing current lags produce zero support and NaNs. Missing ticks are never substituted with prior observed rows.

All feature outcome ticks must be strictly before each prediction tick. The temporal feature test checks that changing future outcomes does not change the inputs; a synthetic reversal verifies signed lifts. Run `python -m unittest lag_model.test_features -v`.

Three HGB structures (40, 60, 80 leaves) train for 150 rounds, no random-row early stopping, seed 42/101/202, train histories and sampled labels only. Candidate probabilities are blended with unchanged champion at .25, .5, .75, 1. Tuning selects against a zero-weight champion fallback. Existing audit episodes are no longer fresh. Original teammate files are never modified.

Outputs include report.json, selection.json, validation_predictions.csv, submission.csv and entry.zip. Reproduce with `python -m lag_model.run predict --model-dir OUTPUT_DIRECTORY --data public --out reproduced.csv`.
