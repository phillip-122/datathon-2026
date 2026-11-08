# V12 new system

An isolated V10 extension. No edits to teammate source or champion packages.

Train and benchmark from repository root:

```
.venv/Scripts/python.exe Phillip_Lab/lab.py train-v12 --data public --out Phillip_Lab/runs/v12_new_sys
```

The baseline archive defaults to the frozen v10_short_history_subpop_triage_solution
package. Its validation and test predictions are regenerated in isolated subprocesses.
Input probability constraints are estimated from released tuning true probability
targets. Three condition-balanced episode folds within tuning are used to select
method and blend weight: each fold estimates constraints only from its training
episodes. Audit and test targets are never fitted. Prior adaptive use of public
validation and baseline selection means audit is not a fresh unbiased test.

Candidates are: exact Euclidean projection onto the high-payout-ratio simplex;
projection onto the convex region compatible with two conditional 1x-share regimes;
and Bayesian soft regime inference with half-lives 24, 64 or infinite ticks.
Only nonzero historical payouts update the conditional regime likelihood. A 50/50
state prior is fixed. Probability mixtures retain state uncertainty. No hidden
generator source, private seed reconstruction, hard state routing or test labels.

The projection solves a convex quadratic problem by enumerating triangle faces.
For a truth vector in that region, projection cannot increase Euclidean error.
The conservation-style V11 map is included as a separate benchmark. Estimated
constraints need not hold on hidden episodes or the adaptation round.

Offline entry command, after installing root requirements.txt:

```
python -m v12_new_sys.run predict --model-dir . --data /data --out /output/submission.csv
```

The generated ZIP includes the frozen V10 package and selected V12 parameters.
Report includes exact official folds, component comparisons, hashes and paired
condition-stratified episode-bootstrap uncertainty (2000 resamples, seed 42).
That interval does not correct prior model-selection bias. Original inference uses
two numerical threads and runs without network access.

## First verified benchmark

Selected soft Bayesian payout-regime inference, 24-tick half-life, weight 0.75.
Full validation 38.789732; audit 38.888688; cross-fitted tuning 38.705632.
Saved V10: 38.500702 full, 38.604845 audit. Conservation-style V11 on saved V10:
38.747367 full, 38.874541 audit. Paired audit bootstrap gain versus saved V10:
[-0.012015, 0.576206] points. This does not establish a statistically decisive gain.

Provenance issue: the local frozen V10 ZIP regenerates 38.331653 full, not the
38.500702 saved CSV. Maximum probability discrepancy is 0.07270446. The active
unpacked entry generates the same lower result. V12 reports both benchmarks,
uses regenerated probabilities for all learning/selection/inference, and never
looks up validation/test answers. The archived ZIP also wraps files in entry/
and has a nonconforming reproduce.json; V12 normalizes this in its own output.
Original teammate files are untouched. Do not claim a clean same-baseline
incremental comparison until the saved V10/inference discrepancy is resolved.

Final output: Phillip_Lab/runs/v12_new_sys_confirmed. All 15 tests and official
CSV checks passed. Standalone packaged inference must match submission.csv.
No active champion promotion or Git push is performed automatically.
