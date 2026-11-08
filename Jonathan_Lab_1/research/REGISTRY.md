# House Edge Research Experiment Registry

**Current Champion:** `EXP-012`
**Last Updated:** `2026-11-07 18:20:47`

| Exp ID | Hypothesis | Author | Tuning Pts | Audit Pts | Full Pts | Mean Loss Ratio | Decision | Notes |
|---|---|---|---|---|---|---|---|---|
| BASE-001 | Official pooled starter reference baseline | organizers | 30.0000 | 30.0000 | 30.0000 | 1.0000 | **BASELINE** | Starter baseline reference |
| EXP-006 | HistGradientBoosting with 211 features, multi-cutoff step 6, leaves=60, l2=2.0 | initial_iteration | 35.1863 | 34.9486 | 35.0743 | 0.7107 | **CHAMPION** | Frozen champion v1 (runs/hgb_run/submission.csv, 35.0743 pts) |
| EXP-007 | HGB-Prior Blend (alpha=1.00 uniform, cond={'stable': 1.0, 'short_history': 1.0, 'drift': 0.94, 'shared_change': 1.0}) | BlendAgent | 35.1863 | 34.9486 | 35.0743 | 0.7107 | **REJECTED** | Tuned on tuning split. Uniform best alpha=1.0000. Cond best drift alpha=0.94 (failed audit). Decision: REJECTED |
| EXP-008 | Principled empirical base rates and explicit missingness indicators | FeatureAgent | 34.7985 | 34.4992 | 34.6577 | 0.7312 | **REJECTED** | All conditions degraded: short_history ratio 0.7826->0.8063, drift 0.7588->0.7950 |
| EXP-009 | Early cutoff sampling (tick 12) + drift velocity features | TemporalSpecialist | 35.3378 | 34.9474 | 35.1547 | 0.7067 | **REJECTED** | Missed audit gate by 0.0012. Early cutoffs added label noise. |
| EXP-010 | Drift velocity features with safe tick 30 cutoffs | TemporalDriftSpecialist | 34.8703 | 34.6534 | 34.7644 | 0.7259 | **REJECTED** | Velocity diffs on sparse windows introduced noise. |
| EXP-011 | Specialist HistGradientBoosting models by machine type | ModelSpecialist | 30.9509 | 30.3597 | 30.6602 | 0.9569 | **REJECTED** | Splitting training data by type fragmented sample size and broke shared episode learning. |
| EXP-012 | Logit temperature scaling T=1.06 (Brier score optimization) | CalibrationSpecialist | 35.2282 | 34.9660 | 35.1057 | 0.7091 | **CHAMPION** | Passed both gates: Tune 35.2282 > 35.1863, Audit 34.9660 > 34.9486. Promoted to Champion v2. |

## Condition Breakdown by Experiment

| Exp ID | Stable Ratio | Short History Ratio | Drift Ratio | Shared Change Ratio |
|---|---|---|---|---|
| BASE-001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| EXP-006 | 0.6684 | 0.7826 | 0.7588 | 0.6328 |
| EXP-007 | 0.6684 | 0.7826 | 0.7588 | 0.6328 |
| EXP-008 | 0.6763 | 0.8063 | 0.7950 | 0.6473 |
| EXP-009 | 0.6826 | 0.7970 | 0.7489 | 0.5985 |
| EXP-010 | 0.6979 | 0.7798 | 0.8112 | 0.6147 |
| EXP-011 | 0.9184 | 1.0386 | 1.0159 | 0.8548 |
| EXP-012 | 0.6686 | 0.7838 | 0.7580 | 0.6261 |
