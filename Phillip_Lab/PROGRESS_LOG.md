# Experiment progression

1. Frequency baseline: pooled_recent approximately 30.826 full-validation points.
2. Feature-based XGBoost: 31.851; extended depth search: 34.323.
3. Neural/XGBoost ensemble: 35.773. Local and hierarchical alternatives did not improve its selected blend.
4. Jonathan V3 independently reproduced at 36.756. A global team blend gave mixed results and was not preferred.
5. Historical masking / recency variants: no convincing improvement.
6. Dedicated win model: no tuning improvement; champion fallback retained.
7. Changing-lag relationships: 37.090 full, 37.050 audit. Selected 25% specialist / 75% Jonathan V3.
8. Observation gaps / normalized peer changes: 37.017, below changing-lag model.
9. Calibration selected with 3-fold episode-cross-fitted tuning: 37.380 cross-fitted tuning. Final fit on tuning episodes: 37.476 audit, 37.481 full (partly in-sample). Audit bootstrap gain interval approximately +0.105 to +0.773 points, conditional on prior selection history.

All experiments used released data. No hidden test outcomes. Original teammate source remains unchanged. Repeated public validation inspection limits generalization claims. Old runs are research artifacts, not automatically promoted submissions.

Organization: moved Phillip-owned source under src, experiment outputs under ignored runs, starter baselines under archive, and frozen upload pairs under compiled_versions. lab.py provides stable commands after relocation.
