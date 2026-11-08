# Compiled submission versions

The active upload pair is in `active_champion_submission/`: `submission.csv` and `submission_entry.zip`. Keep them together. ZIPs are self-contained and retain original module names. MANIFEST.json records file sizes and SHA-256 hashes.

| Version | Method | Full validation | Audit |
|---|---|---:|---:|
| v1_neural_xgboost | Neural/XGBoost ensemble | 35.772767 | 35.008673 |
| v2_changing_lags | Jonathan V3 + changing-lag specialist | 37.090165 | 37.050250 |
| v3_calibrated | v2 + cross-fitted tuning-selected calibration | 37.481400 | 37.476425 |

v3 is copied to active_champion_submission. Its final tuning and full-validation scores include calibration fitting episodes; use cross-fitted tuning and audit results to assess it. The audit has been inspected in earlier experiments. No hidden-test score is known. Older full local runs remain under ../runs and are ignored by Git.
