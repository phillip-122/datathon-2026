# Active Phillip submission

Upload **submission.csv** and **submission_entry.zip** from this directory.

Full validation 37.481400/60 (partly calibration fitting data); audit 37.476425/60 (previously inspected); cross-fitted tuning 37.380272/60. This is the calibrated changing-lag blend based on Jonathan V3. See report.json for folds, scores and uncertainty. The frozen ZIP reproduces probabilities exactly in local checks. SHA-256 hashes are recorded in MANIFEST.json. Nothing is uploaded automatically.

From repository root: `.venv/Scripts/python.exe Phillip_Lab/lab.py predict --out Phillip_Lab/runs/reproduced.csv`.
