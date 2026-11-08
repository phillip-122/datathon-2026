> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# Isolated HGB improvement experiments

Jonathan's champion remains unchanged. From repository root:

`python -m hgb_upgrade.run train --data public --out house-edge-output-hgb-upgrade-02`

Compares recency weighting (64-tick half-life), deterministic short-history masking (40 recent + 24 older observations), and both. Fits three structures per variant, fixed 150 rounds, no random-row early stopping. Selects blends at weights 0.25, 0.5, 0.75, 1 using tuning episodes only, with unchanged champion fallback. Existing audit episodes are no longer untouched.

Training uses only released train history/labels. Feature code is copied from the frozen V3 package. A temporal mask test is in test_data.py. The training runner extracts the current active champion archive into its output directory and uses its fitted baseline priors. Outputs include selected model, individual candidate models, score report, CSV and reproducible ZIP. Dependencies are pinned in generated entry/requirements.txt.

Reproduce: `python -m hgb_upgrade.run predict --model-dir house-edge-output-hgb-upgrade --data public --out reproduced.csv`
