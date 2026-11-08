> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# Verified structure and peer-change experiment

Run: `python -m pdf_model.run train --data public --out house-edge-output-structure-peers-02`.

The supplied data-pattern report correctly identifies type, signal and episode grouping, but its explanation of short histories is incorrect. In released train data, short-history queries each have 64 observed payouts. Cutoff denotes the prediction tick, not observation count. Condition labels are not available at test time and are excluded from input features.

This isolated experiment adds observation-gap geometry, uninterrupted recent history, and peer win-rate changes centered on each peer's older behavior to the previously tested changing-lag features. All inputs use ticks strictly before the example cutoff. The test verifies sparse-history counts and independence from future observations. Three HGB structures and fixed 150 rounds match the earlier lag experiment; candidate blends are selected against unchanged Jonathan champion using tuning episodes only. Compare report.json with the earlier lag report before promoting. Audit episodes were inspected repeatedly.

Predict: `python -m pdf_model.run predict --model-dir OUTPUT_DIRECTORY --data public --out reproduced.csv`.

Original teammate source and packages are unchanged. The generated entry.zip contains the selected saved model, frozen feature dependencies and pinned package requirements.
