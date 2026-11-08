> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# Dedicated win probability experiment

From repository root with the existing environment:

`.venv/Scripts/python.exe -m win_model.run train --data public --out house-edge-output-win-02`

Tests binary log-loss and squared-error models with 15 or 31 leaves, 200 rounds, learning rate .05, L2 5, seed 42. A separate four-class classifier is fitted on positive payouts only. Each win model is tested with champion payout-size probabilities or the separate classifier, at blend weights .25, .5, .75, 1. Tuning episodes select the winner, including unchanged champion fallback. No new model is fitted on validation answers. Existing audit episodes have been inspected before.

Features and cutoff-safe training generation are reused from hgb_upgrade. Jonathan's active champion package is extracted to the output directory; original teammate files are unchanged. Generated entry.zip includes all inference dependencies and pinned requirements. Probability composition tests: `python -m unittest win_model.test_probability -v`.

First run: best attempted blend scored 36.689457 tuning versus champion 36.819389, so fallback retained champion. This experiment did not improve it. See house-edge-output-win/report.json for all comparisons.
