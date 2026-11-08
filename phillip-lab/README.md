# Phillip House Edge models

Run commands from the repository root with Python 3.11+.

Install: `python -m pip install -r my_model/requirements.txt`

Tests: `python -m unittest my_model.test_features combo_model.test_models -v`

Reproduce committed champion: `python -m combo_model.run predict --model-dir phillip-lab/champion --data public --out reproduced.csv`

Train a new ensemble: `python -m combo_model.run train --data public --out house-edge-output-new --epochs 100`

Train XGBoost with transition features: `python -m my_model.train --data public --out house-edge-output-new-trees --depths 4 6 8 10 12 --max-rounds 1500 --checkpoint-every 100`

The champion is the earlier neural/XGBoost ensemble (35.772767 full validation points). Its original feature code is frozen in combo_model/frozen_features.py. It is distinct from the later transition-feature experiments. Artifacts and matching CSV/ZIP are in champion/. Jonathan's model is kept separately in Jonathan_Lab_1 and scored 36.756095 on the same full validation data. Audit data has been inspected repeatedly. Test scores are unknown.

Comparison: `python compare_models.py --a FIRST_VALIDATION.csv --b SECOND_VALIDATION.csv`

Prediction refits deterministic local models and loads shared saved parameters. No network is needed after dependencies are installed. Scratch outputs, environments and duplicate training data are excluded from Git. optimizer.py is a separate manual casino experiment GUI, unrelated to House Edge training.
