> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# Episode-cross-fitted probability calibration

Run: `python -m calibrated_model.run train --data public --source house-edge-output-changing-lags --out house-edge-output-calibrated-02`.

The six-parameter calibration map is softmax(exp(a)*log(p)+bias). It adjusts a frozen model's confidence and class biases. Regularization choices .01, .1, 1 are compared with identity using three episode-grouped, condition-balanced folds within tuning episodes. Every cross-fitted tuning prediction is made by a calibrator fitted on other episodes. The selected calibrator is then fitted to all tuning true probability targets. Audit and test labels are never fitted. The final tuning and full-validation scores partly include calibration fitting data and must not be presented as fully held out. Base models and audit were selected/inspected in prior experiments, so this process does not create a new unbiased holdout.

The report includes out-of-fold tuning scores, audit scores, fold assignments and a paired condition-stratified episode-bootstrap confidence interval with 2000 draws, seed 42. That interval does not correct prior experiment selection bias. Tests check identity, finite normalized output and fitting a known synthetic bias on a separate sample.

First run: cross-fitted tuning 37.380272 versus raw 37.124762; audit 37.476425 versus raw 37.050250; full 37.481400 (partly calibration training data). Audit gain bootstrap interval [0.1053, 0.7726] points. Full score is not a fresh held-out estimate.

Reproduce: `python -m calibrated_model.run predict --model-dir house-edge-output-calibrated --data public --out reproduced.csv`.

The generated entry.zip includes frozen base model, feature dependencies, calibration parameters and requirements. Prediction uses no labels or network. Both CSVs passed official checks and extracted-entry predictions matched exactly. Original teammate files remain unchanged.
