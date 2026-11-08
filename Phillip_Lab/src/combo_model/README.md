> Current layout: use `Phillip_Lab/lab.py` from the repository root. See `Phillip_Lab/README.md` for supported commands. Earlier command examples below describe the original experiment layout; packaged offline commands remain unchanged.

# House Edge combined model

Python 3.11+, pinned dependencies in requirements.txt. No new dependencies beyond
NumPy, SciPy and XGBoost. Original tree features are frozen to match the saved tree.

Train from the kit root:

```
.venv/Scripts/python.exe -m combo_model.run train --data public --out house-edge-output-combo --epochs 100
```

Models include L2-regularized per-machine multinomial logistic regression, a
shared logistic model used as the center of per-machine L2 penalties (empirical
Bayes-style partial pooling), a 64-unit tanh neural network over eight exact tick
lags with explicit missing indicators, the existing XGBoost model, and pooled_recent.
The neural model is a small feed-forward sequence-window model, not a recurrent
network. All components predict five payout probabilities.

Global training uses training histories and labels only. Historical examples use
strictly earlier payouts. Local models fit only each query's earlier observed
outcomes, with half-life weighting and fixed L2 strengths. Missing ticks remain
unknown. Global neural training uses seed 42 and Adam, 100 epochs by default.
Convex ensemble weights minimize the official normalized squared error using
tuning episodes only. Audit episodes are scored after weight selection and have
been inspected in earlier work. A component may receive zero weight.

Reproduce from extracted entry root, with dependencies prepared before offline run:

```
python -m combo_model.run predict --model-dir . --data /data --out /output/submission.csv
```

Prediction refits small local models on supplied past history; this is deterministic
and uses no future outcomes or network. Two CPU threads are requested. Saved shared
parameters and tree model are loaded from the package. The runner checks repeated
test predictions agree within 1e-8. Submit submission.csv with matching entry.zip.
