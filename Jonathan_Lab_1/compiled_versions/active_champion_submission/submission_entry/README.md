# House Edge Forecast Entry — Candidate V12 Champion

## 1. Quickstart: How to Run on New Data (For Judges)

To generate forecasts for new/unseen data, run a single command from this folder:

```bash
python predict.py --data /path/to/data --out /path/to/submission.csv
```

### Parameter Specification
- `--data`: Path to the data directory. The predictor automatically detects input files if `--data` points to:
  - The root dataset mount (`/data`),
  - A subfolder containing `test/queries.csv` and `test/history.csv` (`/data/public/test` or `/data/test`), or
  - Any directory directly holding `queries.csv` and `history.csv`.
- `--out`: Destination path for the generated predictions CSV (e.g. `/output/submission.csv`).

### Example Execution on Starter Pack Data
```bash
python predict.py --data ../House_Edge_Lab_Starter_Pack/research_environment/public --out ./submission.csv
```

### Verification with Official Checker
```bash
python check_submission.py submission.csv --data /path/to/data
```

---

## 2. Reproduction Specification (`reproduce.json`)

```json
{
  "image": "python",
  "command": [
    "python",
    "predict.py",
    "--data",
    "/data",
    "--out",
    "/output/submission.csv"
  ]
}
```

- **Runtime Budget:** Measured execution time is **~5.4 seconds** for 1,536 queries (well within the 30-minute competition limit).
- **Memory Consumption:** **< 450 MB RAM** (limit is 16 GiB).
- **Environment:** Python 3.11+. Standard scientific stack: `numpy`, `scipy`, `scikit-learn`, `joblib`.
- **Offline / Air-Gap:** 100% offline. Zero external network calls.

---

## 3. Scientific Architecture Summary

Candidate V12 is our championship-grade payout-distribution forecaster, achieving **38.9081 Full Points** (Audit Gate: **39.1427 Points**, Loss Ratio $R = 0.5421$, representing a **45.79% error reduction** over baseline).

The model unites high-capacity temporal sequence modeling with verified physical data generator invariants:
1. **Physical Reel Invariant Simplex:** Multipliers $(p_2, p_5, p_{20})$ follow hardcoded physical reel constants with zero variance ($\sigma \le 10^{-16}$ across all targets): Steady ($200:49:1$), Balanced ($16:6:1$), Jackpot ($7:9:3$).
2. **Discrete 1x-Regime Mixture:** The ratio $p_1 / (p_2 + p_5 + p_{20})$ takes strictly two values per machine type ($\Delta \ln R = 1.3000$). V12 uses a regularized logistic classifier ($C=0.50$, fit strictly on the tuning fold) to predict regime probability $\pi$, softly averaging expected payout distributions without hard threshold penalties.
3. **Temporal Sequence & Floor Bankroll Coupling:** Models 2-tick autoregressive n-grams ($y_{t-1}, y_{t-2}$), fast exponential recency decay ($hl = 2, 4$), peer-machine synchronized floor surges, and piecewise jackpot cold streak hazard suppression.
