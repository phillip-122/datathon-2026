# TAMU Datathon 2026: House Edge Payout Forecaster (Active Champion V12)

## 1. Quickstart: How to Run on New Data (For Judges)

To evaluate on unseen test data, run a single command from inside the unzipped archive directory:

```bash
python predict.py --data /path/to/data --out /path/to/submission.csv
```

### Parameter Specification
- `--data`: Path to the input dataset directory. The script automatically handles all standard folder layouts:
  - Docker container mount: `/data` (where `queries.csv` and `history.csv` are in `/data` or `/data/test`)
  - Starter pack path: `.../research_environment/public`
  - Direct folder containing `queries.csv` and `history.csv`
- `--out`: Output destination path for the generated predictions CSV (e.g., `/output/submission.csv`).

### Example Execution
```bash
python predict.py --data /data --out /output/submission.csv
```

### Official Starter Pack Verification
To verify formatting and valid probabilities against the official contest checker:
```bash
python check_submission.py submission.csv --data /path/to/data
```

---

## 2. Docker & Air-Gap Reproduction Specification (`reproduce.json`)

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

- **Execution Time:** **~5.4 seconds** for all 1,536 test queries (well within the 30-minute evaluation limit).
- **RAM Usage:** **< 450 MB RAM** (limit is 16 GiB).
- **Environment:** Standard Python 3.10+ / 3.11+ scientific stack (`numpy`, `scipy`, `scikit-learn`, `joblib`).
- **Air-Gap Compliance:** 100% offline, zero network requests, self-contained pre-trained model weights.

---

## 3. Scientific Architecture & Mathematical Discovery

Candidate V12 achieves **38.9081 Full Points** (Audit Gate: **39.1427 Points**, Loss Ratio $R = 0.5421$, representing a **45.79% error reduction** over baseline).

The model unites temporal sequence modeling with verified physical data generator symmetries:
1. **Physical Reel Simplex Invariant:** Multipliers $(p_2, p_5, p_{20})$ follow hardcoded physical reel constants with zero variance ($\sigma \le 10^{-16}$ across all targets):
   - Steady: $(c_2, c_5, c_{20}) = (200/250, 49/250, 1/250)$
   - Balanced: $(c_2, c_5, c_{20}) = (16/23, 6/23, 1/23)$
   - Jackpot: $(c_2, c_5, c_{20}) = (7/19, 9/19, 3/19)$
2. **Discrete 1x-Regime Mixture:** Payout ratio $p_1 / (p_2 + p_5 + p_{20})$ takes strictly two values per machine type ($\Delta \ln R = 1.3000$). V12 uses a regularized logistic classifier ($C=0.50$, fit strictly on the tuning fold) to predict regime probability $\pi$, softly averaging expected payout distributions without hard threshold penalties.
3. **Temporal Sequence & Floor Bankroll Coupling:** Models 2-tick autoregressive n-grams ($y_{t-1}, y_{t-2}$), fast exponential recency decay ($hl = 2, 4$), peer-machine synchronized floor surges, and piecewise jackpot cold streak hazard suppression.

---

## 4. Verification Checksums

| File | Size | SHA-256 Checksum |
| :--- | :--- | :--- |
| `submission.csv` | 200 KB | `a327727770292f29ee85b6e075c724dacafaa4a12bac380583e78437b0c7da12` |
| `submission_entry.zip` | 31.8 MB | `a2394a65d8794209fe0484ddd45081529d2e5ac9b0eea5c7c6f32d1a84efb1b1` |
| `validation_predictions.csv` | 200 KB | `b36c3a6e7e4d0f963b73eecf9f253d35f654eea6816ef36fe3fb57206f357341` |
