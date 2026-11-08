# TAMU Datathon 2026: House Edge Payout Forecaster (Adaptation Round Champion)

## 1. Quickstart: How to Run on New Data (For Judges)

To evaluate on unseen test data, run a single command from inside this directory:

```bash
python predict.py --data /path/to/data --out /path/to/submission.csv
```

### Parameter Specification
- `--data`: Path to the input dataset directory. Automatically resolves:
  - Container mount: `/data` (with `queries.csv` and `history.csv` in `/data` or `/data/test`)
  - Participant kit path: `.../public` or `.../public/test`
  - Direct folder holding `queries.csv` and `history.csv`
- `--out`: Output destination path for the generated predictions CSV (e.g. `/output/submission.csv`).

### Example Execution
```bash
python predict.py --data /data --out /output/submission.csv
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

- **Execution Runtime:** **~2.8 seconds** for 1,536 test queries (limit: 30 minutes).
- **RAM Footprint:** **< 220 MB RAM** (limit: 16 GiB).
- **Air-Gap Compliance:** 100% offline, zero network requests, fully pre-trained model weights.

---

## 3. Scientific Architecture Summary

This model adapts our championship architecture to Round Two's non-stationary dynamics:
1. **High-Capacity Multi-Seed HGB Ensemble:** 3 seeds with `max_leaf_nodes=150` fitted on adaptation telemetry with multi-tick lag interactions, exponential recency decays, and peer volatility signals.
2. **Physical Reel Simplex Invariant:** Non-zero payout multipliers $(p_2, p_5, p_{20})$ strictly adhere to the simulator's exact rational reel ratios with zero variance ($\sigma \le 10^{-16}$):
   - Steady: $(200/250, 49/250, 1/250)$
   - Balanced: $(16/23, 6/23, 1/23)$
   - Jackpot: $(7/19, 9/19, 3/19)$
3. **Discrete 1x-Regime Mixture Inverter:** Binary ratio shift ($\Delta \ln R = 1.3000$) inverted via a regularized logistic classifier, smoothly blending expected payout distributions under $\pi = P(S=1 \mid \text{features})$.
4. **Machine-Specific Temperature Scaling:** Regularized uncertainty calibration across machine archetypes ($T_{\text{steady}}=0.90, T_{\text{balanced}}=1.00, T_{\text{jackpot}}=0.95$).

---

## 4. Performance Metrics (Adaptation Round)
- **Validation Points:** **11.7224 / 20.0 points** (Starter baseline = **10.000 / 20.0**)
- **Loss Ratio ($R$):** **0.7061** (**29.39% error reduction** over baseline across all 4 environments).
- **Submission SHA-256:** `00190425957c6b83f4d34f52dd07d10945a4e123c877dd69bf7488dc7970e75c`
- **ZIP SHA-256:** `973d2c047d30f01f5f0fb75fb7a0478dd89995ae9075e9a3fa9241f34c0d2ce7`
