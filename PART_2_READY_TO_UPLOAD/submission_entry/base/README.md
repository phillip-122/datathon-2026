# TAMU Datathon 2026: House Edge Payout Forecaster (Candidate A4 Unified Champion)

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

1. **High-Capacity Multi-Seed HGB Ensemble:** 3 seeds with `max_leaf_nodes=150` fitted on adaptation telemetry.
2. **Physical Reel Simplex Invariant:** Non-zero payout multipliers $(p_2, p_5, p_{20})$ strictly adhere to the simulator's exact rational reel ratios with zero variance ($\sigma \le 10^{-16}$):
   - Steady: $(200/250, 49/250, 1/250)$
   - Balanced: $(16/23, 6/23, 1/23)$
   - Jackpot: $(7/19, 9/19, 3/19)$
3. **Full-Horizon Discrete 1x-Regime Mixture Inverter:** Binary ratio shift ($\Delta \ln R = 1.3000$) inverted via an L2-regularized logistic classifier over full-horizon spin counts ($H=\infty$, AUC $> 0.72$).
4. **Short-History Triage:** Peer bankroll coupling and hazard suppression for $N \le 64$ queries.

---

## 4. Performance Metrics (Adaptation Round)
- **Validation Points:** **11.7266 / 20.0 points** (Starter baseline = **10.000 / 20.0**)
- **Loss Ratio ($R$):** **0.7055** (**29.45% error reduction** over baseline across all 4 environments).
