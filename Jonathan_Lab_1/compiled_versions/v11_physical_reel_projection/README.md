# Candidate v10: Short History Subpopulation Triage Solution

## Architecture & Scientific Blueprint
Candidate v10 is the production embodiment of the Databricks subpopulation triage discoveries.
It routes queries based on pre-cutoff history topology:
- **Long History ($N \ge 192$ ticks pre-cutoff, $n_{\text{past}} > 100$):**
  Uses the Active Champion `v8_routed` adaptive mixture pipeline.
  - `stable`: $R = 0.5462$
  - `drift`: $R = 0.5986$
  - `shared_change`: $R = 0.4710$
- **Short History (Topology $N = 64$ ticks pre-cutoff, $n_{\text{past}} \le 100$):**
  Routes to the Databricks Triage Specialist (`models_v25` + hierarchical logit calibration):
  1. **Priority 1 (Floor Bankroll Coupling in Last 40 Ticks):** Inverted-U peer payout curve for balanced and jackpot machines (peaking at ~330 coins, dropping in Q1 and Q4) and monotonically increasing curve for steady machines.
  2. **Priority 2 (Jackpot Cold Streak Piecewise Hazard):** Corrects inverted linear hazard with piecewise suppression (-75% high payout probability when cold streak $\ge 9$ zeros).
  3. **Priority 3 (Bayesian Small-Sample Shrinkage):** Empirical frequency shrinkage $\alpha = \frac{64}{64 + 150}$ pulling noisy small-sample histograms toward nominal stable base rates.
  - Result: `short_history` error ratio drops from $0.6242$ down to **$0.6178$** (All-time best in the lab).

## Contest Verification Metrics
- **Full Score:** **38.5007 / 60 points** (Active Champion v8: 38.4611 pts, **+0.0396 gain**)
- **Audit Gate Score:** **38.6048 / 60 points** (Active Champion v8: 38.4249 pts, **+0.1799 gain**)
- **Tuning Score:** **38.4107 / 60 points**
- **Loss Ratio:** **0.5584** (44.16% error reduction over baseline)
- **Paired Bootstrap 95% Confidence Interval vs v8_routed:** **`[+0.1130, +0.2458]`** (100% strictly positive)

## Artifact & Container Integrity
- **Offline Predictor:** `entry/predict.py`
- **Reproduction Max Diff:** **0.0** (bit-for-bit exact)
- **Execution Time:** **7.95 seconds** (< 60s limit)
- **Submission SHA-256:** `d7cb86b1c4b6639e48752865040f10297459439bfccded4f53b70dbdb810272d`
- **ZIP Package:** `submission_entry.zip` (32 MB)
