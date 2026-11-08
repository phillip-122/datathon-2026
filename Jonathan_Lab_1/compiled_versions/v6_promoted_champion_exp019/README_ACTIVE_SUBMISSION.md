# Active Google Form Submission — Champion v6 (`EXP-019`)

Upload **`submission.csv`** and **`submission_entry.zip`** from this directory to the TAMU Datathon Google Form.

### Official Performance Metrics (DUAL PROMOTION GATE PASSED)
- **Full Validation Score:** **37.999049 / 60.000000** (Full Loss Ratio $R = 0.578987$, **42.10% Error Reduction**)
- **Held-Out Audit Score:** **37.994812 / 60.000000** (Audit Loss Ratio $R = 0.579163$)
- **Held-Out Tuning Score:** **38.000420 / 60.000000** (Tuning Loss Ratio $R = 0.578930$)
- **3-Fold Episode Cross-Fitted Tuning Score:** **37.795779 / 60.000000**
- **Paired Condition-Stratified Bootstrap 95% CI Gain vs Champion v4:** **[+0.1652, +0.8490] Points** (Strictly positive)

### Dual Promotion Gate Verification vs Champion v5 (`EXP-018`):
- **Tuning Points Gate:** **38.000420** > 37.883155 (**PASSED [+0.117265]**)
- **Audit Points Gate:**  **37.994812** > 37.849509 (**PASSED [+0.145303]**)
- **Full Points Gate:**   **37.999049** > 37.868275 (**PASSED [+0.130774]**)

### Condition Error Breakdown ($R = \frac{\text{MSE}}{\text{Baseline MSE}}$)
- **`shared_change`**: $R = 0.4774$ (**52.26% error reduction**)
- **`stable`**: $R = 0.5487$ (**45.13% error reduction**)
- **`drift`**: $R = 0.6376$ (**36.24% error reduction**)
- **`short_history`**: $R = 0.6522$ (**34.78% error reduction**)

### Machine Type Breakdown
- **`jackpot`**: $R = 0.5532$ (Brier: $0.062844$)
- **`balanced`**: $R = 0.5659$ (Brier: $0.125065$)
- **`steady`**: $R = 0.5932$ (Brier: $0.145155$)

### Architecture Details
- **Discrete Sequence Momentum Priors**: 2-tick autoregressive pair sequence `(payout_{t-1}, payout_{t-2})`.
- **Bayesian Smoothing**: Dirichlet/Laplace smoothing with $\alpha = 10.0$ on train history.
- **Dual Prior Specialists**: 50% Global Transition Table + 50% Machine-Type Conditioned Transition Table.
- **Ensemble Blend**: 30% Discrete Lag Specialists + 70% Champion Changing-Lag Backbone.
- **Hierarchical Calibration**: Machine-type adaptive temperature and class biases fitted on Tuning split.

### Clean-Room Verification Status
- **Official Format Check:** `VALID FORMAT: 1536 queries | main | test`
- **Offline Reproduction Runtime:** `< 3.5 seconds` (Air-gapped limit: 30 minutes)
- **Maximum Probability Deviation:** **`0.00000000`** (Exact machine-level equivalence)
- **Offline Reproduction Command:**
  ```bash
  python predict_v6.py --model-dir . --data public --out runs/reproduced.csv
  ```

### Upload Package Manifest & Hashes
| File | Byte Size | SHA-256 Hash |
| :--- | :--- | :--- |
| **`submission.csv`** | 201,173 | `ef443805d7e3ab10080e806177c7475be42d389b95198836b029bd57b863ccdd` |
| **`submission_entry.zip`** | 18,106,558 | `582ee28c920c3936060e9b1d0a57ae7004fe6068be4da122324547356284cf19` |
| **`report.json`** | 5,545 | `c1adb5052e87fee6e663942743d40526c89c305cdc6240467924aa4152c48658` |
