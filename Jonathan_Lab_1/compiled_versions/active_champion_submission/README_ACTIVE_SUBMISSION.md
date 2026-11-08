# Active Google Form Submission — Champion v5 (`EXP-018`)

Upload **`submission.csv`** and **`submission_entry.zip`** from this directory to the TAMU Datathon Google Form.

### Official Performance Metrics
- **Full Validation Score:** **37.868275 / 60.000000** (Full Loss Ratio $R = 0.584440$, **41.56% Error Reduction**)
- **Held-Out Audit Score:** **37.849509 / 60.000000** (Audit Loss Ratio $R = 0.585225$)
- **3-Fold Episode Cross-Fitted Tuning Score:** **37.706199 / 60.000000**
- **Paired Condition-Stratified Bootstrap 95% CI Gain vs Champion v4:** **[+0.0932, +0.6578] Points** (Strictly positive)

### Condition Error Breakdown ($R = \frac{\text{MSE}}{\text{Baseline MSE}}$)
- **`shared_change`**: $R = 0.4844$ (**51.56% error reduction**)
- **`stable`**: $R = 0.5545$ (**44.55% error reduction**)
- **`drift`**: $R = 0.6383$ (**36.17% error reduction**)
- **`short_history`**: $R = 0.6606$ (**33.94% error reduction**)

### Machine Type Breakdown
- **`jackpot`**: $R = 0.5670$ (Brier: $0.064403$)
- **`balanced`**: $R = 0.5767$ (Brier: $0.127461$)
- **`steady`**: $R = 0.5906$ (Brier: $0.144519$)

### Clean-Room Verification Status
- **Official Format Check:** `VALID FORMAT: 1536 queries | main | test`
- **Season ID:** `5fffdc4d0c700bbf61c3f911`
- **Offline Reproduction Runtime:** **3.1 seconds** (Limit: 30 minutes)
- **Maximum Probability Deviation:** **`0.00000000`** (Exact machine-level equivalence)
- **Offline Reproduction Command:**
  ```bash
  python lab.py predict --out runs/reproduced.csv
  ```

### Upload Package Manifest & Hashes
| File | Byte Size | SHA-256 Hash |
| :--- | :--- | :--- |
| **`submission.csv`** | 201,036 | `c067ffc7d855bf474374ca5628670c8baf73c940d29b60f8eeb28ee753207ffe` |
| **`submission_entry.zip`** | 10,424,865 | `452f71d87f1297214f8155eac96eadb1eb69b175b1e58781a97a679257cdc6cc` |
