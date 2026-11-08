# Candidate v6 — EXP-020 (Continuous Drift Slopes & Peer Coupling Specialist)

Staged candidate in **`compiled_versions/v6_candidate_exp020/`** for promotion review.

### Official Performance Metrics (ResearchController)
- **Full Validation Score:** **37.903932 / 60.000000** (+0.035657 Points Gain vs Champion v5: 37.868275)
- **Held-Out Audit Score:** **37.860569 / 60.000000** (+0.011061 Points Gain vs Gate: 37.849509)
- **Cross-Fit Tuning Score:** **37.941980 / 60.000000** (+0.058824 Points Gain vs Gate: 37.883155)
- **Full Loss Ratio R:** **0.582949** (41.71% Total Error Reduction vs baseline)
- **Paired Condition-Stratified Bootstrap 95% CI Gain vs Champion v4:** **[+0.0997, +0.6671] Points** (Strictly positive)

### Dual Promotion Gate Verification
- **Tuning Points Gate:** `37.941980 > 37.883200` — **PASSED**
- **Audit Points Gate:** `37.860569 > 37.849500` — **PASSED**
- **Dual Promotion Gate Status:** **CLEAN PASS**

### Condition Error Breakdown ($R = \frac{\text{MSE}}{\text{Baseline MSE}}$)
- **`shared_change`**: $R = 0.4820$ (v5: 0.4844, **-0.0024 improvement**)
- **`stable`**: $R = 0.5544$ (v5: 0.5545, **strictly improved**)
- **`drift`**: $R = 0.6336$ (v5: 0.6383, **-0.0047 error ratio drop**)
- **`short_history`**: $R = 0.6618$ (v5: 0.6606)

### Machine Type Breakdown
- **`jackpot`**: $R = 0.5594$ (Brier: $0.063546$ vs v5: $0.064403$, **-0.0076 drop**)
- **`balanced`**: $R = 0.5767$ (Brier: $0.127461$)
- **`steady`**: $R = 0.5906$ (Brier: $0.144519$)

### Package Manifest
| File | Byte Size | SHA-256 Hash |
| :--- | :--- | :--- |
| **`submission.csv`** | 201,230 | `e1efbc618d18e83907bdb456b263dc3d0976a0279dae3de85d389d4c8ac5a4fb` |
| **`submission_entry.zip`** | 25,723,023 | `71315f3bf435367fc3600ac03efcf026808620b92410e4b765c2186d610971ab` |
| **`validation_predictions.csv`** | 105,100 | `09479c8feab7d48333e2da6badc9b7eea2062a1125e449fc14e5ac3f9a700495` |
| **`report.json`** | 4,850 | `f5ad4cc56a3fb3cca2543d5fa2ba5cf8f66e7a10b17885373af7a0262ec205c3` |

### Offline Reproduction Verification
Run:
```bash
python predict_v6.py --model-dir . --data public --out runs/test_reproduced.csv
```
