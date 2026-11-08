# House Edge — Master Lab Outcomes & Directory Index

**Challenge:** TAMU Datathon 2026 — House Edge (Slot Machine Forecasting)  
**Main Round Score:** Champion v10 Promoted (**38.5007 Points**, Audit: **38.6048 Points**, Full $R = 0.5584$)  
**Status:** Champion v10 active submission package compiled, cryptographically verified, and fully reproducible.

---

## 1. Google Form Submission Package (CURRENT ACTIVE CHAMPION V10)

All active submission files are consolidated into **one dedicated folder**:

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Champion v10 — Short History Subpopulation Triage at **38.5007 points**, Audit: **38.6048 points**, $R = 0.5584$)*

| Upload Item | Filename | Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Forecast CSV** | `submission.csv` | 201,581 bytes | `d7cb86b1c4b6639e48752865040f10297459439bfccded4f53b70dbdb810272d` |
| **Code ZIP** | `submission_entry.zip` | 33,298,379 bytes | `d131bf3d28cf066ff3488a9d19585b50380a1a1671fcc5c2721cc950153b76d3` |

*Official checker status:*
```text
VALID FORMAT: 1536 queries | main | test
Season: 5fffdc4d0c700bbf61c3f911
```

---

## 2. Version History & Metric Progression

Technical points formula: $\text{Points} = \frac{60}{1 + R}$, where $R = \frac{\text{MSE}}{\text{Baseline MSE}}$.

| Version | Directory / Source | Loss Ratio ($R$) | Error Reduction | Full Points | Audit Points | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Starter Baseline** | `outputs-baseline/` | $1.0000$ | $0.0\%$ | **30.0000** | 30.0000 | Baseline reference |
| **Pooled Recent** | `outputs-baseline/` | $0.9463$ | $5.4\%$ | **30.8259** | 30.8259 | Recency reference |
| **Champion v1 (`EXP-006`)** | [`compiled_versions/v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | $0.7107$ | $28.9\%$ | **35.0743** | 34.9486 | Frozen Fallback |
| **Champion v2 (`EXP-012`)** | [`compiled_versions/v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | $0.7091$ | $29.1\%$ | **35.1057** | 34.9660 | Calibrated Temperature |
| **Champion v3 (`EXP-013`)** | [`compiled_versions/v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | $0.6324$ | $36.76\%$ | **36.7561** | 36.6807 | 3-Seed HGB + Fast-HL |
| **Champion v4 (`EXP-014`)** | [`compiled_versions/v4_promoted_champion_exp014/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v4_promoted_champion_exp014/) | $0.6008$ | $39.92\%$ | **37.4814** | 37.4764 | Hybrid Specialist |
| **Champion v5 (`EXP-018`)** | [`compiled_versions/v5_promoted_champion_exp018/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v5_promoted_champion_exp018/) | $0.5844$ | $41.56\%$ | **37.8683** | 37.8495 | Previous Champion |
| **Champion v6 (`EXP-019`)** | [`compiled_versions/v6_promoted_champion_exp019/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v6_promoted_champion_exp019/) | $0.5790$ | $42.10\%$ | **37.9990** | 37.9948 | Prior Champion v6 |
| **Champion v8 (`EXP-022`)** | [`compiled_versions/v8_candidate_solution_a/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v8_candidate_solution_a/) | $0.5616$ | $43.84\%$ | **38.4227** | 38.3854 | Adaptive Mixture v8 |
| **Champion v10 (`EXP-025`)** | [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/) | **$0.5584$** | **$44.16\%$** | **38.5007** | **38.6048** | **ACTIVE CHAMPION v10** |

### Condition Breakdown for Active Champion v10:
- **`shared_change`**: $R = 0.4710$ (**52.90% error reduction**)
- **`stable`**: $R = 0.5462$ (**45.38% error reduction**)
- **`drift`**: $R = 0.5986$ (**40.14% error reduction**)
- **`short_history`**: $R = 0.6178$ (**38.22% error reduction — All-time Lab Record**)

### Machine Type Breakdown for Active Champion v5:
- **`jackpot`**: $R = 0.5670$ (Brier: $0.064403$)
- **`balanced`**: $R = 0.5767$ (Brier: $0.127461$)
- **`steady`**: $R = 0.5906$ (Brier: $0.144519$)

---

## 3. Reproduction & Clean-Room Audit Metrics
- **Offline Reproduction Command:** `python lab.py predict`
- **Reproduction Runtime:** **3.1 seconds** (limit: 30 minutes)
- **Max Probability Cell Numerical Deviation:** **$0.00000000$** (exact bitwise match)
- **Format Verification:** `VALID FORMAT: 1536 queries | main | test`

---

## 4. Databricks Quantitative Insights & Paradigm Shifts toward 39 Points

### Verified Super-Query Findings:
1. **Query 1 (Jackpot Zero-Clamp Evaluation):** REJECTED. Ground-truth $p_{20}$ in `validation/targets.csv` is $0.001003$ for steady machines. Hard zero-clamping increased continuous MSE ($37.8683 \rightarrow 37.8676$). Preserved regularized calibration.
2. **Query 2 (3rd-Order Triplet Priors in EXP-019):** REJECTED. Scored $37.7449$ ($-0.123$ vs v5). Proves discrete frequency tables dilute continuous gradient-boosted tree splits.
3. **Query 3 (Late-Episode Surge Peak at Ticks 180–200):** CONFIRMED & PROMOTED. Synchronous shocks concentrate in the 180–200 tick window. Modeling continuous trend slopes and peer coupling across this window (`EXP-020`) surged Tuning score to **$38.0499$** and dropped drift loss ratio to **$0.6212$**.

### The Big Shifts to Reach 39.0 Points ($R = 0.5385$):
- **Shift 1: Latent Machine Archetype Discovery:** Cluster machines across all episodes by their 8-lag autocorrelation vector to uncover the 2–3 discrete sub-archetypes per machine type (e.g. *Momentum* vs *Mean-Reverting*).
- **Shift 2: Parametric Drift Reversal Kernel:** Fit the exact trajectory $f(t)$ of autocorrelation inversion across tick buckets to cancel non-stationarity in `drift`.
- **Shift 3: Lakehouse Distributed Tree Distillation:** Leverage Databricks distributed compute to train LightGBM over 1,000,000+ synthetic multi-cutoff slices, eliminating variance in `short_history`.
- Full quantitative records maintained in [`research/DATABRICKS_EMPIRICAL_INSIGHTS.md`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/research/DATABRICKS_EMPIRICAL_INSIGHTS.md).
