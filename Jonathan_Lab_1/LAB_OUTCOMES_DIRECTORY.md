# House Edge — Master Lab Outcomes & Directory Index

**Challenge:** TAMU Datathon 2026 — House Edge (Slot Machine Forecasting)  
**Main Round Score:** Champion v5 Promoted (**37.8683 Points**, Audit: **37.8495 Points**, Full $R = 0.5844$)  
**Status:** Champion v5 active submission package compiled, cryptographically verified, and fully reproducible.

---

## 1. Google Form Submission Package (CURRENT ACTIVE CHAMPION V6)

All active submission files are consolidated into **one dedicated folder**:

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Champion v6 — EXP-019 at **37.9990 points**, Audit: **37.9948 points**, $R = 0.5790$)*

| Upload Item | Filename | Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Forecast CSV** | `submission.csv` | 201,036 bytes | `6256f6ba433c6981fae83069c9b5bfb1b702ec8c909e1e99813c9fb6475704a4` |
| **Code ZIP** | `submission_entry.zip` | 10,432,608 bytes | `d0c8d1976a44547b7bb221d607e1124d776495df7b51ae5dcb4a4802c608f654` |

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
| **Champion v6 (`EXP-019`)** | [`compiled_versions/v6_promoted_champion_exp019/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v6_promoted_champion_exp019/) | **$0.5790$** | **$42.10\%$** | **37.9990** | **37.9948** | **ACTIVE CHAMPION** |

### Condition Breakdown for Active Champion v6:
- **`shared_change`**: $R = 0.4774$ (**52.26% error reduction**)
- **`stable`**: $R = 0.5487$ (**45.13% error reduction**)
- **`drift`**: $R = 0.6376$ (**36.24% error reduction**)
- **`short_history`**: $R = 0.6522$ (**34.78% error reduction**)

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
