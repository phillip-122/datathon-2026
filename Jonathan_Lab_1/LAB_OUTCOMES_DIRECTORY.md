# House Edge — Master Lab Outcomes & Directory Index

**Challenge:** TAMU Datathon 2026 — House Edge (Slot Machine Forecasting)  
**Main Round Score:** Champion v5 Promoted (**37.8683 Points**, Audit: **37.8495 Points**, Full $R = 0.5844$)  
**Status:** Champion v5 active submission package compiled, cryptographically verified, and fully reproducible.

---

## 1. Google Form Submission Package (CURRENT ACTIVE CHAMPION V5)

All active submission files are consolidated into **one dedicated folder**:

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)

| Upload Item | Filename | Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Forecast CSV** | `submission.csv` | 201,036 bytes | `c067ffc7d855bf474374ca5628670c8baf73c940d29b60f8eeb28ee753207ffe` |
| **Code ZIP** | `submission_entry.zip` | 10,424,865 bytes | `452f71d87f1297214f8155eac96eadb1eb69b175b1e58781a97a679257cdc6cc` |

*Official checker status:*
```text
VALID FORMAT: 1536 queries | main | test
Season: 5fffdc4d0c700bbf61c3f911
SHA-256: c067ffc7d855bf474374ca5628670c8baf73c940d29b60f8eeb28ee753207ffe
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
| **Champion v4 (`EXP-014`)** | [`compiled_versions/v4_promoted_champion_exp014/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v4_promoted_champion_exp014/) | $0.6008$ | $39.92\%$ | **37.4814** | 37.4764 | Superceded Champion |
| **Champion v5 (`EXP-018`)** | [`compiled_versions/v5_promoted_champion_exp018/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v5_promoted_champion_exp018/) | **$0.5844$** | **$41.56\%$** | **37.8683** | **37.8495** | **ACTIVE CHAMPION** |

### Condition Breakdown for Active Champion v5:
- **`shared_change`**: $R = 0.4844$ (**51.56% error reduction**)
- **`stable`**: $R = 0.5545$ (**44.55% error reduction**)
- **`drift`**: $R = 0.6383$ (**36.17% error reduction**)
- **`short_history`**: $R = 0.6606$ (**33.94% error reduction**)

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
