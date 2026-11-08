# House Edge — Master Lab Outcomes & Directory Index

**Challenge:** TAMU Datathon 2026 — House Edge (Slot Machine Forecasting)  
**Main Round Deadline:** Saturday, Nov 7 at 9:00 PM Central  
**Round Two (Adaptation) Kickoff:** Saturday, Nov 7 at 9:00 PM Central  
**Status:** Champion v3 Promoted (**36.7561 Points**). Submissions verified and locked.

---

## 1. Google Form Submission Package (UPLOAD THESE TWO FILES)

All final files for upload are consolidated into **one dedicated folder**:

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)

| Upload Item | Filename | Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Forecast CSV** | `submission.csv` | 199,879 bytes | `f713536a8e6d64e59046019fc1e6c1573b05089a27bc7186766aab2deddec2e9` |
| **Code ZIP** | `submission_entry.zip` | 1,496,512 bytes | `32a527de145ef74d460f1684c50e49c4db6442d12d2d929848ebb161e4cde5ab` |

*Official checker status:*
```text
VALID FORMAT: 1536 queries | main | test
Season: 5fffdc4d0c700bbf61c3f911
SHA-256: f713536a8e6d64e59046019fc1e6c1573b05089a27bc7186766aab2deddec2e9
```

---

## 2. Version History & Metric Progression

Technical points formula: $\text{Points} = \frac{60}{1 + R}$, where $R = \frac{\text{MSE}}{\text{Baseline MSE}}$.

| Version | Directory / Source | Loss Ratio ($R$) | Error Reduction | Technical Points | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Starter Baseline** | `outputs-baseline/` | $1.0000$ | $0.0\%$ | **30.0000 / 60** | Baseline reference |
| **Pooled Recent** | `outputs-baseline/` | $0.9463$ | $5.4\%$ | **30.8259 / 60** | Recency reference |
| **Champion v1 (`EXP-006`)** | [`compiled_versions/v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | $0.7107$ | $28.9\%$ | **35.0743 / 60** | Frozen Fallback |
| **Champion v2 (`EXP-012`)** | [`compiled_versions/v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | $0.7091$ | $29.1\%$ | **35.1057 / 60** | Previous Champion |
| **Champion v3 (`EXP-013`)** | [`compiled_versions/v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | **$0.6324$** | **$36.76\%$** | **36.7561 / 60** | **ACTIVE CHAMPION** |

### Condition Breakdown for Active Champion v3:
- **`stable`**: $R = 0.5697$ (**43.0% error reduction**)
- **`shared_change`**: $R = 0.5328$ (**46.7% error reduction**)
- **`drift`**: $R = 0.7053$ (**29.5% error reduction**)
- **`short_history`**: $R = 0.7218$ (**27.8% error reduction**)

---

## 3. Reproduction & Clean-Room Audit Metrics
- **Offline Runtime:** **2.63 seconds** (limit: 30 minutes)
- **Peak RSS RAM:** **336.3 MB** (limit: 16 GB)
- **Max Probability Cell Deviation:** **0.0000000000** (Bit-for-bit SHA-256 match)
- **Row Sum Deviation:** $< 2.22 \times 10^{-16}$ (limit: $1 \times 10^{-6}$)
