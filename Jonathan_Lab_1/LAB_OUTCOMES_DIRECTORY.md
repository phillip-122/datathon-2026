# House Edge — Master Lab Outcomes & Directory Index

**Challenge:** TAMU Datathon 2026 — House Edge (Slot Machine Forecasting)  
**Main Round Deadline:** Saturday, Nov 7 at 9:00 PM Central  
**Round Two (Adaptation) Kickoff:** Saturday, Nov 7 at 9:00 PM Central  
**Status:** All autonomous agents halted. Clean, verified submission packages prepared.

---

## 1. Google Form Submission Package (UPLOAD THESE TWO FILES)

All final files for upload have been consolidated into **one dedicated folder** to eliminate any ambiguity:

📁 **Directory:** [`compiled_versions/final_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/final_submission/)

| Upload Item | Filename | Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Forecast CSV** | `submission.csv` | 199,919 bytes | `1b479d986d35562be35302127d51b22d23228d3601b4b7691010a3db158d8783` |
| **Code ZIP** | `submission_entry.zip` | 751,322 bytes | `73e1bbe40de68bf63b7847b0d940697978458248b05e0c47c4d4f3ec5a9e026c` |

*Official checker status:*
```text
VALID FORMAT: 1536 queries | main | test
Season: 5fffdc4d0c700bbf61c3f911
SHA-256: 1b479d986d35562be35302127d51b22d23228d3601b4b7691010a3db158d8783
```

---

## 2. Is Our Submission Ready to Reproduce Predictions by Category?

### **YES — 100% Offline, End-to-End Verified.**

The organizer evaluation harness runs our ZIP inside a clean, air-gapped Docker container:
```bash
python predict.py --data /data/public --out /output/submission.csv
```

### What Happens Inside the Container:
1. **Dynamic Path Resolution:** [`predict.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/submission_entry/predict.py#L21-L29) automatically discovers `queries.csv` and `history.csv` whether the data is mounted at `/data`, `/data/test`, or `/data/public/test`.
2. **Category Handling:** For all 1,536 test queries across all four hidden categories:
   - **`stable`**: Uses long-term machine type, signal priors, and exact lag sequences.
   - **`short_history`**: Backed by the 194-state empirical transition tensor (`empirical_transition_priors.csv`) and Laplace smoothing.
   - **`drift`**: Half-life recency weights ($6, 12, 24, 48$ ticks) and streak counters capture shifting win probabilities.
   - **`shared_change`**: Cross-machine peer features (`peer_win_rate`, `peer_mean_pay`, `peer_t1_wins`) detect episode-wide regime changes before the cutoff.
3. **Probability Strictness:** Logit probabilities pass through temperature calibration ($T=1.06$), max-subtracted softmax, and `math.fsum` normalization to ensure $|\sum p_i - 1.0| \le 2.22 \times 10^{-16}$.
4. **Clean-Room Test Results:**
   - Reproduction time: **2.37 seconds** (limit: 30 minutes).
   - Peak RAM (RSS): **331 MB** (limit: 16 GB).
   - Max numerical discrepancy: **0.0000000000** (Bit-for-bit identical SHA-256 hash match).

---

## 3. Version History & Metric Progression

Technical points formula: $\text{Points} = \frac{60}{1 + R}$, where $R = \frac{\text{MSE}}{\text{Baseline MSE}}$.

| Version | Directory / Source | Loss Ratio ($R$) | Error Reduction | Technical Points | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Starter Baseline** | `outputs-baseline/` | $1.0000$ | $0.0\%$ | **30.0000 / 60** | Baseline reference |
| **Pooled Recent** | `outputs-baseline/` | $0.9463$ | $5.4\%$ | **30.8259 / 60** | Recency reference |
| **Champion v1 (`EXP-006`)** | [`compiled_versions/v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | $0.7107$ | $28.9\%$ | **35.0743 / 60** | Frozen Fallback |
| **Champion v2 (`EXP-012`)** | [`compiled_versions/v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | **$0.7091$** | **$29.1\%$** | **35.1057 / 60** | **ACTIVE CHAMPION** |

### Detailed Performance by Condition ($R$-values):
- **`stable`**: $R = 0.6686$ (33.1% error reduction over baseline)
- **`shared_change`**: $R = 0.6261$ (37.4% error reduction over baseline)
- **`drift`**: $R = 0.7580$ (24.2% error reduction over baseline)
- **`short_history`**: $R = 0.7838$ (21.6% error reduction over baseline)

---

## 4. Databricks Empirical Iteration & Compiled Champion v3 Resolution

**Status: ABLATED & VERIFIED (Nov 7, 2026)**

Following an iterative benchmark on the 50/50 audit split, we separated true signal from overfit noise:

1. **Ablation Post-Mortem:**
   - **Transition Tensor:** Overfit to training episodes (Tuning rose to $35.71$, but Audit dropped to $34.52$). Purged from artifacts.
   - **Peer Escalation Table:** Rigid lookup degraded performance ($34.92$ pts) relative to continuous rolling features.
2. **The Compiled Winning Resolution (Machine-Type Volatility Calibration):**
   - Derived from Databricks Finding 2 (divergent machine volatility):
     - **Steady ($T=1.00$):** High baseline win rate ($52-64\%$) and zero jackpots; preserves crisp confidence without softening.
     - **Balanced ($T=1.06$):** Standard intermediate calibration.
     - **Jackpot ($T=1.12$):** High volatility ($16-31\%$ win rate); softens overconfident tails, sharply reducing quadratic Brier loss.
3. **Audit Gate Status:**
   - **Full Points:** **$35.1556$** (Beats active champion $35.1057$ by $+0.0499$ pts)
   - **Tuning Points:** **$35.3044$** (Beats gate $> 35.2282$)
   - **Audit Points:** **$34.9857$** (Beats gate $> 34.9660$)
   - Documentation & directives recorded in [`research/DATABRICKS_EMPIRICAL_INSIGHTS.md`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/research/DATABRICKS_EMPIRICAL_INSIGHTS.md).

> **Safe Experimentation Rule:** Both promotion gates have passed. Ready to compile into `compiled_versions/` upon user confirmation. Master submission in `compiled_versions/final_submission/` remains locked until confirmed.
