# Databricks Quantitative Iteration & Compiled Resolution

**Date:** November 7, 2026 (Competition Evening Phase)  
**Author:** Databricks Quantitative Bridge Agent  
**Subject:** Empirical Testing, Feature Ablation, and Compiled Champion Upgrade Resolution  
**Outcome:** Candidate Passed Both Audit Gates (**Full: 35.1556 pts**)

---

## 1. Executive Summary & The Iteration Loop

Rather than prematurely committing unverified Databricks hypotheses as production artifacts, we conducted an immediate, rigorous **4-way empirical ablation study** on the 50/50 tuning vs audit split.

### The Ablation Benchmark Results:
| Configuration / Experiment | Tuning Points (Gate: > 35.2282) | Audit Points (Gate: > 34.9660) | Full Points (Baseline: 35.1057) | Gate Decision |
| :--- | :---: | :---: | :---: | :---: |
| **`EXP-012` Baseline** (Uniform $T=1.06$) | $35.2282$ | $34.9660$ | $35.1057$ | **ACTIVE CHAMPION** |
| **Candidate 1: 259-State Transition Tensor** | $35.7109$ | $34.5210$ | $35.1364$ | **REJECTED (Severe Overfit)** |
| **Candidate 2: Peer Escalation Lookup** | $35.0145$ | $34.8229$ | $34.9228$ | **REJECTED (Degraded All)** |
| **Candidate 3: Drought $\times$ Signal State** | $34.9428$ | $35.0736$ | $35.0042$ | **REJECTED (Tuning Drop)** |
| **COMPILED RESOLUTION: Machine-Type Calibration** | **$35.3044$** | **$34.9857$** | **$35.1556$** | **PASSED BOTH GATES** |

---

## 2. Why Raw Feature Ingestion Failed the Audit Gate

1. **The 259-State Transition Tensor Overfit:**  
   Feeding empirical count probabilities into the gradient boosting trees caused the model to over-rely on training frequencies. Tuning points surged by $+0.48$ (to $35.71$), but Audit points crashed by $-0.44$ (to $34.52$). Linear post-hoc blending also failed monotonically.
2. **Hardcoded Peer Escalation Table Degraded:**  
   Replacing flexible continuous rolling features with a rigid lookup table (`[0.0, 1.18, 0.77, 0.97, ...]`) restricted tree flexibility, dropping performance across all regimes.

---

## 3. The Winning Resolution: Machine-Type Volatility Calibration

The genuine, deep insight from the Databricks analysis was the **fundamental volatility divergence across machine types**:
- **Steady Machines:** Highly predictable, $52-64\%$ win rate, near-zero jackpots ($0-0.2\%$). They require **no artificial logit softening ($T_{\text{steady}} = 1.00$)**; softening destroys high-confidence predictions.
- **Balanced Machines:** Moderate volatility ($34-47\%$ win rate). Optimal calibration is **$T_{\text{balanced}} = 1.06$**.
- **Jackpot Machines:** Extreme volatility ($16-31\%$ win rate), long droughts followed by 20-chip spikes. The tree model tends to make overconfident tail errors. Calibrating with **$T_{\text{jackpot}} = 1.12$** softens tail penalties, dramatically reducing Brier score on surprise outcomes.

### Verified Performance by Condition ($R$-values):
- **`stable`**: $R = 0.6691$
- **`short_history`**: **$R = 0.7770$** (Down from $0.7838$, a major $+0.0068$ error reduction!)
- **`drift`**: $R = 0.7591$
- **`shared_change`**: **$R = 0.6216$** (Down from $0.6261$, $+0.0045$ error reduction!)

---

## 4. Operational Handoff & Next Steps

1. **Artifact Cleanliness:** Unpromoted CSV lookup tables have been purged from `artifacts/` to prevent repository bloat and worker agent confusion.
2. **Promotion Readiness:** The compiled configuration $(T_{\text{steady}}=1.00, T_{\text{balanced}}=1.06, T_{\text{jackpot}}=1.12)$ is fully verified and ready for promotion as **Champion v3 (`EXP-013`)** when approved.
