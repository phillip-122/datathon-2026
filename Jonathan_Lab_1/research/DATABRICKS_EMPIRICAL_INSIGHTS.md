# Databricks Quantitative Iteration & Compiled Research Index

**Last Updated:** November 8, 2026 (00:20 Central)  
**Author:** Databricks Quantitative Bridge Agent  
**Subject:** Empirical Testing, Ablation Post-Mortem, Super-Query Evaluation, and Paradigm Shifts toward 39 Points  
**Active Repository Champion:** `EXP-018` (Champion v5, **Full: 37.8683 pts**, Loss Ratio $R = 0.5844$)

---

## 1. Executive Summary: The Evolution of Findings

```
========================================================================================
MILESTONE PROGRESSION & EMPIRICAL POST-MORTEM
========================================================================================
1. Phase 1 (19:00): Machine-Type Calibration (Steady T=1.00, Balanced T=1.06, Jackpot T=1.12)
   - Passed both audit gates, promoted through Champion v3 -> v4 -> v5 (37.8683 pts).
2. Phase 2 (23:55): 3-Page Databricks Super-Query Suite Evaluated:
   - Query 1 (Jackpot Zero-Clamp): REJECTED. Evaluated on continuous target vector;
     ground-truth p_20 is ~0.001003. Hard zeroing increased MSE loss (37.8683 -> 37.8676).
   - Query 2 (3rd-Order Triplet Priors): REJECTED in EXP-019. Scored 37.7449 (-0.123 vs v5).
     Discrete count tables add variance and dilute continuous tree splits.
   - Query 3 (Synchronous Surge at Ticks 180-200): PROMOTED via EXP-020.
     Continuous trend slopes and peer coupling in this window pushed Tuning to 38.0499!
========================================================================================
```

---

## 2. Quantitative Super-Query Post-Mortem

### Finding A: Why Jackpot Zero-Clamping Fails ($p_{20} = 0$)
- **Empirical Test:** Clamping $p_{20} = 0.0$ for steady machines with $\text{signal}=-1$.
- **Validation Score Impact:** $37.8683 \rightarrow 37.8676$ ($-0.0007$ pts).
- **Mathematical Root Cause:**  
  The competition evaluates Mean Squared Error against the simulator's continuous probability vector in `validation/targets.csv`:
  $$\text{True Mean } p_{20} \text{ for Steady Machines with Signal}=-1 = \mathbf{0.001003} \quad (0.10\%)$$
  Setting $\hat{p}_{20} = 0.0$ incurs a guaranteed squared error $(0.0010 - 0.0)^2 = 1.0 \times 10^{-6}$ on every query.
- **The Correct Rule:** Do not hard-clamp to 0. Use regularized hierarchical temperature calibration ($T_{\text{steady}} = 0.824, T_{\text{jackpot}} = 0.863$) which smoothly shrinks rare tails to $\approx 0.0002$.

### Finding B: 3rd-Order Triplet Priors in `EXP-019`
- **Empirical Test:** Tested in `task-208` using `test_discrete_lags.py` with Dirichlet $\alpha=10.0$ on 296 triplet patterns.
- **Validation Score Impact:** Scored **37.7449 points** ($R = 0.5896$), falling short of Champion v5 ($37.8683$ pts) by $-0.123$ points.
- **Root Cause:** Discrete categorical tables create rigid step-discontinuities that conflict with gradient boosted tree splits.

### Finding C: Synchronous Surges Concentrate at Ticks 180–200 (`EXP-020`)
- **Empirical Test:** Databricks Query 3 proved 5+ simultaneous machine wins peak at tick window 180–199 (403 surges).
- **Implementation in `EXP-020`:** Modeled continuous differential trend slopes (`win_rate_16 - win_rate_48`, `peer_slope`) and pairwise peer coupling over this exact pre-cutoff window.
- **Validation Score Impact:**
  - **Tuning Score:** Surged to **$38.0499$ points** (breaking the 38-point barrier!).
  - **Drift Loss Ratio:** Dropped from $0.6383 \rightarrow \mathbf{0.6212}$ (a major $+0.0171$ error reduction).

---

## 3. The Path to 39.0 Points: The Mathematical Gap

To reach **$39.00$ technical points**, the competition formula requires:
$$\text{Points} = \frac{60}{1 + R} = 39.00 \implies R = 0.5385 \quad (46.15\% \text{ error reduction})$$
Our current Active Champion v5 sits at **$37.8683$ ($R = 0.5844$, $41.56\%$ error reduction)**. The remaining gap is **$\Delta R = -0.0459$**.

Where the remaining error reduction lives:
1. `shared_change`: $R = 0.4844$ (nearly saturated, $>51\%$ error reduction)
2. `stable`: $R = 0.5545$ (approaching Bayes noise floor)
3. **`drift` ($R = 0.6383 \rightarrow 0.5500$):** **Gap = 0.088** $\leftarrow$ PRIMARY ERROR POOL
4. **`short_history` ($R = 0.6606 \rightarrow 0.5800$):** **Gap = 0.080** $\leftarrow$ SECONDARY ERROR POOL

---

## 4. The Next Big Paradigm Shifts to Exploit Databricks

Discrete lookup tables and rule-based clamps have hit their ceiling. To unlock 39.0 points, we must exploit Databricks for **Large-Scale Simulator Reverse-Engineering**:

### Shift 1: The Simulator's Latent Machine Archetype Genome
- **The Clue from Rules/Data:** *"Two machines of the same type can react differently or oppositely. Useful patterns can repeat across machines, while their direction differs."*
- **The Hypothesis:** Within each nominal machine type (`steady`, `balanced`, `jackpot`), the simulator generates from a finite set of $K$ discrete parameter archetypes (e.g., *Momentum-Driven* vs. *Mean-Reverting*).
- **The Databricks Super-Task:**
  Cluster all machines across all training episodes by their 8-lag autocorrelation vector $(r_1, \dots, r_8)$ and payout variance using k-means / GMM. 
  - How many clusters $K$ exist per machine type? (e.g. 2 steady clusters, 2 jackpot clusters).
  - If we discover the discrete cluster centroids, we can infer each machine's archetype within its first 24 ticks and feed an `archetype_id` feature into the tree!

### Shift 2: The Parametric Drift Reversal Kernel
- **The Clue from Rules/Data:** *"Some relationships gradually reverse."*
- **The Hypothesis:** Drift is not random noise; it is a deterministic parametric function of tick position $t / \text{cutoff}$ (e.g., $w(t) = \tanh(\kappa(t - t_0))$ or a smooth cosine phase shift).
- **The Databricks Super-Task:**
  Fit the exact empirical trajectory of lag-1 and lag-2 autocorrelation across tick buckets $t \in [0, 250]$ in drift episodes. Extract the mathematical kernel $f(t)$ that governs the sign inversion.
  - Adding $f(t) \times \text{lag1}$ directly cancels the drift non-stationarity, collapsing drift error to stable levels.

### Shift 3: Full-Lakehouse Distributed Tree Distillation
- Locally on CPU, we train on $\approx 37,000$ multi-cutoff slices (`step_ticks=6`).
- Databricks can generate **1,000,000+ synthetic multi-cutoff slices** with exact 40/24 topology masking across all episodes and train a massive LightGBM booster. We export the trained booster JSON/model directly into `submission_entry/` for offline inference ($< 2\,\text{s}$).
