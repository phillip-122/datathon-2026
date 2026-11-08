# EXP-030: The Analytical Two-Variable Physical Inverter

## 1. Executive Summary
Experiment EXP-030 executes the clean-sheet analytical physical inversion of the slot machine simulator.
By uncovering the exact mechanics of the simulator, we proved that the system has only **two latent degrees of freedom**:
1. $p_0$: continuous failure hazard ($P(\text{payout} = 0)$).
2. $S \in \{0, 1\}$: binary payout multiplier state governing $p_1 / (p_2 + p_5 + p_{20})$.

When both latent variables are known (Oracle test), the closed-form physical projection achieves **60.0000 / 60 points** ($R = 1.95 \times 10^{-18}$ error ratio).
Furthermore, if $p_0$ is known exactly and $S$ is completely unmodeled (setting $\pi = 0.50$), the physical projection still achieves **56.6941 / 60 points** ($R = 0.0583$).

## 2. Experimental Setup (Clean-Sheet Inverter)
In accordance with instructions, this experiment uses **zero ensemble code or historical tree backbones from v8/v10**:
- **1-D Hazard Estimator for $p_0$:**
  - Features: drought streak (`streak_0`), win rates (`wr_8`, `wr_16`, `wr_32`), `floor_bankroll_16`, `machine_type`.
  - Architecture: Regularized Shallow Random Forest Regressor (`max_depth=3`, `min_samples_leaf=15`, `n_estimators=100`).
  - Fit strictly on the Tuning Fold (48 episodes, 384 queries).
- **Binary Classifier for $S \in \{0, 1\}$:**
  - Features: recent ratio of 1x vs (2x+), lag payouts 1 to 8, `machine_type`.
  - Architecture: $L_2$-regularized Logistic Regression ($C=0.10$).
  - Fit strictly on the Tuning Fold (48 episodes, 384 queries).
- **Evaluation:** Out-of-sample evaluation via `ResearchController` with the Dual-Gate protocol and 2,000 paired bootstrap resamples.

## 3. Quantitative Comparison vs Active Champion v10

| Metric | Baseline Champion v10 | Clean-Sheet Inverter EXP-030 | Difference |
| :--- | :---: | :---: | :---: |
| **Full Technical Points** | **38.5007** | **31.9493** | -6.5514 |
| **Tuning Technical Points** | **38.4107** | **32.8384** | -5.5723 |
| **Audit Technical Points** | **38.6048** | **30.9938** | -7.6110 |
| **Tuning-Audit Spread** | **0.1941** | **1.8446** | +1.6505 |
| **Overall Mean Loss Ratio $R$** | **0.5584** | **0.8780** | +0.3196 |
| $\bullet$ `stable` Loss Ratio | **0.5462** | **0.8523** | +0.3061 |
| $\bullet$ `short_history` Loss Ratio | **0.6178** | **0.8997** | +0.2819 |
| $\bullet$ `drift` Loss Ratio | **0.5986** | **0.8938** | +0.2952 |
| $\bullet$ `shared_change` Loss Ratio | **0.4710** | **0.8664** | +0.3954 |
| **Paired Bootstrap 95% CI vs v10** | [0.0000, 0.0000] | **[-8.7291, -6.4912]** | Strictly negative |

## 4. Key Scientific Insights & Failure Mode Analysis

1. **Physical Soundness vs Estimation Sparsity:**
   - The physical formulas $c_2, c_5, c_{20}$ and $R_{\text{low}}, R_{\text{high}}$ are exact down to machine epsilon ($2.22 \times 10^{-16}$ error across all rows).
   - However, a 1-D hazard estimator using ONLY the 5 specified features (`streak_0`, `wr_8`, `wr_16`, `wr_32`, `floor_bankroll_16`) lacks the fine-grained sequence dynamics (autoregressive 2-tick n-gram transitions, exponential recency half-life $hl=2,4$, and peer-coupling slopes).
   - The standalone 1-D estimator has an MSE of $0.1158$ on $p_0$ ($r = 0.5285$), whereas Champion v10 achieves an MSE of $0.0663$ ($r = 0.7604$).
2. **Error Amplification in the Closed Form:**
   - Because $W_{\text{high}} = (1 - p_0) \cdot K(\pi)$ and $p_1 = (1 - p_0) - W_{\text{high}}$, any error in $p_0$ linearly scales all win payout probabilities.
   - When $\hat{p}_0$ deviates by $\pm 0.35$ during a regime switch, the squared error explodes across all 5 classes simultaneously.
3. **Implication for Champion v11:**
   - Do NOT replace the predictive engine with a clean-sheet 5-feature hazard estimator.
   - INSTEAD, use the physical formulas as a **post-hoc projection operator** on top of the strong ensemble's $p_0$ and $S$ predictions to enforce hard physical constraints.
