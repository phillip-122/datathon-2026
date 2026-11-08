# House Edge Research Experiment Registry & Dual Scoreboards

**Active Submission Champion:** `EXP-018 (Champion v5)`  
**Scoring Paradigm:** $$\text{Technical Points} = \frac{60}{1 + R}$$, where $$R = \frac{\text{MSE}}{\text{Baseline MSE}}$$

---

## 1. Competition Scoreboard

Tracks technical points, loss ratios, and generalization across strict episode splits.

| Exp ID | Hypothesis / Candidate | Author | Tuning Pts | Audit Pts | Full Pts | Mean Loss Ratio ($R$) | Decision | Bootstrap Gain CI95 |
|---|---|---|---|---|---|---|---|---|
| BASE-001 | Official pooled starter baseline | organizers | 30.0000 | 30.0000 | 30.0000 | 1.0000 | **BASELINE** | [0.0, 0.0] |
| EXP-006 | HGB 211 features, multi-cutoff step 6, leaves=60 | initial_iteration | 35.1863 | 34.9486 | 35.0743 | 0.7107 | **CHAMPION v1** | [+4.81, +5.20] |
| EXP-012 | Logit temperature scaling T=1.06 | CalibrationSpecialist | 35.2282 | 34.9660 | 35.1057 | 0.7091 | **CHAMPION v2** | [+4.84, +5.22] |
| EXP-013 | 3-Seed HGB Ensemble + Fast Recency + Machine Volatility | EnsembleSpecialist | 36.8194 | 36.6807 | 36.7561 | 0.6324 | **CHAMPION v3** | [+6.51, +6.95] |
| EXP-014 | Calibrated Changing-Lag Blend (75% V3 + 25% Specialist + 6-Param Cal) | Phillip/Jonathan Hybrid | 37.4841 | 37.4764 | 37.4814 | 0.6008 | **CHAMPION v4** | [+7.32, +7.68] |
| EXP-015 | Hierarchical Machine-Type Calibration (T_eff = T_global + Delta_T_m) | CalibrationAgent | 37.5562 | 37.5171 | 37.5389 | 0.5983 | **BENCHMARK** | [-0.037, +0.115] vs v4 |
| EXP-016 | Exact 40/24 Observation-Pattern Augmentation (Pre-Calibration) | TopologyAgent | 37.2835 | 37.2281 | 37.2575 | 0.6105 | **BENCHMARK** | [+0.11, +0.22] uncal vs v4-base |
| EXP-017 | Continuous Regime & Adaptive Recency Descriptors (Pre-Calibration) | RegimeAgent | 37.2377 | 37.0690 | 37.1577 | 0.6148 | **BENCHMARK** | [-0.05, +0.12] uncal vs v4-base |
| EXP-018 | Augmented Topology + Regularized Hierarchical Calibration Blend | Controller / Strategy | 37.8832 | 37.8495 | 37.8683 | 0.5844 | **ACTIVE CHAMPION v5** | [+0.0932, +0.6578] vs v4 |

---
## 2. Scientific Scoreboard: Discovered Physical & Stochastic Mechanisms

Tracks the empirical discovery of underlying simulator structures and causal mechanisms.

| Exp ID | Discovered Mechanism | Simulator / Physical Grounding | Target Weakness | Empirical Evidence |
|---|---|---|---|---|
| EXP-018 | Observation Topology Invariance + Machine-Specific Regularized Uncertainty | Bridging the training/evaluation missingness gap (40/24 topology) alongside regularized shrinkage calibration (Steady T=0.824, Jackpot T=0.863). | Short history error degradation and jackpot burst overconfidence | Full Points reached 37.8683 (Audit: 37.8495). Short history loss ratio dropped to 0.6606, Drift to 0.6383, Jackpot to 0.5670. |
| EXP-015 | Hierarchical Machine Uncertainty & Bias Correction | Steady machines have low variance (T=0.8244); Jackpot machines have heavy-tailed burst volatility requiring higher calibrated entropy (T=0.8631). | High variance on jackpot reels & uniform temperature bias | Full loss ratio dropped to 0.5983; error reduced across all 3 machine types (Jackpot R=0.5972, Balanced R=0.5899, Steady R=0.5968). |
| EXP-016 | Exact Observation Topology Conditioning (40 Recent + 24 Sparse Older) | Short history conditions do not have uniform cutoffs; they present exactly 40 recent and 24 sparse older ticks. Synthetic masking closes feature divergence. | Short history performance degradation (R=0.6932 down to 0.6831 uncalibrated) | Direct synthetic observation masking raised uncalibrated score from 37.0902 to 37.2575. |
| EXP-017 | Continuous Regime & Adaptive Recency Invariance | Entropy, short/long volatility ratios, and rebound/momentum transition shifts provide continuous regime signatures without memorizing transition tables. | Drift adaptation and shared change detection | Drift loss ratio dropped from 0.6797 to 0.6655 in 25% blend. |

---
## 3. Condition Loss Ratio Breakdown

Tracks error reduction ($R < 1.0$) across distinct environment conditions.

| Exp ID | Stable ($R$) | Short History ($R$) | Drift ($R$) | Shared Change ($R$) | Notes |
|---|---|---|---|---|---|
| BASE-001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Baseline reference |
| EXP-006 | 0.6684 | 0.7826 | 0.7588 | 0.6328 | Champion v1 |
| EXP-012 | 0.6686 | 0.7838 | 0.7580 | 0.6261 | Champion v2 |
| EXP-013 | 0.5697 | 0.7218 | 0.7053 | 0.5328 | Champion v3 |
| EXP-014 | 0.5538 | 0.6858 | 0.6662 | 0.4973 | Champion v4 |
| EXP-015 | 0.5511 | 0.6811 | 0.6668 | 0.4943 | Hierarchical Cal only |
| EXP-018 | 0.5545 | 0.6606 | 0.6383 | 0.4844 | ACTIVE CHAMPION v5 |
