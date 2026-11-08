# How We Used Databricks: From Lakehouse Discovery to Edge Intelligence

We started with conventional gradient-boosted models and extensive feature engineering, but performance plateaued around 35.1 points. The challenge was harder than ordinary prediction: machine histories were limited, outcomes were stochastic, and the simulator's underlying rules were unknown. Instead of endlessly tuning models, we pivoted to system identification, using Databricks to investigate the environment itself.

**Creativity — Hypothesis Discovery Through Genie**

We used Databricks Genie as an interactive scientific workbench to test competing explanations. We investigated and rejected a hard floor-budget hypothesis and the assumption of continuous intra-episode temporal drift, redirecting development toward persistent structural properties rather than unsupported assumptions.

**Technical Implementation — Reverse-Engineering the Telemetry**

Across 282,664 historical events, Databricks Delta tables, SQL window functions, and cross-machine joins revealed:

- **Floor synchronization:** Multi-machine joins showed that collective floor activity directly conditioned individual machines—when 6+ machines won simultaneously, zero-payout probability plunged from 64.6% to 51.2%.
- **Physical invariants:** Non-zero payout multipliers exhibited near-zero variance (σ ≤ 10⁻¹⁶), following fixed reel ratios.
- **Discrete regimes:** 1× payout behavior exhibited two distinct mathematical states separated by Δ ln R = 1.3000.
- **Conditional behavior:** Jackpot frequency showed a substantial decline after extended losing streaks, motivating specialized short-history features.
- **Sparse transitions:** Third-order transition counts with Laplace smoothing (α = 2) provided more robust estimates for rare outcomes.

**Meaningful Impact — Lakehouse to Edge**

We translated supported discoveries into statistical priors, regime estimates, and predictive constraints. Databricks handled large-scale discovery; our production system retained only the useful knowledge, executing completely offline in 5.4 seconds with less than 450 MB RAM.

**Results**

- **38.91 / 60:** Final benchmark score (versus 30.0 baseline and 35.1 initial plateau).
- **39.14 / 60:** Untouched audit score.
- **45.8%:** Reduction in normalized prediction error (Loss Ratio R = 0.5421).

The score is not percentage accuracy: it measures improvement over a defined baseline in a stochastic prediction challenge. Our achievement was not eliminating uncertainty, but discovering enough hidden structure to substantially outperform conventional prediction while respecting limited observations and strict deployment constraints.