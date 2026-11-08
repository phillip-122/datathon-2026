# How We Used Databricks: From Lakehouse Discovery to Microsecond Edge Intelligence

When standard gradient-boosted trees plateaued around 35.1 points on the House Edge challenge, we realized the problem: slot machines are noisy, observations are tiny ($N = 64$ to $192$), and predicting raw stochastic outcomes leads to overfitting. Instead of endlessly tweaking models on local CSVs, we pivoted to **system identification**—using Databricks to reverse-engineer the simulator's hidden physical mechanics.

---

### 1. Creativity: Databricks Genie & MCP as an Interactive Scientific Lab

We integrated the **Databricks Model Context Protocol (MCP)** directly with our autonomous AI agent fleet, turning our Databricks SQL Warehouse into an interactive scientific laboratory. 

Rather than running static reports, our agents used **Databricks Genie** to formulate and test competing physical hypotheses:
* **Floor-Budget Hypothesis (Tested & Rejected):** Did the casino floor maintain a fixed payout cap? Genie queries across multi-machine sums showed zero cap variance, saving us hours of chasing false constraints.
* **Continuous Temporal Drift (Tested & Rejected):** Did machines continuously decay? SQL window functions proved payout mechanics remain stable for $78 \text{ to } 230+$ ticks before discrete shifts.
* **The "57-Point Illusion" Disproved:** Textbook math suggested simple 64-spin averages would score 57 points. Databricks telemetry audits proved the actual simulator has hidden drought states ($p_0 = 0.906$ after 8 zeros vs $0.467$ after wins), causing naive frequency counting to fail ($R = 1.0006$). We mathematically proved the real empirical Bayes noise ceiling is bounded near 39–41 points.

---

### 2. Technical Implementation: Delta Lake & The Medallion Architecture

We structured our telemetry pipeline across a clean **Databricks Medallion Architecture** over 280,608 historical spins:

* **Bronze Layer (Raw Ingestion):** Ingested all multi-round raw spin histories, query contexts, and condition logs into partitioned Delta Lake tables.
* **Silver Layer (Enriched Telemetry & Autoregressive Windows):** Built cutoff-safe SQL transformations calculating rolling win rates, floor dispersion ratios, and 8-tick autoregressive transition matrices. Multi-machine joins revealed collective floor dynamics: when 6+ neighboring machines won simultaneously, individual zero-payout probability plunged from 64.6% to 51.2%.
* **Gold Layer (Physical Invariant Extraction):** High-precision SQL aggregations isolated two immutable physical laws of the simulator:
  1. **Physical Reel Simplex Invariant ($\sigma \le 10^{-16}$):** Non-zero payouts $(p_2, p_5, p_{20})$ obey exact rational reel constants with zero variance across all machines: Steady ($200:49:1$), Balanced ($16:6:1$), and Jackpot ($7:9:3$).
  2. **Discrete $1\times$-Regime Invariant ($\Delta \ln R = 1.3000$):** The ratio $p_1 / (p_2 + p_5 + p_{20})$ takes strictly two values per machine type, separated by an exact constant log-shift.

---

### 3. Rapid Adaptation: Diagnosing Round 2 Mutations in Minutes

When Round 2 opened with secretly mutated simulator dynamics, our frozen Round 1 model experienced negative transfer ($9.84 / 20$). Within minutes, our agent ran cross-dataset telemetry audits in Databricks to isolate the exact shifts:
* **The Invariants Held:** Ground-truth validation queries proved the Reel Simplex and $1\times$-regime levels remained 100% identical down to machine precision ($\sigma \le 10^{-16}$).
* **Lag Memory Mutated:** The Main Round's dominant 7-tick periodic rhythm (+0.257 payout spike) completely vanished (+0.146 background noise). Memory shifted to **Lag 2 on balanced machines** (2.2× stronger than Lag 1) and **Lags 5 & 8 on jackpot machines**.
* **Signal Polarity Inverted:** Steady machines with $\text{signal} = 0$ flipped from the coldest state ($p_0 = 0.4695$) to the hottest state ($p_0 = 0.3600$).

Because Databricks gave us instant visibility into the simulator's mutation, we rapidly retrained our temporal tree branches while locking down our physical invariants, immediately surging from **$9.84 \to 11.764 / 20.0$** ($29.99\%$ error reduction).

---

### 4. Meaningful Impact: Lakehouse Discovery to Offline Edge Intelligence

Databricks was the heavy-lifting discovery engine; our production predictor retained only the distilled truth. We exported the extracted invariant tensors and retrained models into a self-contained **4.5 MB offline package** that runs anywhere:
* **Sub-Second Execution:** Predicts 1,536 test queries in **2.8 seconds** on standard CPU with less than **220 MB RAM**.
* **Zero Network Dependencies:** 100% air-gapped, fully compliant with strict competition container rules.

### Results
* **Round 1 (Main Challenge):** **$39.1427$ audit points** out of $60$ (Loss Ratio $R = 0.5421$, **$45.79\%$ error reduction** over baseline).
* **Round 2 (Adaptation Challenge):** **$11.7643$ points** out of $20$ (Loss Ratio $R = 0.7001$, **$29.99\%$ error reduction** over baseline, equivalent to $35.29 / 60$).
* **Speed:** 2.8s total runtime across 1,536 queries.
