# House Edge Lab --- Progress & Learning Log

Use this as the lab's living notebook. Update it after every meaningful
experiment or architectural decision. Keep the record concise but
reproducible.

## 0. Project status

-   **Current phase:** Main Round finalized & ready for Round 2 Adaptation; Champion v4 active
-   **Current best model:** `CalibratedChangingLagBlend` (75% Jonathan V3 3-Seed HGB Ensemble + 25% Changing-Lag Specialist with 3-fold episode-cross-fitted low-capacity logit calibration)
-   **Best cross-validation score:** 37.3803 cross-fitted tuning / **37.4764 audit** (split-half episodes)
-   **Official validation score:** **37.4814 / 60 technical points** (mean loss ratio: 0.6008, 39.92% error reduction)
-   **Last reproducible run ID:** EXP-014 (`compiled_versions/v4_promoted_champion_exp014`)
-   **Frozen fallback run ID:** EXP-006 (`compiled_versions/v1_frozen_champion_exp006`, 35.0743 pts)
-   **Known blockers:** None
-   **Next highest-value action:** Await Round 2 (adaptation) kit release; run adaptation fine-tuning and calibration pipeline.

## 1. Setup checklist

-   [x] Original starter ZIP copied into `starter_kit/` (`research_environment`)
-   [x] Starter-kit file inventory verified
-   [x] Official rules and data dictionary read
-   [x] Python version recorded
-   [x] Isolated environment created (`.venv` Python 3.13)
-   [x] Dependencies installed from an explicit specification (`pandas`, `scipy`, `scikit-learn`, `numpy`)
-   [x] Starter baseline runs unchanged
-   [x] Baseline output schema verified
-   [x] Baseline metrics recorded
-   [x] Episode-grouped validation strategy implemented (tuning/audit 50-50 split + full validation scorer)
-   [x] Reproduction command tested

### Environment record

  Item                 Value
  -------------------- ----------------------------------------------------
  OS                   macOS (Darwin arm64)
  Python               3.13.12
  pandas               2.2.3
  NumPy                2.2.3
  scikit-learn         1.6.1
  scipy                1.15.2
  CPU / RAM            Apple Silicon M-series / 16+ GiB RAM
  Install command      `source .venv/bin/activate && pip install pandas scipy scikit-learn`

## 2. Data map

Record exact filenames, columns, row counts, and semantics after
inspecting the kit. Do not guess.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Dataset/file                  Purpose                 Rows Key identifiers               Target/labels                   Notes
  ----------------------------- -------------------- ------- ----------------------------- ------------------------------- -------------------------------------------------
  Train queries.csv             Fitting models           768 query_id, episode_id,         -                               96 episodes, 8 machines per ep, cutoffs 192-256
                                                             machine_id                                                    
  Train history.csv             Gameplay history     139,944 episode_id, machine_id, tick  payout                          Ticks strictly < cutoff, payouts in {0,1,2,5,20}
  Train labels.csv              Next-spin outcomes       768 query_id                      outcome                         Observed payout at cutoff tick
  Train conditions.csv          Condition labels         768 query_id                      condition                       24 ep per condition (stable, short, drift, shared)
  Validation queries.csv        Validation queries       768 query_id, episode_id,         -                               96 episodes, 8 machines per ep, cutoffs 192-256
                                                             machine_id                                                    
  Validation history.csv        Validation history   141,392 episode_id, machine_id, tick  payout                          Ticks strictly < cutoff
  Validation targets.csv        True conditional p       768 query_id                      p_0, p_1, p_2, p_5, p_20, outcome Ground truth probability distribution from simulator
  Validation conditions.csv     Condition labels         768 query_id                      condition                       24 ep per condition
  Validation reference.csv      Frozen baseline p        768 query_id                      p_0, p_1, p_2, p_5, p_20        Frozen pooled starter predictions (denominator)
  Test queries.csv              Evaluation queries     1,536 query_id, episode_id,         -                               192 episodes, 8 machines per ep, cutoffs 192-256
                                                             machine_id                                                    
  Test history.csv              Test history         282,664 episode_id, machine_id, tick  payout                          Ticks strictly < cutoff
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Data semantics verified

-   [x] **What uniquely identifies a query?** `query_id` (1 query per machine per episode).
-   [x] **What identifies a machine?** `machine_id` within `episode_id`. Always group by `(episode_id, machine_id)`.
-   [x] **What identifies an episode?** `episode_id`. Contains exactly 8 machines that share identical cutoff and condition.
-   [x] **What defines the cutoff tick?** `cutoff` is the tick to predict. Only ticks `< cutoff` are permitted as features.
-   [x] **Are payout histories ordered?** Ticks are integer time-steps. Consecutive ticks are not guaranteed (ticks can be missing).
-   [x] **How are missing ticks represented?** Absent rows in `history.csv`. A missing tick is UNKNOWN, not a zero payout!
-   [x] **Which observations are available at inference?** All history rows with `tick < cutoff`. Other machines' earlier history in the same episode is permitted.
-   [x] **Which fields are absent at test time?** `targets.csv`, `labels.csv`, `conditions.csv`, and `reference.csv`.
-   [x] **What is the exact target meaning?** True probability vector `[p_0, p_1, p_2, p_5, p_20]` at the cutoff tick.
-   [x] **What is the official scoring metric?** Sum of squared errors $\sum (p_i - q_i)^2$ averaged per episode, then averaged per condition, divided by reference error $B[c]$, averaged across all 4 conditions to give $R$. Technical points $= 60 / (1 + R)$.

## 3. Hypothesis backlog

Status values: `NEW`, `PLANNED`, `RUNNING`, `SUPPORTED`, `REJECTED`,
`INCONCLUSIVE`, `PARKED`.

  ---------------------------------------------------------------------------------------
  ID          Hypothesis      Why plausible      Test             Status      Evidence /
                                                                              next step
  ----------- --------------- ------------------ ---------------- ----------- -----------
  H-001       Machine type    Different machine  Type-only        NEW         
              predicts        categories may     baseline                     
              different       have distinct                                   
              payout          behavior                                        
              distributions                                                   

  H-002       Signal          Signal effect may  Type × signal    NEW         
              interacts with  vary by type       comparison                   
              machine type                                                    

  H-003       Individual      Observed payouts   Compare          NEW         
              history         reveal             population vs                
              improves        machine-specific   individual                   
              prediction      behavior           estimates                    

  H-004       Longer history  Estimates become   History-length   NEW         
              deserves more   more stable with   stratification /             
              weight          observations       shrinkage                    

  H-005       Recent          Behavior may drift Compare rolling  NEW         
              observations    or be              windows and                  
              matter more     state-dependent    decay                        

  H-006       Lag/sequence    Order may matter   Lag and sequence NEW         
              features add    beyond frequencies ablation                     
              information                                                     

  H-007       Other machines  Machines may share Same-episode,    NEW         
              reveal shared   hidden conditions  past-only                    
              episode state                      features                     

  H-008       Behavior        A single           Change-point /   NEW         
              changes by      stationary         rolling                      
              regime          distribution may   diagnostics                  
                              be inadequate                                   

  H-009       A gated         Evidence quality   Compare fixed    NEW         
              ensemble beats  varies by query    and learned                  
              one global                         mixtures                     
              model                                                           
  ---------------------------------------------------------------------------------------

## 4. Experiment registry

Create one entry per run. Never overwrite the only copy of a previous
successful run.

### Experiment template

#### EXP-\_\_\_\_ --- \[short title\]

-   **Date:**
-   **Owner / agent:**
-   **Status:** Planned / Running / Complete
-   **Hypothesis:**
-   **Question being answered:**
-   **Data split / episode folds:**
-   **Features:**
-   **Model / configuration:**
-   **Baseline being compared:**
-   **Random seed(s):**
-   **Command:**
-   **Code revision / hash:**
-   **Runtime / memory:**
-   **Official metric or CV metric:**
-   **Per-class diagnostics:**
-   **Calibration diagnostics:**
-   **Failure cases:**
-   **Leakage checks:**
-   **Result:** Keep / Revise / Reject / Inconclusive
-   **Interpretation:**
-   **Next experiment:**
-   **Artifact paths:**

------------------------------------------------------------------------

## 5. Run summary table

  -----------------------------------------------------------------------------------------------------------------------------------------
  Run ID     Hypothesis / change             Fold strategy    Model           Val Points   Mean Loss Ratio  Runtime  Decision
  ---------- ------------------------------- ---------------- --------------- ------------ ---------------- -------- ----------------------
  BASE-001   Untouched starter baseline      Full validation  pooled          30.0000      1.000000         0.18s    Keep (Ref)
  BASE-002   Recency decay + context prior   Full validation  pooled_recent   30.8259      0.946285         0.19s    Baseline
  EXP-001    Starter candidates (12 models)  50/50 Tune/Audit pooled_recent   30.5645 (aud)0.9631 (aud)     ~2.5s    Baseline
  EXP-002    Grid search prior & half-life   50/50 Tune/Audit pooled_recent   30.6233 (aud)0.9436 (aud)     ~4.0s    Improved baseline
             (prior_strength=10, hl=16)      Full validation  pooled_recent   30.9023      0.940800         0.19s    Keep
  EXP-003    LogisticRegression + Features   Full validation  LR (C=0.01)     31.5742      0.898800         ~3.0s    Step change
  EXP-004    HistGradientBoosting + Baseline Full validation  HGB (leaves=31) 32.0204      0.873800         ~5.0s    Strong improvement
  EXP-005    211 Advanced Cutoff Features,   50/50 Tune/Audit HGB             34.1988 (aud)0.7545 (aud)     ~10.0s   Breakthrough
             Multi-cutoff Slices (step=6)    Full validation  HGB (leaves=31) 34.3739      0.745500         ~10.0s   Keep
  EXP-006    HGB Tree Capacity & L2 Tuning   50/50 Tune/Audit HGB             34.9486 (aud)0.7167 (aud)     ~12.0s   Frozen Fallback v1
             (leaves=60, l2=2.0, min_leaf=25)Full validation  HGB             35.0743      0.710700         12.03s   (frozen_champion_v1)
  EXP-007    HGB-Prior Blend Optimization    50/50 Tune/Audit HGB+Base        34.9486 (aud)0.710700         ~3.0s    Reject (alpha=1.0 best)
  EXP-008    Missingness Defaults (Base rate)50/50 Tune/Audit HGB             34.4992 (aud)0.731200         ~10.6s   Reject (flags diluted splits)
  EXP-009    Early Cutoff (t=12) + Drift Vel 50/50 Tune/Audit HGB             34.9474 (aud)0.706700         ~12.5s   Reject (label noise at t<30)
  EXP-010    Drift Velocity (t=30 safe cut)  50/50 Tune/Audit HGB             34.6534 (aud)0.725900         ~12.0s   Reject (sparse window noise)
  EXP-011    Specialist Models by Type       50/50 Tune/Audit HGB x 3         30.3597 (aud)0.956900         ~16.0s   Reject (sample fragmentation)
  EXP-012    Logit Temperature Scaling T=1.0650/50 Tune/Audit HGB-Calibrated  34.9660 (aud)0.709100         12.00s   **PROMOTED CHAMPION V2**
             (Brier score calibration)       Full validation  HGB-Calibrated  35.1057      0.709100         12.00s   (promoted_champion_v2)
  -----------------------------------------------------------------------------------------------------------------------------------------

## 6. Learning history / architecture decisions

  -------------------------------------------------------------------------------------------------------------------------------------------
  Date       Decision or discovery           Supporting experiments  Confidence  Consequence
  ---------- ------------------------------- ----------------------- ----------- -------------------------------------------------------------
  2026-11-07 Simulator has hot/cold regimes   Data Audit, EXP-003-005 Very High   True $p_0$ is strongly bimodal (modes at ~0.04 and ~0.99).
             with strong lag-8 interactions                                      Frequency pooling alone is inadequate; nonlinear models excel.
  2026-11-07 Multi-cutoff training slicing   EXP-005                 Very High   Extracting cutoffs every 6 ticks expands training data from
             from history is permitted                                           768 to 21,200 examples while preserving cutoff safety.
  2026-11-07 HistGradientBoosting natively   EXP-005, EXP-006        Very High   HGB outputs well-calibrated multiclass softmax distributions
             models nonlinear dynamics                                           achieving 35.0743 points (+5.07 points over pooled baseline).
  2026-11-07 Offline artifact deployment     Verification test       Absolute    Model fits in 705 KB, predicts in < 1.5s on CPU, 100%
             using joblib and scikit-learn                                       deterministic, identical SHA256 output.
  -------------------------------------------------------------------------------------------------------------------------------------------

### Current architecture

``` text
Input queries.csv + history.csv
  → Cutoff validation (strict t < cutoff filtering)
  → Advanced Cutoff-Safe Feature Extraction (211 features):
      - Machine Type & Signal interactions (steady, balanced, jackpot x {-1, 0, 1})
      - Multi-scale recency decay counts (half-lives 6, 12, 24, 48 ticks)
      - Rolling window win-rates & mean payouts (windows 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
      - Lags 1 to 8 exact payouts and pairwise interaction categories
      - Win / loss streak counters and ticks since last win / big win
      - Same-episode peer machine cross-gameplay activity
      - Hierarchical Dirichlet context and global priors
  → Frozen HistGradientBoostingClassifier (leaves=60, l2=2.0)
  → Exact probability normalization (sum strictly = 1.0)
  → submission.csv
```

## 7. Agent activity log

Agents should not simply say "this feature helps." Require a run ID and
evidence.

  ---------------------------------------------------------------------------
  Date        Agent / role   Task        Output      Experiment   Accepted?
                                         produced    IDs          
  ----------- -------------- ----------- ----------- ------------ -----------
              Data auditor                                        

              Temporal                                            
              specialist                                          

              Episode                                             
              specialist                                          

              Evaluation                                          
              critic                                              

              Orchestrator                                        
  ---------------------------------------------------------------------------

## 8. Leakage and validity checklist

Before accepting any result:

-   [x] Split is grouped by episode where appropriate.
-   [x] No training feature uses information after its historical cutoff.
-   [x] No future payout is used.
-   [x] No observations from unrelated episodes are mixed.
-   [x] Cross-machine features use only same-episode, earlier observations.
-   [x] Missing ticks are not converted into zero payouts.
-   [x] Preprocessing is fitted only on the training fold.
-   [x] Validation targets are not accidentally used as training labels.
-   [x] Feature availability matches test-time availability.
-   [x] The comparison uses the official metric or a justified proxy.
-   [x] Result is reproducible from the recorded command and configuration.

## 9. Finalization checklist

-   [x] Select the best validated architecture (`HGBForecaster`).
-   [x] Remove unused research and agent dependencies from inference.
-   [x] Freeze model parameters and feature definitions.
-   [x] Train final model only on data permitted by the rules (`public/train`).
-   [x] Save all required model artifacts (`artifacts/models/hgb_production`).
-   [x] Verify class order `[0, 1, 2, 5, 20]`.
-   [x] Verify all probabilities are finite and in `[0, 1]`.
-   [x] Verify each row sums to one within tolerance (`1e-6`).
-   [x] Verify query IDs and row ordering (all 1,536 test queries verified).
-   [x] Run from a clean environment with networking disabled.
-   [x] Test the exact organizer reproduction command (`python predict.py --data /data --out /output/submission.csv`).
-   [x] Compare regenerated predictions with submitted CSV (100% byte-for-byte SHA256 match).
-   [x] Check runtime, memory, and package contents (uncompressed < 1 MB, runtime ~1.5s).
-   [x] Archive source, config, dependencies, hashes, and final metrics.

## 10. Session handoff

**What changed?**
Constructed a complete high-performance machine learning pipeline using `HistGradientBoostingClassifier` with 211 cutoff-safe features, multi-cutoff historical slicing, and hierarchical priors. Built and validated the submission package `submission_entry.zip` and `runs/hgb_run/submission.csv`.

**What did we learn?**
The simulator exhibits regime switching (bimodal $p_0$ around 0.04 and 0.99) driven by lag 1-8 sequence interactions, streak lengths, and machine-type specific parameters. Standard frequency pooling models cannot capture these nonlinear dynamics. HGB achieved a 25-35% error reduction across all four conditions.

**Which experiment IDs support it?**
EXP-004, EXP-005, EXP-006.

**What remains uncertain?**
Round 2 adaptation data release at 9 PM Central will introduce changed dynamics.

**What should the next session do first?**
Submit the two Main round files (`runs/hgb_run/submission.csv` and `submission_entry.zip`) to the Main Google Form before 9 PM Central cutoff. At 9 PM, extract adaptation kit, run validation check, and fine-tune/refit HGB pipeline.

**Exact command to resume:**
`source .venv/bin/activate && python House_Edge_Lab_Starter_Pack/research_environment/check_submission.py runs/hgb_run/submission.csv`
