# Compiled Versions Directory — TAMU Datathon 2026

This directory houses all compiled, versioned, and verified champion packages for the **House Edge** challenge. Each champion milestone is archived in its own dedicated version folder so the lab never loses prior iterations as new champions are developed.

---

## 1. Active Champion Submission (CURRENT BEST TO UPLOAD)

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Pointing to **Champion v12 — Probabilistic 1x-Regime Mixture & Physical Reel Inverter** at **38.9081 Full Points** / **39.1427 Audit Points**)*

Files ready for the Google Form:
1. **`submission.csv`** (200,432 bytes)  
   `SHA-256: a327727770292f29ee85b6e075c724dacafaa4a12bac380583e78437b0c7da12`
2. **`submission_entry.zip`** (33,299,612 bytes)  
   `SHA-256: dcb7372d0f9fde14f022ecd666f907c753c7c2ccb28fc5dfde1d3eb23f921218`

---

## 2. Champion Version Archive (Every Champion Preserved Separately)

| Folder | Champion ID | Method / Architecture | Full Validation | Audit | Loss Ratio ($R$) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| [`v12_promoted_champion_probabilistic_regime/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v12_promoted_champion_probabilistic_regime/) | `EXP-033` | Probabilistic 1x-Regime Mixture ($S \in \{0, 1\}$) + Exact Physical Reel Projections ($200:49:1$, $16:6:1$, $7:9:3$) | **38.9081** | **39.1427** | **0.5421** | **ACTIVE CHAMPION v12 (FINAL)** |
| [`v11_physical_reel_projection/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v11_physical_reel_projection/) | `EXP-027` | Physical Reel Simplex Projection onto Candidate v10 | **38.7474** | **38.8745** | **0.5544** | Candidate v11 (Preserved Fallback) |
| [`v10_promoted_champion_short_history_triage/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v10_promoted_champion_short_history_triage/) | `EXP-025` | Short History Subpopulation Triage (Peer bankroll coupling + piecewise cold hazard + empirical Bayes shrinkage) | **38.5007** | **38.6048** | **0.5584** | Previous Champion v10 |
| [`v8_candidate_solution_a/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v8_candidate_solution_a/) | `EXP-022` | Machine-Type Adaptive Mixture (15/28/35) + Regularized Hierarchical Calibration | **38.4227** | **38.3854** | **0.5616** | Previous Champion v8 |
| [`v6_promoted_champion_exp019/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v6_promoted_champion_exp019/) | `EXP-019` | Discrete Autoregressive Lag Sequences (2-tick n-grams + Dirichlet transition priors) + Dual Prior Ensemble + Hierarchical Calibrator | **37.9990** | **37.9948** | **0.5790** | Previous Champion v6 |
| [`v5_promoted_champion_exp018/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v5_promoted_champion_exp018/) | `EXP-018` | 40/24 Topology-Augmented Ensemble + Hierarchical Machine Calibration | **37.8683** | **37.8495** | **0.5844** | Previous Champion v5 |
| [`v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | `EXP-013` | 3-Seed HGB Ensemble + Fast Recency ($hl=2,4$) + Machine Volatility Calibration | **36.7561** | **36.6807** | **0.6324** | Previous Champion v3 |
| [`v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | `EXP-006` | HistGradientBoosting (leaves=60, L2=2.0, multi-cutoff step 6) | **35.0743** | **34.9486** | **0.7107** | Initial Frozen Baseline |

---

## 3. Quick Verification Commands

From workspace root:
```bash
python House_Edge_Lab_Starter_Pack/research_environment/check_submission.py \
  --data House_Edge_Lab_Starter_Pack/research_environment/public \
  compiled_versions/active_champion_submission/submission.csv
```

Verify SHA-256 cryptographic parity:
```bash
shasum -a 256 compiled_versions/active_champion_submission/*
```
