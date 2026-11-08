# Compiled Versions Directory — TAMU Datathon 2026

This directory houses all compiled, versioned, and verified champion packages for the **House Edge** challenge. Each champion milestone is archived in its own dedicated version folder so the lab never loses prior iterations as new champions are developed.

---

## 1. Active Champion Submission (CURRENT BEST TO UPLOAD)

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Currently pointing to Champion v3 — EXP-013 at **36.7561 points**)*

Files ready for the Google Form:
1. **`submission.csv`** (200,063 bytes)  
   `SHA-256: f713536a8e6d64e59046019fc1e6c1573b05089a27bc7186766aab2deddec2e9`
2. **`submission_entry.zip`** (2,595,518 bytes)  
   `SHA-256: 32a527de145ef74d460f1684c50e49c4db6442d12d2d929848ebb161e4cde5ab`

---

## 2. Champion Version Archive (Every Champion Preserved Separately)

| Folder | Champion ID | Model Description | Score (out of 60) | Loss Ratio ($R$) | Status | Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| [`v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | `EXP-013` | 3-Seed HGB Ensemble + Fast Recency ($hl=2,4$) + Machine Volatility Calibration | **36.7561** | **0.6324** (36.76% error reduction) | Verified (2.63s, 336 MB) | **CURRENT ACTIVE CHAMPION** |
| [`v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | `EXP-012` | HGB (211 features) + Logit Temperature Calibration ($T=1.06$) | **35.1057** | **0.7091** (29.1% error reduction) | Verified | Previous Champion |
| [`v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | `EXP-006` | HistGradientBoosting (leaves=60, L2=2.0, multi-cutoff step 6) | **35.0743** | **0.7107** (28.9% error reduction) | Verified | Safe Frozen Fallback |

If a new candidate (e.g., `EXP-014`) beats 36.7561 on both Tuning and Audit gates, it will be added as `v4_promoted_champion_exp014/` and `active_champion_submission/` will be updated accordingly.

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
