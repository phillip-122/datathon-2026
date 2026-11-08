# Compiled Versions Directory — TAMU Datathon 2026

This directory houses all compiled, versioned, and verified champion packages for the **House Edge** challenge. Each champion milestone is archived in its own dedicated version folder so the lab never loses prior iterations as new champions are developed.

---

## 1. Active Champion Submission (CURRENT BEST TO UPLOAD)

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Currently pointing to Champion v4 — EXP-014 at **37.4814 points**)*

Files ready for the Google Form:
1. **`submission.csv`** (200,982 bytes)  
   `SHA-256: a4b42a156c12797865ebadf9f51c14ce565f1c9b50c5fdf9d04a969b455f29a5`
2. **`submission_entry.zip`** (10,222,980 bytes)  
   `SHA-256: b66683c407384e08811067c9ef08fa070656fd9aa462d14c5271691b33ee79ad`

---

## 2. Champion Version Archive (Every Champion Preserved Separately)

| Folder | Champion ID | Method / Architecture | Full Validation | Audit | Loss Ratio ($R$) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| [`v4_promoted_champion_exp014/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v4_promoted_champion_exp014/) | `EXP-014` | Calibrated Changing-Lag Blend (75% Jonathan V3 + 25% Changing-Lag Specialist + 3-Fold Episode-Cross-Fitted Calibration) | **37.4814** | **37.4764** | **0.6008** | **CURRENT ACTIVE CHAMPION** |
| [`v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | `EXP-013` | 3-Seed HGB Ensemble + Fast Recency ($hl=2,4$) + Machine Volatility Calibration | **36.7561** | **36.6807** | **0.6324** | Previous Champion |
| [`v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | `EXP-012` | HGB (211 features) + Logit Temperature Calibration ($T=1.06$) | **35.1057** | **34.9660** | **0.7091** | Previous Champion |
| [`v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | `EXP-006` | HistGradientBoosting (leaves=60, L2=2.0, multi-cutoff step 6) | **35.0743** | **34.9486** | **0.7107** | Safe Frozen Fallback |

---

## 3. Quick Verification Commands

From workspace root:
```bash
python lab.py predict
```

Verify submission format:
```bash
python House_Edge_Lab_Starter_Pack/research_environment/check_submission.py \
  --data ../public \
  compiled_versions/active_champion_submission/submission.csv
```

Verify SHA-256 cryptographic parity:
```bash
shasum -a 256 compiled_versions/active_champion_submission/*
```
