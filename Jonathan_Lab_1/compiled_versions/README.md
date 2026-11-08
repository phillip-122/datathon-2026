# Compiled Versions Directory — TAMU Datathon 2026

This directory houses all compiled, frozen, and verified model submission packages for the **House Edge** challenge, keeping the root workspace clean and uncluttered.

---

## 1. Final Google Form Submission (UPLOAD THESE TWO FILES)

📁 **Directory:** [`compiled_versions/final_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/final_submission/)

Before the **9:00 PM Central deadline**, upload ONLY these two files to the official Google Form:

1. **`submission.csv`** (199,919 bytes)
   - **SHA-256:** `1b479d986d35562be35302127d51b22d23228d3601b4b7691010a3db158d8783`
   - Forecasts for 1,536 test queries across all four hidden categories.
   - Header: `query_id,p_0,p_1,p_2,p_5,p_20` (all rows sum to 1.0 within $10^{-6}$).

2. **`submission_entry.zip`** (751,322 bytes)
   - **SHA-256:** `73e1bbe40de68bf63b7847b0d940697978458248b05e0c47c4d4f3ec5a9e026c`
   - Complete, self-contained offline reproduction package (runs `predict.py` in 2.37s).

---

## 2. Version Archive

| Folder | Champion ID | Model Description | Score (out of 60) | Loss Ratio ($R$) | Verification Status | Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| [`v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | `EXP-012` | HGB (211 features) + Logit Temperature Calibration ($T=1.06$) | **35.1057** | **0.7091** (29.1% error reduction) | 100% Bit-for-bit clean-room match | **Active Champion** (copied to `final_submission/`) |
| [`v1_frozen_champion_exp006/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v1_frozen_champion_exp006/) | `EXP-006` | HistGradientBoosting (leaves=60, L2=2.0, multi-cutoff step 6) | **35.0743** | **0.7107** (28.9% error reduction) | 100% Bit-for-bit clean-room match | **Safe Fallback** |

---

## 3. Quick Verification Commands

From the workspace root, verify the submission format:
```bash
python House_Edge_Lab_Starter_Pack/research_environment/check_submission.py \
  --data House_Edge_Lab_Starter_Pack/research_environment/public \
  compiled_versions/final_submission/submission.csv
```

Verify SHA-256 cryptographic parity:
```bash
shasum -a 256 compiled_versions/final_submission/*
```
