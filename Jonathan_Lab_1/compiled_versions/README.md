# Compiled Versions Directory — TAMU Datathon 2026

This directory houses all compiled, versioned, and verified champion packages for the **House Edge** challenge. Each champion milestone is archived in its own dedicated version folder so the lab never loses prior iterations as new champions are developed.

---

## 1. Active Champion Submission (CURRENT BEST TO UPLOAD)

📁 **Directory:** [`compiled_versions/active_champion_submission/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/active_champion_submission/)  
*(Currently pointing to **Champion v6 — EXP-019** at **37.9990 Full Points** / **37.9948 Audit Points**)*

Files ready for the Google Form:
1. **`submission.csv`** (201,173 bytes)  
   `SHA-256: ef443805d7e3ab10080e806177c7475be42d389b95198836b029bd57b863ccdd`
2. **`submission_entry.zip`** (18,106,558 bytes)  
   `SHA-256: 582ee28c920c3936060e9b1d0a57ae7004fe6068be4da122324547356284cf19`

---

## 2. Champion Version Archive (Every Champion Preserved Separately)

| Folder | Champion ID | Method / Architecture | Full Validation | Audit | Loss Ratio ($R$) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| [`v6_promoted_champion_exp019/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v6_promoted_champion_exp019/) | `EXP-019` | Discrete Autoregressive Lag Sequences (2-tick n-grams + Dirichlet transition priors) + Dual Prior Ensemble + Hierarchical Calibrator | **37.9990** | **37.9948** | **0.5790** | **ACTIVE CHAMPION v6** |
| [`v5_promoted_champion_exp018/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v5_promoted_champion_exp018/) | `EXP-018` | 40/24 Topology-Augmented Ensemble + Hierarchical Machine Calibration | **37.8683** | **37.8495** | **0.5844** | Previous Champion v5 |
| [`v4_promoted_champion_exp014/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v4_promoted_champion_exp014/) | `EXP-014` | Calibrated Changing-Lag Blend (75% V3 + 25% Specialist + 3-Fold Episode Calibration) | **37.4814** | **37.4764** | **0.6008** | Previous Champion v4 |
| [`v3_promoted_champion_exp013/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v3_promoted_champion_exp013/) | `EXP-013` | 3-Seed HGB Ensemble + Fast Recency ($hl=2,4$) + Machine Volatility Calibration | **36.7561** | **36.6807** | **0.6324** | Previous Champion v3 |
| [`v2_promoted_champion_exp012/`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1/compiled_versions/v2_promoted_champion_exp012/) | `EXP-012` | HGB (211 features) + Logit Temperature Calibration ($T=1.06$) | **35.1057** | **34.9660** | **0.7091** | Previous Champion v2 |
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
