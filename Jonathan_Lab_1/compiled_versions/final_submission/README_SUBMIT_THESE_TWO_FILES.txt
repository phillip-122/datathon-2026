TAMU Datathon 2026 — House Edge
FINAL MAIN ROUND SUBMISSION FILES
Deadline: Saturday, Nov 7 at 9:00 PM Central

Upload ONLY these two files to the official Google Form:
1. submission.csv (199,919 bytes)
   SHA-256: 1b479d986d35562be35302127d51b22d23228d3601b4b7691010a3db158d8783
   - 1,536 test queries across all 4 categories (stable, short_history, drift, shared_change)
   - Header: query_id,p_0,p_1,p_2,p_5,p_20
   - All rows sum to 1.0 within 1e-6

2. submission_entry.zip (751,322 bytes)
   SHA-256: 73e1bbe40de68bf63b7847b0d940697978458248b05e0c47c4d4f3ec5a9e026c
   - Root-level README.md and reproduce.json
   - Air-gapped offline inference entrypoint: predict.py
   - Model weights and offline transition table in artifacts/
   - Clean-room reproduction verified: 2.37s runtime, 331 MB RAM

Model: HGBForecaster with Temperature Calibration (T = 1.06)
Score: 35.1057 / 60 Technical Points (Mean Loss Ratio R = 0.7091, 29.1% error reduction)
Format Verification: VALID FORMAT: 1536 queries | main | test
