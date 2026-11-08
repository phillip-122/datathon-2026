TAMU Datathon 2026 — House Edge
ACTIVE CHAMPION SUBMISSION FILES (CURRENT BEST: CHAMPION V3 - EXP-013)
Main Round Cutoff: Saturday, Nov 7 at 9:00 PM Central

Two submission files ready for Google Form upload:
1. submission.csv (200,063 bytes)
   SHA-256: f713536a8e6d64e59046019fc1e6c1573b05089a27bc7186766aab2deddec2e9
   - 1,536 test queries across all 4 categories (stable, short_history, drift, shared_change)
   - Header: query_id,p_0,p_1,p_2,p_5,p_20
   - All rows sum to 1.0 within 1e-15

2. submission_entry.zip (2,595,518 bytes)
   SHA-256: 32a527de145ef74d460f1684c50e49c4db6442d12d2d929848ebb161e4cde5ab
   - Root-level README.md and reproduce.json
   - Air-gapped offline inference entrypoint: predict.py
   - 3-seed ensemble weights (seeds 42, 101, 202) and 223 cutoff-safe features in artifacts/
   - Clean-room reproduction verified: 2.63s runtime, 336.3 MB RAM

Model: HGBEnsembleForecaster with Fast Recency Half-Lives [2, 4, 6, 12, 24, 48] & Machine-Type Volatility Calibration
Score: 36.7561 / 60 Technical Points (Mean Loss Ratio R = 0.6324, 36.76% error reduction)
Format Verification: VALID FORMAT: 1536 queries | main | test
Archive Copy: compiled_versions/v3_promoted_champion_exp013/
