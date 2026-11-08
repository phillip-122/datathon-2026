# Champion Candidate v5 (EXP-016)

- **Champion ID:** EXP-016
- **Architecture:** Continuous Drift Trend Slopes & Jackpot Hazard (227 cutoff-safe features)
  - 65% 3-Seed HGB Ensemble [42, 101, 202] (leaves=80, l2=1.5)
  - 35% Changing-Lag Specialist
  - 3-Fold Episode Cross-Fitted Calibration (penalty=0.005)
- **Validation Technical Points:** **37.6687 / 60** (+0.1873 over Champion v4)
- **Holdout Audit Points:** **37.6093 / 60** (+0.1329 over Champion v4)
- **Cross-Fit Tuning Points:** **37.5912 / 60** (+0.2109 over Champion v4)
- **Overall Loss Ratio:** **0.5928** (Sub-0.60 milestone achieved)
- **Condition Loss Ratios:**
  - drift: 0.6450 (-0.0212 error ratio drop)
  - short_history: 0.6761 (-0.0097 drop)
  - shared_change: 0.4931 (-0.0042 drop)
  - stable: 0.5571 (steady)

## Verification
Format verified with check_submission.py:
VALID FORMAT: 1536 queries | main | test
