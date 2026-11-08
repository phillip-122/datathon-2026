# Frozen Phillip V12 new system

This is Phillip/Codex V12, distinct from Jonathan EXP-033 V12.

Full validation: 38.789732. Audit: 38.888688. Saved V10 comparison: 38.500702 full, 38.604845 audit. Gain: +0.289030 full, +0.283843 audit. Paired audit bootstrap interval vs saved V10: [-0.012015, 0.576206]; not a decisive improvement. Audit has been repeatedly inspected.

Method: soft Bayesian payout-regime inference, 24-tick half-life, 75% blend with regenerated packaged V10. Constraint values estimated within episode-cross-fitted tuning folds for selection, then refitted on all tuning targets. No audit/test targets fitted. The archived local V10 package regenerates 38.331653 rather than its saved CSV's 38.500702; both are recorded in report.json. This prevents treating this as a clean same-baseline ablation.

The submission ZIP contains frozen inference code, all model artifacts and v12.json. It passed the starter archive contract; packaged inference reproduced submission.csv byte-for-byte. CSV format checks and 15 tests passed. Root file hashes are in MANIFEST.json. Never overwrite this version; create a new folder for changes. The active champion was not changed by this freeze.

Files for this frozen version: submission.csv and submission_entry.zip. Full validation and model parameters accompany them for comparison. Nothing is submitted automatically. Jonathan's new reported V12 (38.9081 full / 39.1427 audit) is not included here and requires its own independently verified package.
