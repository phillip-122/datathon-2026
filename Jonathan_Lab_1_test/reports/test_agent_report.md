# House Edge — Test Agent Run Report

## Run identity
- Status (execution/format only): **PASS**
- Timestamp (local): 2026-11-07T17:30:38-0600
- Python: Python 3.13.12
- Predictor: /Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/final_inference/predict.py
- Predictor SHA-256: a7645871c73a7bd52faa0b0f41c066f251501a7f2a4895f130ede4ce3911f808
- Model artifact hashes:
  - base_forecaster.json: 90304d111764e789264bb2a0bd161accb9db289c8f59851b5c3d02777e19e4fb
  - hgb_model.joblib: b7316b494dfcebe90f1de5d6b9be68003473d90192e22363bfb87f083a415470
  - meta.json: 78e27607355db2b6882a1bbc2c656a969e73092d4a3dabc9a98c7c3036e4e49f
- Exact command: python3 "/Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/final_inference/predict.py" --data "/Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/public" --out "/Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/output/submission.csv"

## Inputs
- test/history.csv data rows: 282664
- test/queries.csv data rows: 1536
- test/sample_submission.csv data rows: 1536
- metadata.json path: /Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/public/metadata.json

## Execution
- Predictor exit code: 0
- Runtime (seconds): 3
- Log: reports/test_agent_run.log

## Output
- Path: /Users/jonathankalsky/Developer/Datathon26/datathon-2026/Jonathan_Lab_1_test/output/submission.csv
- Columns: query_id,p_0,p_1,p_2,p_5,p_20
- Data rows: 1536
- File size (bytes): 200384
- SHA-256: 2ae6e79b6a1834100fb2f1ac99c699321611436c769f2d68d43e07759f330756

## Official checker
- Exit code: 0
- Result: PASS

## Interpretation
- Format / interface status: PASS
- Predictive performance: **UNKNOWN — hidden test targets are not available.**
- A passing checker establishes output/interface compliance only; it does not establish predictive accuracy or competition score.
