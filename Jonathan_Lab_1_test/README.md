# House Edge — Test Agent Pack

This pack stages the official **test inputs** and runs a frozen final predictor against them. It does not train, tune, select models, or estimate the hidden competition score.

## Folder layout expected

Place this folder at:

`datathon-2026/Jonathan_Lab_1/House_Edge_Lab_Starter_Pack/test_agent/`

The original competition kit should remain at `datathon-2026/` (three directories above this pack's `test_agent` folder).

## 1. Stage the test environment

From this folder:

```bash
chmod +x scripts/*.sh
./scripts/stage_test_data.sh
```

This copies only:
- `public/test/history.csv`
- `public/test/queries.csv`
- `public/test/sample_submission.csv`
- `public/metadata.json`
- `check_submission.py`
- `slot_forecasting/`

It deliberately does **not** copy train or validation data into the test-agent workspace.

## 2. Add the frozen predictor

Put the final, already-selected predictor at:

`final_inference/predict.py`

It must support the official interface:

```bash
python3 predict.py --data PATH_TO_PUBLIC_DIR --out PATH_TO_OUTPUT_CSV
```

The script can also be called with an alternate predictor path:

```bash
./scripts/run_test_agent.sh /absolute/path/to/predict.py
```

## 3. Run

```bash
./scripts/run_test_agent.sh
```

Outputs:
- `output/submission.csv`
- `reports/test_agent_report.md`
- `reports/test_agent_run.log`

The official test checker verifies schema/query coverage/format. It does **not** reveal or validate the hidden test score. The report therefore labels predictive performance as **UNKNOWN**.

## Important discipline

Treat the test set as a final inference target. Do not use test rows to tune features, choose models, change thresholds, or make performance claims. If the run fails, debug interface/runtime issues without fitting to test outcomes. Keep a hash of the frozen predictor and its model artifacts before running.
