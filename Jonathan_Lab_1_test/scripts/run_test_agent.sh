#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICTOR="${1:-$HERE/final_inference/predict.py}"
DATA="$HERE/public"
OUTDIR="$HERE/output"
REPORTDIR="$HERE/reports"
OUT="$OUTDIR/submission.csv"
LOG="$REPORTDIR/test_agent_run.log"
REPORT="$REPORTDIR/test_agent_report.md"

mkdir -p "$OUTDIR" "$REPORTDIR"
: > "$LOG"

if [[ ! -f "$PREDICTOR" ]]; then
  echo "ERROR: Predictor not found: $PREDICTOR" | tee -a "$LOG"
  echo "Place the frozen predictor at final_inference/predict.py or pass its path as argument." | tee -a "$LOG"
  exit 2
fi
for f in "$DATA/test/history.csv" "$DATA/test/queries.csv" "$DATA/test/sample_submission.csv" "$DATA/metadata.json" "$HERE/tools/check_submission.py"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing staged file: $f" | tee -a "$LOG"
    echo "Run scripts/stage_test_data.sh first." | tee -a "$LOG"
    exit 2
  fi
done

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
  else sha256sum "$1" | awk '{print $1}'; fi
}
count_csv_rows() { awk 'END { if (NR > 0) print NR-1; else print 0 }' "$1"; }

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "$HERE/.venv/bin/activate" ]]; then
    source "$HERE/.venv/bin/activate"
  elif [[ -f "$HERE/../Jonathan_Lab_1/.venv/bin/activate" ]]; then
    source "$HERE/../Jonathan_Lab_1/.venv/bin/activate"
  fi
fi

PRED_HASH="$(sha256_file "$PREDICTOR")"
ARTIFACTS_DIR="$(dirname "$PREDICTOR")/artifacts"
ARTIFACT_HASHES=""
if [[ -d "$ARTIFACTS_DIR" ]]; then
  for art in "$ARTIFACTS_DIR"/*; do
    if [[ -f "$art" ]]; then
      ARTIFACT_HASHES="${ARTIFACT_HASHES}  - $(basename "$art"): $(sha256_file "$art")"$'\n'
    fi
  done
else
  ARTIFACT_HASHES="  (none)"$'\n'
fi
PYVER="$(python3 --version 2>&1)"
START="$(date +%s)"
PRED_EXIT=0
python3 "$PREDICTOR" --data "$DATA" --out "$OUT" 2>&1 | tee -a "$LOG" || PRED_EXIT=${PIPESTATUS[0]}
END="$(date +%s)"
ELAPSED=$((END-START))

CHECK_EXIT=not_run
CHECK_MSG="Predictor did not produce an output file."
if [[ "$PRED_EXIT" -eq 0 && -f "$OUT" ]]; then
  set +e
  PYTHONPATH="$HERE/tools${PYTHONPATH:+:$PYTHONPATH}" python3 "$HERE/tools/check_submission.py" "$OUT" --data "$DATA" --split test 2>&1 | tee -a "$LOG"
  CHECK_EXIT=${PIPESTATUS[0]}
  set -e
  if [[ "$CHECK_EXIT" -eq 0 ]]; then CHECK_MSG="PASS"; else CHECK_MSG="FAIL (see run log)"; fi
fi

HISTORY_ROWS="$(count_csv_rows "$DATA/test/history.csv")"
QUERY_ROWS="$(count_csv_rows "$DATA/test/queries.csv")"
SAMPLE_ROWS="$(count_csv_rows "$DATA/test/sample_submission.csv")"
OUT_ROWS="not_created"
OUT_COLS="not_created"
OUT_BYTES="not_created"
OUT_HASH="not_created"
if [[ -f "$OUT" ]]; then
  OUT_ROWS="$(count_csv_rows "$OUT")"
  OUT_COLS="$(head -n 1 "$OUT")"
  OUT_BYTES="$(wc -c < "$OUT" | tr -d ' ')"
  OUT_HASH="$(sha256_file "$OUT")"
fi
STATUS=PASS
if [[ "$PRED_EXIT" -ne 0 || "$CHECK_EXIT" != "0" ]]; then STATUS=FAIL; fi
STAMP="$(date '+%Y-%m-%dT%H:%M:%S%z')"

cat > "$REPORT" <<REPORT_EOF
# House Edge — Test Agent Run Report

## Run identity
- Status (execution/format only): **$STATUS**
- Timestamp (local): $STAMP
- Python: $PYVER
- Predictor: $PREDICTOR
- Predictor SHA-256: $PRED_HASH
- Model artifact hashes:
$ARTIFACT_HASHES- Exact command: python3 "$PREDICTOR" --data "$DATA" --out "$OUT"

## Inputs
- test/history.csv data rows: $HISTORY_ROWS
- test/queries.csv data rows: $QUERY_ROWS
- test/sample_submission.csv data rows: $SAMPLE_ROWS
- metadata.json path: $DATA/metadata.json

## Execution
- Predictor exit code: $PRED_EXIT
- Runtime (seconds): $ELAPSED
- Log: reports/test_agent_run.log

## Output
- Path: $OUT
- Columns: $OUT_COLS
- Data rows: $OUT_ROWS
- File size (bytes): $OUT_BYTES
- SHA-256: $OUT_HASH

## Official checker
- Exit code: $CHECK_EXIT
- Result: $CHECK_MSG

## Interpretation
- Format / interface status: $STATUS
- Predictive performance: **UNKNOWN — hidden test targets are not available.**
- A passing checker establishes output/interface compliance only; it does not establish predictive accuracy or competition score.
REPORT_EOF

cat "$REPORT"
if [[ "$STATUS" != PASS ]]; then exit 1; fi
