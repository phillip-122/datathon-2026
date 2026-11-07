#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Locate competition kit root
if [[ -f "$HERE/../public/metadata.json" ]]; then
  KIT_ROOT="$(cd "$HERE/.." && pwd)"
elif [[ -f "$HERE/../../public/metadata.json" ]]; then
  KIT_ROOT="$(cd "$HERE/../.." && pwd)"
elif [[ -f "$HERE/../../../public/metadata.json" ]]; then
  KIT_ROOT="$(cd "$HERE/../../.." && pwd)"
else
  echo "ERROR: Could not locate competition kit root containing public/metadata.json" >&2
  exit 2
fi

mkdir -p "$HERE/public/test" "$HERE/tools"

for f in history.csv queries.csv sample_submission.csv; do
  src="$KIT_ROOT/public/test/$f"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: Required test input not found: $src" >&2
    echo "Check that this pack is inside Jonathan_Lab_1/House_Edge_Lab_Starter_Pack/test_agent." >&2
    exit 2
  fi
  cp "$src" "$HERE/public/test/$f"
done

if [[ ! -f "$KIT_ROOT/public/metadata.json" ]]; then
  echo "ERROR: Missing $KIT_ROOT/public/metadata.json" >&2
  exit 2
fi
cp "$KIT_ROOT/public/metadata.json" "$HERE/public/metadata.json"
cp "$KIT_ROOT/check_submission.py" "$HERE/tools/check_submission.py"
rm -rf "$HERE/tools/slot_forecasting"
cp -R "$KIT_ROOT/slot_forecasting" "$HERE/tools/slot_forecasting"

printf 'Staged test-only inputs in %s\n' "$HERE/public"
printf 'Copied official checker and slot_forecasting package into %s/tools\n' "$HERE"
printf 'No train or validation data was copied.\n'
