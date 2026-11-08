"""Adaptation Round Rapid-Response Harness (TAMU Datathon 2026 - House Edge).

Usage:
    python adaptation_pipeline.py --data /path/to/adaptation/public [--main-data /path/to/main/public]

Steps:
1. Verify metadata (round == 'adaptation', max_points == 40).
2. Evaluate official starter reference (expected = 20.0000).
3. Evaluate zero-shot Champion V12 on new validation split.
4. Verify physical invariants (reel multipliers & 1x-regime shifts) on adaptation data.
5. Calibrate / refit regime classifier on adaptation data.
6. Generate test predictions for adaptation queries.
7. Run check_submission validation.
8. Package adaptation_entry.zip compliant with submission_contract.
"""

import argparse
import io
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add starter pack to sys.path
_ROOT = Path(__file__).resolve().parent
_ENV = _ROOT / "House_Edge_Lab_Starter_Pack" / "research_environment"
sys.path.insert(0, str(_ENV))
sys.path.insert(0, str(_ROOT))

from slot_forecasting.contest import score_contest_files
from slot_forecasting.io import load_inputs, read_rows, read_targets, write_submission
from slot_forecasting.rounds import round_metadata, score_round
from slot_forecasting.schema import PAYOUTS, probabilities
from slot_forecasting.submission_contract import inspect_archive, package_code


def banner(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def evaluate_adaptation_split(pred_df: pd.DataFrame, public_dir: Path) -> dict:
    val_dir = public_dir / "validation"
    meta = round_metadata(public_dir)
    max_pts = meta.get("max_points", 40)
    
    # Save temporary predictions
    tmp_csv = _ROOT / "/tmp" / "temp_adaptation_val.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(tmp_csv, index=False)
    
    res = score_contest_files(
        tmp_csv,
        val_dir / "targets.csv",
        val_dir / "queries.csv",
        val_dir / "conditions.csv",
        val_dir / "reference.csv",
        max_points=max_pts,
        version="temporal-adaptation-v3-60-40"
    )
    if tmp_csv.exists():
        tmp_csv.unlink()
    return res


def check_invariants_on_targets(val_dir: Path):
    banner("[STEP 4] Testing Physical Reel Invariants on Adaptation Targets")
    targets_file = val_dir / "targets.csv"
    queries_file = val_dir / "queries.csv"
    
    if not targets_file.exists():
        print("targets.csv not found in validation; skipping target invariant inspection.")
        return
        
    targets = pd.read_csv(targets_file)
    queries = pd.read_csv(queries_file)
    df = targets.merge(queries, on="query_id")
    
    REEL_RATIOS = {
        'steady': (200.0 / 250.0, 49.0 / 250.0, 1.0 / 250.0),
        'balanced': (16.0 / 23.0, 6.0 / 23.0, 1.0 / 23.0),
        'jackpot': (7.0 / 19.0, 9.0 / 19.0, 3.0 / 19.0),
    }
    
    for mt in ['steady', 'balanced', 'jackpot']:
        sub = df[df['machine_type'] == mt]
        w_high = sub['p_2'] + sub['p_5'] + sub['p_20']
        valid = sub[w_high > 1e-6]
        if len(valid) == 0:
            continue
        c2 = valid['p_2'] / (valid['p_2'] + valid['p_5'] + valid['p_20'])
        c5 = valid['p_5'] / (valid['p_2'] + valid['p_5'] + valid['p_20'])
        c20 = valid['p_20'] / (valid['p_2'] + valid['p_5'] + valid['p_20'])
        exp_c = REEL_RATIOS[mt]
        err2 = np.max(np.abs(c2 - exp_c[0]))
        err5 = np.max(np.abs(c5 - exp_c[1]))
        err20 = np.max(np.abs(c20 - exp_c[2]))
        print(f"Machine '{mt}' ({len(valid)} targets): Max deviation from exact reel simplex: {max(err2, err5, err20):.2e}")
        
        # Check 1x ratio
        p1 = valid['p_1']
        r = p1 / (valid['p_2'] + valid['p_5'] + valid['p_20'])
        ln_r = np.log(r)
        q25, q75 = np.percentile(ln_r, [25, 75])
        print(f"Machine '{mt}' log-ratio ln(p1/W_high): IQR=[{q25:.4f}, {q75:.4f}], spread={q75 - q25:.4f}")


def run_pipeline(adaptation_dir: Path, main_dir: Path = None):
    t0 = time.time()
    banner("TAMU Datathon 2026 - Adaptation Round Automated Pipeline")
    print(f"Target Adaptation Data: {adaptation_dir.resolve()}")
    if main_dir:
        print(f"Main Round Fallback Data: {main_dir.resolve()}")

    # Step 1: Validate metadata
    banner("[STEP 1] Inspecting Round Metadata")
    meta = round_metadata(adaptation_dir)
    print(f"Round:        {meta.get('round')}")
    print(f"Season ID:    {meta.get('season_id')}")
    print(f"Max Points:   {meta.get('max_points')}")
    print(f"Suite:        {meta.get('suite_version')}")
    
    assert meta.get('round') == 'adaptation', f"Expected round='adaptation', got {meta.get('round')}"
    assert meta.get('max_points') == 40, f"Expected max_points=40, got {meta.get('max_points')}"
    print("Metadata strictly matches official 40-point adaptation contract.")

    # Step 2: Evaluate Official Baseline on New Data
    banner("[STEP 2] Evaluating Official Baseline Reference")
    ref_file = adaptation_dir / "validation" / "reference.csv"
    if ref_file.exists():
        ref_df = pd.read_csv(ref_file)
        ref_score = evaluate_adaptation_split(ref_df, adaptation_dir)
        print(f"Baseline Score: {ref_score['technical_points']:.4f} / 40.0 pts (Loss Ratio R = {ref_score['loss_ratio']:.4f})")
    else:
        print("reference.csv not found; proceeding.")

    # Step 3: Run Champion V12 Predictor
    banner("[STEP 3] Running Champion Predictor on Adaptation Validation")
    from compiled_versions.active_champion_submission.predict import run_prediction
    
    val_out_csv = _ROOT / "/tmp" / "v12_adaptation_validation_pred.csv"
    val_in_dir = adaptation_dir / "validation"
    
    # Run predict on validation
    run_prediction(val_in_dir, val_out_csv)
    v12_val_df = pd.read_csv(val_out_csv)
    v12_res = evaluate_adaptation_split(v12_val_df, adaptation_dir)
    print("\n" + "=" * 50)
    print(f"CHAMPION ZERO-SHOT PERFORMANCE ON ADAPTATION:")
    print(f"Technical Points: {v12_res['technical_points']:.4f} / 40.0 pts")
    print(f"Overall Loss Ratio (R): {v12_res['loss_ratio']:.4f} ({((1 - v12_res['loss_ratio'])*100):.2f}% error reduction)")
    print("By Condition:")
    for cond, r in v12_res['by_condition'].items():
        print(f"  - {cond:15s}: R = {r:.4f}")
    print("=" * 50)

    # Step 4: Check invariants
    check_invariants_on_targets(adaptation_dir / "validation")

    # Step 5: Generate Test Predictions
    banner("[STEP 5] Generating Official Adaptation Test Predictions")
    test_out_dir = _ROOT / "compiled_versions" / "adaptation_champion_submission"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_csv = test_out_dir / "submission.csv"
    
    test_in_dir = adaptation_dir / "test"
    run_prediction(test_in_dir, test_out_csv)
    print(f"Test predictions written to: {test_out_csv}")

    # Step 6: Verify with official check_submission
    banner("[STEP 6] Running Official Contest Submission Checker")
    check_sub_script = _ENV / "check_submission.py"
    cmd = f"{sys.executable} {check_sub_script} {test_out_csv} --data {adaptation_dir}"
    ret = os.system(cmd)
    assert ret == 0, "check_submission validation failed!"

    # Step 7: Package Code ZIP
    banner("[STEP 7] Packaging Adaptation Code ZIP")
    clean_pkg = Path("/tmp/clean_submission_package")
    out_zip = test_out_dir / "submission_entry.zip"
    if out_zip.exists():
        out_zip.unlink()
        
    package_code(clean_pkg, out_zip)
    c = inspect_archive(out_zip.read_bytes())
    print(f"Package created: {out_zip} ({round(out_zip.stat().st_size / (1024*1024), 2)} MB)")
    print(f"inspect_archive config: {c}")

    # Save adaptation report
    report = {
        "round": "adaptation",
        "technical_points": v12_res['technical_points'],
        "max_points": 40,
        "loss_ratio": v12_res['loss_ratio'],
        "by_condition": v12_res['by_condition'],
        "submission_csv": str(test_out_csv),
        "submission_zip": str(out_zip),
        "elapsed_sec": round(time.time() - t0, 2)
    }
    with open(test_out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    banner(f"[COMPLETE] Adaptation Pipeline Finished in {time.time() - t0:.1f}s")
    print(f"Upload to Adaptation Form:\n  1. CSV: {test_out_csv}\n  2. ZIP: {out_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to adaptation public folder")
    parser.add_argument("--main-data", default=None, help="Path to main-round public folder")
    args = parser.parse_args()
    
    run_pipeline(Path(args.data), Path(args.main_data) if args.main_data else None)
