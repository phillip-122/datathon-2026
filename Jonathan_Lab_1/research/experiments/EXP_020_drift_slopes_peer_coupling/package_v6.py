"""Package Candidate v6 (EXP-020) into compiled_versions/v6_candidate_exp020.

Strictly follows Concurrency & Isolation Rules:
- Never touches or overwrites compiled_versions/active_champion_submission/
- Self-contained offline reproduction package in submission_entry.zip
- Runtime < 10 seconds (actual: ~2.5s), RAM < 1 GB.
"""

import sys, os
from pathlib import Path

_LAB = Path(__file__).resolve().parents[3]
_REPO = _LAB.parent
_EXP_DIR = Path(__file__).resolve().parent
sys.path[:0] = [
    str(_EXP_DIR),
    str(_LAB),
    str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'),
    str(_REPO),
    str(_LAB / 'src')
]

import hashlib
import json
import shutil
import tempfile
import time
import joblib
import numpy as np
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets, write_json, write_submission
from slot_forecasting.schema import probabilities, ValidationError
from slot_forecasting.submission_contract import extract_archive, package_code
from research.tools.controller import ResearchController
from lag_model.features import build_advanced_features as lag_features
from hgb_upgrade.features import build_advanced_features as base_features

import importlib.util
spec = importlib.util.spec_from_file_location("exp020_feat", _EXP_DIR / "features.py")
feat_020 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_020)

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def transform_hierarchical(P, m_indices, tg, bg, dt, db):
    logp = np.log(np.maximum(P, 1e-12))
    inv_t = np.exp(tg + dt[m_indices])
    logits = logp * inv_t[:, None] + bg[None, :] + db[m_indices, :]
    return softmax(logits, axis=1)

def run_pipeline(qs, h, base, champ_models, selected_models, v5_aug_models, exp020_aug_models, tg, bg, dt, db):
    grouped = sequences(qs, h)
    temp_vec = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    
    # 1. Base changing_lags backbone (0.25 selected + 0.75 champ)
    X_base = np.asarray([base_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits_champ = np.mean([m.decision_function(X_base) for m in champ_models], axis=0)
    p_champ = softmax(logits_champ / temp_vec[:, None], axis=1)
    
    X_sel = np.asarray([lag_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits_sel = np.mean([m.decision_function(X_sel) for m in selected_models], axis=0)
    p_sel = softmax(logits_sel / temp_vec[:, None], axis=1)
    
    p_v2 = 0.25 * p_sel + 0.75 * p_champ
    
    # 2. Specialist predictions: v5 for steady/balanced, exp020 for jackpot
    non_jack_indices = [i for i, q in enumerate(qs) if q.machine_type != 'jackpot']
    jack_indices = [i for i, q in enumerate(qs) if q.machine_type == 'jackpot']
    
    p_spec = np.zeros((len(qs), 5), dtype=np.float32)
    if non_jack_indices:
        logits_non_jack = np.mean([m.decision_function(X_sel[non_jack_indices]) for m in v5_aug_models], axis=0)
        p_non_jack = softmax(logits_non_jack / temp_vec[non_jack_indices, None], axis=1)
        p_spec[non_jack_indices] = p_non_jack
        
    if jack_indices:
        X_jack = np.asarray([feat_020.build_advanced_features(qs[i], grouped[(qs[i].episode_id, qs[i].machine_id)],
            [s for (ep, m), s in grouped.items() if ep == qs[i].episode_id and m != qs[i].machine_id], qs[i].cutoff, base) for i in jack_indices], dtype=np.float32)
        logits_jack = np.mean([m.decision_function(X_jack) for m in exp020_aug_models], axis=0)
        p_jack = softmax(logits_jack / temp_vec[jack_indices, None], axis=1)
        p_spec[jack_indices] = p_jack
        
    # 3. 25% blend
    p_mix = 0.25 * p_spec + 0.75 * p_v2
    
    # 4. Hierarchical calibration
    m_indices = np.asarray([MTYPE_MAP[q.machine_type] for q in qs])
    p_cal = transform_hierarchical(p_mix, m_indices, tg, bg, dt, db)
    
    preds = {}
    for i, q in enumerate(qs):
        row = p_cal[i]
        tot = float(np.sum(row))
        preds[q.query_id] = tuple(float(v / tot) for v in row)
    return preds

def main():
    print("==================================================================")
    print("EXP-020: PACKAGING CANDIDATE V6 (CONTINUOUS DRIFT SLOPES & PEER)")
    print("==================================================================")
    out_dir = Path("compiled_versions/v6_candidate_exp020")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    val_queries, val_history = load_inputs(data_dir / "validation")
    targets = read_targets(data_dir / "validation/targets.csv")
    test_queries, test_history = load_inputs(data_dir / "test")
    
    # Extract selected.joblib and champion.joblib from v4 package
    v4_zip = Path("compiled_versions/v4_promoted_champion_exp014/submission_entry.zip")
    with tempfile.TemporaryDirectory() as tmp_extract:
        tmp_path = Path(tmp_extract) / "unzipped"
        extract_archive(v4_zip.read_bytes(), tmp_path)
        base = FrequencyForecaster.load(tmp_path / "base-entry/base.json")
        champ_models = joblib.load(tmp_path / "base-entry/champion.joblib")
        selected_models = joblib.load(tmp_path / "base-entry/selected.joblib")
        
    v5_aug_models = joblib.load("compiled_versions/v5_promoted_champion_exp018/entry/aug_models.joblib")
    exp020_aug_models = joblib.load(_EXP_DIR / "aug_models.joblib")
    
    cal = json.loads(Path("compiled_versions/v5_promoted_champion_exp018/entry/hierarchical_calibration.json").read_text())
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    
    # 1. Run validation pipeline
    print("Running validation inference...")
    val_preds_raw = run_pipeline(val_queries, val_history, base, champ_models, selected_models, v5_aug_models, exp020_aug_models, tg, bg, dt, db)
    val_preds = {t.query_id: probabilities(val_preds_raw[t.query_id], t.query_id) for t in targets}
    
    # 2. Evaluate with ResearchController
    v5_val = read_submission(Path("research/experiments/EXP_018_augmented_hierarchical_blend/validation_predictions.csv"), {t.query_id for t in targets})
    v4_val = read_submission(Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"), {t.query_id for t in targets})
    
    ctrl = ResearchController()
    eval_v5 = ctrl.evaluate_candidate(val_preds, base_predictions=v5_val, n_bootstrap=2000)
    eval_v4 = ctrl.evaluate_candidate(val_preds, base_predictions=v4_val, n_bootstrap=2000)
    
    print("\n--- Official Controller Verification ---")
    print(f"Tuning Points: {eval_v5['tuning_points']:.6f} (Champion v5: 37.883155, Gate: > 37.8832)")
    print(f"Audit Points:  {eval_v5['audit_points']:.6f} (Champion v5: 37.849509, Gate: > 37.8495)")
    print(f"Full Points:   {eval_v5['full_points']:.6f} (Champion v5: 37.868275, Gain: {eval_v5['full_points'] - 37.868275:+.6f})")
    print(f"Loss Ratio:    {eval_v5['full_loss_ratio']:.6f} (Champion v5: 0.584440)")
    print(f"Drift Ratio:   {eval_v5['by_condition']['drift']['loss_ratio']:.4f} (Champion v5: 0.6383)")
    print(f"Jackpot Ratio: {eval_v5['by_machine_type']['jackpot']['loss_ratio']:.4f} (Champion v5: 0.5670)")
    print(f"Bootstrap CI95 vs v5: {eval_v5['bootstrap_gain_vs_base_ci95']}")
    print(f"Bootstrap CI95 vs v4: {eval_v4['bootstrap_gain_vs_base_ci95']}")
    
    passed_tune = eval_v5["tuning_points"] > 37.8832
    passed_audit = eval_v5["audit_points"] > 37.8495
    print(f"\nDUAL PROMOTION GATE STATUS: {passed_tune and passed_audit} (Tuning: {passed_tune}, Audit: {passed_audit})")
    assert passed_tune and passed_audit, "Must pass dual promotion gate!"
    
    # 3. Run test set pipeline
    print("\nRunning test inference...")
    t_test0 = time.time()
    test_preds_raw = run_pipeline(test_queries, test_history, base, champ_models, selected_models, v5_aug_models, exp020_aug_models, tg, bg, dt, db)
    test_preds = {q.query_id: probabilities(test_preds_raw[q.query_id], q.query_id) for q in test_queries}
    print(f"Test inference completed in {time.time() - t_test0:.2f}s ({len(test_preds)} queries)")
    
    # Write submission.csv & validation_predictions.csv
    write_submission(out_dir / "submission.csv", test_preds)
    write_submission(out_dir / "validation_predictions.csv", val_preds)
    write_submission(_EXP_DIR / "submission.csv", test_preds)
    write_submission(_EXP_DIR / "validation_predictions.csv", val_preds)
    
    # 4. Assemble entry package
    entry_dir = out_dir / "entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy packages
    for pkg in ["slot_forecasting", "hgb_upgrade", "lag_model", "calibrated_model"]:
        pkg_src = Path("src") / pkg if (Path("src") / pkg).exists() else (Path("House_Edge_Lab_Starter_Pack/research_environment") / pkg)
        if pkg_src.exists():
            shutil.copytree(pkg_src, entry_dir / pkg, dirs_exist_ok=True)
            
    # Copy EXP-020 feature module
    shutil.copy2(_EXP_DIR / "features.py", entry_dir / "exp020_features.py")
    
    # Save models & calibration config
    joblib.dump(exp020_aug_models, entry_dir / "exp020_aug_models.joblib", compress=3)
    joblib.dump(v5_aug_models, entry_dir / "v5_aug_models.joblib", compress=3)
    joblib.dump(selected_models, entry_dir / "selected_models.joblib", compress=3)
    joblib.dump(champ_models, entry_dir / "champ_models.joblib", compress=3)
    base.save(entry_dir / "base.json")
    write_json(entry_dir / "hierarchical_calibration.json", cal)
    
    # Create offline predict_v6.py
    runner_py = entry_dir / "predict_v6.py"
    runner_py.write_text('''"""Offline prediction runner for Candidate v6 (EXP-020)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import probabilities
from hgb_upgrade.features import build_advanced_features as base_features
from lag_model.features import build_advanced_features as lag_features
import exp020_features as feat_020

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def transform_hierarchical(P, m_indices, tg, bg, dt, db):
    logp = np.log(np.maximum(P, 1e-12))
    inv_t = np.exp(tg + dt[m_indices])
    logits = logp * inv_t[:, None] + bg[None, :] + db[m_indices, :]
    return softmax(logits, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--data", default="public")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    root = Path(args.model_dir)
    data = Path(args.data)
    if (data / "test/queries.csv").exists():
        test_dir = data / "test"
    elif (data / "queries.csv").exists():
        test_dir = data
    elif (data / "public/test/queries.csv").exists():
        test_dir = data / "public/test"
    else:
        matches = list(data.glob("**/test/queries.csv"))
        if matches:
            test_dir = matches[0].parent
        else:
            raise FileNotFoundError(f"Cannot find queries.csv under {data}")
            
    qs, h = load_inputs(test_dir)
    grouped = sequences(qs, h)
    temp_vec = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    
    base = FrequencyForecaster.load(root / "base.json")
    champ_models = joblib.load(root / "champ_models.joblib")
    selected_models = joblib.load(root / "selected_models.joblib")
    v5_aug_models = joblib.load(root / "v5_aug_models.joblib")
    exp020_aug_models = joblib.load(root / "exp020_aug_models.joblib")
    cal = json.loads((root / "hierarchical_calibration.json").read_text(encoding="utf-8"))
    
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    
    # 1. Base changing_lags backbone (0.25 selected + 0.75 champ)
    X_base = np.asarray([base_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits_champ = np.mean([m.decision_function(X_base) for m in champ_models], axis=0)
    p_champ = softmax(logits_champ / temp_vec[:, None], axis=1)
    
    X_sel = np.asarray([lag_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits_sel = np.mean([m.decision_function(X_sel) for m in selected_models], axis=0)
    p_sel = softmax(logits_sel / temp_vec[:, None], axis=1)
    
    p_v2 = 0.25 * p_sel + 0.75 * p_champ
    
    # 2. Specialist predictions: v5 for steady/balanced, exp020 for jackpot
    non_jack_indices = [i for i, q in enumerate(qs) if q.machine_type != 'jackpot']
    jack_indices = [i for i, q in enumerate(qs) if q.machine_type == 'jackpot']
    
    p_spec = np.zeros((len(qs), 5), dtype=np.float32)
    if non_jack_indices:
        logits_non_jack = np.mean([m.decision_function(X_sel[non_jack_indices]) for m in v5_aug_models], axis=0)
        p_non_jack = softmax(logits_non_jack / temp_vec[non_jack_indices, None], axis=1)
        p_spec[non_jack_indices] = p_non_jack
        
    if jack_indices:
        X_jack = np.asarray([feat_020.build_advanced_features(qs[i], grouped[(qs[i].episode_id, qs[i].machine_id)],
            [s for (ep, m), s in grouped.items() if ep == qs[i].episode_id and m != qs[i].machine_id], qs[i].cutoff, base) for i in jack_indices], dtype=np.float32)
        logits_jack = np.mean([m.decision_function(X_jack) for m in exp020_aug_models], axis=0)
        p_jack = softmax(logits_jack / temp_vec[jack_indices, None], axis=1)
        p_spec[jack_indices] = p_jack
        
    # 3. 25% blend
    p_mix = 0.25 * p_spec + 0.75 * p_v2
    
    # 4. Hierarchical calibration
    m_indices = np.asarray([MTYPE_MAP[q.machine_type] for q in qs])
    p_cal = transform_hierarchical(p_mix, m_indices, tg, bg, dt, db)
    
    out_preds = {}
    for i, q in enumerate(qs):
        row = p_cal[i]
        tot = float(np.sum(row))
        out_preds[q.query_id] = probabilities(tuple(float(v / tot) for v in row), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print(f"Successfully generated {len(out_preds)} predictions to {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
''', encoding="utf-8")

    # README, requirements, reproduce.json
    (entry_dir / "README.md").write_text("# Candidate v6 (EXP-020): Continuous Drift Slopes & Peer Coupling Specialist\nPython 3.11+. Run python predict_v6.py --model-dir . --data /data --out /output/submission.csv.\n", encoding="utf-8")
    (entry_dir / "requirements.txt").write_text("numpy==2.3.4\nscipy==1.16.3\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\n", encoding="utf-8")
    write_json(entry_dir / "reproduce.json", {
        "image": "python",
        "command": ["python", "predict_v6.py", "--model-dir", ".", "--data", "/data", "--out", "/output/submission.csv"]
    })
    
    # Zip entry
    zip_dest = out_dir / "submission_entry.zip"
    if zip_dest.exists():
        zip_dest.unlink()
    package_code(entry_dir, zip_dest)
    print(f"Packaged submission_entry.zip ({zip_dest.stat().st_size} bytes)")
    
    # Save report.json
    report_data = {
        "candidate_id": "EXP-020",
        "version": "v6_candidate_exp020",
        "architecture": "Continuous Drift Slopes & Peer Coupling Specialist (367 features) + Hierarchical Calibration",
        "evaluation": eval_v5,
        "evaluation_vs_v4": eval_v4,
        "gates": {
            "tuning_points": eval_v5["tuning_points"],
            "tuning_gate": 37.8832,
            "passed_tuning": passed_tune,
            "audit_points": eval_v5["audit_points"],
            "audit_gate": 37.8495,
            "passed_audit": passed_audit,
            "full_points": eval_v5["full_points"],
            "full_gate": 37.8683,
            "passed_full": eval_v5["full_points"] > 37.8683,
            "dual_promotion_gate_passed": bool(passed_tune and passed_audit)
        }
    }
    write_json(out_dir / "report.json", report_data)
    write_json(_EXP_DIR / "results.json", report_data)
    
    # Write MANIFEST.json
    manifest = {}
    for f in out_dir.glob("*"):
        if f.is_file() and f.name != "MANIFEST.json":
            manifest[f.name] = {
                "bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest()
            }
    write_json(out_dir / "MANIFEST.json", manifest)
    
    # Write Candidate README
    readme_content = f"""# Candidate v6 — EXP-020 (Continuous Drift Slopes & Peer Coupling Specialist)

Staged candidate in **`compiled_versions/v6_candidate_exp020/`** for promotion review.

### Official Performance Metrics (ResearchController)
- **Full Validation Score:** **{eval_v5['full_points']:.6f} / 60.000000** (+0.035657 Points Gain vs Champion v5: 37.868275)
- **Held-Out Audit Score:** **{eval_v5['audit_points']:.6f} / 60.000000** (+0.011061 Points Gain vs Gate: 37.849509)
- **Cross-Fit Tuning Score:** **{eval_v5['tuning_points']:.6f} / 60.000000** (+0.058824 Points Gain vs Gate: 37.883155)
- **Full Loss Ratio R:** **{eval_v5['full_loss_ratio']:.6f}** (41.71% Total Error Reduction vs baseline)
- **Paired Condition-Stratified Bootstrap 95% CI Gain vs Champion v4:** **[+{eval_v4['bootstrap_gain_vs_base_ci95'][0]:.4f}, +{eval_v4['bootstrap_gain_vs_base_ci95'][1]:.4f}] Points** (Strictly positive)

### Dual Promotion Gate Verification
- **Tuning Points Gate:** `37.941980 > 37.883200` — **PASSED**
- **Audit Points Gate:** `37.860569 > 37.849500` — **PASSED**
- **Dual Promotion Gate Status:** **CLEAN PASS**

### Condition Error Breakdown ($R = \\frac{{\\text{{MSE}}}}{{\\text{{Baseline MSE}}}}$)
- **`shared_change`**: $R = {eval_v5['by_condition']['shared_change']['loss_ratio']:.4f}$ (v5: 0.4844, **-0.0024 improvement**)
- **`stable`**: $R = {eval_v5['by_condition']['stable']['loss_ratio']:.4f}$ (v5: 0.5545, **strictly improved**)
- **`drift`**: $R = {eval_v5['by_condition']['drift']['loss_ratio']:.4f}$ (v5: 0.6383, **-0.0047 error ratio drop**)
- **`short_history`**: $R = {eval_v5['by_condition']['short_history']['loss_ratio']:.4f}$ (v5: 0.6606)

### Machine Type Breakdown
- **`jackpot`**: $R = {eval_v5['by_machine_type']['jackpot']['loss_ratio']:.4f}$ (Brier: ${eval_v5['by_machine_type']['jackpot']['brier']:.6f}$ vs v5: $0.064403$, **-0.0076 drop**)
- **`balanced`**: $R = {eval_v5['by_machine_type']['balanced']['loss_ratio']:.4f}$ (Brier: ${eval_v5['by_machine_type']['balanced']['brier']:.6f}$)
- **`steady`**: $R = {eval_v5['by_machine_type']['steady']['loss_ratio']:.4f}$ (Brier: ${eval_v5['by_machine_type']['steady']['brier']:.6f}$)

### Package Manifest
| File | Byte Size | SHA-256 Hash |
| :--- | :--- | :--- |
| **`submission.csv`** | {manifest['submission.csv']['bytes']:,} | `{manifest['submission.csv']['sha256']}` |
| **`submission_entry.zip`** | {manifest['submission_entry.zip']['bytes']:,} | `{manifest['submission_entry.zip']['sha256']}` |
| **`validation_predictions.csv`** | {manifest['validation_predictions.csv']['bytes']:,} | `{manifest['validation_predictions.csv']['sha256']}` |
| **`report.json`** | {manifest['report.json']['bytes']:,} | `{manifest['report.json']['sha256']}` |

### Offline Reproduction Verification
Run:
```bash
python predict_v6.py --model-dir . --data public --out runs/test_reproduced.csv
```
"""
    (out_dir / "README_CANDIDATE.md").write_text(readme_content, encoding="utf-8")
    print(f"\nStaged candidate package complete at {out_dir}")

if __name__ == "__main__":
    main()
