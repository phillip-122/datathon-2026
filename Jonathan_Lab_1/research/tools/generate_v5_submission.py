"""Generate, package, and verify Champion v5 (EXP-018) submission entry."""

import sys, os
from pathlib import Path
_LAB = Path(__file__).resolve().parents[2]
_REPO = _LAB.parent
sys.path[:0] = [str(_LAB), str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO), str(_LAB / 'src')]

import hashlib
import json
import shutil
import tempfile
import time
import joblib
import numpy as np
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_queries, read_submission, write_json, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from slot_forecasting.submission_contract import extract_archive, package_code
from hgb_upgrade.features import build_advanced_features as build_base_features
from lag_model.features import build_advanced_features as build_lag_features, PAYOUT_TO_IDX

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def predict_champ(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_base_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base_forecaster
        ) for q in qs
    ], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def predict_aug(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_lag_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base_forecaster
        ) for q in qs
    ], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def transform_hierarchical(P, m_indices, tg, bg, dt, db):
    logp = np.log(np.maximum(P, 1e-12))
    inv_t = np.exp(tg + dt[m_indices])
    logits = logp * inv_t[:, None] + bg[None, :]
    if db is not None:
        logits += db[m_indices, :]
    return softmax(logits, axis=1)

def main():
    data_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    test_queries, test_history = load_inputs(data_dir / "test")
    
    # 1. Load v4 champion zip to extract base models
    v4_zip = Path("compiled_versions/v4_promoted_champion_exp014/submission_entry.zip")
    with tempfile.TemporaryDirectory() as tmp_extract:
        tmp_path = Path(tmp_extract) / "unzipped"
        extract_archive(v4_zip.read_bytes(), tmp_path)
        base_forecaster = FrequencyForecaster.load(tmp_path / "base-entry/base.json")
        champ_models = joblib.load(tmp_path / "base-entry/champion.joblib")
        
        # Load EXP-018 augmented models
        exp018_dir = Path("research/experiments/EXP_018_augmented_hierarchical_blend")
        aug_models = joblib.load(exp018_dir / "aug_models.joblib")
        exp018_report = json.loads((exp018_dir / "report.json").read_text(encoding="utf-8"))
        cal_params = exp018_report["parameters"]
        
        tg = cal_params["theta_global"]
        bg = np.asarray(cal_params["beta_global"])
        dt = np.asarray([cal_params["delta_theta_machine"][m] for m in MTYPES])
        db = np.asarray([cal_params["delta_beta_machine"][m] for m in MTYPES])
        
        print(f"Predicting on {len(test_queries)} test queries with Champion v3/v4...")
        champ_preds = predict_champ(test_queries, test_history, base_forecaster, champ_models)
        
        print(f"Predicting on {len(test_queries)} test queries with 40/24 Augmented Ensemble...")
        aug_preds = predict_aug(test_queries, test_history, base_forecaster, aug_models)
        
        # 25% blend
        raw_test_P = []
        m_indices = []
        for q in test_queries:
            qid = q.query_id
            p_aug = np.asarray(aug_preds[qid])
            p_champ = np.asarray(champ_preds[qid])
            p_mix = 0.25 * p_aug + 0.75 * p_champ
            raw_test_P.append(p_mix)
            m_indices.append(MTYPE_MAP[q.machine_type])
            
        raw_test_P = np.asarray(raw_test_P)
        m_indices = np.asarray(m_indices)
        
        # Apply Hierarchical Calibration
        cal_test_P = transform_hierarchical(raw_test_P, m_indices, tg, bg, dt, db)
        
        # Format final submission
        final_test_preds = {}
        for i, q in enumerate(test_queries):
            raw_p = cal_test_P[i]
            tot = float(np.sum(raw_p))
            norm = tuple(float(v / tot) for v in raw_p)
            final_test_preds[q.query_id] = probabilities(norm, q.query_id)
            
    # Save destination folders:
    # 1. compiled_versions/v5_promoted_champion_exp018
    # 2. compiled_versions/active_champion_submission
    v5_dir = Path("compiled_versions/v5_promoted_champion_exp018")
    v5_dir.mkdir(parents=True, exist_ok=True)
    
    write_submission(v5_dir / "submission.csv", final_test_preds)
    
    # Create offline reproduction package
    entry_dir = v5_dir / "entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy packages
    for pkg in ["slot_forecasting", "hgb_upgrade", "lag_model", "calibrated_model"]:
        pkg_src = Path("src") / pkg if (Path("src") / pkg).exists() else (Path("House_Edge_Lab_Starter_Pack/research_environment") / pkg)
        if pkg_src.exists():
            shutil.copytree(pkg_src, entry_dir / pkg, dirs_exist_ok=True)
            
    # Copy models and weights
    joblib.dump(aug_models, entry_dir / "aug_models.joblib", compress=3)
    joblib.dump(champ_models, entry_dir / "champ_models.joblib", compress=3)
    base_forecaster.save(entry_dir / "base.json")
    
    # Save calibration config
    write_json(entry_dir / "hierarchical_calibration.json", cal_params)
    
    # Write runner predict script inside entry
    runner_py = entry_dir / "predict_v5.py"
    runner_py.write_text('''"""Offline prediction runner for Champion v5."""
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
from hgb_upgrade.features import build_advanced_features as build_base_features
from lag_model.features import build_advanced_features as build_lag_features

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def predict_champ(qs, h, base, models):
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    X = np.asarray([build_base_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def predict_aug(qs, h, base, models):
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    X = np.asarray([build_lag_features(q, grouped[(q.episode_id, q.machine_id)],
        [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id], q.cutoff, base) for q in qs], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def transform(P, m_indices, tg, bg, dt, db):
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
        # Fallback search
        matches = list(data.glob("**/test/queries.csv"))
        if matches:
            test_dir = matches[0].parent
        else:
            raise FileNotFoundError(f"Cannot find queries.csv under {data}")
            
    qs, h = load_inputs(test_dir)
    
    base = FrequencyForecaster.load(root / "base.json")
    champ_models = joblib.load(root / "champ_models.joblib")
    aug_models = joblib.load(root / "aug_models.joblib")
    cal = json.loads((root / "hierarchical_calibration.json").read_text())
    
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    
    p_champ = predict_champ(qs, h, base, champ_models)
    p_aug = predict_aug(qs, h, base, aug_models)
    
    P_raw = []
    m_indices = []
    for q in qs:
        p_mix = 0.25 * np.asarray(p_aug[q.query_id]) + 0.75 * np.asarray(p_champ[q.query_id])
        P_raw.append(p_mix)
        m_indices.append(MTYPE_MAP[q.machine_type])
        
    P_cal = transform(np.asarray(P_raw), np.asarray(m_indices), tg, bg, dt, db)
    out_preds = {}
    for i, q in enumerate(qs):
        raw = P_cal[i]
        tot = float(np.sum(raw))
        out_preds[q.query_id] = probabilities(tuple(float(v / tot) for v in raw), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print("Predictions saved to:", Path(args.out).resolve())

if __name__ == "__main__":
    main()
''', encoding="utf-8")

    # README & reproduce.json
    (entry_dir / "README.md").write_text("# Champion v5 (EXP-018): Observation Augmentation + Hierarchical Calibration\nPython 3.11+. Run python predict_v5.py --model-dir . --data /data --out /output/submission.csv.\n", encoding="utf-8")
    (entry_dir / "requirements.txt").write_text("numpy==2.3.4\nscipy==1.16.3\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\n", encoding="utf-8")
    write_json(entry_dir / "reproduce.json", {
        "image": "python",
        "command": ["python", "predict_v5.py", "--model-dir", ".", "--data", "/data", "--out", "/output/submission.csv"]
    })
    
    # Package into submission_entry.zip
    zip_dest = v5_dir / "submission_entry.zip"
    if zip_dest.exists():
        zip_dest.unlink()
    package_code(entry_dir, zip_dest)
    
    # Save full report
    write_json(v5_dir / "report.json", exp018_report)
    
    # Write MANIFEST.json
    manifest = {}
    for f in v5_dir.glob("*"):
        if f.is_file() and f.name != "MANIFEST.json":
            manifest[f.name] = {
                "bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest()
            }
    write_json(v5_dir / "MANIFEST.json", manifest)
    
    print("\n=======================================================")
    print("CHAMPION V5 PACKAGE CREATED")
    print("=======================================================")
    print(f"Directory: {v5_dir}")
    for k, v in manifest.items():
        print(f"  {k:22s} ({v['bytes']} bytes) SHA-256: {v['sha256']}")
        
    # Copy to compiled_versions/active_champion_submission
    active_dir = Path("compiled_versions/active_champion_submission")
    for f in ["submission.csv", "submission_entry.zip", "report.json", "MANIFEST.json"]:
        shutil.copy2(v5_dir / f, active_dir / f)
    print(f"Promoted to {active_dir}")

if __name__ == "__main__":
    main()
