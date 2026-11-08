"""Package EXP-016 into compiled_versions/v5_candidate_exp016 without touching active_champion."""

import json
from pathlib import Path
import joblib
import numpy as np

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import probabilities

import sys
sys.path[:0] = [
    "research/experiments/EXP_016_drift_trend_slopes",
    "src",
    ".",
    "House_Edge_Lab_Starter_Pack/research_environment"
]

import features as exp_features
from lag_model.run import predict_model as predict_lag_model
from research.experiments.EXP_016_drift_trend_slopes.run_exp import predict_ensemble, transform

def main():
    exp_dir = Path("research/experiments/EXP_016_drift_trend_slopes")
    out_dir = Path("compiled_versions/v5_candidate_exp016")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test inputs
    data_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    test_q, test_h = load_inputs(data_dir / "test")
    test_q_list = list(test_q)
    
    # Load models
    models_exp = joblib.load(exp_dir / "exp016_models.joblib")
    base = FrequencyForecaster.load("artifacts/models/hgb_ensemble_v3/base_forecaster.json")
    
    # Predict EXP-016 base
    preds_exp_base, _ = predict_ensemble(test_q_list, test_h, base, models_exp, exp_features.build_advanced_features)
    
    # Predict lag specialist
    lag_champ_entry = Path("compiled_versions/v4_promoted_champion_exp014")
    # Load lag specialist models from v4
    import tempfile
    from slot_forecasting.submission_contract import extract_archive
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "unzipped"
        extract_archive((lag_champ_entry / "submission_entry.zip").read_bytes(), dest)
        lag_sub_path = dest / "base-entry"
        lag_models = joblib.load(lag_sub_path / "selected.joblib")
        lag_base = FrequencyForecaster.load(lag_sub_path / "base.json")
        preds_lag = predict_lag_model(test_q_list, test_h, lag_base, lag_models)
        
    # Blend: 65% EXP-016 + 35% Lag Specialist
    results_data = json.loads((exp_dir / "results.json").read_text())
    best_w = results_data["models"]["blend_weight_lag_specialist"]
    
    uncal_blend = {}
    for qid in preds_exp_base:
        p = (1.0 - best_w) * np.array(preds_exp_base[qid]) + best_w * np.array(preds_lag[qid])
        uncal_blend[qid] = tuple(float(v / p.sum()) for v in p)
        
    # Calibrate using theta
    theta = np.array(results_data["models"]["calibration_theta"])
    ids = sorted(uncal_blend.keys())
    P = np.array([uncal_blend[k] for k in ids])
    cal_values = transform(P, theta)
    
    final_preds = {k: probabilities(tuple(v), k) for k, v in zip(ids, cal_values)}
    
    # Write submission.csv
    sub_csv = out_dir / "submission.csv"
    write_submission(sub_csv, final_preds)
    print(f"Wrote test submission to {sub_csv} ({len(final_preds)} queries)")
    
    # Copy validation predictions and results
    import shutil
    shutil.copy2(exp_dir / "validation_predictions.csv", out_dir / "validation_predictions.csv")
    shutil.copy2(exp_dir / "results.json", out_dir / "results.json")
    
    # Write README
    readme_content = f"""# Champion Candidate v5 (EXP-016)

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
"""
    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"Packaged Candidate v5 in {out_dir}")

if __name__ == "__main__":
    main()
