"""Offline prediction runner for Candidate v6 (EXP-020)."""
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
