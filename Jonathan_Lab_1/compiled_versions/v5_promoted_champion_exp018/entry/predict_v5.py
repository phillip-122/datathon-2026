"""Offline prediction runner for Champion v5."""
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
