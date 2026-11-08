"""Self-contained offline predictor for Champion Candidate v7 (EXP-022 Config 1: Drift Velocity Specialist)."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from collections import defaultdict
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import build_advanced_features as build_lag_features
from feature_factory import (
    extract_exp020_drift_features,
    compute_cutoff_safe_archetype,
    get_discrete_archetype_state,
    ArchetypeConditionedTransitionTable,
    MTYPES, MTYPE_MAP
)

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def extract_features(query, past, peer_past, cutoff, base, trans_table):
    feats = build_lag_features(query, past, peer_past, cutoff, base)
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    drift_feats = extract_exp020_drift_features(query, past, peer_past, cutoff)
    
    if "1" in ("2C", "3"):
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    else:
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=0)
    feats += trans_feats
    
    if "1" == "1":
        feats += drift_feats
    elif "1" == "3":
        feats += drift_feats
        feats += [1.0 if arch_conf == -1 else 0.0, 1.0 if arch_conf == 0 else 0.0, 1.0 if arch_conf == 1 else 0.0]
    return feats

def transform_hierarchical(P, M, cal):
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    logP = np.log(np.maximum(P, 1e-12))
    inv_T = np.exp(tg + dt[M])
    logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
    return softmax(logits, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--data", default="public")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    root = Path(args.model_dir)
    data = Path(args.data)
    test_dir = data / "test" if (data / "test/queries.csv").exists() else data
    if (data / "public/test/queries.csv").exists(): test_dir = data / "public/test"
    
    qs, h = load_inputs(test_dir)
    base = FrequencyForecaster.load(root / "base.json")
    models = joblib.load(root / "models.joblib")
    cal = json.loads((root / "hierarchical_calibration.json").read_text())
    
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    
    # Fit or load transition table
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    # Using fallback priors from base
    trans_table.global_prior = np.asarray(base.global_prior, dtype=np.float32)
    for m in MTYPES: trans_table.mt_prior[m] = np.asarray(base.global_prior, dtype=np.float32)
    
    X = np.asarray([
        extract_features(
            q,
            {t: p for t, p in grouped[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table
        ) for q in qs
    ], dtype=np.float32)
    
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    raw_probs = softmax(logits / temp[:, None], axis=1)
    
    M = np.asarray([MTYPE_MAP[q.machine_type] for q in qs])
    P_cal = transform_hierarchical(raw_probs, M, cal)
    
    out_preds = {}
    for i, q in enumerate(qs):
        row = P_cal[i]
        out_preds[q.query_id] = probabilities(tuple(float(v / np.sum(row)) for v in row), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print("Predictions saved to:", Path(args.out).resolve())

if __name__ == "__main__":
    main()
