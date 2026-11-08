"""Self-contained offline predictor for v8_candidate_solution_a (Solution A Optimized).

Formulation:
- Champion v6 backbone (Discrete Autoregressive Lag Sequences + Changing-Lag HGB + Hierarchical Calibrator)
- EXP-022 Config 3 component (EXP-020 Drift Features + Archetype-Conditioned Transitions + Confidence Gating)
- Condition-adaptive mixture weights per machine type:
    steady: 0.15 Config 3 + 0.85 Champion v6
    balanced: 0.28 Config 3 + 0.72 Champion v6
    jackpot: 0.35 Config 3 + 0.65 Champion v6
- Regularized Hierarchical Calibration fitted on tuning split (lambda_base=0.01, lambda_hier=0.08)
"""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from collections import defaultdict, Counter
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

def extract_features_cfg3(query, past, peer_past, cutoff, base, trans_table):
    feats = build_lag_features(query, past, peer_past, cutoff, base)
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    drift_feats = extract_exp020_drift_features(query, past, peer_past, cutoff)
    trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    feats += trans_feats
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
    if (data / "public/test/queries.csv").exists():
        test_dir = data / "public/test"
    
    qs, h = load_inputs(test_dir)
    grouped = defaultdict(dict)
    for r in h:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    M = np.asarray([MTYPE_MAP[q.machine_type] for q in qs])
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    
    # 1. Predict Champion v6 component offline
    from predict_v6 import (
        predict_ensemble as v6_predict_ensemble,
        predict_champ as v6_predict_champ,
        DiscreteLagTransitionTable as V6DiscreteLagTransitionTable,
        transform_hierarchical as v6_transform_hierarchical
    )
    base_champ = FrequencyForecaster.load(root / "base_champ.json")
    champ_models = joblib.load(root / "champ_models.joblib")
    base = FrequencyForecaster.load(root / "base.json")
    models_glob = joblib.load(root / "models_glob.joblib")
    models_mt = joblib.load(root / "models_mt.joblib")
    table_data = json.loads((root / "transition_tables.json").read_text())
    v6_table = V6DiscreteLagTransitionTable(table_data)
    v6_cal = json.loads((root / "hierarchical_calibration.json").read_text())
    
    p_champ = v6_predict_champ(qs, h, base_champ, champ_models)
    p_glob = v6_predict_ensemble(qs, h, base, models_glob, v6_table, use_mt=False)
    p_mt = v6_predict_ensemble(qs, h, base, models_mt, v6_table, use_mt=True)
    
    p_v6_raw = []
    for q in qs:
        p_disc = 0.5 * np.asarray(p_glob[q.query_id]) + 0.5 * np.asarray(p_mt[q.query_id])
        p_mix = 0.30 * p_disc + 0.70 * np.asarray(p_champ[q.query_id])
        p_v6_raw.append(p_mix)
    dt_v6 = np.asarray([v6_cal["delta_theta_machine"][m] for m in MTYPES])
    db_v6 = np.asarray([v6_cal["delta_beta_machine"][m] for m in MTYPES])
    p_v6_cal = v6_transform_hierarchical(np.asarray(p_v6_raw), M, v6_cal["theta_global"], np.asarray(v6_cal["beta_global"]), dt_v6, db_v6)
    
    # 2. Predict Config 3 component
    models_cfg3 = joblib.load(root / "models_cfg3.joblib")
    trans_data = json.loads((root / "trans_table_cfg3.json").read_text())
    trans_table_cfg3 = ArchetypeConditionedTransitionTable(alpha=trans_data["alpha"])
    trans_table_cfg3.global_prior = np.asarray(trans_data["global_prior"], dtype=np.float32)
    for m in MTYPES:
        trans_table_cfg3.mt_prior[m] = np.asarray(trans_data["mt_prior"][m], dtype=np.float32)
    for k_str, cnt in trans_data["pair_counts"].items():
        p1, p2 = [int(x) for x in k_str.split(",")]
        trans_table_cfg3.pair_counts[(p1, p2)] = Counter({int(out): c for out, c in cnt.items()})
    for arch_str, p_dict in trans_data["archetype_pair_counts"].items():
        arch = int(arch_str)
        for k_str, cnt in p_dict.items():
            p1, p2 = [int(x) for x in k_str.split(",")]
            trans_table_cfg3.archetype_pair_counts[arch][(p1, p2)] = Counter({int(out): c for out, c in cnt.items()})
            
    X_cfg3 = np.asarray([
        extract_features_cfg3(
            q,
            {t: p for t, p in grouped[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table_cfg3
        ) for q in qs
    ], dtype=np.float32)
    logits_cfg3 = np.mean([m.decision_function(X_cfg3) for m in models_cfg3], axis=0)
    raw_cfg3 = softmax(logits_cfg3 / temp[:, None], axis=1)
    
    # 3. Condition-adaptive mixture weights per machine type
    cal_v8 = json.loads((root / "hierarchical_calibration_v8.json").read_text())
    w_mtype = cal_v8.get("blend_weights", {"steady": 0.15, "balanced": 0.28, "jackpot": 0.35})
    w_vec = np.asarray([w_mtype[q.machine_type] for q in qs])
    
    test_blended = w_vec[:, None] * raw_cfg3 + (1.0 - w_vec[:, None]) * p_v6_cal
    test_blended = test_blended / np.sum(test_blended, axis=1, keepdims=True)
    
    # 4. Hierarchical calibration
    p_final = transform_hierarchical(test_blended, M, cal_v8)
    p_final = p_final / np.sum(p_final, axis=1, keepdims=True)
    
    out_preds = {}
    for i, q in enumerate(qs):
        row = p_final[i]
        out_preds[q.query_id] = probabilities(tuple(float(v / np.sum(row)) for v in row), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print("Predictions saved to:", Path(args.out).resolve())

if __name__ == "__main__":
    main()
