"""Offline prediction runner for Champion Candidate v6 (EXP-019)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from hgb_upgrade.features import build_advanced_features as build_base_features
from lag_model.features import build_advanced_features as build_lag_features

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

class DiscreteLagTransitionTable:
    def __init__(self, data):
        self.alpha = data["alpha"]
        self.global_prior = np.array(data["global_prior"], dtype=np.float32)
        self.mt_prior = {m: np.array(data["mt_prior"][m], dtype=np.float32) for m in MTYPES}
        self.pair_counts = {tuple(map(int, k.split(","))): {int(p): c for p, c in cnt.items()} for k, cnt in data["pair_counts"].items()}
        self.mt_pair_counts = {m: {tuple(map(int, k.split(","))): {int(p): c for p, c in cnt.items()} for k, cnt in data["mt_pair_counts"][m].items()} for m in MTYPES}
        
    def get_features(self, query, past, cutoff, use_machine_type=False):
        a = self.alpha
        mt = query.machine_type
        prior = self.mt_prior[mt] if use_machine_type else self.global_prior
        p1 = past.get(cutoff - 1)
        p2 = past.get(cutoff - 2)
        if p1 is not None and p2 is not None:
            s2 = (p1, p2)
            c2 = self.mt_pair_counts[mt].get(s2, {}) if use_machine_type else self.pair_counts.get(s2, {})
            n2 = sum(c2.values())
            return [(c2.get(p, 0) + a * prior[i]) / (n2 + a) for i, p in enumerate(PAYOUTS)]
        else:
            return list(prior)

def build_features(query, sequence, peer_sequences, cutoff, base_forecaster, transition_table, use_mt):
    past = {t: p for t, p in sequence.items() if t < cutoff}
    peers = [{t: p for t, p in s.items() if t < cutoff} for s in peer_sequences]
    base_f = build_lag_features(query, past, peers, cutoff, base_forecaster)
    trans_f = transition_table.get_features(query, past, cutoff, use_machine_type=use_mt)
    return base_f + trans_f

def predict_ensemble(qs, h, base_forecaster, models, transition_table, use_mt):
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    X = np.asarray([
        build_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base_forecaster,
            transition_table,
            use_mt
        ) for q in qs
    ], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def predict_champ(qs, h, base_forecaster, models):
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
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
    
    base = FrequencyForecaster.load(root / "base.json")
    base_champ = FrequencyForecaster.load(root / "base_champ.json")
    table_data = json.loads((root / "transition_tables.json").read_text(encoding="utf-8"))
    table = DiscreteLagTransitionTable(table_data)
    
    models_glob = joblib.load(root / "models_glob.joblib")
    models_mt = joblib.load(root / "models_mt.joblib")
    champ_models = joblib.load(root / "champ_models.joblib")
    
    cal = json.loads((root / "hierarchical_calibration.json").read_text(encoding="utf-8"))
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    
    p_champ = predict_champ(qs, h, base_champ, champ_models)
    p_glob = predict_ensemble(qs, h, base, models_glob, table, use_mt=False)
    p_mt = predict_ensemble(qs, h, base, models_mt, table, use_mt=True)
    
    P_raw = []
    m_indices = []
    for q in qs:
        qid = q.query_id
        p_disc = 0.5 * np.asarray(p_glob[qid]) + 0.5 * np.asarray(p_mt[qid])
        p_mix = 0.30 * p_disc + 0.70 * np.asarray(p_champ[qid])
        P_raw.append(p_mix)
        m_indices.append(MTYPE_MAP[q.machine_type])
        
    P_cal = transform_hierarchical(np.asarray(P_raw), np.asarray(m_indices), tg, bg, dt, db)
    out_preds = {}
    for i, q in enumerate(qs):
        raw = P_cal[i]
        tot = float(np.sum(raw))
        out_preds[q.query_id] = probabilities(tuple(float(v / tot) for v in raw), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print(f"Predictions saved to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
