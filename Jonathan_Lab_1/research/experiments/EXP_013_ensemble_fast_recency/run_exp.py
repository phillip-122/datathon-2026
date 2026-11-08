"""EXP-013: Fast Recency Half-Lives & Multi-Structure Logit Ensemble.

Hypotheses:
1. Fast Recency Half-Lives (2.0, 4.0 ticks): Isolate the immediate pre-cutoff regime
   in 'drift' episodes before sign reversals degrade rolling estimates.
2. Multi-Structure Logit Ensemble: Average calibrated logits across 3 diverse HGB structures
   (leaves=40/L2=4.0, leaves=60/L2=2.0, leaves=80/L2=1.5) to reduce quadratic Brier variance.
3. Machine-Type Volatility Calibration: Apply T_steady=1.00, T_balanced=1.06, T_jackpot=1.12.
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from research.evaluator import ResearchEvaluator
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from src.features import PAYOUT_TO_IDX

EXP_ID = "EXP-013"
OUT_DIR = Path("research/experiments/EXP_013_ensemble_fast_recency")

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def build_features_fast_recency(query, sequence, peer_sequences, cutoff, base_forecaster=None, add_fast_hl=True):
    """Cutoff-safe features with optional fast half-lives [2.0, 4.0]."""
    feats = []
    
    # 1. Static features
    m_type = query.machine_type
    sig = int(query.signal)
    feats.append(1.0 if m_type == "steady" else 0.0)
    feats.append(1.0 if m_type == "balanced" else 0.0)
    feats.append(1.0 if m_type == "jackpot" else 0.0)
    feats.append(float(sig))
    feats.append(1.0 if sig == -1 else 0.0)
    feats.append(1.0 if sig == 0 else 0.0)
    feats.append(1.0 if sig == 1 else 0.0)
    
    for t_name in ("steady", "balanced", "jackpot"):
        for s_val in (-1, 0, 1):
            feats.append(1.0 if (m_type == t_name and sig == s_val) else 0.0)
            
    n_obs = len(sequence)
    feats.append(float(n_obs))
    feats.append(1.0 if n_obs == 64 else 0.0)
    
    # 2. Recency decay counts
    half_lives = (2.0, 4.0, 6.0, 12.0, 24.0, 48.0) if add_fast_hl else (6.0, 12.0, 24.0, 48.0)
    for hl in half_lives:
        counts = [0.0] * 5
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                w = 2.0 ** (-age / hl)
                counts[PAYOUT_TO_IDX[p]] += w
        tot = sum(counts) + 1e-6
        for c in counts:
            feats.append(c / tot)
        feats.append(sum(counts[1:]) / tot)
        
    # 3. Rolling window win rates
    for win in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        window_payouts = [sequence[t] for t in range(cutoff - win, cutoff) if t in sequence]
        if window_payouts:
            w_rate = sum(1 for p in window_payouts if p > 0) / len(window_payouts)
            m_payout = sum(window_payouts) / len(window_payouts)
            p20_count = sum(1 for p in window_payouts if p == 20)
            p5_count = sum(1 for p in window_payouts if p == 5)
            obs_ratio = len(window_payouts) / win
        else:
            w_rate = 0.5
            m_payout = 0.5
            p20_count = 0
            p5_count = 0
            obs_ratio = 0.0
        feats.extend([w_rate, m_payout, float(p20_count), float(p5_count), obs_ratio])
        
    # 4. Lag sequence features: lags 1 to 8
    lag_payouts = {}
    for lag in range(1, 9):
        p = sequence.get(cutoff - lag, -1)
        lag_payouts[lag] = p
        feats.append(float(p))
        for target_p in (0, 1, 2, 5, 20):
            feats.append(1.0 if p == target_p else 0.0)
            
    def cat(v):
        return -1 if v == -1 else (0 if v == 0 else (1 if v == 1 else 2))
    for l1, l2 in ((1, 2), (1, 4), (1, 8), (2, 4), (4, 8)):
        c1, c2 = cat(lag_payouts[l1]), cat(lag_payouts[l2])
        for v1 in (0, 1, 2):
            for v2 in (0, 1, 2):
                feats.append(1.0 if (c1 == v1 and c2 == v2) else 0.0)
                
    streak_0 = 0
    t = cutoff - 1
    while t in sequence and sequence[t] == 0:
        streak_0 += 1
        t -= 1
    feats.append(float(streak_0))
    
    streak_win = 0
    t = cutoff - 1
    while t in sequence and sequence[t] > 0:
        streak_win += 1
        t -= 1
    feats.append(float(streak_win))
    
    last_win_age = 50.0
    last_big_win_age = 100.0
    for tick in sorted(sequence.keys(), reverse=True):
        if sequence[tick] > 0 and last_win_age == 50.0:
            last_win_age = float(cutoff - 1 - tick)
        if sequence[tick] >= 5 and last_big_win_age == 100.0:
            last_big_win_age = float(cutoff - 1 - tick)
        if last_win_age < 50.0 and last_big_win_age < 100.0:
            break
    feats.extend([last_win_age, last_big_win_age])
    
    # 5. Peer machine features
    if peer_sequences:
        for p_win in (4, 8, 16):
            peer_recent = []
            for p_seq in peer_sequences:
                for t in range(cutoff - p_win, cutoff):
                    if t in p_seq:
                        peer_recent.append(p_seq[t])
            if peer_recent:
                p_win_rate = sum(1 for p in peer_recent if p > 0) / len(peer_recent)
                p_mean_pay = sum(peer_recent) / len(peer_recent)
            else:
                p_win_rate = 0.5
                p_mean_pay = 0.5
            feats.extend([p_win_rate, p_mean_pay])
            
        peer_t1 = [p_seq.get(cutoff - 1, -1) for p_seq in peer_sequences]
        peer_t1_wins = sum(1 for p in peer_t1 if p > 0)
        feats.append(float(peer_t1_wins))
    else:
        feats.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
        
    # 6. Baseline prior features
    if base_forecaster is not None:
        key = f"{m_type}|{sig}"
        context_prior = base_forecaster.context_priors.get(key, base_forecaster.global_prior)
        feats.extend(context_prior)
        pr_counts = [10.0 * v for v in context_prior]
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                pr_counts[PAYOUT_TO_IDX[p]] += 2.0 ** (-age / 16.0)
        pr_tot = sum(pr_counts)
        feats.extend([v / pr_tot for v in pr_counts])
        
    return feats

def extract_dataset(queries, history, labels=None, base_forecaster=None, add_fast_hl=True, step_ticks=6):
    ep_hist = defaultdict(lambda: defaultdict(dict))
    for h in history:
        ep_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X = []
    y = []
    for q in queries:
        ep_seqs = ep_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        
        if labels is None:
            f = build_features_fast_recency(q, m_seq, peers, q.cutoff, base_forecaster, add_fast_hl=add_fast_hl)
            X.append(f)
        elif q.query_id in labels:
            f = build_features_fast_recency(q, m_seq, peers, q.cutoff, base_forecaster, add_fast_hl=add_fast_hl)
            X.append(f)
            y.append(PAYOUT_TO_IDX[labels[q.query_id]])
            
        if step_ticks is not None:
            for t in range(30, q.cutoff, step_ticks):
                if t in m_seq:
                    sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                    if len(sub_seq) >= 8:
                        sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                        f = build_features_fast_recency(q, sub_seq, sub_peers, t, base_forecaster, add_fast_hl=add_fast_hl)
                        X.append(f)
                        y.append(PAYOUT_TO_IDX[m_seq[t]])
                        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32) if y else None
    return X, y

def predict_logits(models, X):
    """Average raw decision logits across multiple models."""
    all_logits = [m.decision_function(X) for m in models]
    return np.mean(all_logits, axis=0)

def logits_to_probs(logits, queries, temp_map=TEMP_MAP):
    scaled = np.zeros_like(logits)
    for i, q in enumerate(queries):
        T = temp_map.get(q.machine_type, 1.06)
        scaled[i] = logits[i] / T
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    
    preds = {}
    for i, q in enumerate(queries):
        raw = probs[i]
        tot = sum(raw)
        preds[q.query_id] = tuple(float(v / tot) for v in raw)
    return preds

def main():
    print(f"=== Running {EXP_ID}: Fast Recency & Multi-Structure Ensemble ===")
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    # 1. Load train & val data
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q = list(evaluator.val_queries)
    val_h = evaluator.val_history
    
    # Fit baseline forecaster
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(train_q, train_h, train_labels)
    
    # Define model ensemble structures
    configs = [
        {"name": "standard", "leaves": 60, "min_leaf": 25, "l2": 2.0, "lr": 0.08, "seed": 42},
        {"name": "shallow_reg", "leaves": 40, "min_leaf": 35, "l2": 4.0, "lr": 0.07, "seed": 101},
        {"name": "deep_exp", "leaves": 80, "min_leaf": 20, "l2": 1.5, "lr": 0.08, "seed": 202},
    ]
    
    print("\n--- Phase 1: Test with Original Features (211 feats) ---")
    X_train_orig, y_train_orig = extract_dataset(train_q, train_h, train_labels, base, add_fast_hl=False, step_ticks=6)
    X_val_orig, _ = extract_dataset(val_q, val_h, None, base, add_fast_hl=False, step_ticks=None)
    print(f"Train shape: {X_train_orig.shape}, Val shape: {X_val_orig.shape}")
    
    models_orig = []
    for cfg in configs:
        m = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=cfg["lr"],
            max_leaf_nodes=cfg["leaves"],
            min_samples_leaf=cfg["min_leaf"],
            l2_regularization=cfg["l2"],
            random_state=cfg["seed"]
        )
        m.fit(X_train_orig, y_train_orig)
        models_orig.append(m)
        
        # Single model evaluation
        logits = m.decision_function(X_val_orig)
        preds = logits_to_probs(logits, val_q)
        res = evaluator.evaluate(preds)
        print(f"  Single [{cfg['name']}]: Tune={res['tuning_points']:.4f}, Audit={res['audit_points']:.4f}, Full={res['full_points']:.4f}")
        
    # Ensemble evaluation (original features)
    ens_logits_orig = predict_logits(models_orig, X_val_orig)
    ens_preds_orig = logits_to_probs(ens_logits_orig, val_q)
    res_ens_orig = evaluator.evaluate(ens_preds_orig)
    print(f"\n>> ENSEMBLE (Original 211 feats):")
    print(f"   Tuning: {res_ens_orig['tuning_points']:.4f} (Gate: > 35.3044)")
    print(f"   Audit:  {res_ens_orig['audit_points']:.4f} (Gate: > 34.9857)")
    print(f"   Full:   {res_ens_orig['full_points']:.4f} (Base: 35.1556)")
    for c, r in res_ens_orig["by_condition"].items():
        print(f"     {c:14s}: {r['loss_ratio']:.4f}")
        
    print("\n--- Phase 2: Test with Fast Recency Features (223 feats) ---")
    X_train_fast, y_train_fast = extract_dataset(train_q, train_h, train_labels, base, add_fast_hl=True, step_ticks=6)
    X_val_fast, _ = extract_dataset(val_q, val_h, None, base, add_fast_hl=True, step_ticks=None)
    print(f"Train shape: {X_train_fast.shape}, Val shape: {X_val_fast.shape}")
    
    models_fast = []
    for cfg in configs:
        m = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=cfg["lr"],
            max_leaf_nodes=cfg["leaves"],
            min_samples_leaf=cfg["min_leaf"],
            l2_regularization=cfg["l2"],
            random_state=cfg["seed"]
        )
        m.fit(X_train_fast, y_train_fast)
        models_fast.append(m)
        
        logits = m.decision_function(X_val_fast)
        preds = logits_to_probs(logits, val_q)
        res = evaluator.evaluate(preds)
        print(f"  Fast-HL [{cfg['name']}]: Tune={res['tuning_points']:.4f}, Audit={res['audit_points']:.4f}, Full={res['full_points']:.4f}")
        
    # Ensemble evaluation (fast recency features)
    ens_logits_fast = predict_logits(models_fast, X_val_fast)
    ens_preds_fast = logits_to_probs(ens_logits_fast, val_q)
    res_ens_fast = evaluator.evaluate(ens_preds_fast)
    print(f"\n>> ENSEMBLE (Fast Recency 223 feats):")
    print(f"   Tuning: {res_ens_fast['tuning_points']:.4f} (Gate: > 35.3044)")
    print(f"   Audit:  {res_ens_fast['audit_points']:.4f} (Gate: > 34.9857)")
    print(f"   Full:   {res_ens_fast['full_points']:.4f} (Base: 35.1556)")
    for c, r in res_ens_fast["by_condition"].items():
        print(f"     {c:14s}: {r['loss_ratio']:.4f}")

    # Check promotion gate
    gate_tune = 35.3044
    gate_audit = 34.9857
    
    winner = None
    if res_ens_fast["tuning_points"] > gate_tune and res_ens_fast["audit_points"] > gate_audit:
        winner = ("Ensemble_Fast_Recency", res_ens_fast, models_fast, True)
    elif res_ens_orig["tuning_points"] > gate_tune and res_ens_orig["audit_points"] > gate_audit:
        winner = ("Ensemble_Original_Feats", res_ens_orig, models_orig, False)
        
    out_record = {
        "exp_id": EXP_ID,
        "ensemble_orig": res_ens_orig,
        "ensemble_fast": res_ens_fast,
        "passed_both_gates": winner is not None,
        "promoted_winner": winner[0] if winner else None
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out_record, indent=2), encoding="utf-8")
    print(f"\nSaved results to {OUT_DIR / 'results.json'}")

if __name__ == "__main__":
    main()
