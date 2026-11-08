"""EXP-008: Missing-History Defaults & Imputation Fix.

Hypothesis: In the zero-heavy slot machine distribution (where base win rate is ~15-20%
and payout expectation is ~0.35), hardcoding 0.5 fallbacks for empty rolling windows and
peer machines introduces severe positive bias and distortion.
Replacing 0.5 with empirical machine-type base rates and providing explicit boolean missingness
indicators will allow gradient boosted decision trees to cleanly separate unobserved regimes
from active gameplay, reducing error on short histories and newly initialized machines.
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from research.evaluator import ResearchEvaluator
from research.run_experiment_helper import run_standard_eval
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels
from slot_forecasting.schema import PAYOUTS

EXP_ID = "EXP-008"
OUT_DIR = Path("research/experiments/EXP_008_missingness")
PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def compute_empirical_base_rates(train_queries, train_history):
    """Compute empirical win rates and payout averages per machine_type."""
    hist_by_m = defaultdict(list)
    for h in train_history:
        hist_by_m[(h.episode_id, h.machine_id)].append(h.payout)
        
    rates = {}
    type_payouts = defaultdict(list)
    for q in train_queries:
        payouts = hist_by_m[(q.episode_id, q.machine_id)]
        type_payouts[q.machine_type].extend(payouts)
        
    all_payouts = [h.payout for h in train_history]
    global_win_rate = sum(1 for p in all_payouts if p > 0) / len(all_payouts)
    global_mean_payout = sum(all_payouts) / len(all_payouts)
    
    for m_type in ("steady", "balanced", "jackpot"):
        p = type_payouts[m_type]
        if p:
            w_rate = sum(1 for x in p if x > 0) / len(p)
            m_pay = sum(p) / len(p)
        else:
            w_rate = global_win_rate
            m_pay = global_mean_payout
        rates[m_type] = {"win_rate": w_rate, "mean_payout": m_pay}
        
    rates["global"] = {"win_rate": global_win_rate, "mean_payout": global_mean_payout}
    return rates

def build_features_v2(query, sequence, peer_sequences, cutoff, base_rates, base_forecaster=None):
    """Cutoff-safe features with principled base-rate defaults and missingness indicators."""
    feats = []
    m_type = query.machine_type
    sig = int(query.signal)
    type_rate = base_rates.get(m_type, base_rates["global"])
    
    # 1. Static features & interactions
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
    feats.append(1.0 if n_obs < 16 else 0.0) # short history indicator
    
    # 2. Recency decay counts
    for hl in (6.0, 12.0, 24.0, 48.0):
        counts = [0.0] * 5
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                w = 2.0 ** (-age / hl)
                counts[PAYOUT_TO_IDX[p]] += w
        tot = sum(counts)
        if tot > 0:
            for c in counts:
                feats.append(c / tot)
            feats.append(sum(counts[1:]) / tot)
            feats.append(1.0) # has_decay_obs indicator
        else:
            # Fall back to machine type prior distribution instead of arbitrary numbers
            for c in (0.80, 0.12, 0.05, 0.02, 0.01):
                feats.append(c)
            feats.append(0.20)
            feats.append(0.0) # has_decay_obs = 0
            
    # 3. Rolling window win rates, payout sums, and non-zero counts
    for win in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        window_payouts = [sequence[t] for t in range(cutoff - win, cutoff) if t in sequence]
        if window_payouts:
            w_rate = sum(1 for p in window_payouts if p > 0) / len(window_payouts)
            m_payout = sum(window_payouts) / len(window_payouts)
            p20_count = sum(1 for p in window_payouts if p == 20)
            p5_count = sum(1 for p in window_payouts if p == 5)
            obs_ratio = len(window_payouts) / win
            is_empty = 0.0
        else:
            w_rate = type_rate["win_rate"]
            m_payout = type_rate["mean_payout"]
            p20_count = 0
            p5_count = 0
            obs_ratio = 0.0
            is_empty = 1.0
        feats.extend([w_rate, m_payout, float(p20_count), float(p5_count), obs_ratio, is_empty])
        
    # 4. Lag sequence features: lags 1 to 8
    lag_payouts = {}
    for lag in range(1, 9):
        p = sequence.get(cutoff - lag, -1)
        lag_payouts[lag] = p
        feats.append(float(p))
        feats.append(1.0 if p == -1 else 0.0) # is_missing indicator
        for target_p in (0, 1, 2, 5, 20):
            feats.append(1.0 if p == target_p else 0.0)
            
    # Pair interaction categories
    def cat(v):
        return -1 if v == -1 else (0 if v == 0 else (1 if v == 1 else 2))
    for l1, l2 in ((1, 2), (1, 4), (1, 8), (2, 4), (4, 8)):
        c1, c2 = cat(lag_payouts[l1]), cat(lag_payouts[l2])
        for v1 in (0, 1, 2):
            for v2 in (0, 1, 2):
                feats.append(1.0 if (c1 == v1 and c2 == v2) else 0.0)
                
    # Streaks
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
    
    # Ticks since last win (>0) and last big win (>=5)
    last_win_age = float(cutoff) # actual duration since cutoff start if no win
    last_big_win_age = float(cutoff)
    has_win = 0.0
    has_big_win = 0.0
    for tick in sorted(sequence.keys(), reverse=True):
        if sequence[tick] > 0 and has_win == 0.0:
            last_win_age = float(cutoff - 1 - tick)
            has_win = 1.0
        if sequence[tick] >= 5 and has_big_win == 0.0:
            last_big_win_age = float(cutoff - 1 - tick)
            has_big_win = 1.0
        if has_win == 1.0 and has_big_win == 1.0:
            break
    feats.extend([last_win_age, last_big_win_age, has_win, has_big_win])
    
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
                p_has_data = 1.0
            else:
                p_win_rate = type_rate["win_rate"]
                p_mean_pay = type_rate["mean_payout"]
                p_has_data = 0.0
            feats.extend([p_win_rate, p_mean_pay, p_has_data])
            
        peer_t1 = [p_seq.get(cutoff - 1, -1) for p_seq in peer_sequences]
        peer_t1_wins = sum(1 for p in peer_t1 if p > 0)
        feats.append(float(peer_t1_wins))
    else:
        feats.extend([
            type_rate["win_rate"], type_rate["mean_payout"], 0.0,
            type_rate["win_rate"], type_rate["mean_payout"], 0.0,
            type_rate["win_rate"], type_rate["mean_payout"], 0.0,
            0.0
        ])
        
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

def main():
    print(f"=== Starting {EXP_ID}: Missing-History Defaults & Imputation Fix ===")
    t_start = time.time()
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q, val_h = evaluator.val_queries, evaluator.val_history
    
    # 1. Base rates & baseline forecaster
    base_rates = compute_empirical_base_rates(train_q, train_h)
    print("Computed base rates:", base_rates)
    
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    
    # 2. Index train history
    train_hist = defaultdict(lambda: defaultdict(dict))
    for h in train_h:
        train_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    # 3. Extract training features with multi-cutoff step 6
    print("Extracting training features...")
    X_train, y_train = [], []
    for q in train_q:
        ep_seqs = train_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        
        if q.query_id in train_labels:
            f = build_features_v2(q, m_seq, peers, q.cutoff, base_rates, base_forecaster)
            X_train.append(f)
            y_train.append(PAYOUT_TO_IDX[train_labels[q.query_id]])
            
        for t in range(30, q.cutoff, 6):
            if t in m_seq:
                sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                if len(sub_seq) >= 8:
                    sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                    f = build_features_v2(q, sub_seq, sub_peers, t, base_rates, base_forecaster)
                    X_train.append(f)
                    y_train.append(PAYOUT_TO_IDX[m_seq[t]])
                    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train shape: {X_train.shape}")
    
    # 4. Extract validation features
    val_hist = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        val_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X_val = []
    val_q_list = list(val_q)
    for q in val_q_list:
        ep_seqs = val_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_features_v2(q, m_seq, peers, q.cutoff, base_rates, base_forecaster)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    print(f"X_val shape: {X_val.shape}")
    
    # 5. Fit HGB
    print("Fitting HistGradientBoostingClassifier (leaves=60, l2=2.0)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=60,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=42
    )
    hgb.fit(X_train, y_train)
    probs = hgb.predict_proba(X_val)
    
    preds = {}
    for i, q in enumerate(val_q_list):
        raw = probs[i]
        tot = sum(raw)
        preds[q.query_id] = tuple(float(v / tot) for v in raw)
        
    res = evaluator.evaluate(preds)
    
    print("\n--- RESULTS COMPARISON ---")
    print(f"Champion EXP-006: Tune: 35.1863 | Audit: 34.9486 | Full: 35.0743")
    print(f"EXP-008:          Tune: {res['tuning_points']:.4f} | Audit: {res['audit_points']:.4f} | Full: {res['full_points']:.4f}")
    print(f"Mean loss ratio:  {res['full_loss_ratio']:.4f}")
    for c, row in res["by_condition"].items():
        print(f"  {c:14s}: ratio={row['loss_ratio']:.4f} (err={row['error']:.4f}, ref={row['reference_error']:.4f})")
        
    decision = "PROMOTED" if (res["tuning_points"] > 35.1863 and res["audit_points"] > 34.9486) else "REJECTED"
    
    run_standard_eval(
        exp_id=EXP_ID,
        hypothesis="Principled empirical base rates and explicit missingness indicators",
        author="FeatureAgent",
        predictions=preds,
        out_dir=OUT_DIR,
        notes=f"Decision: {decision}. Total time: {time.time() - t_start:.1f}s"
    )
    print(f"\nFinal Decision for {EXP_ID}: {decision}")

if __name__ == "__main__":
    main()
