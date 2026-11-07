"""Feature extraction and model evaluation script for House Edge."""
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.schema import PAYOUTS

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def build_features(query, sequence, peer_sequences, cutoff, base_forecaster=None):
    """Extract strictly cutoff-safe features available before `cutoff`."""
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
    
    # Interaction: type x signal (9 features)
    for t_name in ("steady", "balanced", "jackpot"):
        for s_val in (-1, 0, 1):
            feats.append(1.0 if (m_type == t_name and sig == s_val) else 0.0)
            
    # History length so far
    n_obs = len(sequence)
    feats.append(float(n_obs))
    feats.append(1.0 if n_obs == 64 else 0.0)
    
    # 2. Recency decay counts for half-lives [8, 16, 32]
    for hl in (8.0, 16.0, 32.0):
        counts = [0.0] * 5
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                w = 2.0 ** (-age / hl)
                counts[PAYOUT_TO_IDX[p]] += w
        tot = sum(counts) + 1e-6
        for c in counts:
            feats.append(c / tot)
        feats.append(sum(counts[1:]) / tot) # decayed win rate
        
    # 3. Rolling window win rates and mean payouts
    for win in (1, 2, 3, 5, 8, 12, 16, 24, 32):
        window_payouts = [sequence[t] for t in range(cutoff - win, cutoff) if t in sequence]
        if window_payouts:
            w_rate = sum(1 for p in window_payouts if p > 0) / len(window_payouts)
            m_payout = sum(window_payouts) / len(window_payouts)
            obs_ratio = len(window_payouts) / win
        else:
            w_rate = 0.5
            m_payout = 0.5
            obs_ratio = 0.0
        feats.extend([w_rate, m_payout, obs_ratio])
        
    # 4. Lag sequence features: lags 1 to 8
    for lag in range(1, 9):
        p = sequence.get(cutoff - lag, -1)
        feats.append(float(p))
        for target_p in (0, 1, 2, 5, 20):
            feats.append(1.0 if p == target_p else 0.0)
            
    # Streaks and recency
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
    
    # Ticks since last win
    last_win_age = 50.0
    for tick in sorted(sequence.keys(), reverse=True):
        if sequence[tick] > 0:
            last_win_age = float(cutoff - 1 - tick)
            break
    feats.append(last_win_age)
    
    # 5. Peer machine features
    if peer_sequences:
        peer_recent = []
        for p_seq in peer_sequences:
            for t in range(cutoff - 16, cutoff):
                if t in p_seq:
                    peer_recent.append(p_seq[t])
        if peer_recent:
            peer_win_rate = sum(1 for p in peer_recent if p > 0) / len(peer_recent)
        else:
            peer_win_rate = 0.5
        feats.append(peer_win_rate)
        
        # Peer wins at cutoff - 1
        peer_t1 = [p_seq.get(cutoff - 1, -1) for p_seq in peer_sequences]
        peer_t1_wins = sum(1 for p in peer_t1 if p > 0)
        feats.append(float(peer_t1_wins))
    else:
        feats.extend([0.5, 0.0])
        
    # 6. Baseline prior features if base_forecaster provided
    if base_forecaster is not None:
        key = f"{m_type}|{sig}"
        context_prior = base_forecaster.context_priors.get(key, base_forecaster.global_prior)
        feats.extend(context_prior)
        # pooled recent prediction for this sequence
        pr_counts = [10.0 * v for v in context_prior]
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                pr_counts[PAYOUT_TO_IDX[p]] += 2.0 ** (-age / 16.0)
        pr_tot = sum(pr_counts)
        feats.extend([v / pr_tot for v in pr_counts])
        
    return feats


def run_experiment():
    print("Loading datasets...")
    public = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    train_q, train_h = load_inputs(public / "train")
    val_q, val_h = load_inputs(public / "validation")
    train_labels = read_labels(public / "train/labels.csv", train_q)
    validation_targets = read_targets(public / "validation/targets.csv")
    group_rows = read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(public / "validation/reference.csv")
    
    # Fit baseline
    print("Fitting baseline forecaster...")
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    
    # Baseline score on full validation
    base_preds = base_forecaster.predict(val_q, val_h)
    base_sc = contest_score(base_preds, validation_targets, groups, reference, max_points=60.0)
    print(f"Base Pooled Recent (10, 16) full validation points: {base_sc['technical_points']:.4f}")
    
    # Organize train history by (episode_id, machine_id) -> {tick: payout}
    print("Indexing train history...")
    train_hist = defaultdict(lambda: defaultdict(dict))
    for h in train_h:
        train_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    train_q_map = {q.query_id: q for q in train_q}
    
    # Generate training examples across multiple cutoffs
    print("Generating multi-cutoff training examples from train...")
    X_train = []
    y_train = []
    
    for q in train_q:
        ep_seqs = train_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        sorted_ticks = sorted(m_seq.keys())
        if not sorted_ticks:
            continue
            
        # Target at query cutoff
        if q.query_id in train_labels:
            cutoff = q.cutoff
            peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
            f = build_features(q, m_seq, peers, cutoff, base_forecaster)
            X_train.append(f)
            y_train.append(PAYOUT_TO_IDX[train_labels[q.query_id]])
            
        # Also sample historical cutoffs (every 10 ticks from tick 40 up to cutoff - 1)
        for t in range(40, q.cutoff, 10):
            if t in m_seq:
                # Features using only ticks < t
                sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                if len(sub_seq) >= 10:
                    sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in ep_seqs.values() if p_seq is not m_seq]
                    f = build_features(q, sub_seq, sub_peers, t, base_forecaster)
                    X_train.append(f)
                    y_train.append(PAYOUT_TO_IDX[m_seq[t]])
                    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Generated X_train shape: {X_train.shape}, class counts: {np.bincount(y_train)}")
    
    # Extract validation features
    print("Extracting validation features...")
    val_hist = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        val_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X_val = []
    val_queries_ordered = list(val_q)
    for q in val_queries_ordered:
        ep_seqs = val_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_features(q, m_seq, peers, q.cutoff, base_forecaster)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    
    # Skip repeated grid search, jump straight to ensemble testing

    print("\n--- Testing Ensemble of HGB + LR + Baseline ---")
    hgb = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_val)
    
    lr = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)
    
    best_score = 0.0
    best_weights = None
    best_condition_breakdown = None
    
    for w_hgb in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for w_lr in [0.0, 0.1, 0.2, 0.25, 0.3]:
            w_base = 1.0 - w_hgb - w_lr
            if w_base < 0:
                continue
            preds = {}
            for i, q in enumerate(val_queries_ordered):
                b_p = base_preds[q.query_id]
                h_p = hgb_probs[i]
                l_p = lr_probs[i]
                combined = tuple(w_hgb * h_p[j] + w_lr * l_p[j] + w_base * b_p[j] for j in range(5))
                s = sum(combined)
                preds[q.query_id] = tuple(v / s for v in combined)
                
            sc = contest_score(preds, validation_targets, groups, reference, max_points=60.0)
            pts = sc['technical_points']
            if pts > best_score:
                best_score = pts
                best_weights = (w_hgb, w_lr, w_base)
                best_condition_breakdown = sc['by_condition']
            print(f"w=(HGB:{w_hgb:.2f}, LR:{w_lr:.2f}, Base:{w_base:.2f}) -> points: {pts:.4f}")

    print(f"\nBEST ENSEMBLE: HGB={best_weights[0]:.2f}, LR={best_weights[1]:.2f}, Base={best_weights[2]:.2f}")
    print(f"BEST TECHNICAL POINTS: {best_score:.4f} / 60")
    for c, row in best_condition_breakdown.items():
        print(f"   {c:14s}: loss_ratio={row['loss_ratio']:.4f}")

if __name__ == "__main__":
    run_experiment()
