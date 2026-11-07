"""Advanced Feature Engineering and Model Exploration for House Edge."""
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.schema import PAYOUTS

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def build_advanced_features(query, sequence, peer_sequences, cutoff, base_forecaster=None):
    """Cutoff-safe advanced features."""
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
    
    # Interaction: type x signal
    for t_name in ("steady", "balanced", "jackpot"):
        for s_val in (-1, 0, 1):
            feats.append(1.0 if (m_type == t_name and sig == s_val) else 0.0)
            
    n_obs = len(sequence)
    feats.append(float(n_obs))
    feats.append(1.0 if n_obs == 64 else 0.0)
    
    # 2. Recency decay counts for multiple half-lives
    for hl in (6.0, 12.0, 24.0, 48.0):
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
        
    # 3. Rolling window win rates, payout sums, and non-zero counts
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
            
    # Pair interaction features: (lag1, lag2), (lag1, lag4), (lag1, lag8), (lag2, lag4)
    # Simplify category: 0, 1, 2+ (or -1 if missing)
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
            
        # Peer wins at cutoff - 1
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

def main():
    print("=== ADVANCED MODEL EXPERIMENT ===")
    start_time = time.time()
    public = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    train_q, train_h = load_inputs(public / "train")
    val_q, val_h = load_inputs(public / "validation")
    train_labels = read_labels(public / "train/labels.csv", train_q)
    validation_targets = read_targets(public / "validation/targets.csv")
    group_rows = read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(public / "validation/reference.csv")
    
    tuning_eps, audit_eps = episode_folds(validation_targets, groups)
    
    # Baseline
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(train_q, train_h, train_labels)
    base_preds = base.predict(val_q, val_h)
    
    # Train index
    train_hist = defaultdict(lambda: defaultdict(dict))
    for h in train_h:
        train_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    # Multi-cutoff generation: step size 6
    print("Extracting training features (sampling step = 6 ticks)...")
    X_train, y_train = [], []
    for q in train_q:
        ep_seqs = train_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        
        # At cutoff
        if q.query_id in train_labels:
            f = build_advanced_features(q, m_seq, peers, q.cutoff, base)
            X_train.append(f)
            y_train.append(PAYOUT_TO_IDX[train_labels[q.query_id]])
            
        # Historical ticks every 6 ticks
        for t in range(30, q.cutoff, 6):
            if t in m_seq:
                sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                if len(sub_seq) >= 8:
                    sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                    f = build_advanced_features(q, sub_seq, sub_peers, t, base)
                    X_train.append(f)
                    y_train.append(PAYOUT_TO_IDX[m_seq[t]])
                    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train shape: {X_train.shape}, classes: {np.bincount(y_train)}")
    
    # Val features
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
        f = build_advanced_features(q, m_seq, peers, q.cutoff, base)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    print(f"X_val shape: {X_val.shape}")
    
    # Fit HistGradientBoostingClassifier
    print("\nTraining HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=42
    )
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_val)
    
    # Fit LogisticRegression
    print("Training LogisticRegression...")
    lr = LogisticRegression(C=0.02, max_iter=500, solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)
    
    # Fit ExtraTreesClassifier
    print("Training ExtraTreesClassifier...")
    et = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)
    et.fit(X_train, y_train)
    et_probs = et.predict_proba(X_val)
    
    # Evaluate individual models and blends on Tuning fold, Audit fold, and Full validation
    models = {
        "Baseline": base_preds,
        "HGB": {q.query_id: tuple(hgb_probs[i]) for i, q in enumerate(val_queries_ordered)},
        "LR": {q.query_id: tuple(lr_probs[i]) for i, q in enumerate(val_queries_ordered)},
        "ET": {q.query_id: tuple(et_probs[i]) for i, q in enumerate(val_queries_ordered)},
    }
    
    for name, pred_dict in models.items():
        t_sc = subset_score(pred_dict, validation_targets, groups, reference, tuning_eps, 60.0)
        a_sc = subset_score(pred_dict, validation_targets, groups, reference, audit_eps, 60.0)
        f_sc = contest_score(pred_dict, validation_targets, groups, reference, max_points=60.0)
        print(f"{name:10s} -> Tuning: {t_sc['technical_points']:.4f} | Audit: {a_sc['technical_points']:.4f} | Full: {f_sc['technical_points']:.4f}")
        if name == "HGB":
            for c, row in f_sc['by_condition'].items():
                print(f"   HGB {c:14s}: loss_ratio={row['loss_ratio']:.4f}")
        
    # Search blends on TUNING episodes only
    print("\nSearching optimal ensemble blend on TUNING episodes...")
    best_tune = 0.0
    best_w = None
    
    for w_hgb in [0.3, 0.4, 0.5, 0.6]:
        for w_et in [0.0, 0.1, 0.2, 0.3]:
            for w_lr in [0.0, 0.1, 0.15, 0.2]:
                w_base = 1.0 - w_hgb - w_et - w_lr
                if w_base < 0:
                    continue
                combo = {}
                for i, q in enumerate(val_queries_ordered):
                    b = base_preds[q.query_id]
                    h = hgb_probs[i]
                    e = et_probs[i]
                    l = lr_probs[i]
                    p = tuple(w_hgb * h[j] + w_et * e[j] + w_lr * l[j] + w_base * b[j] for j in range(5))
                    s = sum(p)
                    combo[q.query_id] = tuple(v / s for v in p)
                t_sc = subset_score(combo, validation_targets, groups, reference, tuning_eps, 60.0)
                pts = t_sc['technical_points']
                if pts > best_tune:
                    best_tune = pts
                    best_w = (w_hgb, w_et, w_lr, w_base)
                    
    print(f"Selected blend on tuning: HGB={best_w[0]:.2f}, ET={best_w[1]:.2f}, LR={best_w[2]:.2f}, Base={best_w[3]:.2f}")
    print(f"Tuning fold points: {best_tune:.4f}")
    
    # Now evaluate chosen blend on Audit and Full validation
    final_preds = {}
    for i, q in enumerate(val_queries_ordered):
        b = base_preds[q.query_id]
        h = hgb_probs[i]
        e = et_probs[i]
        l = lr_probs[i]
        p = tuple(best_w[0] * h[j] + best_w[1] * e[j] + best_w[2] * l[j] + best_w[3] * b[j] for j in range(5))
        s = sum(p)
        final_preds[q.query_id] = tuple(v / s for v in p)
        
    audit_sc = subset_score(final_preds, validation_targets, groups, reference, audit_eps, 60.0)
    full_sc = contest_score(final_preds, validation_targets, groups, reference, max_points=60.0)
    
    print(f"\nAUDIT FOLD POINTS: {audit_sc['technical_points']:.4f}")
    print(f"FULL VALIDATION POINTS: {full_sc['technical_points']:.4f} / 60")
    print(f"Mean loss ratio: {full_sc['mean_loss_ratio']:.4f}")
    for c, row in full_sc['by_condition'].items():
        print(f"   {c:14s}: loss_ratio={row['loss_ratio']:.4f} (episodes={row['episodes']})")
    print(f"Total experiment time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
