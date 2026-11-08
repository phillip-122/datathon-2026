"""Ablation study of Machine-Level Polarity features for Adaptation Round.
Uses exact HGBForecaster pipeline matching Candidate A2.
"""
import sys
import time
from pathlib import Path
from collections import defaultdict
import copy

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "House_Edge_Lab_Starter_Pack" / "research_environment"))

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from slot_forecasting.io import load_inputs, read_labels, read_targets, read_submission, read_rows
from slot_forecasting.contest import contest_score
from slot_forecasting.schema import probabilities
from slot_forecasting.baseline import FrequencyForecaster
from src.features import PAYOUT_TO_IDX, build_advanced_features
from src.model import DEFAULT_TEMP_MAP

REEL_RATIOS = {
    'steady': (200.0 / 250.0, 49.0 / 250.0, 1.0 / 250.0),
    'balanced': (16.0 / 23.0, 6.0 / 23.0, 1.0 / 23.0),
    'jackpot': (7.0 / 19.0, 9.0 / 19.0, 3.0 / 19.0),
}
REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

def extract_features(query, sequence, peer_sequences, cutoff, base_forecaster, mode="all"):
    # Base 223 features
    feats = build_advanced_features(query, sequence, peer_sequences, cutoff, base_forecaster)
    
    if mode == "none":
        return feats
        
    valid_ticks = [t for t in sequence if t < cutoff]
    
    # --- Feature Set A: Self Lag-2 Polarity ---
    if mode in ("lag2", "all", "lag2_peer", "lag2_regime"):
        n_win_t2 = 0
        n_win_given_win = 0
        n_loss_t2 = 0
        n_win_given_loss = 0
        for t in valid_ticks:
            t_prev = t - 2
            if t_prev in sequence and t_prev < cutoff:
                w_t = (sequence[t] > 0)
                w_prev = (sequence[t_prev] > 0)
                if w_prev:
                    n_win_t2 += 1
                    if w_t: n_win_given_win += 1
                else:
                    n_loss_t2 += 1
                    if w_t: n_win_given_loss += 1
        tot = n_win_t2 + n_loss_t2
        if tot > 0:
            p_base = (n_win_given_win + n_win_given_loss) / tot
            p_win_win = (n_win_given_win + 2.0 * p_base) / (n_win_t2 + 2.0)
            p_win_loss = (n_win_given_loss + 2.0 * p_base) / (n_loss_t2 + 2.0)
            lag2_pol = p_win_win - p_win_loss
        else:
            lag2_pol = 0.0
            
        lag2_val = sequence.get(cutoff - 2, -1)
        lag2_state = 1.0 if lag2_val > 0 else (-1.0 if lag2_val == 0 else 0.0)
        lag2_dir = lag2_pol * lag2_state
        feats.extend([lag2_pol, lag2_dir])
        
    # --- Feature Set B: Peer Contagion Polarity ---
    if mode in ("peer", "all", "lag2_peer", "peer_regime"):
        y_m = []
        y_peers = []
        for t in valid_ticks:
            peer_vals = [p_seq[t] for p_seq in peer_sequences if (t in p_seq and t < cutoff)]
            if peer_vals:
                y_m.append(1.0 if sequence[t] > 0 else 0.0)
                y_peers.append(sum(1 for p in peer_vals if p > 0) / len(peer_vals))
        if len(y_m) >= 6:
            arr_m = np.array(y_m, dtype=np.float32)
            arr_p = np.array(y_peers, dtype=np.float32)
            mean_m = float(np.mean(arr_m))
            mean_p = float(np.mean(arr_p))
            cov_peer = float(np.mean((arr_m - mean_m) * (arr_p - mean_p)))
            std_m = float(np.std(arr_m))
            std_p = float(np.std(arr_p))
            corr_peer = float(cov_peer / (std_m * std_p)) if (std_m > 1e-4 and std_p > 1e-4) else 0.0
        else:
            cov_peer = 0.0
            corr_peer = 0.0
            mean_p = 0.5
            
        peer_t1 = [p_seq[cutoff - 1] for p_seq in peer_sequences if (cutoff - 1 in p_seq and cutoff - 1 < cutoff)]
        if peer_t1:
            peer_t1_rate = sum(1 for p in peer_t1 if p > 0) / len(peer_t1)
            peer_dir_t1 = corr_peer * (peer_t1_rate - mean_p)
        else:
            peer_dir_t1 = 0.0
        feats.extend([cov_peer, corr_peer, peer_dir_t1])
        
    # --- Feature Set C: Machine Regime Trend ---
    if mode in ("regime", "all", "lag2_regime", "peer_regime"):
        ticks_32 = [t for t in range(cutoff - 32, cutoff) if t in sequence]
        if len(ticks_32) >= 4:
            x = np.array([float(t - (cutoff - 32)) / 32.0 for t in ticks_32])
            y = np.array([1.0 if sequence[t] == 1 else (-1.0 if sequence[t] in (2, 5, 20) else 0.0) for t in ticks_32], dtype=np.float32)
            var_x = float(np.var(x))
            regime_slope_ols = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / var_x) if var_x > 1e-5 else 0.0
            
            last_16 = [t for t in range(cutoff - 16, cutoff) if t in sequence]
            prior_16 = [t for t in range(cutoff - 32, cutoff - 16) if t in sequence]
            diff_last = sum(1 for t in last_16 if sequence[t] == 1) - sum(1 for t in last_16 if sequence[t] in (2, 5, 20))
            diff_prior = sum(1 for t in prior_16 if sequence[t] == 1) - sum(1 for t in prior_16 if sequence[t] in (2, 5, 20))
            regime_slope_split = float(diff_last - diff_prior)
        else:
            regime_slope_ols = 0.0
            regime_slope_split = 0.0
        feats.extend([regime_slope_ols, regime_slope_split])
        
    return feats


def run_single_experiment(mode, train_q, train_h, train_labels, val_q, val_h, val_targets, groups, reference, history_val, inverter_slope=False):
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    
    ep_hist_train = defaultdict(lambda: defaultdict(dict))
    for h in train_h:
        ep_hist_train[h.episode_id][h.machine_id][h.tick] = h.payout
        
    ep_hist_val = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        ep_hist_val[h.episode_id][h.machine_id][h.tick] = h.payout
        
    # Build training dataset
    X_train = []
    y_train = []
    for q in train_q:
        ep_seqs = ep_hist_train[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        
        # Label at cutoff
        f = extract_features(q, m_seq, peers, q.cutoff, base_forecaster, mode=mode)
        X_train.append(f)
        y_train.append(PAYOUT_TO_IDX[train_labels[q.query_id]])
        
        # Earlier cutoffs
        for t in range(30, q.cutoff, 5):
            if t in m_seq:
                sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                if len(sub_seq) >= 8:
                    sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                    f = extract_features(q, sub_seq, sub_peers, t, base_forecaster, mode=mode)
                    X_train.append(f)
                    y_train.append(PAYOUT_TO_IDX[m_seq[t]])
                    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    # Build validation dataset
    X_val = []
    for q in val_q:
        ep_seqs = ep_hist_val[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = extract_features(q, m_seq, peers, q.cutoff, base_forecaster, mode=mode)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    
    # Train 3-seed HGB ensemble
    seeds = [42, 101, 202]
    all_logits = []
    for s in seeds:
        m = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.065,
            max_leaf_nodes=150,
            min_samples_leaf=12,
            l2_regularization=1.2,
            random_state=s
        )
        m.fit(X_train, y_train)
        all_logits.append(m.decision_function(X_val))
    avg_logits = np.mean(all_logits, axis=0)
    
    # Internal HGB temperature scaling matching HGBForecaster.predict()
    scaled = np.zeros_like(avg_logits)
    for i, q in enumerate(val_q):
        T = DEFAULT_TEMP_MAP.get(q.machine_type, 1.06)
        scaled[i] = avg_logits[i] / T
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    val_mat = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    
    # Simplex projection
    queries_val = pd.read_csv("adaptation-participants/public/validation/queries.csv")
    for i, q in queries_val.iterrows():
        mt = q['machine_type']
        c2, c5, c20 = REEL_RATIOS[mt]
        whigh = val_mat[i, 2] + val_mat[i, 3] + val_mat[i, 4]
        val_mat[i, 2] = whigh * c2
        val_mat[i, 3] = whigh * c5
        val_mat[i, 4] = whigh * c20
    val_mat = val_mat / np.sum(val_mat, axis=1, keepdims=True)
    
    # 1x-Regime Mixture Inverter
    grouped_h = {}
    for _, row in history_val.iterrows():
        grouped_h.setdefault((row['episode_id'], row['machine_id']), []).append(row)
        
    targets_df = pd.read_csv("adaptation-participants/public/validation/targets.csv").set_index('query_id')
    feature_rows = []
    for i, q in queries_val.iterrows():
        qid = q['query_id']
        mt = q['machine_type']
        r_low, r_high = REGIME_RATIOS[mt]
        t_row = targets_df.loc[qid]
        w_high_true = t_row['p_2'] + t_row['p_5'] + t_row['p_20']
        true_ratio = t_row['p_1'] / max(1e-6, w_high_true)
        true_regime = 0 if abs(true_ratio - r_low) < abs(true_ratio - r_high) else 1
        
        p0 = val_mat[i, 0]
        p1 = val_mat[i, 1]
        whigh = val_mat[i, 2] + val_mat[i, 3] + val_mat[i, 4]
        r_hat = p1 / max(1e-6, whigh)
        x = np.log(max(1e-6, r_hat))
        x_low = np.log(r_low)
        x_high = np.log(r_high)
        x_mid = 0.5 * (x_low + x_high)
        implied_logit = x - x_mid
        
        past = [s['payout'] for s in grouped_h.get((q['episode_id'], q['machine_id']), []) if s['tick'] < q['cutoff']]
        c1_16 = sum(1 for p in past[-16:] if p == 1)
        chigh_16 = sum(1 for p in past[-16:] if p in [2, 5, 20])
        c1_32 = sum(1 for p in past[-32:] if p == 1)
        chigh_32 = sum(1 for p in past[-32:] if p in [2, 5, 20])
        
        row_dict = {
            'implied_logit': implied_logit,
            'diff_16': c1_16 - chigh_16,
            'diff_32': c1_32 - chigh_32,
            'is_balanced': 1 if mt == 'balanced' else 0,
            'is_jackpot': 1 if mt == 'jackpot' else 0,
            'true_regime': true_regime
        }
        
        if inverter_slope:
            ticks_32 = [s['tick'] for s in grouped_h.get((q['episode_id'], q['machine_id']), []) if s['tick'] >= q['cutoff'] - 32 and s['tick'] < q['cutoff']]
            if len(ticks_32) >= 4:
                x_ols = np.array([float(t - (q['cutoff'] - 32)) / 32.0 for t in ticks_32])
                p_map = {s['tick']: s['payout'] for s in grouped_h.get((q['episode_id'], q['machine_id']), [])}
                y_ols = np.array([1.0 if p_map[t] == 1 else (-1.0 if p_map[t] in (2, 5, 20) else 0.0) for t in ticks_32], dtype=np.float32)
                var_x = float(np.var(x_ols))
                row_dict['regime_slope_ols'] = float(np.mean((x_ols - np.mean(x_ols)) * (y_ols - np.mean(y_ols))) / var_x) if var_x > 1e-5 else 0.0
            else:
                row_dict['regime_slope_ols'] = 0.0
        feature_rows.append(row_dict)
        
    fdf = pd.DataFrame(feature_rows)
    feats = ['implied_logit', 'diff_16', 'diff_32', 'is_balanced', 'is_jackpot']
    if inverter_slope:
        feats.append('regime_slope_ols')
        
    lr_m = LogisticRegression(C=0.50, random_state=42)
    lr_m.fit(fdf[feats], fdf['true_regime'])
    pi = lr_m.predict_proba(fdf[feats])[:, 1]
    
    inv_preds = []
    for i, q in queries_val.iterrows():
        mt = q['machine_type']
        r_low, r_high = REGIME_RATIOS[mt]
        c2, c5, c20 = REEL_RATIOS[mt]
        p0 = val_mat[i, 0]
        W = max(1e-6, 1.0 - p0)
        p = pi[i]
        w_low = W / (1.0 + r_low)
        p1_low = W - w_low
        w_high_state = W / (1.0 + r_high)
        p1_high = W - w_high_state
        mix_p1 = (1.0 - p) * p1_low + p * p1_high
        mix_whigh = (1.0 - p) * w_low + p * w_high_state
        inv_preds.append([p0, mix_p1, mix_whigh * c2, mix_whigh * c5, mix_whigh * c20])
        
    inv_mat = np.array(inv_preds)
    inv_mat = inv_mat / np.sum(inv_mat, axis=1, keepdims=True)
    
    # Outer temperature scaling
    mt_arr = queries_val['machine_type'].values
    t_vec = np.where(mt_arr == 'steady', 0.90, np.where(mt_arr == 'balanced', 1.00, 0.95))
    log_p = np.log(np.maximum(inv_mat, 1e-12))
    final_mat = softmax(log_p / t_vec[:, None], axis=1)
    
    pred_dict = {
        q.query_id: probabilities(tuple(final_mat[i]), q.query_id)
        for i, q in enumerate(val_q)
    }
    score_res = contest_score(pred_dict, val_targets, groups, reference, max_points=20.0)
    return score_res


def main():
    data_dir = Path("adaptation-participants/public")
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q, val_h = load_inputs(data_dir / "validation")
    val_targets = read_targets(data_dir / "validation/targets.csv")
    reference = read_submission(data_dir / "validation/reference.csv", {q.query_id: q for q in val_q})
    group_rows = read_rows(data_dir / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    history_val = pd.read_csv(data_dir / "validation/history.csv")
    
    modes = [
        ("none", False, "Candidate A2 Baseline (no polarity)"),
        ("lag2", False, "+ Lag-2 Polarity Only"),
        ("peer", False, "+ Peer Contagion Only"),
        ("regime", False, "+ Regime Trend (HGB Only)"),
        ("none", True, "+ Regime Trend (Inverter Only)"),
        ("regime", True, "+ Regime Trend (HGB & Inverter)"),
        ("all", False, "+ All Polarity Features (HGB Only)"),
        ("all", True, "+ All Polarity (HGB & Inverter)"),
    ]
    
    results = []
    print("=" * 75)
    print("STARTING POLARITY ABLATION BENCHMARKS")
    print("=" * 75)
    for mode, inv_slope, label in modes:
        t0 = time.time()
        print(f"\n--> Running: {label}...")
        res = run_single_experiment(mode, train_q, train_h, train_labels, val_q, val_h, val_targets, groups, reference, history_val, inverter_slope=inv_slope)
        elapsed = time.time() - t0
        print(f"    Finished in {elapsed:.1f}s | Points: {res['technical_points']:.6f} | R: {res['mean_loss_ratio']:.6f}")
        for c, d in res['by_condition'].items():
            print(f"      {c:15s}: R = {d['loss_ratio']:.4f}")
        results.append((label, res['technical_points'], res['mean_loss_ratio'], res['by_condition']))
        
    print("\n" + "=" * 75)
    print("FINAL SUMMARY OF POLARITY ABLATION")
    print("=" * 75)
    print(f"{'Configuration':40s} | {'Points':10s} | {'Loss Ratio R':12s}")
    print("-" * 75)
    for label, pts, r, _ in results:
        print(f"{label:40s} | {pts:10.6f} | {r:12.6f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
