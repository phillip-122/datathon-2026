"""EXP-024: Candidate v9 (Soft-Thresholded Coupled Router).

Integrates:
1. Soft-thresholded differential drift slopes (kappa = 1.50).
2. Floor bankroll coupling: peer win rate dispersion & peer slope dispersion (p < 10^-11).
3. Continuous Bayesian machine genome shrinkage (mom * N / (N + 20)).
4. Exact deterministic router separating short_history (n_past <= 100) from long_history.
5. Benchmark directly against Active Champion v8_routed (38.4611 pts).
"""
import sys, os
from pathlib import Path
import time
import json
import joblib
import numpy as np
from collections import defaultdict, Counter
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import (
    load_inputs, read_labels, read_rows, read_targets,
    read_submission, write_submission
)
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import build_advanced_features as build_lag_features
from lag_model.features import PAYOUT_TO_IDX
from research.tools.controller import ResearchController

from feature_factory_v24 import (
    extract_soft_thresholded_drift_features,
    extract_bankroll_floor_coupling_features,
    compute_cutoff_safe_archetype,
    get_discrete_archetype_state,
    StandardTransitionTable,
    MTYPES, MTYPE_MAP
)

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def mask_exact_40_24(past, cutoff, rng):
    recent = {t: p for t, p in past.items() if cutoff - 40 <= t < cutoff}
    older = sorted(t for t in past if t < cutoff - 40)
    keep_older = rng.choice(older, min(24, len(older)), replace=False) if older else []
    return {**recent, **{int(t): past[t] for t in keep_older}}

def fit_hierarchical_calibrator(P, Q, M, sample_weights=None, lambda_base=0.01, lambda_hier=0.08):
    N, C = P.shape
    assert C == 5
    logP = np.log(np.maximum(P, 1e-12))
    
    def loss_func(params):
        tg = params[0]
        bg = params[1:6]
        dt = params[6:9]
        db = params[9:24].reshape(3, 5)
        
        inv_T = np.exp(tg + dt[M])
        logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
        log_denom = np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        pred_log_probs = logits - log_denom
        
        errs = np.sum((np.exp(pred_log_probs) - Q)**2, axis=1)
        if sample_weights is not None:
            data_loss = np.sum(sample_weights * errs)
        else:
            data_loss = np.mean(errs)
            
        reg = (
            lambda_base * (tg**2 + np.sum(bg**2)) +
            lambda_hier * (np.sum(dt**2) + np.sum(db**2))
        )
        return data_loss + reg

    init_params = np.zeros(24)
    res = minimize(loss_func, init_params, method="L-BFGS-B", options={"maxiter": 250})
    return res.x

def transform_hierarchical(P, M, theta):
    tg = theta[0]
    bg = theta[1:6]
    dt = theta[6:9]
    db = theta[9:24].reshape(3, 5)
    logP = np.log(np.maximum(P, 1e-12))
    inv_T = np.exp(tg + dt[M])
    logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
    return softmax(logits, axis=1)

def extract_features_v24(query, past, peer_past, cutoff, base_forecaster, trans_table):
    # Base features (275)
    feats = build_lag_features(query, past, peer_past, cutoff, base_forecaster)
    
    # Standard discrete Markov transitions (5)
    trans_feats = trans_table.get_features(query, past, cutoff)
    feats += trans_feats
    
    # Soft-thresholded drift velocity (4)
    drift_feats = extract_soft_thresholded_drift_features(query, past, peer_past, cutoff, kappa=1.50)
    feats += drift_feats
    
    # Bankroll floor coupling features (3)
    coupling_feats = extract_bankroll_floor_coupling_features(peer_past, cutoff)
    feats += coupling_feats
    
    # Continuous Bayesian machine genome shrinkage & gating (5)
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    tot = nw + nl
    mom_shrunk = mom * (tot / (tot + 20.0))
    
    genome_feats = [
        float(mom_shrunk),
        1.0 if arch_conf == -1 else 0.0,
        1.0 if arch_conf == 0 else 0.0,
        1.0 if arch_conf == 1 else 0.0,
        float(mom_shrunk * drift_feats[0]) # streak-drift interaction
    ]
    feats += genome_feats
    return feats

def build_training_dataset(qs, history, labels, base_forecaster, trans_table, rng_seed=42):
    grouped = sequences(qs, history)
    X = []
    y = []
    rng = np.random.default_rng(rng_seed)
    for q in qs:
        seq = grouped[(q.episode_id, q.machine_id)]
        peers = [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id]
        ticks = [t for t in range(30, q.cutoff, 8) if t in seq] + [q.cutoff]
        for t in ticks:
            past = {k: v for k, v in seq.items() if k < t}
            if len(past) < 8: continue
            earlier = [{k: v for k, v in s.items() if k < t} for s in peers]
            outcome = labels[q.query_id] if t == q.cutoff else seq[t]
            X.append(extract_features_v24(q, past, earlier, t, base_forecaster, trans_table))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(extract_features_v24(q, m_past, m_earlier, t, base_forecaster, trans_table))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def main():
    out_dir = Path("research/experiments/EXP_024_soft_coupled_router")
    ctrl = ResearchController()
    val_queries = ctrl.val_queries
    val_history = ctrl.val_history
    targets = ctrl.val_targets
    groups = ctrl.groups
    reference = ctrl.reference
    tuning = ctrl.tuning_eps
    audit = ctrl.audit_eps
    
    train_dir = ctrl.data_dir / "train"
    t_queries, t_history = load_inputs(train_dir)
    t_labels = read_labels(train_dir / "labels.csv", t_queries)
    
    print(f"Loaded {len(t_queries)} train queries, {len(val_queries)} val queries.")
    
    # 0. Active Champion v8_routed baseline
    v8_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", {t.query_id for t in targets})
    eval_v8 = ctrl.evaluate_candidate(v8_preds, n_bootstrap=1000)
    print(f"\nACTIVE CHAMPION V8_ROUTED BENCHMARK:")
    print(f"  Full:   {eval_v8['full_points']:.4f} pts | Loss Ratio: {eval_v8['full_loss_ratio']:.4f}")
    print(f"  Tuning: {eval_v8['tuning_points']:.4f} pts | Audit: {eval_v8['audit_points']:.4f} pts")
    for c, r in eval_v8['by_condition'].items():
        print(f"    - {c:14s}: {r['loss_ratio']:.4f}")
        
    # 1. Base forecaster & transition table
    print("\nFitting base forecaster and standard Markov transition table...")
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, t_labels)
    trans_table = StandardTransitionTable(alpha=10.0)
    trans_table.fit(t_queries, t_history)
    
    # 2. Build training dataset (full 28,154 rows)
    t0 = time.time()
    print("Building full 28,154-sample training dataset with 40/24 topology masking...")
    X_train, y_train = build_training_dataset(t_queries, t_history, t_labels, base, trans_table)
    print(f"  Train data generated: {X_train.shape} in {time.time() - t0:.1f}s")
    
    # 3. Validation feature extraction
    grouped_val = sequences(val_queries, val_history)
    t1 = time.time()
    print("Extracting validation features...")
    X_val = np.asarray([
        extract_features_v24(
            q,
            {t: p for t, p in grouped_val[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_val.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table
        ) for q in val_queries
    ], dtype=np.float32)
    print(f"  Validation features extracted: {X_val.shape} in {time.time() - t1:.1f}s")
    
    # 4. Train 3-seed HGB ensemble
    print("Training 3-seed ensemble (leaves=60, 80, 40)...")
    models = []
    for leaves, l2, s in [(60, 2.0, 42), (80, 1.5, 101), (40, 4.0, 202)]:
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_train, y_train)
        models.append(m)
        
    val_logits = np.mean([m.decision_function(X_val) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in val_queries])
    raw_v24 = softmax(val_logits / temp[:, None], axis=1)
    raw_preds = {q.query_id: raw_v24[i] for i, q in enumerate(val_queries)}
    
    # 5. Routing and Calibration
    query_map = {q.query_id: q for q in val_queries}
    target_qids = [t.query_id for t in targets]
    M_val = np.asarray([MTYPE_MAP[query_map[qid].machine_type] for qid in target_qids])
    T = [t for t in targets if t.episode_id in tuning]
    M_tuning = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    Q_tuning = np.asarray([t.probabilities for t in T])
    
    n_past_val = {q.query_id: len([t for t in grouped_val[(q.episode_id, q.machine_id)] if t < q.cutoff]) for q in val_queries}
    
    # Load Champion v6 predictions for blending long regimes
    v6_preds = read_submission("compiled_versions/v6_promoted_champion_exp019/validation_predictions.csv", set(target_qids))
    
    # Tuning weights
    def compute_weights(rows):
        w = []
        for t in rows:
            cond = groups[t.query_id]
            ct = [r for r in rows if groups[r.query_id] == cond]
            eps = {r.episode_id for r in ct}
            cnts = {e: sum(r.episode_id == e for r in ct) for e in eps}
            base_err = sum(np.sum((np.asarray(reference[r.query_id]) - r.probabilities)**2) / cnts[r.episode_id] for r in ct) / len(eps)
            w.append(1.0 / (4.0 * len(eps) * cnts[t.episode_id] * max(1e-9, base_err)))
        return np.asarray(w)
        
    w_tuning = compute_weights(T)
    
    print("\nEvaluating Candidate v9 Router Options:")
    
    # Option A: Long-history uses machine-type adaptive blend with v6 + regularized calibration; short-history uses v24 alone
    # Option B: Long-history uses machine-type adaptive blend with v6; short-history uses optimal blend
    for short_mode in ["v24_pure", "v24_v6_blend_50", "v8_routed_short"]:
        for w_long in [(0.15, 0.28, 0.35), (0.20, 0.30, 0.40), (0.10, 0.25, 0.30)]:
            w_map = {"steady": w_long[0], "balanced": w_long[1], "jackpot": w_long[2]}
            
            # Form long-history blend
            long_blended = {}
            for q in val_queries:
                qid = q.query_id
                w = w_map[q.machine_type]
                pb = w * np.asarray(raw_preds[qid]) + (1.0 - w) * np.asarray(v6_preds[qid])
                long_blended[qid] = pb / np.sum(pb)
                
            # Calibrate long-history on tuning split
            P_tuning_long = np.asarray([long_blended[t.query_id] for t in T if n_past_val[t.query_id] > 100])
            Q_tuning_long = np.asarray([t.probabilities for t in T if n_past_val[t.query_id] > 100])
            M_tuning_long = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T if n_past_val[t.query_id] > 100])
            w_tuning_long = compute_weights([t for t in T if n_past_val[t.query_id] > 100])
            
            theta_long = fit_hierarchical_calibrator(P_tuning_long, Q_tuning_long, M_tuning_long, w_tuning_long, lambda_base=0.01, lambda_hier=0.08)
            P_val_long_cal = transform_hierarchical(np.asarray([long_blended[t.query_id] for t in targets]), M_val, theta_long)
            
            # Short-history calibration
            P_tuning_short = np.asarray([raw_preds[t.query_id] for t in T if n_past_val[t.query_id] <= 100])
            Q_tuning_short = np.asarray([t.probabilities for t in T if n_past_val[t.query_id] <= 100])
            M_tuning_short = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T if n_past_val[t.query_id] <= 100])
            w_tuning_short = compute_weights([t for t in T if n_past_val[t.query_id] <= 100])
            theta_short = fit_hierarchical_calibrator(P_tuning_short, Q_tuning_short, M_tuning_short, w_tuning_short, lambda_base=0.01, lambda_hier=0.08)
            P_val_short_cal = transform_hierarchical(np.asarray([raw_preds[t.query_id] for t in targets]), M_val, theta_short)
            
            cfg3_val = read_submission("compiled_versions/v7_candidate_exp022_config3_unified/validation_predictions.csv", set(target_qids))
            
            final_routed = {}
            for i, t in enumerate(targets):
                qid = t.query_id
                if n_past_val[qid] <= 100:
                    if short_mode == "v24_pure":
                        final_routed[qid] = tuple(float(v) for v in P_val_short_cal[i])
                    elif short_mode == "v8_routed_short":
                        final_routed[qid] = tuple(float(v) for v in cfg3_val[qid])
                    elif short_mode == "v24_v6_blend_50":
                        p_b = 0.5 * P_val_short_cal[i] + 0.5 * np.asarray(cfg3_val[qid])
                        final_routed[qid] = tuple(float(v / sum(p_b)) for v in p_b)
                else:
                    final_routed[qid] = tuple(float(v) for v in P_val_long_cal[i])
                    
            res = ctrl.evaluate_candidate(final_routed, base_predictions=v8_preds, n_bootstrap=1000)
            print(f"ShortMode: {short_mode:18s} | LongWeights: {w_long} -> Full: {res['full_points']:.4f} pts (Tune: {res['tuning_points']:.4f}, Audit: {res['audit_points']:.4f}) | Loss: {res['full_loss_ratio']:.4f} | CI vs v8: [{res['bootstrap_gain_vs_base_ci95'][0]:.4f}, {res['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
            
    print("\nEXP-024 Grid Evaluation complete!")

if __name__ == "__main__":
    main()
