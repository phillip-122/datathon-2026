"""EXP-023: Soft-Thresholded Regime Recognition & Bayesian Genome Shrinkage."""
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

from feature_factory_v23 import (
    extract_soft_thresholded_drift_features,
    compute_cutoff_safe_archetype,
    get_discrete_archetype_state,
    ArchetypeConditionedTransitionTable,
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

def fit_hierarchical_calibrator(P, Q, M, sample_weights=None, lambda_base=0.01, lambda_hier=0.05):
    n_classes = Q.shape[1]
    n_machines = len(MTYPES)
    n_params = 1 + n_classes + n_machines + n_machines * n_classes
    
    def unpack(theta):
        tg = theta[0]
        bg = theta[1:1+n_classes]
        dt = theta[1+n_classes:1+n_classes+n_machines]
        db = theta[1+n_classes+n_machines:].reshape((n_machines, n_classes))
        return tg, bg, dt, db

    def objective(theta):
        tg, bg, dt, db = unpack(theta)
        logP = np.log(np.maximum(P, 1e-12))
        inv_T = np.exp(tg + dt[M])
        logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
        P_cal = softmax(logits, axis=1)
        diff = P_cal - Q
        if sample_weights is not None:
            loss = np.sum(sample_weights[:, None] * (diff ** 2))
        else:
            loss = np.mean(diff ** 2)
        reg = (
            lambda_base * (tg**2 + np.sum(bg**2)) +
            lambda_hier * (np.sum(dt**2) + np.sum(db**2))
        )
        return loss + reg

    bounds = [(-2.0, 2.0)] + [(-5.0, 5.0)] * n_classes + [(-1.5, 1.5)] * n_machines + [(-3.0, 3.0)] * (n_machines * n_classes)
    x0 = np.zeros(n_params)
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
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

def extract_features_v23(query, past, peer_past, cutoff, base_forecaster, trans_table, config):
    feats = build_lag_features(query, past, peer_past, cutoff, base_forecaster)
    
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    tot = nw + nl
    mom_shrunk = mom * (tot / (tot + 20.0))
    
    # Drift features
    if config == "raw_drift":
        drift_feats = extract_soft_thresholded_drift_features(query, past, peer_past, cutoff, kappa=0.0)
    elif config in ("soft_k125", "soft_shrunk_genome", "full_unified"):
        drift_feats = extract_soft_thresholded_drift_features(query, past, peer_past, cutoff, kappa=1.25)
    elif config == "soft_k150":
        drift_feats = extract_soft_thresholded_drift_features(query, past, peer_past, cutoff, kappa=1.50)
    else:
        drift_feats = [0.0, 0.0, 0.0, 0.0]
        
    # Transitions
    if config == "full_unified":
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    else:
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=0)
    feats += trans_feats
    feats += drift_feats
    
    # Genome representations
    if config in ("soft_shrunk_genome", "full_unified"):
        feats += [
            float(mom_shrunk),
            1.0 if arch_conf == -1 else 0.0,
            1.0 if arch_conf == 0 else 0.0,
            1.0 if arch_conf == 1 else 0.0
        ]
        
    return feats

def build_training_dataset(qs, history, labels, base_forecaster, trans_table, config, rng_seed=42):
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
            X.append(extract_features_v23(q, past, earlier, t, base_forecaster, trans_table, config))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(extract_features_v23(q, m_past, m_earlier, t, base_forecaster, trans_table, config))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def main():
    out_dir = Path("research/experiments/EXP_023_soft_threshold_regime_shrinkage")
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
    
    print(f"Loaded {len(t_queries)} train, {len(val_queries)} val, {len(targets)} validation targets.")
    
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, t_labels)
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    trans_table.fit(t_queries, t_history)
    
    grouped_val = sequences(val_queries, val_history)
    query_map = {q.query_id: q for q in val_queries}
    target_qids = [t.query_id for t in targets]
    M_val = np.asarray([MTYPE_MAP[query_map[qid].machine_type] for qid in target_qids])
    T = [t for t in targets if t.episode_id in tuning]
    M_tuning = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    Q_tuning = np.asarray([t.probabilities for t in T])
    
    v6_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", set(target_qids))
    eval_champ = ctrl.evaluate_candidate(v6_preds, n_bootstrap=1000)
    print(f"\nCHAMPION V6 BENCHMARK:")
    print(f"  Full:   {eval_champ['full_points']:.4f} pts | Loss Ratio: {eval_champ['full_loss_ratio']:.4f}")
    print(f"  Tuning: {eval_champ['tuning_points']:.4f} pts | Audit: {eval_champ['audit_points']:.4f} pts")
    for c, r in eval_champ['by_condition'].items():
        print(f"    - {c:14s}: {r['loss_ratio']:.4f}")
        
    configs = [
        ("raw_drift", "Raw Continuous Drift Slopes (EXP-020 replica, kappa=0.0)"),
        ("soft_k125", "Soft-Thresholded Drift Slopes (kappa=1.25)"),
        ("soft_k150", "Soft-Thresholded Drift Slopes (kappa=1.50)"),
        ("soft_shrunk_genome", "Soft Drift (k=1.25) + Bayesian Continuous Shrunk Genome"),
        ("full_unified", "Unified: Soft Drift (k=1.25) + Shrunk Genome + Conditioned Transitions")
    ]
    
    results = {}
    
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
    
    for cfg_code, desc in configs:
        print(f"\n============================================================")
        print(f"RUNNING CONFIG: {cfg_code}")
        print(f"Description: {desc}")
        print(f"============================================================")
        t0 = time.time()
        
        X_train, y_train = build_training_dataset(t_queries, t_history, t_labels, base, trans_table, cfg_code)
        print(f"  Train data generated: {X_train.shape} in {time.time() - t0:.1f}s")
        
        t1 = time.time()
        X_val = np.asarray([
            extract_features_v23(
                q,
                {t: p for t, p in grouped_val[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
                [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_val.items() if ep == q.episode_id and m != q.machine_id],
                q.cutoff,
                base,
                trans_table,
                cfg_code
            ) for q in val_queries
        ], dtype=np.float32)
        print(f"  Validation features extracted: {X_val.shape} in {time.time() - t1:.1f}s")
        
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
        val_probs = softmax(val_logits / temp[:, None], axis=1)
        raw_preds = {q.query_id: val_probs[i] for i, q in enumerate(val_queries)}
        
        # 1. Candidate Alone (calibrated)
        P_tuning_raw = np.asarray([raw_preds[t.query_id] for t in T])
        theta_native = fit_hierarchical_calibrator(P_tuning_raw, Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
        P_val_native = transform_hierarchical(np.asarray([raw_preds[t.query_id] for t in targets]), M_val, theta_native)
        val_native_preds = {t.query_id: tuple(float(v) for v in P_val_native[i]) for i, t in enumerate(targets)}
        eval_native = ctrl.evaluate_candidate(val_native_preds, base_predictions=v6_preds, n_bootstrap=1000)
        
        # 2. Invariant Mixture Calibration (25% Candidate + 75% v6, identical to EXP-022)
        blended_preds = {}
        for q in val_queries:
            p_c = np.asarray(raw_preds[q.query_id])
            p_v6 = np.asarray(v6_preds[q.query_id])
            pb = 0.25 * p_c + 0.75 * p_v6
            blended_preds[q.query_id] = pb / np.sum(pb)
            
        P_tuning_blend = np.asarray([blended_preds[t.query_id] for t in T])
        theta_blend = fit_hierarchical_calibrator(P_tuning_blend, Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
        P_val_blend = transform_hierarchical(np.asarray([blended_preds[t.query_id] for t in targets]), M_val, theta_blend)
        val_blend_preds = {t.query_id: tuple(float(v) for v in P_val_blend[i]) for i, t in enumerate(targets)}
        eval_blend = ctrl.evaluate_candidate(val_blend_preds, base_predictions=v6_preds, n_bootstrap=1000)
        
        print(f"\n  --- RESULTS FOR {cfg_code} ---")
        print(f"  [Candidate Alone]")
        print(f"    Full:   {eval_native['full_points']:.4f} pts (Tuning: {eval_native['tuning_points']:.4f}, Audit: {eval_native['audit_points']:.4f})")
        print(f"    Loss Ratio: {eval_native['full_loss_ratio']:.4f} | Bootstrap CI: [{eval_native['bootstrap_gain_vs_base_ci95'][0]:.4f}, {eval_native['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
        print(f"    By Condition: { {c: round(v['loss_ratio'], 4) for c, v in eval_native['by_condition'].items()} }")
        
        print(f"  [Unified Mixture (25% / 75%)]")
        print(f"    Full:   {eval_blend['full_points']:.4f} pts (Tuning: {eval_blend['tuning_points']:.4f}, Audit: {eval_blend['audit_points']:.4f})")
        print(f"    Loss Ratio: {eval_blend['full_loss_ratio']:.4f} | Bootstrap CI: [{eval_blend['bootstrap_gain_vs_base_ci95'][0]:.4f}, {eval_blend['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
        print(f"    By Condition: { {c: round(v['loss_ratio'], 4) for c, v in eval_blend['by_condition'].items()} }")
        
        results[cfg_code] = {
            "desc": desc,
            "feature_dim": X_train.shape[1],
            "native": {
                "full_points": eval_native['full_points'],
                "tuning_points": eval_native['tuning_points'],
                "audit_points": eval_native['audit_points'],
                "loss_ratio": eval_native['full_loss_ratio'],
                "bootstrap_ci": eval_native['bootstrap_gain_vs_base_ci95'],
                "by_condition": {c: v['loss_ratio'] for c, v in eval_native['by_condition'].items()}
            },
            "mixture": {
                "full_points": eval_blend['full_points'],
                "tuning_points": eval_blend['tuning_points'],
                "audit_points": eval_blend['audit_points'],
                "loss_ratio": eval_blend['full_loss_ratio'],
                "bootstrap_ci": eval_blend['bootstrap_gain_vs_base_ci95'],
                "by_condition": {c: v['loss_ratio'] for c, v in eval_blend['by_condition'].items()}
            }
        }
        
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nEXP-023 complete! Saved results to {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
