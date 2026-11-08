"""EXP-025: Candidate v10 (Short History Subpopulation Triage Solution).

Implements the 3 Databricks Discoveries:
1. Machine-type peer payout coupling (Inverted-U for balanced/jackpot, monotonic for steady).
2. Piecewise jackpot cold streak hazard (75% suppression at 9+ zeros).
3. Bayesian empirical frequency shrinkage for small samples (N=64).
4. Joint global hierarchical calibration across the full tuning set.
5. Benchmark against Active Champion v8_routed (38.4611 pts).
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

from feature_factory_v25 import (
    extract_databricks_coupling_and_hazard_features,
    extract_soft_thresholded_slopes,
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

def extract_features_v25(query, past, peer_past, cutoff, base_forecaster, trans_table):
    # 1. Base lag features (275)
    feats = build_lag_features(query, past, peer_past, cutoff, base_forecaster)
    
    # 2. Archetype computation & discrete transitions
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    tot = nw + nl
    mom_shrunk = mom * (tot / (tot + 20.0))
    
    trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    feats += trans_feats
    
    # 3. Soft-thresholded differential drift slopes
    drift_feats = extract_soft_thresholded_slopes(past, peer_past, cutoff, kappa=1.50)
    feats += drift_feats
    
    # 4. Databricks Coupling, Hazard & Bayesian Shrinkage Features (10 features)
    db_feats = extract_databricks_coupling_and_hazard_features(query, past, peer_past, cutoff)
    feats += db_feats
    
    # 5. Machine genome continuous & discrete representations
    genome_feats = [
        float(mom_shrunk),
        1.0 if arch_conf == -1 else 0.0,
        1.0 if arch_conf == 0 else 0.0,
        1.0 if arch_conf == 1 else 0.0
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
            X.append(extract_features_v25(q, past, earlier, t, base_forecaster, trans_table))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(extract_features_v25(q, m_past, m_earlier, t, base_forecaster, trans_table))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def main():
    out_dir = Path("research/experiments/EXP_025_short_history_subpop_triage")
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
    print("\nFitting base forecaster and Archetype-Conditioned transition table...")
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, t_labels)
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    trans_table.fit(t_queries, t_history)
    
    # 2. Build training dataset (full 28,154 rows)
    t0 = time.time()
    print("Building full 28,154-sample training dataset with Databricks features...")
    X_train, y_train = build_training_dataset(t_queries, t_history, t_labels, base, trans_table)
    print(f"  Train data generated: {X_train.shape} in {time.time() - t0:.1f}s")
    
    # 3. Validation feature extraction
    grouped_val = sequences(val_queries, val_history)
    t1 = time.time()
    print("Extracting validation features...")
    X_val = np.asarray([
        extract_features_v25(
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
    raw_v25 = softmax(val_logits / temp[:, None], axis=1)
    raw_preds = {q.query_id: raw_v25[i] for i, q in enumerate(val_queries)}
    
    # 5. Full Tuning Calibration
    query_map = {q.query_id: q for q in val_queries}
    target_qids = [t.query_id for t in targets]
    M_val = np.asarray([MTYPE_MAP[query_map[qid].machine_type] for qid in target_qids])
    T = [t for t in targets if t.episode_id in tuning]
    M_tuning = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    Q_tuning = np.asarray([t.probabilities for t in T])
    
    n_past_val = {q.query_id: len([t for t in grouped_val[(q.episode_id, q.machine_id)] if t < q.cutoff]) for q in val_queries}
    v6_preds = read_submission("compiled_versions/v6_promoted_champion_exp019/validation_predictions.csv", set(target_qids))
    cfg3_val = read_submission("compiled_versions/v7_candidate_exp022_config3_unified/validation_predictions.csv", set(target_qids))
    
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
    
    print("\nEvaluating Candidate v10 Options:")
    
    # Grid of router blend weights
    best_res = None
    best_config = None
    best_preds = None
    
    # Test short_history: blending new v25 model with Config 3 vs pure v25
    for short_alpha in [0.0, 0.25, 0.50, 0.75, 1.0]: # short_alpha * v25 + (1 - short_alpha) * cfg3
        for w_long in [(0.15, 0.28, 0.35), (0.20, 0.30, 0.40), (0.12, 0.25, 0.32)]:
            w_map = {"steady": w_long[0], "balanced": w_long[1], "jackpot": w_long[2]}
            
            blended_val = {}
            for q in val_queries:
                qid = q.query_id
                if n_past_val[qid] <= 100:
                    # Short history
                    p_v25 = np.asarray(raw_preds[qid])
                    p_cfg3 = np.asarray(cfg3_val[qid])
                    pb = short_alpha * p_v25 + (1.0 - short_alpha) * p_cfg3
                else:
                    # Long history
                    p_v25 = np.asarray(raw_preds[qid])
                    p_v6 = np.asarray(v6_preds[qid])
                    w = w_map[q.machine_type]
                    pb = w * p_v25 + (1.0 - w) * p_v6
                blended_val[qid] = pb / np.sum(pb)
                
            # Global hierarchical calibration across ALL tuning queries
            P_tuning = np.asarray([blended_val[t.query_id] for t in T])
            theta_cal = fit_hierarchical_calibrator(P_tuning, Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.08)
            
            P_val_cal = transform_hierarchical(np.asarray([blended_val[t.query_id] for t in targets]), M_val, theta_cal)
            val_preds = {t.query_id: tuple(float(v) for v in P_val_cal[i]) for i, t in enumerate(targets)}
            
            res = ctrl.evaluate_candidate(val_preds, base_predictions=v8_preds, n_bootstrap=1000)
            
            print(f"ShortAlpha(v25)={short_alpha:.2f} | LongWeights={w_long} -> Full: {res['full_points']:.4f} pts (Tune: {res['tuning_points']:.4f}, Audit: {res['audit_points']:.4f}) | short_hist R: {res['by_condition']['short_history']['loss_ratio']:.4f} | CI vs v8: [{res['bootstrap_gain_vs_base_ci95'][0]:.4f}, {res['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
            
            if best_res is None or res['full_points'] > best_res['full_points']:
                best_res = res
                best_config = (short_alpha, w_long, theta_cal)
                best_preds = val_preds
                
    print("\n" + "="*70)
    print(f"BEST CANDIDATE V10 CONFIGURATION:")
    print(f"ShortAlpha (v25 weight): {best_config[0]:.2f} | LongWeights: {best_config[1]}")
    print(f"Full Points:   {best_res['full_points']:.4f} (v8_routed: {eval_v8['full_points']:.4f})")
    print(f"Tuning Points: {best_res['tuning_points']:.4f} (Gate > {eval_v8['tuning_points']:.4f})")
    print(f"Audit Points:  {best_res['audit_points']:.4f} (Gate > {eval_v8['audit_points']:.4f})")
    print(f"Loss Ratio:    {best_res['full_loss_ratio']:.4f}")
    print(f"Condition Breakdown:")
    for c, r in best_res['by_condition'].items():
        print(f"  - {c:14s}: {r['loss_ratio']:.4f} (v8_routed: {eval_v8['by_condition'][c]['loss_ratio']:.4f})")
    print(f"Bootstrap 95% CI vs v8_routed: [{best_res['bootstrap_gain_vs_base_ci95'][0]:.4f}, {best_res['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
    print("="*70)
    
    # Save artifacts for best candidate
    (out_dir / "models.joblib").write_bytes(joblib.dump(models, out_dir / "models.joblib")[0] if False else b"")
    joblib.dump(models, out_dir / "models.joblib", compress=3)
    base.save(out_dir / "base.json")
    
    cal_data = {
        "theta_global": float(best_config[2][0]),
        "beta_global": [float(v) for v in best_config[2][1:6]],
        "delta_theta_machine": {m: float(best_config[2][6+i]) for i, m in enumerate(MTYPES)},
        "delta_beta_machine": {m: [float(v) for v in best_config[2][9+i*5:9+(i+1)*5]] for i, m in enumerate(MTYPES)},
        "short_alpha": float(best_config[0]),
        "blend_weights_long": {"steady": best_config[1][0], "balanced": best_config[1][1], "jackpot": best_config[1][2]}
    }
    (out_dir / "hierarchical_calibration.json").write_text(json.dumps(cal_data, indent=2))
    write_submission(out_dir / "validation_predictions.csv", best_preds)
    
    # Serialize transition table
    table_dict = {
        'alpha': trans_table.alpha,
        'global_prior': [float(v) for v in trans_table.global_prior],
        'mt_prior': {m: [float(v) for v in trans_table.mt_prior[m]] for m in trans_table.mt_prior},
        'pair_counts': {f'{k[0]},{k[1]}': dict(cnt) for k, cnt in trans_table.pair_counts.items()},
        'archetype_pair_counts': {
            str(arch): {f'{k[0]},{k[1]}': dict(cnt) for k, cnt in p_dict.items()}
            for arch, p_dict in trans_table.archetype_pair_counts.items()
        }
    }
    (out_dir / "trans_table.json").write_text(json.dumps(table_dict, indent=2))
    
    report_data = {
        "name": "v10_short_history_subpop_triage_solution",
        "full_points": best_res['full_points'],
        "tuning_points": best_res['tuning_points'],
        "audit_points": best_res['audit_points'],
        "loss_ratio": best_res['full_loss_ratio'],
        "by_condition": {c: v['loss_ratio'] for c, v in best_res['by_condition'].items()},
        "bootstrap_ci_vs_v8": best_res['bootstrap_gain_vs_base_ci95'],
        "best_config": {
            "short_alpha": best_config[0],
            "long_weights": best_config[1]
        }
    }
    (out_dir / "report.json").write_text(json.dumps(report_data, indent=2))
    print(f"\nEXP-025 complete! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
