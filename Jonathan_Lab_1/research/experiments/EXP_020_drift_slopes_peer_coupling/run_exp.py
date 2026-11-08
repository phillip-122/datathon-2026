"""EXP-020: Continuous Drift Slopes & Peer Coupling Specialist.

Hypothesis:
In drift episodes (R = 0.6383), past history degrades over time.
Differential velocity features (slope_16_48, slope_8_24, peer_slope, jackpot_hazard)
combined with pairwise peer coupling (co-win rate, t-1/t-2 co-wins, payout cross-correlation
with highest-variance peer machine) will cut the drift loss ratio towards < 0.6000
while preserving/enhancing performance across stable, short_history, and shared_change.

Ensemble & Training Architecture:
1. 367 cutoff-safe features (223 base + 128 changing lags + 4 differential velocity + 12 peer coupling).
2. 40/24 topology masking (40 recent + 24 older ticks) across historical ticks >= 64.
3. 3-Seed HGB Ensemble [42, 101, 202] with leaves [40, 60, 80] and L2 [4.0, 2.0, 1.5].
4. Changing-lag backbone blend.
5. Hierarchical machine-type log-temperature + bias calibration with 3-fold episode cross-fitting.
6. Complete evaluation via ResearchController with paired bootstrap CI vs Champion v5.
"""

import sys, os
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent
_LAB = Path(__file__).resolve().parents[3]
_REPO = _LAB.parent
sys.path[:0] = [
    str(_EXP_DIR),
    str(_LAB),
    str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'),
    str(_REPO),
    str(_LAB / 'src')
]

import importlib.util
spec = importlib.util.spec_from_file_location("exp020_features", _EXP_DIR / "features.py")
exp020_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp020_features)
build_advanced_features = exp020_features.build_advanced_features
PAYOUT_TO_IDX = exp020_features.PAYOUT_TO_IDX

from collections import defaultdict
import hashlib
import json
import time
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from research.tools.controller import ResearchController
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_json, write_submission
from slot_forecasting.schema import probabilities, ValidationError

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def mask_exact_40_24(seq, cutoff, rng):
    recent = {t: p for t, p in seq.items() if cutoff - 40 <= t < cutoff}
    older = sorted(t for t in seq if t < cutoff - 40)
    keep_older = rng.choice(older, min(24, len(older)), replace=False) if older else []
    return {**recent, **{int(t): seq[t] for t in keep_older}}

def build_augmented_training_data(qs, history, labels, base_forecaster, rng_seed=42):
    grouped = sequences(qs, history)
    X = []
    y = []
    rng = np.random.default_rng(rng_seed)
    for q in qs:
        seq = grouped[(q.episode_id, q.machine_id)]
        peers = [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id]
        ticks = [t for t in range(30, q.cutoff, 6) if t in seq] + [q.cutoff]
        for t in ticks:
            past = {k: v for k, v in seq.items() if k < t}
            if len(past) < 8:
                continue
            earlier = [{k: v for k, v in s.items() if k < t} for s in peers]
            outcome = labels[q.query_id] if t == q.cutoff else seq[t]
            X.append(build_advanced_features(q, past, earlier, t, base_forecaster))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(build_advanced_features(q, m_past, m_earlier, t, base_forecaster))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def predict_ensemble(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_advanced_features(
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

def mix_predictions(a, b, w):
    return {k: tuple(w * np.asarray(a[k]) + (1.0 - w) * np.asarray(b[k])) for k in a}

def transform_hierarchical(P, m_indices, tg, bg, dt, db):
    logp = np.log(np.maximum(P, 1e-12))
    inv_t = np.exp(tg + dt[m_indices])
    logits = logp * inv_t[:, None] + bg[None, :]
    if db is not None:
        logits += db[m_indices, :]
    return softmax(logits, axis=1)

def fit_hierarchical(P, Q, m_indices, weights, l_global, l_hier, use_bias=True):
    n_params = 1 + 5 + 3 + (15 if use_bias else 0)
    def unpack(params):
        tg = params[0]
        bg = params[1:6]
        dt = params[6:9]
        db = params[9:].reshape((3, 5)) if use_bias else None
        return tg, bg, dt, db
        
    def objective(params):
        tg, bg, dt, db = unpack(params)
        pred = transform_hierarchical(P, m_indices, tg, bg, dt, db)
        residual = 2.0 * weights[:, None] * (pred - Q)
        derivative = pred * (residual - (residual * pred).sum(axis=1, keepdims=True))
        loss = np.sum(weights[:, None] * (pred - Q) ** 2)
        loss += l_global * (tg ** 2 + np.sum(bg ** 2))
        loss += l_hier * (np.sum(dt ** 2) + (np.sum(db ** 2) if db is not None else 0.0))
        
        logp = np.log(np.maximum(P, 1e-12))
        inv_t = np.exp(tg + dt[m_indices])
        d_tg = np.sum(derivative * logp * inv_t[:, None]) + 2.0 * l_global * tg
        d_bg = np.sum(derivative, axis=0) + 2.0 * l_global * bg
        d_dt = np.zeros(3)
        for m in range(3):
            mask = (m_indices == m)
            if np.any(mask):
                d_dt[m] = np.sum(derivative[mask] * logp[mask] * inv_t[mask, None]) + 2.0 * l_hier * dt[m]
        if use_bias:
            d_db = np.zeros((3, 5))
            for m in range(3):
                mask = (m_indices == m)
                if np.any(mask):
                    d_db[m] = np.sum(derivative[mask], axis=0) + 2.0 * l_hier * db[m]
            grad = np.r_[d_tg, d_bg, d_dt, d_db.ravel()]
        else:
            grad = np.r_[d_tg, d_bg, d_dt]
        return loss, grad

    bounds = [(-0.7, 0.7)] + [(-1.0, 1.0)] * 5 + [(-0.5, 0.5)] * 3
    if use_bias:
        bounds += [(-0.5, 0.5)] * 15
    x0 = np.zeros(n_params)
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x

def main():
    start_time = time.time()
    val_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/validation")
    train_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/train")
    test_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/test")
    
    t_queries, t_history = load_inputs(train_dir)
    t_labels = read_labels(train_dir / "labels.csv", t_queries)
    
    v_queries, v_history = load_inputs(val_dir)
    targets = read_targets(val_dir / "targets.csv")
    group_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(val_dir / "reference.csv")
    tuning, audit = episode_folds(targets, groups)
    query_map = {q.query_id: q for q in v_queries}
    
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(t_queries, t_history, t_labels)
    
    # 1. Train 3-seed ensemble on 40/24 augmented data with new features
    print("==================================================================")
    print("EXP-020: Training Continuous Drift Slopes & Peer Coupling Ensemble")
    print("==================================================================")
    model_path = _EXP_DIR / "aug_models.joblib"
    if model_path.exists() and "--retrain" not in sys.argv:
        print(f"Loading cached ensemble models from {model_path}...")
        models = joblib.load(model_path)
        feat_dim = models[0].n_features_in_
    else:
        t0 = time.time()
        X_tr, y_tr = build_augmented_training_data(t_queries, t_history, t_labels, base)
        feat_dim = X_tr.shape[1]
        print(f"Extracted {len(X_tr)} examples with {feat_dim} features in {time.time() - t0:.1f}s")
        
        models = []
        configs = [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]
        for leaves, l2, s in configs:
            m = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
                min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
            )
            m.fit(X_tr, y_tr)
            models.append(m)
            print(f"  Trained seed {s:3d} (leaves={leaves}, l2={l2})")
        joblib.dump(models, model_path, compress=3)
        
    t_pred0 = time.time()
    aug_preds = predict_ensemble(v_queries, v_history, base, models)
    print(f"Validation inference completed in {time.time() - t_pred0:.2f}s")
    
    # 2. Blend with Champion v4 base (evaluate weights)
    v2_raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
    v2_raw = read_submission(v2_raw_path, {t.query_id for t in targets})
    
    T = [t for t in targets if t.episode_id in tuning]
    Q_tune = np.asarray([t.probabilities for t in T])
    M_tune = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    
    def compute_weights(rows):
        w = []
        for t in rows:
            condition = groups[t.query_id]
            ct = [r for r in rows if groups[r.query_id] == condition]
            episodes = {r.episode_id for r in ct}
            counts = {e: sum(r.episode_id == e for r in ct) for e in episodes}
            baseline = sum(np.sum((np.asarray(reference[r.query_id]) - r.probabilities) ** 2) / counts[r.episode_id] for r in ct) / len(episodes)
            w.append(1.0 / (4.0 * len(episodes) * counts[t.episode_id] * max(1e-9, baseline)))
        return np.asarray(w)
        
    w_T = compute_weights(T)
    
    folds = {}
    for condition in sorted(set(groups.values())):
        eps = sorted({t.episode_id for t in T if groups[t.query_id] == condition}, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        folds.update({e: i % 3 for i, e in enumerate(eps)})
        
    def score(p, eps):
        return subset_score(p, targets, groups, reference, eps, 60.0)
        
    # Grid search blend weight and hierarchical shrinkage penalty
    print("\n==================================================================")
    print("EXP-020: Cross-Fitting Blend Weight & Hierarchical Calibration...")
    print("==================================================================")
    best_overall_ratio = 999.0
    best_w = 0.25
    best_penalty = 0.1
    best_raw_blended = None
    
    for candidate_w in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        raw_b = mix_predictions(aug_preds, v2_raw, candidate_w)
        P_cand = np.asarray([raw_b[t.query_id] for t in T])
        raw_res = score(raw_b, tuning)
        print(f"\n--- Testing Blend Weight w={candidate_w:.2f} (Raw Tuning: Pts={raw_res['technical_points']:.4f}, R={raw_res['mean_loss_ratio']:.4f}) ---")
        
        for l_h in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            cv_values = np.zeros_like(P_cand)
            for fold in range(3):
                train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
                hold_mask = ~train_mask
                theta_fold = fit_hierarchical(
                    P_cand[train_mask], Q_tune[train_mask], M_tune[train_mask],
                    compute_weights([t for t, k in zip(T, train_mask) if k]),
                    0.01, l_h, True
                )
                tg = theta_fold[0]; bg = theta_fold[1:6]; dt = theta_fold[6:9]; db = theta_fold[9:].reshape((3, 5))
                cv_values[hold_mask] = transform_hierarchical(P_cand[hold_mask], M_tune[hold_mask], tg, bg, dt, db)
            fold_preds = {t.query_id: tuple(p) for t, p in zip(T, cv_values)}
            res = score(fold_preds, tuning)
            print(f"  w={candidate_w:.2f}, l_h={l_h:4.2f}: Tuning Pts = {res['technical_points']:.6f} | Loss Ratio = {res['mean_loss_ratio']:.6f}")
            if res["mean_loss_ratio"] < best_overall_ratio:
                best_overall_ratio = res["mean_loss_ratio"]
                best_w = candidate_w
                best_penalty = l_h
                best_raw_blended = raw_b
                
    print(f"\n>>> Selected Optimal Config: Blend Weight w={best_w:.2f}, Shrinkage Penalty l_h={best_penalty:.2f} (Tuning Ratio: {best_overall_ratio:.6f})")
    
    # 3. Fit final calibration on full tuning set
    P_final = np.asarray([best_raw_blended[t.query_id] for t in T])
    theta_final = fit_hierarchical(P_final, Q_tune, M_tune, w_T, 0.01, best_penalty, True)
    tg = theta_final[0]; bg = theta_final[1:6]; dt = theta_final[6:9]; db = theta_final[9:].reshape((3, 5))
    
    # 4. Transform all validation queries
    all_targets = targets
    all_P = np.asarray([best_raw_blended[t.query_id] for t in all_targets])
    all_M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in all_targets])
    all_cal = transform_hierarchical(all_P, all_M, tg, bg, dt, db)
    final_preds = {t.query_id: tuple(p) for t, p in zip(all_targets, all_cal)}
    
    # 5. Full Evaluation via ResearchController
    # Compare against Champion v5 (EXP-018) and Champion v4 (EXP-014)
    v5_preds = read_submission(Path("research/experiments/EXP_018_augmented_hierarchical_blend/validation_predictions.csv"), {t.query_id for t in targets})
    v4_preds = read_submission(Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"), {t.query_id for t in targets})
    
    ctrl = ResearchController()
    eval_v5 = ctrl.evaluate_candidate(final_preds, base_predictions=v5_preds, n_bootstrap=2000)
    eval_v4 = ctrl.evaluate_candidate(final_preds, base_predictions=v4_preds, n_bootstrap=2000)
    
    print("\n==================================================================")
    print("EXP-020: FINAL RESULTS & VERIFICATION")
    print("==================================================================")
    print(f"Tuning Points: {eval_v5['tuning_points']:.6f} (Champion v5: 37.883155, Gate: > 37.8832)")
    print(f"Audit Points:  {eval_v5['audit_points']:.6f} (Champion v5: 37.849509, Gate: > 37.8495)")
    print(f"Full Points:   {eval_v5['full_points']:.6f} (Champion v5: 37.868275)")
    print(f"Full Loss Ratio: {eval_v5['full_loss_ratio']:.6f} (Champion v5: 0.584440)")
    print(f"Paired Audit Bootstrap CI95 vs Champion v5: {eval_v5['bootstrap_gain_vs_base_ci95']}")
    print(f"Paired Audit Bootstrap CI95 vs Champion v4: {eval_v4['bootstrap_gain_vs_base_ci95']}")
    
    print("\nCondition Breakdown:")
    for cond, r in eval_v5["by_condition"].items():
        v5_cond_r = 0.6383 if cond == "drift" else (0.4844 if cond == "shared_change" else (0.6606 if cond == "short_history" else 0.5545))
        delta = r['loss_ratio'] - v5_cond_r
        print(f"  {cond:15s}: Loss Ratio = {r['loss_ratio']:.4f} (v5: {v5_cond_r:.4f}, delta: {delta:+.4f})")
        
    print("\nMachine Type Breakdown:")
    for mt, r in eval_v5["by_machine_type"].items():
        print(f"  {mt:15s}: Loss Ratio = {r['loss_ratio']:.4f} (Brier: {r['brier']:.6f})")
        
    # Check promotion gates
    passed_tuning = eval_v5["tuning_points"] > 37.8832
    passed_audit = eval_v5["audit_points"] > 37.8495
    passed_drift = eval_v5["by_condition"]["drift"]["loss_ratio"] < 0.6383
    dual_gate = passed_tuning and passed_audit
    
    print("\n--- Promotion Gate Checks ---")
    print(f"Tuning Points > 37.8832: {passed_tuning} ({eval_v5['tuning_points']:.6f})")
    print(f"Audit Points > 37.8495:  {passed_audit} ({eval_v5['audit_points']:.6f})")
    print(f"Drift Loss Ratio Drop:   {passed_drift} ({eval_v5['by_condition']['drift']['loss_ratio']:.4f} vs 0.6383)")
    print(f"DUAL PROMOTION GATE:     {dual_gate}")
    
    # 6. Generate test submission on public/test
    print("\nGenerating test submission on public/test...")
    test_queries, test_history = load_inputs(test_dir)
    aug_test_preds = predict_ensemble(test_queries, test_history, base, models)
    
    # Load champ models for test backbone
    champ_models = joblib.load("compiled_versions/v5_promoted_champion_exp018/entry/champ_models.joblib")
    from hgb_upgrade.features import build_advanced_features as build_base_features
    def predict_champ(qs, h, base_forecaster, c_models):
        grouped_h = sequences(qs, h)
        X_c = np.asarray([
            build_base_features(
                q,
                grouped_h[(q.episode_id, q.machine_id)],
                [s for (ep, m), s in grouped_h.items() if ep == q.episode_id and m != q.machine_id],
                q.cutoff,
                base_forecaster
            ) for q in qs
        ], dtype=np.float32)
        logits_c = np.mean([m.decision_function(X_c) for m in c_models], axis=0)
        temp_c = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
        probs_c = softmax(logits_c / temp_c[:, None], axis=1)
        return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs_c)}
        
    champ_test_preds = predict_champ(test_queries, test_history, base, champ_models)
    
    raw_test_P = []
    m_test_indices = []
    for q in test_queries:
        p_mix = best_w * np.asarray(aug_test_preds[q.query_id]) + (1.0 - best_w) * np.asarray(champ_test_preds[q.query_id])
        raw_test_P.append(p_mix)
        m_test_indices.append(MTYPE_MAP[q.machine_type])
        
    raw_test_P = np.asarray(raw_test_P)
    m_test_indices = np.asarray(m_test_indices)
    cal_test_P = transform_hierarchical(raw_test_P, m_test_indices, tg, bg, dt, db)
    
    final_test_preds = {}
    for i, q in enumerate(test_queries):
        raw_p = cal_test_P[i]
        tot = float(np.sum(raw_p))
        final_test_preds[q.query_id] = probabilities(tuple(float(v / tot) for v in raw_p), q.query_id)
        
    # Write artifacts to EXP directory
    write_submission(_EXP_DIR / "validation_predictions.csv", final_preds)
    write_submission(_EXP_DIR / "submission.csv", final_test_preds)
    
    results_payload = {
        "exp_id": "EXP-020",
        "description": "Continuous Drift Slopes & Peer Coupling Specialist",
        "feature_dim": feat_dim,
        "runtime_seconds": time.time() - start_time,
        "parameters": {
            "best_blend_weight": best_w,
            "best_shrinkage_penalty": best_penalty,
            "theta_global": float(tg),
            "beta_global": [float(v) for v in bg],
            "delta_theta_machine": {m: float(dt[i]) for m, i in MTYPE_MAP.items()},
            "delta_beta_machine": {m: [float(v) for v in db[i]] for m, i in MTYPE_MAP.items()}
        },
        "evaluation": eval_v5,
        "evaluation_vs_v4": eval_v4,
        "gates": {
            "passed_tuning": bool(passed_tuning),
            "passed_audit": bool(passed_audit),
            "passed_drift": bool(passed_drift),
            "dual_gate_passed": bool(dual_gate)
        }
    }
    write_json(_EXP_DIR / "results.json", results_payload)
    joblib.dump(models, _EXP_DIR / "aug_models.joblib", compress=3)
    
    print(f"\nArtifacts successfully written to {_EXP_DIR}")
    print(f"Total experiment runtime: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
