"""EXP-018: Controlled Combination Phase — Pattern Augmentation + Hierarchical Calibration.

Controlled Pairwise Combination:
1. Base Specialist: 3-Seed HGB Ensemble trained on Exact 40/24 Observation Topology (Experiment C).
2. Blended 25% with Champion v3/v4 changing-lag backbone.
3. Hierarchical Machine-Type Calibration (Experiment D) fitted with shrinkage regularization
   and 3-fold episode cross-fitting on tuning partition.
4. Evaluation: Complete test with ResearchController, checking if combination breaks 37.60+ points.
"""

import sys, os
from pathlib import Path
_LAB = Path(__file__).resolve().parents[2]
_REPO = _LAB.parent
sys.path[:0] = [str(_LAB), str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO), str(_LAB / 'src')]

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
from lag_model.features import build_advanced_features, PAYOUT_TO_IDX

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
    val_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/validation")
    train_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/train")
    
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
    
    # 1. Train 3-seed ensemble on 40/24 augmented data
    print("=======================================================")
    print("EXP-018: Training 40/24 Augmented Ensemble Specialist...")
    print("=======================================================")
    t0 = time.time()
    X_tr, y_tr = build_augmented_training_data(t_queries, t_history, t_labels, base)
    print(f"Extracted {len(X_tr)} examples in {time.time() - t0:.1f}s")
    
    models = []
    for leaves, l2, s in [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]:
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_tr, y_tr)
        models.append(m)
        print(f"  Trained seed {s} (leaves={leaves})")
        
    aug_preds = predict_ensemble(v_queries, v_history, base, models)
    
    # 2. Blend with Champion v4 base (25% weight)
    v2_raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
    v2_raw = read_submission(v2_raw_path, {t.query_id for t in targets})
    raw_blended = mix_predictions(aug_preds, v2_raw, 0.25)
    
    # 3. Fit Hierarchical Machine-Type Calibration with 3-fold cross-fitting
    print("\n=======================================================")
    print("EXP-018: Cross-Fitting Hierarchical Calibration...")
    print("=======================================================")
    T = [t for t in targets if t.episode_id in tuning]
    P = np.asarray([raw_blended[t.query_id] for t in T])
    Q = np.asarray([t.probabilities for t in T])
    M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    
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
        
    best_loss_ratio = score(raw_blended, tuning)["mean_loss_ratio"]
    best_penalty = 0.5
    for l_h in (0.1, 0.5, 1.0, 2.0, 5.0):
        cv_values = np.zeros_like(P)
        for fold in range(3):
            train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
            hold_mask = ~train_mask
            theta_fold = fit_hierarchical(P[train_mask], Q[train_mask], M[train_mask], compute_weights([t for t, k in zip(T, train_mask) if k]), 0.01, l_h, True)
            tg = theta_fold[0]; bg = theta_fold[1:6]; dt = theta_fold[6:9]; db = theta_fold[9:].reshape((3, 5))
            cv_values[hold_mask] = transform_hierarchical(P[hold_mask], M[hold_mask], tg, bg, dt, db)
        fold_preds = {t.query_id: tuple(p) for t, p in zip(T, cv_values)}
        res = score(fold_preds, tuning)
        print(f"  Penalty l_h={l_h:4.1f}: Tuning Pts = {res['technical_points']:.6f} | Loss Ratio = {res['mean_loss_ratio']:.6f}")
        if res["mean_loss_ratio"] < best_loss_ratio:
            best_loss_ratio = res["mean_loss_ratio"]
            best_penalty = l_h
            
    print(f"\nSelected optimal shrinkage penalty: l_h={best_penalty}")
    theta_final = fit_hierarchical(P, Q, M, w_T, 0.01, best_penalty, True)
    tg = theta_final[0]; bg = theta_final[1:6]; dt = theta_final[6:9]; db = theta_final[9:].reshape((3, 5))
    
    # 4. Transform all validation queries
    all_targets = targets
    all_P = np.asarray([raw_blended[t.query_id] for t in all_targets])
    all_M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in all_targets])
    all_cal = transform_hierarchical(all_P, all_M, tg, bg, dt, db)
    final_preds = {t.query_id: tuple(p) for t, p in zip(all_targets, all_cal)}
    
    # 5. Full Evaluation via ResearchController
    v4_preds = read_submission(Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"), {t.query_id for t in targets})
    ctrl = ResearchController()
    eval_rep = ctrl.evaluate_candidate(final_preds, base_predictions=v4_preds, n_bootstrap=2000)
    
    print("\n=======================================================")
    print("EXP-018: COMBINATION RESULTS (AUGMENTED + HIERARCHICAL)")
    print("=======================================================")
    print(f"Tuning Points: {eval_rep['tuning_points']:.6f} (Loss Ratio: {eval_rep['tuning_loss_ratio']:.6f})")
    print(f"Audit Points:  {eval_rep['audit_points']:.6f} (Loss Ratio: {eval_rep['audit_loss_ratio']:.6f})")
    print(f"Full Points:   {eval_rep['full_points']:.6f} (Loss Ratio: {eval_rep['full_loss_ratio']:.6f})")
    print(f"Paired Audit Bootstrap Gain CI95 vs Champ v4: {eval_rep['bootstrap_gain_vs_base_ci95']}")
    print("\nCondition Breakdown:")
    for cond, r in eval_rep["by_condition"].items():
        print(f"  {cond:15s}: Loss Ratio = {r['loss_ratio']:.4f}")
    print("\nMachine Type Breakdown:")
    for mt, r in eval_rep["by_machine_type"].items():
        print(f"  {mt:15s}: Loss Ratio = {r['loss_ratio']:.4f} (Brier: {r['brier']:.6f})")
        
    out_dir = Path("research/experiments/EXP_018_augmented_hierarchical_blend")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_submission(out_dir / "validation_predictions.csv", final_preds)
    write_json(out_dir / "report.json", {
        "best_penalty": best_penalty,
        "evaluation": eval_rep,
        "parameters": {
            "theta_global": float(tg),
            "beta_global": [float(v) for v in bg],
            "delta_theta_machine": {m: float(dt[i]) for m, i in MTYPE_MAP.items()},
            "delta_beta_machine": {m: [float(v) for v in db[i]] for m, i in MTYPE_MAP.items()}
        }
    })
    joblib.dump(models, out_dir / "aug_models.joblib", compress=3)
    print(f"\nArtifacts written to: {out_dir}")

if __name__ == "__main__":
    main()
