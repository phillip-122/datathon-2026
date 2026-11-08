"""EXP-021: Machine Sub-Archetype Genome + Topology Augmentation + Hierarchical Calibration."""

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

def compute_archetype_features(m_seq, cutoff):
    sorted_ticks = sorted([t for t in m_seq if t < cutoff])
    w_w_num, w_w_den = 1.0, 2.0 # Laplace prior
    w_l_num, w_l_den = 1.0, 2.0
    for i in range(1, len(sorted_ticks)):
        t_curr, t_prev = sorted_ticks[i], sorted_ticks[i-1]
        if t_curr - t_prev == 1:
            curr_win = 1.0 if m_seq[t_curr] > 0 else 0.0
            prev_win = 1.0 if m_seq[t_prev] > 0 else 0.0
            if prev_win == 1.0:
                w_w_num += curr_win
                w_w_den += 1.0
            else:
                w_l_num += curr_win
                w_l_den += 1.0
    p_w_w = w_w_num / w_w_den
    p_w_l = w_l_num / w_l_den
    mom = p_w_w - p_w_l
    l1_payout = m_seq.get(cutoff - 1, -1)
    l1_win = 1.0 if l1_payout > 0 else (0.0 if l1_payout == 0 else 0.5)
    return [
        float(mom),
        float(p_w_w),
        float(p_w_l),
        float(mom * l1_win),
        1.0 if mom < -0.06 else 0.0,
        1.0 if mom > 0.06 else 0.0,
    ]

def build_features_with_archetype(q, past, earlier, t, base_forecaster):
    f = build_advanced_features(q, past, earlier, t, base_forecaster)
    f.extend(compute_archetype_features(past, t))
    return f

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
            X.append(build_features_with_archetype(q, past, earlier, t, base_forecaster))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(build_features_with_archetype(q, m_past, m_earlier, t, base_forecaster))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def predict_ensemble(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_features_with_archetype(
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
    cond_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in cond_rows}
    ref = read_submission(val_dir / "reference.csv")
    
    print("Fitting baseline forecaster...")
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, t_labels)
    
    print("Building Augmented Training Data with Machine Archetype Features...")
    t0 = time.time()
    X_train, y_train = build_augmented_training_data(t_queries, t_history, t_labels, base)
    print(f"X_train shape: {X_train.shape} in {time.time() - t0:.1f}s")
    
    print("Training 3-seed HGB ensemble...")
    models = []
    for s in (0, 1, 2):
        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=60, min_samples_leaf=25, l2_regularization=2.0, random_state=s
        )
        clf.fit(X_train, y_train)
        models.append(clf)
        
    print("Predicting validation set with Archetype Ensemble...")
    val_raw = predict_ensemble(v_queries, v_history, base, models)
    
    v4_preds = read_submission(Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"), {t.query_id for t in targets})
    v6_preds = read_submission(Path("compiled_versions/active_champion_submission/validation_predictions.csv"), {t.query_id for t in targets})
    
    ctrl = ResearchController()
    
    best_res = None
    best_penalty = None
    best_blend_w = None
    
    target_qids = [t.query_id for t in targets]
    q_map = {q.query_id: q for q in v_queries}
    m_indices = np.array([MTYPE_MAP[q_map[qid].machine_type] for qid in target_qids])
    Q = np.array([t.probabilities for t in targets])
    weights = np.ones(len(targets))
    
    tuning_eps, audit_eps = episode_folds(targets, groups)
    tuning_indices = np.array([i for i, t in enumerate(targets) if t.episode_id in tuning_eps])
    
    print("\n--- Tuning Hierarchical Calibration ---")
    for blend_w in [0.15, 0.20, 0.25, 0.30]:
        mixed = mix_predictions(val_raw, v4_preds, blend_w)
        P = np.array([mixed[qid] for qid in target_qids])
        
        for l_hier in [0.01, 0.05, 0.1, 0.2]:
            P_tune = P[tuning_indices]
            Q_tune = Q[tuning_indices]
            m_tune = m_indices[tuning_indices]
            w_tune = weights[tuning_indices]
            
            theta = fit_hierarchical(P_tune, Q_tune, m_tune, w_tune, l_global=0.01, l_hier=l_hier, use_bias=True)
            tg, bg, dt, db = theta[0], theta[1:6], theta[6:9], theta[9:].reshape((3, 5))
            
            P_cal = transform_hierarchical(P, m_indices, tg, bg, dt, db)
            cal_preds = {qid: tuple(P_cal[i] / np.sum(P_cal[i])) for i, qid in enumerate(target_qids)}
            
            eval_rep = ctrl.evaluate_candidate(cal_preds, base_predictions=v6_preds, n_bootstrap=1000)
            print(f"Blend {blend_w:.2f}, L_hier {l_hier:.2f} -> Full: {eval_rep['full_points']:.4f}, Tune: {eval_rep['tuning_points']:.4f}, Audit: {eval_rep['audit_points']:.4f}")
            
            if best_res is None or eval_rep['full_points'] > best_res['full_points']:
                best_res = eval_rep
                best_penalty = l_hier
                best_blend_w = blend_w

    print("\n=======================================================")
    print(f"BEST CONFIG: blend_w={best_blend_w}, l_hier={best_penalty}")
    print(f"Full Points:   {best_res['full_points']:.6f} (Active Champion v6: 37.999049)")
    print(f"Tuning Points: {best_res['tuning_points']:.6f} (Target > 38.000420)")
    print(f"Audit Points:  {best_res['audit_points']:.6f} (Target > 37.994812)")
    print(f"Full Loss Ratio: {best_res['full_loss_ratio']:.6f}")
    for cond, r in best_res["by_condition"].items():
        print(f"  {cond:15s}: Loss Ratio = {r['loss_ratio']:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    main()
