"""EXP-019: Discrete Autoregressive Lag Sequences Specialist.

Champion Candidate v6 Architecture:
1. Discrete N-Gram Transition Priors:
   - lag_pair = (payout_{t-1}, payout_{t-2})
   - Bayesian Dirichlet/Laplace smoothing with alpha=10.0:
     P(Y_t = k | state) = (Count(state, k) + alpha * P_prior(k)) / (Count(state) + alpha)
   - Dual Prior Ensembles:
     (a) Global transition prior P_prior(k)
     (b) Machine-type conditioned prior P_prior(k | machine_type)
   - Each appends 5 smoothed transition probabilities as continuous features into the
     Champion v5 topology-masked (40 recent + 24 older ticks) HGB ensemble.
2. 3-Seed HGB Ensemble [42, 101, 202] (leaves: 40, 60, 80) for each prior specialist.
3. 30% Blend with Champion Changing-Lag backbone (70% Champion Changing-Lag + 30% Discrete Lag Specialist).
4. Machine-Type Hierarchical Calibration fitted on 50/50 Tuning partition.
5. Dual Gate Verification:
   - Tuning Points > 37.8832
   - Audit Points > 37.8495
"""

import sys, os
from pathlib import Path
_LAB = Path(__file__).resolve().parents[2]
_REPO = _LAB.parent
sys.path[:0] = [str(_LAB), str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO), str(_LAB / 'src')]

from collections import defaultdict, Counter
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
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import build_advanced_features as base_advanced_features, PAYOUT_TO_IDX

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

class DiscreteLagTransitionTable:
    def __init__(self, alpha=10.0):
        self.alpha = float(alpha)
        self.global_counts = Counter()
        self.mt_counts = defaultdict(Counter)
        self.pair_counts = defaultdict(Counter)
        self.mt_pair_counts = defaultdict(lambda: defaultdict(Counter))
        self.global_prior = None
        self.mt_prior = {}
        
    def fit(self, queries, history):
        q_mtype = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
        grouped = sequences(queries, history)
        
        for (ep, m), seq in grouped.items():
            mt = q_mtype.get((ep, m))
            ticks = sorted(seq.keys())
            for i, t in enumerate(ticks):
                outcome = seq[t]
                self.global_counts[outcome] += 1
                if mt:
                    self.mt_counts[mt][outcome] += 1
                if i >= 2 and ticks[i-1] == t - 1 and ticks[i-2] == t - 2:
                    s2 = (seq[t-1], seq[t-2])
                    self.pair_counts[s2][outcome] += 1
                    if mt:
                        self.mt_pair_counts[mt][s2][outcome] += 1
                        
        tot = sum(self.global_counts.values())
        self.global_prior = np.array([self.global_counts[p] / tot for p in PAYOUTS], dtype=np.float32)
        for mt in MTYPES:
            mt_tot = sum(self.mt_counts[mt].values())
            self.mt_prior[mt] = np.array([self.mt_counts[mt][p] / mt_tot for p in PAYOUTS], dtype=np.float32)

    def get_features(self, query, past, cutoff, use_machine_type=False):
        a = self.alpha
        mt = query.machine_type
        prior = self.mt_prior[mt] if use_machine_type else self.global_prior
        
        p1 = past.get(cutoff - 1)
        p2 = past.get(cutoff - 2)
        
        if p1 is not None and p2 is not None:
            s2 = (p1, p2)
            c2 = self.mt_pair_counts[mt][s2] if use_machine_type else self.pair_counts[s2]
            n2 = sum(c2.values())
            return [(c2[p] + a * prior[i]) / (n2 + a) for i, p in enumerate(PAYOUTS)]
        else:
            return list(prior)

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "global_prior": [float(v) for v in self.global_prior],
            "mt_prior": {m: [float(v) for v in self.mt_prior[m]] for m in MTYPES},
            "pair_counts": {f"{k[0]},{k[1]}": {str(p): int(c) for p, c in cnt.items()} for k, cnt in self.pair_counts.items()},
            "mt_pair_counts": {m: {f"{k[0]},{k[1]}": {str(p): int(c) for p, c in cnt.items()} for k, cnt in self.mt_pair_counts[m].items()} for m in MTYPES}
        }

def build_features(query, sequence, peer_sequences, cutoff, base_forecaster, transition_table, use_mt):
    past = {t: p for t, p in sequence.items() if t < cutoff}
    peers = [{t: p for t, p in s.items() if t < cutoff} for s in peer_sequences]
    base_f = base_advanced_features(query, past, peers, cutoff, base_forecaster)
    trans_f = transition_table.get_features(query, past, cutoff, use_machine_type=use_mt)
    return base_f + trans_f

def build_training_data(qs, history, labels, base_forecaster, transition_table, use_mt, rng_seed=42):
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
            X.append(build_features(q, past, earlier, t, base_forecaster, transition_table, use_mt))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(build_features(q, m_past, m_earlier, t, base_forecaster, transition_table, use_mt))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def predict_ensemble(qs, h, base_forecaster, models, transition_table, use_mt):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base_forecaster,
            transition_table,
            use_mt
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
    out_dir = Path("research/experiments/EXP_019_discrete_lag_sequences")
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 1. Base frequency forecaster
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(t_queries, t_history, t_labels)
    
    # 2. Fit Bayesian discrete lag transition table (alpha=10.0)
    print("===================================================================")
    print("EXP-019: Fitting Bayesian Discrete Transition Priors (alpha=10.0)...")
    print("===================================================================")
    table = DiscreteLagTransitionTable(alpha=10.0)
    table.fit(t_queries, t_history)
    
    # 3. Build & Train Global Prior Specialist
    print("\n--- Training Specialist A: Global Discrete Lag Ensemble ---")
    t0 = time.time()
    X_glob, y_glob = build_training_data(t_queries, t_history, t_labels, base, table, use_mt=False)
    print(f"  Extracted global training data: {X_glob.shape} in {time.time() - t0:.1f}s")
    
    models_glob = []
    for leaves, l2, s in [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]:
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_glob, y_glob)
        models_glob.append(m)
        print(f"    Trained seed {s} (leaves={leaves})")
        
    preds_glob = predict_ensemble(v_queries, v_history, base, models_glob, table, use_mt=False)
    
    # 4. Build & Train Machine-Type Prior Specialist
    print("\n--- Training Specialist B: Machine-Type Discrete Lag Ensemble ---")
    t0 = time.time()
    X_mt, y_mt = build_training_data(t_queries, t_history, t_labels, base, table, use_mt=True)
    print(f"  Extracted MT training data: {X_mt.shape} in {time.time() - t0:.1f}s")
    
    models_mt = []
    for leaves, l2, s in [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]:
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_mt, y_mt)
        models_mt.append(m)
        print(f"    Trained seed {s} (leaves={leaves})")
        
    preds_mt = predict_ensemble(v_queries, v_history, base, models_mt, table, use_mt=True)
    
    # 5. Dual Specialist Averaging
    preds_discrete = {
        qid: tuple(0.5 * np.array(preds_glob[qid]) + 0.5 * np.array(preds_mt[qid]))
        for qid in preds_glob
    }
    
    # 6. Blend with Champion Changing-Lag Backbone (30% weight to Discrete Lag Specialists)
    v2_raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
    v2_raw = read_submission(v2_raw_path, {t.query_id for t in targets})
    raw_blended = mix_predictions(preds_discrete, v2_raw, 0.30)
    
    # 7. Cross-Fitting & Hierarchical Machine-Type Calibration
    print("\n===================================================================")
    print("EXP-019: Machine-Type Hierarchical Calibration on Tuning Split...")
    print("===================================================================")
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
    best_penalty = 0.04
    for l_h in (0.02, 0.04, 0.05, 0.1, 0.2, 0.5):
        cv_values = np.zeros_like(P)
        for fold in range(3):
            train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
            hold_mask = ~train_mask
            theta_fold = fit_hierarchical(P[train_mask], Q[train_mask], M[train_mask], compute_weights([t for t, k in zip(T, train_mask) if k]), 0.01, l_h, True)
            tg = theta_fold[0]; bg = theta_fold[1:6]; dt = theta_fold[6:9]; db = theta_fold[9:].reshape((3, 5))
            cv_values[hold_mask] = transform_hierarchical(P[hold_mask], M[hold_mask], tg, bg, dt, db)
        fold_preds = {t.query_id: tuple(p) for t, p in zip(T, cv_values)}
        res = score(fold_preds, tuning)
        print(f"  Penalty l_h={l_h:4.2f}: Tuning Pts = {res['technical_points']:.6f} | Loss Ratio = {res['mean_loss_ratio']:.6f}")
        if res["mean_loss_ratio"] < best_loss_ratio:
            best_loss_ratio = res["mean_loss_ratio"]
            best_penalty = l_h
            
    print(f"\nSelected optimal shrinkage penalty from 3-fold cross-fit: l_h={best_penalty}")
    theta_final = fit_hierarchical(P, Q, M, w_T, 0.01, best_penalty, True)
    tg = theta_final[0]; bg = theta_final[1:6]; dt = theta_final[6:9]; db = theta_final[9:].reshape((3, 5))
    
    # 8. Transform validation set & Full Controller Evaluation
    all_targets = targets
    all_P = np.asarray([raw_blended[t.query_id] for t in all_targets])
    all_M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in all_targets])
    all_cal = transform_hierarchical(all_P, all_M, tg, bg, dt, db)
    final_val_preds = {t.query_id: tuple(float(v / sum(p)) for v in p) for t, p in zip(all_targets, all_cal)}
    
    # Evaluate against Champion v4 and Champion v5
    v4_preds = read_submission(Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"), {t.query_id for t in targets})
    v5_preds = read_submission(Path("research/experiments/EXP_018_augmented_hierarchical_blend/validation_predictions.csv"), {t.query_id for t in targets})
    ctrl = ResearchController()
    eval_rep = ctrl.evaluate_candidate(final_val_preds, base_predictions=v5_preds, n_bootstrap=2000)
    eval_rep_v4 = ctrl.evaluate_candidate(final_val_preds, base_predictions=v4_preds, n_bootstrap=2000)
    
    print("\n===================================================================")
    print("EXP-019: FINAL VERIFICATION RESULTS (DUAL PROMOTION GATE EVALUATION)")
    print("===================================================================")
    print(f"Tuning Points: {eval_rep['tuning_points']:.6f} (Gate Target > 37.883155, diff = {eval_rep['tuning_points'] - 37.883155:+.6f})")
    print(f"Audit Points:  {eval_rep['audit_points']:.6f} (Gate Target > 37.849509, diff = {eval_rep['audit_points'] - 37.849509:+.6f})")
    print(f"Full Points:   {eval_rep['full_points']:.6f} (Champion v5 = 37.868275, diff = {eval_rep['full_points'] - 37.868275:+.6f})")
    print(f"Full Loss Ratio R: {eval_rep['full_loss_ratio']:.6f} (Champion v5 = 0.584440, reduction = {(0.584440 - eval_rep['full_loss_ratio'])/0.584440*100:+.2f}%)")
    print(f"Paired Condition-Stratified Bootstrap 95% CI vs Champion v5: {eval_rep['bootstrap_gain_vs_base_ci95']}")
    print(f"Paired Condition-Stratified Bootstrap 95% CI vs Champion v4: {eval_rep_v4['bootstrap_gain_vs_base_ci95']}")
    
    print("\nCondition Loss Ratio Breakdown:")
    for cond, r in eval_rep["by_condition"].items():
        print(f"  {cond:15s}: Loss Ratio = {r['loss_ratio']:.4f}")
        
    print("\nMachine Type Breakdown:")
    for mt, r in eval_rep["by_machine_type"].items():
        print(f"  {mt:15s}: Loss Ratio = {r['loss_ratio']:.4f} (Brier: {r['brier']:.6f})")
        
    pass_tuning = eval_rep["tuning_points"] > 37.883155
    pass_audit = eval_rep["audit_points"] > 37.849509
    print("\nDual Promotion Gate Status:")
    print(f"  Tuning Gate: {'PASSED [OK]' if pass_tuning else 'FAILED [X]'}")
    print(f"  Audit Gate:  {'PASSED [OK]' if pass_audit else 'FAILED [X]'}")
    
    # 9. Predict on Test Set
    print("\n===================================================================")
    print("EXP-019: Generating Predictions on Public Test Set...")
    print("===================================================================")
    test_queries, test_history = load_inputs(test_dir)
    
    # Load champion base models from v4/v5 entry for Changing-Lag predictions on test
    import tempfile
    from slot_forecasting.submission_contract import extract_archive
    from hgb_upgrade.features import build_advanced_features as build_base_features
    v4_zip = Path("compiled_versions/v4_promoted_champion_exp014/submission_entry.zip")
    with tempfile.TemporaryDirectory() as tmp_extract:
        tmp_path = Path(tmp_extract) / "unzipped"
        extract_archive(v4_zip.read_bytes(), tmp_path)
        base_v4 = FrequencyForecaster.load(tmp_path / "base-entry/base.json")
        champ_models = joblib.load(tmp_path / "base-entry/champion.joblib")
        
        # Predict champion backbone
        def predict_champ_test(qs, h, base_f, models):
            grouped = sequences(qs, h)
            X = np.asarray([
                build_base_features(
                    q,
                    grouped[(q.episode_id, q.machine_id)],
                    [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
                    q.cutoff,
                    base_f
                ) for q in qs
            ], dtype=np.float32)
            logits = np.mean([m.decision_function(X) for m in models], axis=0)
            temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
            probs = softmax(logits / temp[:, None], axis=1)
            return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}
            
        test_champ_preds = predict_champ_test(test_queries, test_history, base_v4, champ_models)
        
    test_glob_preds = predict_ensemble(test_queries, test_history, base, models_glob, table, use_mt=False)
    test_mt_preds = predict_ensemble(test_queries, test_history, base, models_mt, table, use_mt=True)
    
    test_discrete = {
        q.query_id: tuple(0.5 * np.array(test_glob_preds[q.query_id]) + 0.5 * np.array(test_mt_preds[q.query_id]))
        for q in test_queries
    }
    
    raw_test_P = []
    test_m_indices = []
    for q in test_queries:
        qid = q.query_id
        p_disc = np.asarray(test_discrete[qid])
        p_champ = np.asarray(test_champ_preds[qid])
        p_mix = 0.30 * p_disc + 0.70 * p_champ
        raw_test_P.append(p_mix)
        test_m_indices.append(MTYPE_MAP[q.machine_type])
        
    raw_test_P = np.asarray(raw_test_P)
    test_m_indices = np.asarray(test_m_indices)
    cal_test_P = transform_hierarchical(raw_test_P, test_m_indices, tg, bg, dt, db)
    
    final_test_preds = {}
    for i, q in enumerate(test_queries):
        raw_p = cal_test_P[i]
        tot = float(np.sum(raw_p))
        norm = tuple(float(v / tot) for v in raw_p)
        final_test_preds[q.query_id] = probabilities(norm, q.query_id)
        
    # 10. Save experiment artifacts
    write_submission(out_dir / "validation_predictions.csv", final_val_preds)
    write_submission(out_dir / "submission.csv", final_test_preds)
    
    report_dict = {
        "experiment_id": "EXP-019",
        "name": "Discrete Autoregressive Lag Sequences Specialist",
        "dual_gate_status": {
            "passed": bool(pass_tuning and pass_audit),
            "tuning_passed": bool(pass_tuning),
            "audit_passed": bool(pass_audit)
        },
        "blend_weight_discrete": 0.30,
        "best_penalty": best_penalty,
        "evaluation_vs_v5": eval_rep,
        "evaluation_vs_v4": eval_rep_v4,
        "parameters": {
            "theta_global": float(tg),
            "beta_global": [float(v) for v in bg],
            "delta_theta_machine": {m: float(dt[i]) for m, i in MTYPE_MAP.items()},
            "delta_beta_machine": {m: [float(v) for v in db[i]] for m, i in MTYPE_MAP.items()}
        }
    }
    write_json(out_dir / "results.json", report_dict)
    write_json(out_dir / "report.json", report_dict)
    write_json(out_dir / "transition_tables.json", table.to_dict())
    
    joblib.dump(models_glob, out_dir / "models_glob.joblib", compress=3)
    joblib.dump(models_mt, out_dir / "models_mt.joblib", compress=3)
    base.save(out_dir / "base.json")
    
    print(f"\nAll artifacts successfully saved to: {out_dir}")
    return pass_tuning and pass_audit, report_dict

if __name__ == "__main__":
    main()
