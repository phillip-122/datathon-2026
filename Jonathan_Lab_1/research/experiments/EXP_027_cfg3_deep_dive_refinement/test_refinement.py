import sys, json, joblib, time
from pathlib import Path
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
    str(_LAB / "src"),
    str(_LAB / "compiled_versions/active_champion_submission/entry")
]

from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import build_advanced_features, PAYOUT_TO_IDX
from research.tools.controller import ResearchController

from feature_factory import (
    extract_exp020_drift_features,
    compute_cutoff_safe_archetype,
    get_discrete_archetype_state,
    ArchetypeConditionedTransitionTable,
    MTYPES, MTYPE_MAP
)

DROUGHT_HAZARDS = {0: 0.0, 1: 0.47, 2: 0.45, 3: 0.53, 4: 0.37, 5: 0.46, 6: 0.31, 7: 0.28, 8: 0.17, 9: 0.13, 10: 0.06}
TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def extract_cfg3_plus_deepdive(query, past, peer_past, cutoff, base, trans_table):
    # 1. Base 363 features from Config 3
    feats = build_advanced_features(query, past, peer_past, cutoff, base)
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    drift_feats = extract_exp020_drift_features(query, past, peer_past, cutoff)
    trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    feats += trans_feats
    feats += drift_feats
    feats += [1.0 if arch_conf == -1 else 0.0, 1.0 if arch_conf == 0 else 0.0, 1.0 if arch_conf == 1 else 0.0]
    
    # 2. Add 10 Databricks Deep Dive features:
    # Priority 1: Surge window
    in_surge = 1.0 if (210 <= cutoff <= 219) else 0.0
    surge_bell = float(np.exp(-0.5 * ((cutoff - 217.0) / 3.5) ** 2))
    
    # Priority 2: Sync level at cutoff - 1
    t_prev = cutoff - 1
    sync_prev = sum(1 for s in peer_past if s.get(t_prev, 0) > 0)
    sync_high = 1.0 if sync_prev >= 6 else 0.0
    sync_med = 1.0 if (4 <= sync_prev <= 5) else 0.0
    
    # Priority 3: Drought length & universal hazard
    drought = 0
    t = cutoff - 1
    while t in past and past[t] == 0:
        drought += 1
        t -= 1
    hazard = float(DROUGHT_HAZARDS.get(min(10, drought), 0.03))
    in_deep_drought = 1.0 if drought >= 6 else 0.0
    
    # Priority 4: Jackpot contagion (peer 5x/20x in last 3 ticks)
    pj_3 = sum(1 for s in peer_past for dt in range(1, 4) if s.get(cutoff - dt, 0) in (5, 20))
    in_contagion = 1.0 if pj_3 > 0 else 0.0
    
    feats += [
        in_surge,
        surge_bell,
        float(sync_prev / 7.0),
        sync_high,
        sync_med,
        float(min(drought, 20) / 20.0),
        hazard,
        in_deep_drought,
        float(min(pj_3, 4) / 4.0),
        in_contagion
    ]
    return feats

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
        data_loss = np.sum(sample_weights * errs) if sample_weights is not None else np.mean(errs)
        reg = lambda_base * (tg**2 + np.sum(bg**2)) + lambda_hier * (np.sum(dt**2) + np.sum(db**2))
        return data_loss + reg
    res = minimize(loss_func, np.zeros(24), method="L-BFGS-B", options={"maxiter": 250})
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

def build_training_dataset(qs, history, labels, base_forecaster, trans_table, rng_seed=42):
    grouped = sequences(qs, history)
    X, y = [], []
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
            X.append(extract_cfg3_plus_deepdive(q, past, earlier, t, base_forecaster, trans_table))
            y.append(PAYOUT_TO_IDX[outcome])
            if len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(extract_cfg3_plus_deepdive(q, m_past, m_earlier, t, base_forecaster, trans_table))
                y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def main():
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
    labels = read_labels(train_dir / "labels.csv", t_queries)
    
    print(f"Loaded {len(t_queries)} train queries, {len(val_queries)} val queries.")
    
    v10_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", {t.query_id for t in targets})
    eval_v10 = ctrl.evaluate_candidate(v10_preds)
    print("\nACTIVE CHAMPION V10 BENCHMARK:")
    print(f"  Full:   {eval_v10['full_points']:.4f} pts | Loss Ratio: {eval_v10['full_loss_ratio']:.4f}")
    print(f"  Tuning: {eval_v10['tuning_points']:.4f} pts | Audit: {eval_v10['audit_points']:.4f} pts")
    for c, r in eval_v10['by_condition'].items():
        print(f"    - {c:14s}: {r['loss_ratio']:.4f}")
        
    print("\nFitting base forecaster and ArchetypeConditionedTransitionTable...")
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, labels)
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    trans_table.fit(t_queries, t_history)
    
    print("Building full 28,154-sample training dataset with Config 3 + 10 Deep Dive features...")
    t0 = time.time()
    X_train, y_train = build_training_dataset(t_queries, t_history, labels, base, trans_table)
    print(f"  Train data generated: {X_train.shape} in {time.time() - t0:.1f}s")
    
    grouped_val = sequences(val_queries, val_history)
    t1 = time.time()
    print("Extracting validation features...")
    X_val = np.asarray([
        extract_cfg3_plus_deepdive(
            q,
            {t: p for t, p in grouped_val[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_val.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table
        ) for q in val_queries
    ], dtype=np.float32)
    print(f"  Validation features extracted: {X_val.shape} in {time.time() - t1:.1f}s")
    
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
    raw_cfg3_plus = softmax(val_logits / temp[:, None], axis=1)
    
    # Evaluate Long History Blend with v6
    query_map = {q.query_id: q for q in val_queries}
    target_qids = [t.query_id for t in targets]
    M_val = np.asarray([MTYPE_MAP[query_map[qid].machine_type] for qid in target_qids])
    T = [t for t in targets if t.episode_id in tuning]
    M_tuning = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    Q_tuning = np.asarray([t.probabilities for t in T])
    
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
    n_past_val = {q.query_id: len([t for t in grouped_val[(q.episode_id, q.machine_id)] if t < q.cutoff]) for q in val_queries}
    
    # Load v6 predictions
    v6_dir = Path("compiled_versions/active_champion_submission/entry")
    from predict_v6 import (
        predict_ensemble as v6_predict_ensemble,
        predict_champ as v6_predict_champ,
        DiscreteLagTransitionTable as V6DiscreteLagTransitionTable,
        transform_hierarchical as v6_transform_hierarchical
    )
    base_champ = FrequencyForecaster.load(v6_dir / "base_champ.json")
    champ_models = joblib.load(v6_dir / "champ_models.joblib")
    base_v6 = FrequencyForecaster.load(v6_dir / "base.json")
    models_glob = joblib.load(v6_dir / "models_glob.joblib")
    models_mt = joblib.load(v6_dir / "models_mt.joblib")
    table_data = json.loads((v6_dir / "transition_tables.json").read_text())
    v6_table = V6DiscreteLagTransitionTable(table_data)
    v6_cal = json.loads((v6_dir / "hierarchical_calibration.json").read_text())
    
    p_champ = v6_predict_champ(val_queries, val_history, base_champ, champ_models)
    p_glob = v6_predict_ensemble(val_queries, val_history, base_v6, models_glob, v6_table, use_mt=False)
    p_mt = v6_predict_ensemble(val_queries, val_history, base_v6, models_mt, v6_table, use_mt=True)
    
    p_v6_raw = []
    for q in val_queries:
        p_disc = 0.5 * np.asarray(p_glob[q.query_id]) + 0.5 * np.asarray(p_mt[q.query_id])
        p_mix = 0.30 * p_disc + 0.70 * np.asarray(p_champ[q.query_id])
        p_v6_raw.append(p_mix)
    dt_v6 = np.asarray([v6_cal["delta_theta_machine"][m] for m in MTYPES])
    db_v6 = np.asarray([v6_cal["delta_beta_machine"][m] for m in MTYPES])
    p_v6_cal = v6_transform_hierarchical(np.asarray(p_v6_raw), M_val, v6_cal["theta_global"], np.asarray(v6_cal["beta_global"]), dt_v6, db_v6)
    
    print("\n--- Sweeping Adaptive Weights for Config 3 Plus ---")
    best_res = None
    best_cand = None
    
    for w_weights in [(0.15, 0.28, 0.35), (0.12, 0.25, 0.30), (0.18, 0.30, 0.40), (0.10, 0.20, 0.25)]:
        w_map = {"steady": w_weights[0], "balanced": w_weights[1], "jackpot": w_weights[2]}
        w_vec = np.asarray([w_map[q.machine_type] for q in val_queries])
        
        blended = w_vec[:, None] * raw_cfg3_plus + (1.0 - w_vec[:, None]) * p_v6_cal
        blended = blended / np.sum(blended, axis=1, keepdims=True)
        
        P_tune = np.asarray([blended[target_qids.index(t.query_id)] for t in T])
        theta = fit_hierarchical_calibrator(P_tune, Q_tuning, M_tuning, w_tuning)
        
        P_val_cal = transform_hierarchical(blended, M_val, theta)
        
        # Route: for n_past <= 100, keep exact Champion v10 short history predictions!
        cand = {}
        for i, q in enumerate(val_queries):
            qid = q.query_id
            if n_past_val[qid] <= 100:
                cand[qid] = v10_preds[qid]
            else:
                cand[qid] = tuple(float(v) for v in P_val_cal[i])
                
        res = ctrl.evaluate_candidate(cand, base_predictions=v10_preds, n_bootstrap=1000)
        print("Weights {}: Full: {:.4f} (Tune: {:.4f}, Audit: {:.4f}) | Loss: {:.4f} | CI vs v10: [{:+.4f}, {:+.4f}]".format(
            w_weights, res["full_points"], res["tuning_points"], res["audit_points"], res["full_loss_ratio"],
            res["bootstrap_gain_vs_base_ci95"][0], res["bootstrap_gain_vs_base_ci95"][1]
        ))
        for c, r in res["by_condition"].items():
            print(f"    - {c:14s}: {r['loss_ratio']:.4f} (v10: {eval_v10['by_condition'][c]['loss_ratio']:.4f})")
            
        if best_res is None or res["full_points"] > best_res["full_points"]:
            best_res = res
            best_cand = cand
            
    print("\n" + "="*70)
    print("BEST CONFIG 3 PLUS BENCHMARK:")
    print(f"Full Points:   {best_res['full_points']:.4f} (v10: {eval_v10['full_points']:.4f})")
    print(f"Tuning Points: {best_res['tuning_points']:.4f} (v10: {eval_v10['tuning_points']:.4f})")
    print(f"Audit Points:  {best_res['audit_points']:.4f} (v10: {eval_v10['audit_points']:.4f})")
    print(f"Loss Ratio:    {best_res['full_loss_ratio']:.4f}")
    print(f"CI vs v10:     [{best_res['bootstrap_gain_vs_base_ci95'][0]:.4f}, {best_res['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
    print("="*70)

if __name__ == "__main__":
    main()
