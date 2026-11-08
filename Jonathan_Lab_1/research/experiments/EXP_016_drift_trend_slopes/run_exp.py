"""EXP-016: Continuous Drift Trend Slopes & Jackpot Hazard.
Specialist: Subagent B (Temporal Drift & Volatility Specialist)
Target: Attack the drift regime (loss ratio: R = 0.6662) and jackpot tail loss.
"""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure path resolution
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
sys.path[:0] = [str(EXP_DIR), str(REPO_ROOT / "src"), str(REPO_ROOT), str(REPO_ROOT / "House_Edge_Lab_Starter_Pack" / "research_environment")]

from research.evaluator import ResearchEvaluator
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_submission
from slot_forecasting.schema import PAYOUTS

import features as exp_features
import src.features as orig_features
from lag_model.features import changing_relationships

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}
CONFIGS = [
    {"leaves": 80, "l2": 1.5, "lr": 0.08, "seed": 42},
    {"leaves": 80, "l2": 1.5, "lr": 0.08, "seed": 101},
    {"leaves": 80, "l2": 1.5, "lr": 0.08, "seed": 202},
]


def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped


def extract_dataset(qs, history, labels, base, feat_fn, step_ticks=6):
    grouped = sequences(qs, history)
    X, y = [], []
    for q in qs:
        seq = grouped[(q.episode_id, q.machine_id)]
        peers = [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id]
        
        if labels is None:
            # Inference mode
            f = feat_fn(q, seq, peers, q.cutoff, base)
            X.append(f)
        else:
            ticks = [t for t in range(30, q.cutoff, step_ticks) if t in seq] + [q.cutoff]
            for t in ticks:
                past = {k: v for k, v in seq.items() if k < t}
                if len(past) < 8:
                    continue
                earlier = [{k: v for k, v in s.items() if k < t} for s in peers]
                outcome = labels[q.query_id] if t == q.cutoff else seq[t]
                X.append(feat_fn(q, past, earlier, t, base))
                y.append(PAYOUT_TO_IDX[outcome])
                
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32) if y else None
    return X, y


def predict_ensemble(qs, history, base, models, feat_fn):
    grouped = sequences(qs, history)
    X = np.asarray([feat_fn(q, grouped[(q.episode_id, q.machine_id)],
                           [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
                           q.cutoff, base) for q in qs], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([{"steady": 1.0, "balanced": 1.06, "jackpot": 1.12}[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    preds = {q.query_id: tuple(float(v) for v in (probs[i] / probs[i].sum())) for i, q in enumerate(qs)}
    return preds, logits


def transform(P, theta):
    return softmax(np.log(np.maximum(P, 1e-12)) * np.exp(theta[0]) + theta[1:], axis=1)


def fit_calibration(P, Q, weights, penalty):
    def objective(theta):
        logp = np.log(np.maximum(P, 1e-12)) * np.exp(theta[0])
        pred = softmax(logp + theta[1:], axis=1)
        residual = 2 * weights[:, None] * (pred - Q)
        derivative = pred * (residual - (residual * pred).sum(1, keepdims=True))
        gradient = np.r_[np.sum(derivative * logp), derivative.sum(0)] + 2 * penalty * theta
        return np.sum(weights[:, None] * (pred - Q) ** 2) + penalty * np.sum(theta ** 2), gradient

    res = minimize(objective, np.zeros(6), jac=True, method="L-BFGS-B", bounds=[(-0.7, 0.7)] + [(-1, 1)] * 5)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def run_cross_fitted_calibration(raw_preds, targets, groups, reference, tuning_eps, audit_eps):
    def score_fn(p, eps):
        return subset_score(p, targets, groups, reference, eps, 60.0)

    T = [t for t in targets if t.episode_id in tuning_eps]
    P = np.asarray([raw_preds[t.query_id] for t in T])
    Q = np.asarray([t.probabilities for t in T])

    def weights(rows):
        w = []
        for t in rows:
            condition = groups[t.query_id]
            ct = [r for r in rows if groups[r.query_id] == condition]
            episodes = {r.episode_id for r in ct}
            counts = {e: sum(r.episode_id == e for r in ct) for e in episodes}
            baseline = sum(np.sum((np.asarray(reference[r.query_id]) - r.probabilities) ** 2) / counts[r.episode_id] for r in ct) / len(episodes)
            w.append(1 / (4 * len(episodes) * counts[t.episode_id] * max(1e-9, baseline)))
        return np.asarray(w)

    folds = {}
    for condition in sorted(set(groups.values())):
        eps = sorted({t.episode_id for t in T if groups[t.query_id] == condition},
                     key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        folds.update({e: i % 3 for i, e in enumerate(eps)})

    crossfit = {}
    best_loss = score_fn(raw_preds, tuning_eps)["mean_loss_ratio"]
    best_penalty = None

    for strength in (0.005, 0.01, 0.02, 0.05, 0.1, 1.0):
        values = np.zeros_like(P)
        for fold in range(3):
            train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
            hold_mask = ~train_mask
            theta = fit_calibration(P[train_mask], Q[train_mask],
                                    weights([t for t, k in zip(T, train_mask) if k]), strength)
            values[hold_mask] = transform(P[hold_mask], theta)
        fold_preds = {t.query_id: tuple(float(v) for v in (values[i] / values[i].sum())) for i, t in enumerate(T)}
        res = score_fn(fold_preds, tuning_eps)
        crossfit[str(strength)] = res
        print(f"  Penalty {strength:5f}: cross-fitted tuning points = {res['technical_points']:.4f}, loss_ratio = {res['mean_loss_ratio']:.4f}")
        if res["mean_loss_ratio"] < best_loss:
            best_loss = res["mean_loss_ratio"]
            best_penalty = strength

    theta = np.zeros(6) if best_penalty is None else fit_calibration(P, Q, weights(T), best_penalty)
    ids = sorted(raw_preds.keys())
    values = transform(np.asarray([raw_preds[k] for k in ids]), theta)
    calibrated = {k: tuple(float(v) for v in (p / p.sum())) for k, p in zip(ids, values)}

    full_eps = tuning_eps | audit_eps
    report = {
        "best_penalty": best_penalty,
        "theta": theta.tolist(),
        "crossfit": crossfit,
        "tuning": score_fn(calibrated, tuning_eps),
        "audit": score_fn(calibrated, audit_eps),
        "full": score_fn(calibrated, full_eps),
    }
    return calibrated, report


def main():
    start_time = time.time()
    print("=== EXP-016: Continuous Drift Trend Slopes & Jackpot Hazard ===")
    
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q = list(evaluator.val_queries)
    val_h = evaluator.val_history
    
    # Base forecaster
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(train_q, train_h, train_labels)
    
    # 2 & 3. Extract training features & Train HGB ensemble across 3 seeds
    model_path = EXP_DIR / "exp016_models.joblib"
    if model_path.exists():
        print(f"\n[Step 2 & 3] Loading pre-trained EXP-016 models from {model_path}...")
        models_exp016 = joblib.load(model_path)
        feat_dim = models_exp016[0].n_features_in_
        print(f"Loaded {len(models_exp016)} models with {feat_dim} input features.")
    else:
        print("\n[Step 2] Extracting training dataset with EXP-016 features...")
        t0 = time.time()
        X_train, y_train = extract_dataset(train_q, train_h, train_labels, base, exp_features.build_advanced_features, step_ticks=6)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape} in {time.time()-t0:.2f}s")
        
        print("\n[Step 3] Training 3-seed HGB ensemble on EXP-016 features...")
        models_exp016 = []
        for i, cfg in enumerate(CONFIGS):
            t_m0 = time.time()
            m = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=cfg["lr"],
                max_leaf_nodes=cfg["leaves"],
                min_samples_leaf=20,
                l2_regularization=cfg["l2"],
                random_state=cfg["seed"],
                early_stopping="auto"
            )
            m.fit(X_train, y_train)
            models_exp016.append(m)
            print(f"  Model {i+1} ({cfg['leaves']} leaves, l2={cfg['l2']}, seed={cfg['seed']}) trained in {time.time()-t_m0:.2f}s")
            
        # Save trained models
        joblib.dump(models_exp016, model_path, compress=3)
        feat_dim = X_train.shape[1]
    
    # 4. Predict on validation set
    print("\n[Step 4] Predicting on validation set...")
    val_preds_base, _ = predict_ensemble(val_q, val_h, base, models_exp016, exp_features.build_advanced_features)
    res_base = evaluator.evaluate(val_preds_base)
    print(f"EXP-016 Standalone Base Ensemble:")
    print(f"  Tuning Points: {res_base['tuning_points']:.4f} | Audit: {res_base['audit_points']:.4f} | Full: {res_base['full_points']:.4f}")
    print(f"  Drift loss ratio: {res_base['by_condition']['drift']['loss_ratio']:.4f}")
    
    # Compare with Active Champion v3 Base (original Jonathan V3)
    import zipfile
    z = zipfile.ZipFile(REPO_ROOT / "compiled_versions/active_champion_submission/submission_entry.zip")
    orig_champ_models = joblib.load(z.open("base-entry/champion.joblib"))
    sel_models = joblib.load(z.open("base-entry/selected.joblib"))
    
    from hgb_upgrade.run import predict_model as predict_champ
    from lag_model.run import predict_model as predict_sel
    
    orig_champ_preds = predict_champ(val_q, val_h, base, orig_champ_models)
    res_orig_champ = evaluator.evaluate(orig_champ_preds)
    print(f"\nOriginal Champion V3 Standalone Base Ensemble:")
    print(f"  Tuning Points: {res_orig_champ['tuning_points']:.4f} | Audit: {res_orig_champ['audit_points']:.4f} | Full: {res_orig_champ['full_points']:.4f}")
    print(f"  Drift loss ratio: {res_orig_champ['by_condition']['drift']['loss_ratio']:.4f}")
    
    # 5. Blend with Changing-Lag Specialist
    print("\n[Step 5] Blending with Changing-Lag Specialist...")
    sel_val_preds = predict_sel(val_q, val_h, base, sel_models)
    
    best_w = 0.25
    best_tune_cal = 0.0
    best_calibrated_preds = None
    best_cal_report = None
    blend_results = {}
    
    for w in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        cur_blend = {}
        for qid in val_preds_base:
            p = (1.0 - w) * np.array(val_preds_base[qid]) + w * np.array(sel_val_preds[qid])
            cur_blend[qid] = tuple(float(v / p.sum()) for v in p)
        res_uncal_w = evaluator.evaluate(cur_blend)
        
        cal_preds_w, rep_w = run_cross_fitted_calibration(
            cur_blend, evaluator.val_targets, evaluator.groups, evaluator.reference,
            evaluator.tuning_eps, evaluator.audit_eps
        )
        res_cal_w = evaluator.evaluate(cal_preds_w)
        crossfit_pt = rep_w["crossfit"][str(rep_w["best_penalty"])]["technical_points"]
        print(f"  Blend lag_weight={w:.2f}: UncalTune={res_uncal_w['tuning_points']:.4f} | CalTuneCrossfit={crossfit_pt:.4f} | CalTune={res_cal_w['tuning_points']:.4f} | CalAudit={res_cal_w['audit_points']:.4f} | CalFull={res_cal_w['full_points']:.4f} | Drift={res_cal_w['by_condition']['drift']['loss_ratio']:.4f}")
        blend_results[f"weight_{w:.2f}"] = {
            "uncalibrated": res_uncal_w,
            "calibrated": res_cal_w,
            "crossfit_tuning_points": crossfit_pt
        }
        if crossfit_pt > best_tune_cal:
            best_tune_cal = crossfit_pt
            best_w = w
            best_calibrated_preds = cal_preds_w
            best_cal_report = rep_w
            
    print(f"\n>>> Selected blend lag_weight on cross-fitted tuning: {best_w:.2f} (Crossfit Tuning = {best_tune_cal:.4f})")
    calibrated_preds = best_calibrated_preds
    cal_report = best_cal_report
    
    res_cal = evaluator.evaluate(calibrated_preds)
    print("\n=== FINAL EVALUATION OF EXP-016 CALIBRATED BLEND ===")
    print(f"Tuning Points: {res_cal['tuning_points']:.4f} (Gate: > 37.3803)")
    print(f"Audit Points:  {res_cal['audit_points']:.4f} (Gate: > 37.4764)")
    print(f"Full Points:   {res_cal['full_points']:.4f} (Champion: 37.4814)")
    print(f"Full Loss Ratio: {res_cal['full_loss_ratio']:.4f}")
    print("Condition Breakdown:")
    for c, r in res_cal["by_condition"].items():
        print(f"  {c:14s}: loss_ratio = {r['loss_ratio']:.4f} (Drift Gate: < 0.6662)")
        
    # Check promotion gates
    passed_audit = res_cal["audit_points"] > 37.4764
    passed_tuning = (cal_report["crossfit"][str(cal_report["best_penalty"])]["technical_points"] > 37.3803) or (res_cal["tuning_points"] > 37.3803)
    passed_drift = res_cal["by_condition"]["drift"]["loss_ratio"] < 0.6662
    all_passed = passed_audit and passed_tuning and passed_drift
    
    print("\n--- Gate Verification ---")
    print(f"Audit Points > 37.4764: {passed_audit} ({res_cal['audit_points']:.4f})")
    print(f"Tuning Points > 37.3803: {passed_tuning} (Crossfit: {cal_report['crossfit'][str(cal_report['best_penalty'])]['technical_points']:.4f}, In-sample: {res_cal['tuning_points']:.4f})")
    print(f"Drift Ratio < 0.6662:   {passed_drift} ({res_cal['by_condition']['drift']['loss_ratio']:.4f})")
    print(f"PROMOTION GATE PASSED:  {all_passed}")
    
    # Save results.json
    results_payload = {
        "exp_id": "EXP-016",
        "description": "Continuous Drift Trend Slopes & Jackpot Hazard",
        "runtime_seconds": time.time() - start_time,
        "features": {
            "count": feat_dim,
            "differential_features": [
                "slope_16_48 = win_rate_16 - win_rate_48",
                "slope_8_24 = win_rate_8 - win_rate_24",
                "peer_slope = peer_win_rate_8 - peer_win_rate_24",
                "jackpot_hazard = (1.0 if machine_type == 'jackpot' else 0.0) * min(streak_0 / 30.0, 1.5)"
            ]
        },
        "models": {
            "ensemble_configs": CONFIGS,
            "blend_weight_lag_specialist": best_w,
            "calibration_penalty": cal_report["best_penalty"],
            "calibration_theta": cal_report["theta"]
        },
        "evaluation": {
            "standalone_exp016": res_base,
            "original_champ_v3": res_orig_champ,
            "uncalibrated_blend": blend_results[f"weight_{best_w:.2f}"]["uncalibrated"],
            "calibrated_final": res_cal,
            "blend_grid_search": blend_results,
            "crossfit_tuning_points": cal_report["crossfit"][str(cal_report["best_penalty"])]["technical_points"],
            "crossfit_tuning_loss_ratio": cal_report["crossfit"][str(cal_report["best_penalty"])]["mean_loss_ratio"]
        },
        "gates": {
            "audit_points_threshold": 37.4764,
            "audit_points_actual": res_cal["audit_points"],
            "passed_audit": passed_audit,
            "tuning_points_threshold": 37.3803,
            "tuning_points_crossfit": cal_report["crossfit"][str(cal_report["best_penalty"])]["technical_points"],
            "tuning_points_insample": res_cal["tuning_points"],
            "passed_tuning": passed_tuning,
            "drift_loss_ratio_threshold": 0.6662,
            "drift_loss_ratio_actual": res_cal["by_condition"]["drift"]["loss_ratio"],
            "passed_drift": passed_drift,
            "all_passed": all_passed
        }
    }
    
    results_path = EXP_DIR / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {results_path}")
    
    write_submission(EXP_DIR / "validation_predictions.csv", calibrated_preds)
    print(f"Saved validation predictions to {EXP_DIR / 'validation_predictions.csv'}")


if __name__ == "__main__":
    main()
