"""EXP-015: Synthetic Masking for Short-History Generalization.

Hypothesis:
Training data contains dense, continuous histories (up to 256 ticks).
However, short_history test machines have only 64 observed ticks (40 recent + 24 sparse older ticks).
The decision trees learned splits assuming dense data, so they make suboptimal predictions when 75%
of ticks are unobserved.
By exposing the trees to synthetic 64-tick masked histories during training, the models learn the
missing-data topology and improve predictions on short-history queries.

Pipeline:
1. Feature extraction with augment=True (dense + masked synthetic rows via masked()).
2. Train 3-seed multi-structure changing-lag specialist models (leaves 40/60/80, max_iter=150, lr=0.08).
3. Blend 25% new specialist model + 75% Jonathan V3 (artifacts/models/hgb_ensemble_v3/).
4. Apply 3-fold episode-cross-fitted calibration from src/calibrated_model/run.py.
5. Evaluate on independent 50/50 Tuning vs Audit split using ResearchEvaluator.
6. Compare against active champion v4 (EXP-014):
   - Active Champion Audit: 37.4764
   - Active Champion Tuning (cross-fit): 37.3803
   - Active Champion Short History loss ratio: 0.6848 (Audit) / 0.6858 (Full)
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
LAB_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(LAB_DIR),
    str(LAB_DIR / "src"),
    str(BASE_DIR),
    str(LAB_DIR / "House_Edge_Lab_Starter_Pack" / "research_environment"),
    str(BASE_DIR / "Phillip_Lab" / "src"),
]

from research.evaluator import ResearchEvaluator
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_submission, write_json
from src.features import build_advanced_features as build_v3_features
from lag_model.features import build_advanced_features as build_lag_features, PAYOUT_TO_IDX

EXP_ID = "EXP-015"
EXP_DIR = Path(__file__).resolve().parent

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def masked(seq, tick, rng):
    """40 recent ticks (cutoff - 40 to cutoff) + 24 older ticks randomly sampled."""
    recent = {t: p for t, p in seq.items() if tick - 40 <= t < tick}
    old = sorted(t for t in seq if t < tick - 40)
    keep = rng.choice(old, min(24, len(old)), replace=False) if old else []
    return {**recent, **{int(t): seq[t] for t in keep}}

def training_rows(qs, history, labels, base, augment=False):
    grouped = sequences(qs, history)
    X = []
    y = []
    ages = []
    rng = np.random.default_rng(42)
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
            X.append(build_lag_features(q, past, earlier, t, base))
            y.append(PAYOUT_TO_IDX[outcome])
            ages.append(q.cutoff - t)
            if augment and len(past) > 64:
                X.append(build_lag_features(q, masked(past, t, rng), [masked(s, t, rng) for s in earlier], t, base))
                y.append(PAYOUT_TO_IDX[outcome])
                ages.append(q.cutoff - t)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(ages, dtype=np.float32)

def predict_lag_model(qs, h, base, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_lag_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
        )
        for q in qs
    ], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def predict_v3_champion(qs, h, base, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_v3_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
        )
        for q in qs
    ], dtype=np.float32)
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def mix(a, b, w):
    """Linear blend: w * a + (1 - w) * b."""
    return {k: tuple(w * np.asarray(a[k]) + (1.0 - w) * np.asarray(b[k])) for k in a}

def transform_probs(P, theta):
    """Logit scaling theta[0] and class offsets theta[1:]."""
    return softmax(np.log(np.maximum(P, 1e-12)) * np.exp(theta[0]) + theta[1:], axis=1)

def fit_calibration(P, Q, weights, penalty):
    def objective(theta):
        logp = np.log(np.maximum(P, 1e-12)) * np.exp(theta[0])
        pred = softmax(logp + theta[1:], axis=1)
        residual = 2 * weights[:, None] * (pred - Q)
        derivative = pred * (residual - (residual * pred).sum(1, keepdims=True))
        gradient = np.r_[np.sum(derivative * logp), derivative.sum(0)] + 2 * penalty * theta
        return np.sum(weights[:, None] * (pred - Q)**2) + penalty * np.sum(theta**2), gradient

    result = minimize(objective, np.zeros(6), jac=True, method='L-BFGS-B', bounds=[(-0.7, 0.7)] + [(-1.0, 1.0)] * 5)
    if not result.success:
        raise RuntimeError(f"Calibration optimization failed: {result.message}")
    return result.x

def calculate_weights(rows, groups, reference):
    w = []
    for t in rows:
        condition = groups[t.query_id]
        ct = [r for r in rows if groups[r.query_id] == condition]
        episodes = {r.episode_id for r in ct}
        counts = {e: sum(r.episode_id == e for r in ct) for e in episodes}
        baseline = sum(
            np.sum((np.asarray(reference[r.query_id]) - r.probabilities)**2) / counts[r.episode_id]
            for r in ct
        ) / len(episodes)
        w.append(1.0 / (4.0 * len(episodes) * counts[t.episode_id] * max(1e-9, baseline)))
    return np.asarray(w)

def run_experiment():
    print(f"=== Starting {EXP_ID}: Synthetic Masking for Short-History ===")
    t_start = time.time()
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir

    # 1. Load training data
    print(f"Loading training data from {data_dir / 'train'}...")
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)

    # Validation inputs
    val_q = list(evaluator.val_queries)
    val_h = evaluator.val_history
    val_targets = evaluator.val_targets
    groups = evaluator.groups
    reference = evaluator.reference
    tuning = evaluator.tuning_eps
    audit = evaluator.audit_eps

    def score_subset(p, eps):
        return subset_score(p, val_targets, groups, reference, eps, 60.0)

    # Load Jonathan V3 champion
    v3_base_path = LAB_DIR / "artifacts/models/hgb_ensemble_v3/base_forecaster.json"
    v3_models_path = LAB_DIR / "artifacts/models/hgb_ensemble_v3/models.joblib"
    base = FrequencyForecaster.load(v3_base_path)
    v3_models = joblib.load(v3_models_path)
    print("Loaded Jonathan V3 forecaster and 3-seed ensemble.")

    # Generate Jonathan V3 predictions on validation set
    print("Computing Jonathan V3 baseline validation predictions...")
    p_v3 = predict_v3_champion(val_q, val_h, base, v3_models)
    v3_eval = evaluator.evaluate(p_v3)
    print(f"  V3 Champion: Tune={v3_eval['tuning_points']:.4f}, Audit={v3_eval['audit_points']:.4f}, Full={v3_eval['full_points']:.4f}")

    # 2. Extract training rows with augment=True
    print("\nExtracting training features with augment=True (synthetic 64-tick masking)...")
    t0 = time.time()
    X_train, y_train, ages = training_rows(train_q, train_h, train_labels, base, augment=True)
    t_extract = time.time() - t0
    print(f"Extracted {len(y_train)} rows with {X_train.shape[1]} features in {t_extract:.1f}s.")

    # 3. Fit multi-structure HistGradientBoosting specialist models
    print("\nTraining 3-seed multi-structure specialist models on dense + synthetic masked rows...")
    model_configs = [
        {"leaves": 40, "l2": 4.0, "seed": 42},
        {"leaves": 60, "l2": 2.0, "seed": 101},
        {"leaves": 80, "l2": 1.5, "seed": 202},
    ]
    models = []
    for cfg in model_configs:
        t_m0 = time.time()
        print(f"  Fitting leaves={cfg['leaves']}, l2={cfg['l2']}, seed={cfg['seed']}...")
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=cfg["leaves"],
            min_samples_leaf=20,
            l2_regularization=cfg["l2"],
            random_state=cfg["seed"],
            early_stopping=False,
        )
        clf.fit(X_train, y_train)
        models.append(clf)
        print(f"    Finished in {time.time() - t_m0:.1f}s.")

    # Save trained specialist models to experiment directory
    joblib.dump(models, EXP_DIR / "specialist_masked_models.joblib", compress=3)
    print(f"Saved specialist models to {EXP_DIR / 'specialist_masked_models.joblib'}")

    # 4. Predict validation set with new specialist model
    print("\nPredicting validation queries with masked specialist...")
    p_specialist = predict_lag_model(val_q, val_h, base, models)
    spec_eval = evaluator.evaluate(p_specialist)
    print(f"  Specialist alone: Tune={spec_eval['tuning_points']:.4f}, Audit={spec_eval['audit_points']:.4f}, Full={spec_eval['full_points']:.4f}")

    # 5. Blend 25% new specialist + 75% Jonathan V3
    print("\nBlending 25% specialist + 75% Jonathan V3...")
    raw_blend = mix(p_specialist, p_v3, 0.25)
    blend_eval = evaluator.evaluate(raw_blend)
    print(f"  Raw 25/75 Blend (pre-calibrated):")
    print(f"    Tuning Points: {blend_eval['tuning_points']:.4f}")
    print(f"    Audit Points:  {blend_eval['audit_points']:.4f}")
    print(f"    Full Points:   {blend_eval['full_points']:.4f}")
    for c, r in blend_eval["by_condition"].items():
        print(f"      {c:14s}: ratio={r['loss_ratio']:.4f} (err={r['error']:.4f})")

    # 6. Apply 3-fold episode-cross-fitted calibration from src/calibrated_model/run.py
    print("\n--- Applying 3-Fold Episode-Cross-Fitted Calibration ---")
    T = [t for t in val_targets if t.episode_id in tuning]
    P = np.asarray([raw_blend[t.query_id] for t in T])
    Q = np.asarray([t.probabilities for t in T])

    # Deterministic folds by episode hash
    folds = {}
    for condition in sorted(set(groups.values())):
        eps = sorted(
            {t.episode_id for t in T if groups[t.query_id] == condition},
            key=lambda e: hashlib.sha256(e.encode()).hexdigest(),
        )
        folds.update({e: i % 3 for i, e in enumerate(eps)})

    crossfit_results = {}
    best_loss_ratio = score_subset(raw_blend, tuning)["mean_loss_ratio"]
    best_penalty = None

    print(f"Raw tuning mean loss ratio: {best_loss_ratio:.6f} ({score_subset(raw_blend, tuning)['technical_points']:.4f} pts)")

    for strength in (0.01, 0.1, 1.0):
        values = np.zeros_like(P)
        for fold in range(3):
            train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
            hold_mask = ~train_mask
            theta_fold = fit_calibration(
                P[train_mask],
                Q[train_mask],
                calculate_weights([t for t, k in zip(T, train_mask) if k], groups, reference),
                strength,
            )
            values[hold_mask] = transform_probs(P[hold_mask], theta_fold)
        fold_preds = {t.query_id: tuple(p) for t, p in zip(T, values)}
        res_strength = score_subset(fold_preds, tuning)
        crossfit_results[str(strength)] = {
            "technical_points": res_strength["technical_points"],
            "mean_loss_ratio": res_strength["mean_loss_ratio"],
            "by_condition": {
                c: {"loss_ratio": r["loss_ratio"], "error": r["error"]}
                for c, r in res_strength["by_condition"].items()
            },
        }
        print(f"  Penalty {strength:4.2f}: cross-fit tuning={res_strength['technical_points']:.6f} pts (ratio={res_strength['mean_loss_ratio']:.6f})")
        if res_strength["mean_loss_ratio"] < best_loss_ratio:
            best_loss_ratio = res_strength["mean_loss_ratio"]
            best_penalty = strength

    print(f"Selected penalty: {best_penalty} (best cross-fit ratio: {best_loss_ratio:.6f})")

    # Fit final theta on all tuning true probabilities
    if best_penalty is not None:
        theta_final = fit_calibration(P, Q, calculate_weights(T, groups, reference), best_penalty)
    else:
        theta_final = np.zeros(6)

    print(f"Final calibration theta: {list(np.round(theta_final, 6))}")

    # Transform all validation queries (both tuning and audit)
    all_ids = sorted(raw_blend.keys())
    calibrated_matrix = transform_probs(np.asarray([raw_blend[k] for k in all_ids]), theta_final)
    calibrated_preds = {k: tuple(float(v) for v in p) for k, p in zip(all_ids, calibrated_matrix)}

    # 7. Evaluate calibrated model using ResearchEvaluator
    print("\n--- Final Evaluation with ResearchEvaluator ---")
    final_eval = evaluator.evaluate(calibrated_preds)
    tune_score = score_subset(calibrated_preds, tuning)
    audit_score = score_subset(calibrated_preds, audit)

    # 8. Comparison vs Active Champion (v4 / EXP-014)
    gate_audit = 37.4764
    gate_tune_crossfit = 37.3803
    selected_crossfit_tune = (
        crossfit_results[str(best_penalty)]["technical_points"]
        if best_penalty is not None
        else blend_eval["tuning_points"]
    )

    passed_audit = final_eval["audit_points"] > gate_audit
    passed_tune = selected_crossfit_tune > gate_tune_crossfit
    passed_gate = passed_audit and passed_tune

    print(f"\n==================================================")
    print(f"EXPERIMENT RESULTS SUMMARY: {EXP_ID}")
    print(f"==================================================")
    print(f"Cross-fit Tuning Points: {selected_crossfit_tune:.6f} (Gate: > {gate_tune_crossfit:.4f}) -> {'PASS' if passed_tune else 'FAIL'}")
    print(f"Audit Points:            {final_eval['audit_points']:.6f} (Gate: > {gate_audit:.4f}) -> {'PASS' if passed_audit else 'FAIL'}")
    print(f"Full Points:             {final_eval['full_points']:.6f} (Champion v4: 37.4814)")
    print(f"In-sample Tuning Points: {final_eval['tuning_points']:.6f}")
    print(f"Mean Loss Ratio:         {final_eval['full_loss_ratio']:.6f}")
    print("\nCondition Breakdown Comparison (EXP-015 vs Champion v4):")
    v4_cond_ratios = {
        "stable": 0.5538,
        "shared_change": 0.4973,
        "drift": 0.6662,
        "short_history": 0.6858,
    }
    v4_audit_cond_ratios = {
        "stable": 0.5591,
        "shared_change": 0.4917,
        "drift": 0.6726,
        "short_history": 0.6806,
    }
    for c in ("stable", "shared_change", "drift", "short_history"):
        r_full = final_eval["by_condition"][c]["loss_ratio"]
        r_audit = audit_score["by_condition"][c]["loss_ratio"]
        v4_f = v4_cond_ratios[c]
        v4_a = v4_audit_cond_ratios[c]
        print(f"  {c:14s}: Full R = {r_full:.4f} (v4: {v4_f:.4f}, delta: {r_full - v4_f:+.4f}) | Audit R = {r_audit:.4f} (v4: {v4_a:.4f}, delta: {r_audit - v4_a:+.4f})")

    status = "PROMOTED" if passed_gate else "REJECTED"
    print(f"\nPROMOTION GATE DECISION: {status}")

    # 9. Save validation predictions and results.json
    write_submission(EXP_DIR / "validation_predictions.csv", calibrated_preds)
    results_record = {
        "exp_id": EXP_ID,
        "hypothesis": "Synthetic 64-tick masking exposes decision trees to missingness topology, improving short-history generalization.",
        "status": status,
        "passed_gate": passed_gate,
        "gate_thresholds": {
            "audit_points_min": gate_audit,
            "tuning_crossfit_points_min": gate_tune_crossfit,
        },
        "scores": {
            "audit_points": final_eval["audit_points"],
            "full_points": final_eval["full_points"],
            "crossfit_tuning_points": selected_crossfit_tune,
            "insample_tuning_points": final_eval["tuning_points"],
            "full_loss_ratio": final_eval["full_loss_ratio"],
            "audit_loss_ratio": final_eval["audit_loss_ratio"],
            "tuning_loss_ratio": final_eval["tuning_loss_ratio"],
        },
        "by_condition_full": final_eval["by_condition"],
        "by_condition_audit": audit_score["by_condition"],
        "by_condition_tuning": tune_score["by_condition"],
        "calibration": {
            "penalty": best_penalty,
            "theta": list(theta_final),
            "crossfit": crossfit_results,
        },
        "pre_calibration_blend": {
            "full_points": blend_eval["full_points"],
            "audit_points": blend_eval["audit_points"],
            "tuning_points": blend_eval["tuning_points"],
            "by_condition": blend_eval["by_condition"],
        },
        "runtime_seconds": time.time() - t_start,
    }
    write_json(EXP_DIR / "results.json", results_record)
    print(f"Saved results and metadata to {EXP_DIR / 'results.json'}")

    return results_record

def reproduce_offline(data_dir=None, out_path=None):
    """Offline inference reproduction (< 5 seconds) using saved specialist & champion."""
    t0 = time.time()
    data_path = Path(data_dir) if data_dir else (LAB_DIR / "House_Edge_Lab_Starter_Pack/research_environment/public")
    out_file = Path(out_path) if out_path else (EXP_DIR / "reproduced_predictions.csv")

    test_q, test_h = load_inputs(data_path / "validation")
    base = FrequencyForecaster.load(LAB_DIR / "artifacts/models/hgb_ensemble_v3/base_forecaster.json")
    v3_models = joblib.load(LAB_DIR / "artifacts/models/hgb_ensemble_v3/models.joblib")
    specialist_models = joblib.load(EXP_DIR / "specialist_masked_models.joblib")

    results_data = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
    theta = np.asarray(results_data["calibration"]["theta"])

    p_v3 = predict_v3_champion(test_q, test_h, base, v3_models)
    p_spec = predict_lag_model(test_q, test_h, base, specialist_models)
    p_blend = mix(p_spec, p_v3, 0.25)

    all_ids = sorted(p_blend.keys())
    calibrated = transform_probs(np.asarray([p_blend[k] for k in all_ids]), theta)
    final_preds = {k: tuple(float(v) for v in p) for k, p in zip(all_ids, calibrated)}

    write_submission(out_file, final_preds)
    elapsed = time.time() - t0
    print(f"Offline reproduction finished in {elapsed:.2f}s (saved to {out_file})")
    return final_preds

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        reproduce_offline()
    else:
        run_experiment()
