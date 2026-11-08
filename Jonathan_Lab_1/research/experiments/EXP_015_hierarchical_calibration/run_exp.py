"""EXP-015: Hierarchical Machine-Type Calibration with Shrinkage Regularization.

Hypothesis (Experiment D):
Rather than jumping to a complicated meta-model, test a constrained structure:
    T_effective = T_global + Delta_T_machine
with regularization shrinking machine-specific adjustments toward global temperature,
and conditional class offsets.

Methodology:
1. Load raw uncalibrated predictions from changing-lag base model.
2. Formulate hierarchical calibration with shrinkage regularization:
   Global parameters: theta_global (log-temp), beta_global (5 class offsets)
   Machine parameters: Delta_theta_m, Delta_beta_m for m in (steady, balanced, jackpot)
   Penalty = lambda_global * (theta_global^2 + ||beta_global||^2)
           + lambda_hier * sum_m (Delta_theta_m^2 + ||Delta_beta_m||^2)
3. 3-fold episode cross-fitting on tuning partition to select (lambda_global, lambda_hier).
4. Fit on all tuning episodes with chosen penalty.
5. Score on audit partition and full validation set using ResearchController.
6. Compute paired bootstrap 95% CI vs Champion v4 (6-param calibrator) and Champion v2.
"""

import sys, os
from pathlib import Path
_LAB = Path(__file__).resolve().parents[2]
_REPO = _LAB.parent
sys.path[:0] = [str(_LAB), str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO), str(_LAB / 'src')]

import hashlib
import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from research.tools.controller import ResearchController
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets, write_json, write_submission

MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}

def transform_hierarchical(P, m_indices, theta_g, beta_g, d_theta_m=None, d_beta_m=None):
    """Calibrate logits with global + machine adjustments."""
    N = len(P)
    logp = np.log(np.maximum(P, 1e-12))
    
    # Effective inverse temperature
    inv_t = np.exp(theta_g)
    if d_theta_m is not None:
        inv_t = np.exp(theta_g + d_theta_m[m_indices])
    else:
        inv_t = np.full(N, inv_t)
        
    logits = logp * inv_t[:, None] + beta_g[None, :]
    if d_beta_m is not None:
        logits += d_beta_m[m_indices, :]
        
    return softmax(logits, axis=1)

def fit_hierarchical(P, Q, m_indices, weights, l_global, l_hier, use_machine_bias=True):
    """Fit hierarchical calibrator parameters with L-BFGS-B."""
    # Parameter packing:
    # theta_g (1)
    # beta_g (5)
    # d_theta_m (3)
    # if use_machine_bias: d_beta_m (3 * 5 = 15)
    n_params = 1 + 5 + 3 + (15 if use_machine_bias else 0)
    
    def unpack(params):
        tg = params[0]
        bg = params[1:6]
        dt = params[6:9]
        db = params[9:].reshape((3, 5)) if use_machine_bias else None
        return tg, bg, dt, db
        
    def objective(params):
        tg, bg, dt, db = unpack(params)
        pred = transform_hierarchical(P, m_indices, tg, bg, dt, db)
        residual = 2.0 * weights[:, None] * (pred - Q)
        derivative = pred * (residual - (residual * pred).sum(axis=1, keepdims=True))
        
        # Loss
        loss = np.sum(weights[:, None] * (pred - Q) ** 2)
        loss += l_global * (tg ** 2 + np.sum(bg ** 2))
        loss += l_hier * (np.sum(dt ** 2) + (np.sum(db ** 2) if db is not None else 0.0))
        
        # Gradients
        logp = np.log(np.maximum(P, 1e-12))
        inv_t = np.exp(tg + dt[m_indices])
        
        # tg grad
        d_tg = np.sum(derivative * logp * inv_t[:, None]) + 2.0 * l_global * tg
        # bg grad
        d_bg = np.sum(derivative, axis=0) + 2.0 * l_global * bg
        
        # dt grad
        d_dt = np.zeros(3)
        for m in range(3):
            mask = (m_indices == m)
            if np.any(mask):
                d_dt[m] = np.sum(derivative[mask] * logp[mask] * inv_t[mask, None]) + 2.0 * l_hier * dt[m]
                
        # db grad
        if use_machine_bias:
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
    if use_machine_bias:
        bounds += [(-0.5, 0.5)] * 15
        
    x0 = np.zeros(n_params)
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        print("Warning: Optimization did not converge cleanly:", res.message)
    return res.x

def main():
    val_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/validation")
    targets = read_targets(val_dir / "targets.csv")
    group_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(val_dir / "reference.csv")
    queries, _ = load_inputs(val_dir)
    query_map = {q.query_id: q for q in queries}
    
    # Load uncalibrated raw predictions from changing lag base model
    raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
    if not raw_path.exists():
        raw_path = Path("compiled_versions/v2_changing_lags/validation_predictions.csv")
    raw = read_submission(raw_path, {t.query_id for t in targets})
    
    # Champion v4 predictions for bootstrap comparison
    v4_path = Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv")
    v4_preds = read_submission(v4_path, {t.query_id for t in targets})
    
    tuning, audit = episode_folds(targets, groups)
    def score(p, eps):
        return subset_score(p, targets, groups, reference, eps, 60.0)
        
    T = [t for t in targets if t.episode_id in tuning]
    P = np.asarray([raw[t.query_id] for t in T])
    Q = np.asarray([t.probabilities for t in T])
    M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    
    # Weight computation matching official contest objective
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
    
    # 3-Fold Episode splitting within tuning set
    folds = {}
    for condition in sorted(set(groups.values())):
        eps = sorted({t.episode_id for t in T if groups[t.query_id] == condition}, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        folds.update({e: i % 3 for i, e in enumerate(eps)})
        
    # Grid search over penalties
    grid = [
        (0.01, 0.1, False, "hier_temp_l0.1"),
        (0.01, 0.5, False, "hier_temp_l0.5"),
        (0.01, 1.0, False, "hier_temp_l1.0"),
        (0.01, 2.0, False, "hier_temp_l2.0"),
        (0.01, 5.0, False, "hier_temp_l5.0"),
        (0.01, 0.5, True,  "hier_full_l0.5"),
        (0.01, 1.0, True,  "hier_full_l1.0"),
        (0.01, 2.0, True,  "hier_full_l2.0"),
        (0.01, 5.0, True,  "hier_full_l5.0"),
        (0.01, 10.0, True, "hier_full_l10.0"),
    ]
    
    print("--- 3-Fold Episode Cross-Fitting on Tuning Partition ---")
    best_loss_ratio = score(raw, tuning)["mean_loss_ratio"]
    best_config = None
    cv_results = {}
    
    for l_g, l_h, use_bias, desc in grid:
        cv_values = np.zeros_like(P)
        for fold in range(3):
            train_mask = np.asarray([folds[t.episode_id] != fold for t in T])
            hold_mask = ~train_mask
            theta_fold = fit_hierarchical(P[train_mask], Q[train_mask], M[train_mask], compute_weights([t for t, k in zip(T, train_mask) if k]), l_g, l_h, use_bias)
            tg = theta_fold[0]; bg = theta_fold[1:6]; dt = theta_fold[6:9]
            db = theta_fold[9:].reshape((3, 5)) if use_bias else None
            cv_values[hold_mask] = transform_hierarchical(P[hold_mask], M[hold_mask], tg, bg, dt, db)
            
        fold_preds = {t.query_id: tuple(p) for t, p in zip(T, cv_values)}
        res = score(fold_preds, tuning)
        cv_results[desc] = {
            "technical_points": res["technical_points"],
            "loss_ratio": res["mean_loss_ratio"],
            "config": (l_g, l_h, use_bias)
        }
        print(f"[{desc:18s}] Tuning Pts: {res['technical_points']:.6f} | Loss Ratio: {res['mean_loss_ratio']:.6f}")
        if res["mean_loss_ratio"] < best_loss_ratio:
            best_loss_ratio = res["mean_loss_ratio"]
            best_config = (l_g, l_h, use_bias, desc)
            
    print(f"\nBest Cross-Fitted Configuration: {best_config}")
    
    # Fit final hierarchical calibrator on all tuning episodes
    if best_config is not None:
        l_g, l_h, use_bias, desc = best_config
        theta_final = fit_hierarchical(P, Q, M, w_T, l_g, l_h, use_bias)
        tg = theta_final[0]; bg = theta_final[1:6]; dt = theta_final[6:9]
        db = theta_final[9:].reshape((3, 5)) if use_bias else None
        
        print("\n--- Learned Hierarchical Calibration Parameters ---")
        print(f"Global log-temp: {tg:.4f} (Effective global T: {1.0 / np.exp(tg):.4f})")
        print(f"Global class biases: {[round(float(v), 4) for v in bg]}")
        for m_name, m_idx in MTYPE_MAP.items():
            eff_T = 1.0 / np.exp(tg + dt[m_idx])
            print(f"Machine {m_name:8s}: Delta_theta = {dt[m_idx]:+.4f} (Effective T = {eff_T:.4f})")
            if use_bias and db is not None:
                print(f"  Delta biases: {[round(float(v), 4) for v in db[m_idx]]}")
    else:
        print("Hierarchical shrinkage did not beat raw baseline.")
        return

    # Transform all validation queries
    all_targets = targets
    all_P = np.asarray([raw[t.query_id] for t in all_targets])
    all_M = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in all_targets])
    all_cal = transform_hierarchical(all_P, all_M, tg, bg, dt, db)
    hier_calibrated = {t.query_id: tuple(p) for t, p in zip(all_targets, all_cal)}
    
    # Evaluate with ResearchController
    ctrl = ResearchController()
    eval_rep = ctrl.evaluate_candidate(hier_calibrated, base_predictions=v4_preds, n_bootstrap=2000)
    
    print("\n=======================================================")
    print("EXP-015: HIERARCHICAL CALIBRATION AUDIT RESULTS")
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
        
    # Save results
    out_dir = Path("research/experiments/EXP_015_hierarchical_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_submission(out_dir / "validation_predictions.csv", hier_calibrated)
    write_json(out_dir / "report.json", {
        "best_config": best_config,
        "cv_results": cv_results,
        "evaluation": eval_rep,
        "parameters": {
            "theta_global": float(tg),
            "beta_global": [float(v) for v in bg],
            "delta_theta_machine": {m: float(dt[i]) for m, i in MTYPE_MAP.items()},
            "delta_beta_machine": {m: [float(v) for v in db[i]] for m, i in MTYPE_MAP.items()} if db is not None else None
        }
    })
    print(f"\nArtifacts written to: {out_dir}")

if __name__ == "__main__":
    main()
