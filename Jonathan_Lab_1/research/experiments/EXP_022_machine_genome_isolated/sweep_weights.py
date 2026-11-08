import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import softmax

from research.tools.controller import ResearchController
from slot_forecasting.experiments import subset_score
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_submission
from feature_factory import MTYPES, MTYPE_MAP

ctrl = ResearchController()
val_queries = ctrl.val_queries
targets = ctrl.val_targets
groups = ctrl.groups
reference = ctrl.reference
tuning_eps = ctrl.tuning_eps
audit_eps = ctrl.audit_eps

v6_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv")
p_v6 = np.asarray([v6_preds[t.query_id] for t in targets])
p_cfg3_raw = np.load("/tmp/cfg3_val_raw_probs.npy")

query_map = {q.query_id: q for q in val_queries}
target_qids = [t.query_id for t in targets]
M_val = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in targets])
Q_val = np.asarray([t.probabilities for t in targets])

tune_mask = np.asarray([t.episode_id in tuning_eps for t in targets])
audit_mask = np.asarray([t.episode_id in audit_eps for t in targets])

T = [t for t in targets if t.episode_id in tuning_eps]
M_tuning = M_val[tune_mask]
Q_tuning = Q_val[tune_mask]

def compute_tuning_weights(rows):
    w = []
    for t in rows:
        cond = groups[t.query_id]
        ct = [r for r in rows if groups[r.query_id] == cond]
        eps = {r.episode_id for r in ct}
        cnts = {e: sum(r.episode_id == e for r in ct) for e in eps}
        base_err = sum(np.sum((np.asarray(reference[r.query_id]) - r.probabilities)**2) / cnts[r.episode_id] for r in ct) / len(eps)
        w.append(1.0 / (4.0 * len(eps) * cnts[t.episode_id] * max(1e-9, base_err)))
    return np.asarray(w)

w_tuning = compute_tuning_weights(T)

def fit_hierarchical_calibrator(P, Q, M, weights, lambda_base=0.01, lambda_hier=0.05):
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
        loss = np.sum(weights[:, None] * (diff ** 2))
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
    n_classes = P.shape[1]
    n_machines = len(MTYPES)
    tg = theta[0]
    bg = theta[1:1+n_classes]
    dt = theta[1+n_classes:1+n_classes+n_machines]
    db = theta[1+n_classes+n_machines:].reshape((n_machines, n_classes))
    logP = np.log(np.maximum(P, 1e-12))
    inv_T = np.exp(tg + dt[M])
    logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
    return softmax(logits, axis=1)

print("Baseline v6 scores:")
res_v6 = ctrl.evaluate_candidate(v6_preds)
print("Full: %.4f | Tuning: %.4f | Audit: %.4f | Loss Ratio: %.4f" % (
    res_v6["full_points"], res_v6["tuning_points"], res_v6["audit_points"], res_v6["full_loss_ratio"]))

print("\n--- UNIFORM BLEND WEIGHT SWEEP w in [0.05, 0.45] ---")
weights_to_test = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50]
results = []
for w in weights_to_test:
    p_blend = w * p_cfg3_raw + (1.0 - w) * p_v6
    p_blend /= np.sum(p_blend, axis=1, keepdims=True)
    
    # Fit calibrator on tuning split
    theta = fit_hierarchical_calibrator(p_blend[tune_mask], Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
    p_cal = transform_hierarchical(p_blend, M_val, theta)
    preds_dict = {t.query_id: tuple(p_cal[i]) for i, t in enumerate(targets)}
    
    score_full = contest_score(preds_dict, targets, groups, reference, max_points=60.0)
    score_tune = subset_score(preds_dict, targets, groups, reference, tuning_eps, 60.0)
    score_audit = subset_score(preds_dict, targets, groups, reference, audit_eps, 60.0)
    
    cond_str = ", ".join(["%s: %.4f" % (c, score_full["by_condition"][c]["loss_ratio"]) for c in sorted(score_full["by_condition"])])
    print("w=%.2f: Full=%.4f | Tune=%.4f | Audit=%.4f | R=%.4f | Cond: {%s}" % (
        w, score_full["technical_points"], score_tune["technical_points"], score_audit["technical_points"], score_full["mean_loss_ratio"], cond_str))

print("\n--- TEST POST-CALIBRATION BLENDING ---")
# Fit calibrator on raw cfg3 alone on tuning set, then blend with calibrated v6
theta_cfg3 = fit_hierarchical_calibrator(p_cfg3_raw[tune_mask], Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
p_cfg3_cal_standalone = transform_hierarchical(p_cfg3_raw, M_val, theta_cfg3)

for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    p_blend_post = w * p_cfg3_cal_standalone + (1.0 - w) * p_v6
    p_blend_post /= np.sum(p_blend_post, axis=1, keepdims=True)
    preds_dict = {t.query_id: tuple(p_blend_post[i]) for i, t in enumerate(targets)}
    score_full = contest_score(preds_dict, targets, groups, reference, max_points=60.0)
    score_tune = subset_score(preds_dict, targets, groups, reference, tuning_eps, 60.0)
    score_audit = subset_score(preds_dict, targets, groups, reference, audit_eps, 60.0)
    print("post-blend w=%.2f: Full=%.4f | Tune=%.4f | Audit=%.4f | R=%.4f" % (
        w, score_full["technical_points"], score_tune["technical_points"], score_audit["technical_points"], score_full["mean_loss_ratio"]))
