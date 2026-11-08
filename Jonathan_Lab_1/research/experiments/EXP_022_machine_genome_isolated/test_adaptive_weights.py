import numpy as np
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

print("="*60)
print("TESTING MACHINE-TYPE ADAPTIVE BLENDING")
print("="*60)

# Grid search machine-type blend weights w_m in [0.10, 0.40]
best_tune_score = -1
best_w_mtype = None
best_res_mtype = None

# Grid over steady, balanced, jackpot
for w_s in [0.15, 0.20, 0.25, 0.30]:
    for w_b in [0.15, 0.20, 0.25, 0.30]:
        for w_j in [0.15, 0.20, 0.25, 0.30]:
            w_arr = np.asarray([w_s, w_b, w_j])
            w_per_q = w_arr[M_val]
            p_blend = w_per_q[:, None] * p_cfg3_raw + (1.0 - w_per_q[:, None]) * p_v6
            p_blend /= np.sum(p_blend, axis=1, keepdims=True)
            
            theta = fit_hierarchical_calibrator(p_blend[tune_mask], Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
            p_cal = transform_hierarchical(p_blend, M_val, theta)
            preds_dict = {t.query_id: tuple(p_cal[i]) for i, t in enumerate(targets)}
            
            score_tune = subset_score(preds_dict, targets, groups, reference, tuning_eps, 60.0)
            tune_pts = score_tune["technical_points"]
            
            if tune_pts > best_tune_score:
                best_tune_score = tune_pts
                best_w_mtype = (w_s, w_b, w_j)
                score_audit = subset_score(preds_dict, targets, groups, reference, audit_eps, 60.0)
                score_full = contest_score(preds_dict, targets, groups, reference, max_points=60.0)
                best_res_mtype = {
                    "tune": tune_pts,
                    "audit": score_audit["technical_points"],
                    "full": score_full["technical_points"],
                    "R": score_full["mean_loss_ratio"],
                    "by_cond": {c: round(v["loss_ratio"], 4) for c, v in score_full["by_condition"].items()}
                }

print("Best Machine-Type Adaptive Weights on Tuning Set: steady=%.2f, balanced=%.2f, jackpot=%.2f" % best_w_mtype)
print("Performance: Tune=%.4f | Audit=%.4f | Full=%.4f | R=%.4f" % (
    best_res_mtype["tune"], best_res_mtype["audit"], best_res_mtype["full"], best_res_mtype["R"]))
print("Condition breakdown:", best_res_mtype["by_cond"])

print("\n" + "="*60)
print("TESTING CONDITION-ADAPTIVE BLENDING (Oracle on Tuning)")
print("="*60)
# Look at condition breakdown with uniform w to see which condition wants what weight
for cond in ["stable", "short_history", "drift", "shared_change"]:
    print(f"\nCondition: {cond}")
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        p_blend = w * p_cfg3_raw + (1.0 - w) * p_v6
        p_blend /= np.sum(p_blend, axis=1, keepdims=True)
        theta = fit_hierarchical_calibrator(p_blend[tune_mask], Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
        p_cal = transform_hierarchical(p_blend, M_val, theta)
        preds_dict = {t.query_id: tuple(p_cal[i]) for i, t in enumerate(targets)}
        
        # tuning error for this condition
        t_cond = [t for t in targets if t.episode_id in tuning_eps and groups[t.query_id] == cond]
        errs = [np.sum((np.asarray(preds_dict[t.query_id]) - t.probabilities)**2) for t in t_cond]
        ref_errs = [np.sum((np.asarray(reference[t.query_id]) - t.probabilities)**2) for t in t_cond]
        mean_err = np.mean(errs)
        mean_ref = np.mean(ref_errs)
        ratio = mean_err / mean_ref
        print("  w=%.2f -> Tuning Loss Ratio for %s: %.4f" % (w, cond, ratio))
