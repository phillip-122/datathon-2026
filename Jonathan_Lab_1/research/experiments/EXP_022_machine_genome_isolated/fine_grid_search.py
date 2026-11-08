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
targets = ctrl.val_targets
v6_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv")
p_v6 = np.asarray([v6_preds[t.query_id] for t in targets])
p_cfg3_raw = np.load("/tmp/cfg3_val_raw_probs.npy")

query_map = {q.query_id: q for q in ctrl.val_queries}
M_val = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in targets])
Q_val = np.asarray([t.probabilities for t in targets])

tune_mask = np.asarray([t.episode_id in ctrl.tuning_eps for t in targets])
T = [t for t in targets if t.episode_id in ctrl.tuning_eps]
M_tuning = M_val[tune_mask]
Q_tuning = Q_val[tune_mask]

def compute_tuning_weights(rows):
    w = []
    for t in rows:
        cond = ctrl.groups[t.query_id]
        ct = [r for r in rows if ctrl.groups[r.query_id] == cond]
        eps = {r.episode_id for r in ct}
        cnts = {e: sum(r.episode_id == e for r in ct) for e in eps}
        base_err = sum(np.sum((np.asarray(ctrl.reference[r.query_id]) - r.probabilities)**2) / cnts[r.episode_id] for r in ct) / len(eps)
        w.append(1.0 / (4.0 * len(eps) * cnts[t.episode_id] * max(1e-9, base_err)))
    return np.asarray(w)

w_tuning = compute_tuning_weights(T)

def fit_hierarchical_calibrator(P, Q, M, weights, lambda_base=0.01, lambda_hier=0.05):
    n_classes = Q.shape[1]
    n_machines = len(MTYPES)
    n_params = 1 + n_classes + n_machines + n_machines * n_classes
    
    def unpack(theta):
        return theta[0], theta[1:1+n_classes], theta[1+n_classes:1+n_classes+n_machines], theta[1+n_classes+n_machines:].reshape((n_machines, n_classes))

    def objective(theta):
        tg, bg, dt, db = unpack(theta)
        logP = np.log(np.maximum(P, 1e-12))
        inv_T = np.exp(tg + dt[M])
        logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
        diff = softmax(logits, axis=1) - Q
        loss = np.sum(weights[:, None] * (diff ** 2))
        reg = lambda_base * (tg**2 + np.sum(bg**2)) + lambda_hier * (np.sum(dt**2) + np.sum(db**2))
        return loss + reg

    bounds = [(-2.0, 2.0)] + [(-5.0, 5.0)] * n_classes + [(-1.5, 1.5)] * n_machines + [(-3.0, 3.0)] * (n_machines * n_classes)
    res = minimize(objective, np.zeros(n_params), method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
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
print("FINE GRID SEARCH: MACHINE-TYPE WEIGHTS & REGULARIZATION")
print("="*60)

candidates = []

for lam in [0.02, 0.05, 0.08]:
    for ws in [0.10, 0.12, 0.15, 0.18, 0.20]:
        for wb in [0.22, 0.25, 0.28, 0.30]:
            for wj in [0.25, 0.28, 0.30, 0.35]:
                w_arr = np.asarray([ws, wb, wj])
                w_per_q = w_arr[M_val]
                p_blend = w_per_q[:, None] * p_cfg3_raw + (1.0 - w_per_q[:, None]) * p_v6
                p_blend /= np.sum(p_blend, axis=1, keepdims=True)
                
                theta = fit_hierarchical_calibrator(p_blend[tune_mask], Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=lam)
                p_cal = transform_hierarchical(p_blend, M_val, theta)
                preds_dict = {t.query_id: tuple(p_cal[i]) for i, t in enumerate(targets)}
                
                s_tune = subset_score(preds_dict, targets, ctrl.groups, ctrl.reference, ctrl.tuning_eps, 60.0)
                s_audit = subset_score(preds_dict, targets, ctrl.groups, ctrl.reference, ctrl.audit_eps, 60.0)
                s_full = contest_score(preds_dict, targets, ctrl.groups, ctrl.reference, max_points=60.0)
                
                candidates.append({
                    "lam": lam,
                    "ws": ws, "wb": wb, "wj": wj,
                    "tune": s_tune["technical_points"],
                    "audit": s_audit["technical_points"],
                    "full": s_full["technical_points"],
                    "spread": abs(s_tune["technical_points"] - s_audit["technical_points"]),
                    "R": s_full["mean_loss_ratio"],
                    "by_cond": {c: round(v["loss_ratio"], 4) for c, v in s_full["by_condition"].items()},
                    "theta": theta
                })

# Sort by tuning points
candidates.sort(key=lambda x: x["tune"], reverse=True)

print("Top 10 candidates by Tuning Points:")
for i, c in enumerate(candidates[:10]):
    print("[%2d] lam=%.2f | ws=%.2f wb=%.2f wj=%.2f | Tune=%.4f | Audit=%.4f | Full=%.4f | Spread=%.4f | R=%.4f | %s" % (
        i+1, c["lam"], c["ws"], c["wb"], c["wj"], c["tune"], c["audit"], c["full"], c["spread"], c["R"], c["by_cond"]))

# Sort by Audit points
candidates.sort(key=lambda x: x["audit"], reverse=True)
print("\nTop 5 candidates by Audit Points:")
for i, c in enumerate(candidates[:5]):
    print("[%2d] lam=%.2f | ws=%.2f wb=%.2f wj=%.2f | Tune=%.4f | Audit=%.4f | Full=%.4f | Spread=%.4f | R=%.4f | %s" % (
        i+1, c["lam"], c["ws"], c["wb"], c["wj"], c["tune"], c["audit"], c["full"], c["spread"], c["R"], c["by_cond"]))
