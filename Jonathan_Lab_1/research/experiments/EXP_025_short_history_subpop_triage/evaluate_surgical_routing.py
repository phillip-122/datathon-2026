import sys
from pathlib import Path
import numpy as np

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import probabilities
from research.tools.controller import ResearchController

ctrl = ResearchController()
val_queries = ctrl.val_queries
val_history = ctrl.val_history
targets = ctrl.val_targets
groups = ctrl.groups
reference = ctrl.reference
tuning = ctrl.tuning_eps
audit = ctrl.audit_eps

grouped_val = {}
for r in val_history:
    grouped_val.setdefault((r.episode_id, r.machine_id), {})[r.tick] = r.payout

n_past_val = {q.query_id: len([t for t in grouped_val[(q.episode_id, q.machine_id)] if t < q.cutoff]) for q in val_queries}

v8_preds = read_submission("compiled_versions/v8_promoted_champion_routed/validation_predictions.csv", {t.query_id for t in targets})
eval_v8 = ctrl.evaluate_candidate(v8_preds)

v25_val_raw = read_submission("research/experiments/EXP_025_short_history_subpop_triage/validation_predictions.csv", {t.query_id for t in targets})

print("=== BASELINE V8_ROUTED ===")
print("Full:   {:.4f} | Loss Ratio: {:.4f}".format(eval_v8['full_points'], eval_v8['full_loss_ratio']))
print("Tuning: {:.4f} | Audit: {:.4f}".format(eval_v8['tuning_points'], eval_v8['audit_points']))
for c, r in eval_v8['by_condition'].items():
    print("  {:14s}: {:.4f}".format(c, r['loss_ratio']))

print("\n=== EVALUATING SHORT-HISTORY SURGICAL ROUTING ===")
for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    cand = {}
    for q in val_queries:
        qid = q.query_id
        if n_past_val[qid] <= 100:
            # short history
            p_v8 = np.asarray(v8_preds[qid])
            p_v25 = np.asarray(v25_val_raw[qid])
            p = (1.0 - alpha) * p_v8 + alpha * p_v25
            p = p / np.sum(p)
            cand[qid] = tuple(float(x) for x in p)
        else:
            cand[qid] = v8_preds[qid]
    res = ctrl.evaluate_candidate(cand, base_predictions=v8_preds, n_bootstrap=1000)
    print("alpha_v25={:4.2f} -> Full: {:7.4f} (Tune: {:7.4f}, Audit: {:7.4f}) | short_hist R: {:6.4f} | CI: [{:+6.4f}, {:+6.4f}]".format(
        alpha, res['full_points'], res['tuning_points'], res['audit_points'], res['by_condition']['short_history']['loss_ratio'],
        res['bootstrap_gain_vs_base_ci95'][0], res['bootstrap_gain_vs_base_ci95'][1]
    ))
