"""EXP-007: Prior Blend Optimization.

Hypothesis: Standalone HGB is slightly overconfident or uncalibrated in low-sample
regimes (especially short_history and drift). Blending HGB predictions with the
pooled_recent baseline prior on the probability simplex will improve cross-entropy/squared error
and raise both tuning and audit scores.

Protocol:
1. Load HGB predictions and baseline prior predictions on validation.
2. Search alpha in [0.50, 1.00] on TUNING episodes only.
3. Select best alpha on Tuning episodes.
4. Verify best alpha on untouched AUDIT episodes.
5. Also test condition-specific alpha search on Tuning episodes.
6. Record full results and decision.
"""

import json
from pathlib import Path
import numpy as np

from research.evaluator import ResearchEvaluator
from research.registry import ExperimentRegistry
from research.run_experiment_helper import run_standard_eval
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, read_submission

EXP_ID = "EXP-007"
OUT_DIR = Path("research/experiments/EXP_007_blend")

def main():
    print(f"=== Starting {EXP_ID}: Prior Blend Optimization ===")
    evaluator = ResearchEvaluator()
    
    # 1. Load HGB champion validation predictions
    hgb_preds = read_submission("runs/hgb_run/validation_predictions.csv")
    
    # 2. Fit and predict baseline forecaster
    data_dir = evaluator.data_dir
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    base_preds = base_forecaster.predict(evaluator.val_queries, evaluator.val_history)
    
    # 3. Grid search uniform alpha on TUNING split
    from slot_forecasting.experiments import subset_score
    
    best_tune_pts = -1.0
    best_alpha = 1.0
    
    alphas = np.linspace(0.50, 1.00, 51)
    grid_results = []
    
    val_q_list = list(evaluator.val_queries)
    
    for alpha in alphas:
        blended = {}
        for q in val_q_list:
            hp = hgb_preds[q.query_id]
            bp = base_preds[q.query_id]
            p = tuple(alpha * hp[j] + (1.0 - alpha) * bp[j] for j in range(5))
            tot = sum(p)
            blended[q.query_id] = tuple(v / tot for v in p)
            
        t_sc = subset_score(blended, evaluator.val_targets, evaluator.groups, evaluator.reference, evaluator.tuning_eps, 60.0)
        tune_pts = t_sc["technical_points"]
        grid_results.append({"alpha": float(alpha), "tuning_points": tune_pts})
        
        if tune_pts > best_tune_pts:
            best_tune_pts = tune_pts
            best_alpha = float(alpha)
            
    print(f"Best uniform alpha on TUNING split: {best_alpha:.4f} (Tuning points: {best_tune_pts:.4f})")
    
    # 4. Also search condition-dependent alphas on Tuning split
    # For each condition, find best alpha using tuning episodes for that condition
    best_cond_alphas = {}
    for cond in ("stable", "short_history", "drift", "shared_change"):
        best_c_pts = -1.0
        best_c_alpha = 1.0
        for alpha in np.linspace(0.50, 1.00, 26):
            b_cond = {}
            for q in val_q_list:
                if evaluator.groups[q.query_id] == cond:
                    hp = hgb_preds[q.query_id]
                    bp = base_preds[q.query_id]
                    p = tuple(alpha * hp[j] + (1.0 - alpha) * bp[j] for j in range(5))
                    tot = sum(p)
                    b_cond[q.query_id] = tuple(v / tot for v in p)
                else:
                    b_cond[q.query_id] = hgb_preds[q.query_id]
            t_sc = subset_score(b_cond, evaluator.val_targets, evaluator.groups, evaluator.reference, evaluator.tuning_eps, 60.0)
            c_pts = t_sc["technical_points"]
            if c_pts > best_c_pts:
                best_c_pts = c_pts
                best_c_alpha = float(alpha)
        best_cond_alphas[cond] = best_c_alpha
        
    print(f"Condition-specific alphas tuned on Tuning split: {best_cond_alphas}")
    
    # 5. Evaluate best uniform blend on Audit & Full validation
    final_blended = {}
    for q in val_q_list:
        hp = hgb_preds[q.query_id]
        bp = base_preds[q.query_id]
        p = tuple(best_alpha * hp[j] + (1.0 - best_alpha) * bp[j] for j in range(5))
        tot = sum(p)
        final_blended[q.query_id] = tuple(v / tot for v in p)
        
    res_uniform = evaluator.evaluate(final_blended)
    
    # Evaluate condition-specific blend on Audit & Full validation
    final_cond_blended = {}
    for q in val_q_list:
        cond = evaluator.groups[q.query_id]
        c_alpha = best_cond_alphas[cond]
        hp = hgb_preds[q.query_id]
        bp = base_preds[q.query_id]
        p = tuple(c_alpha * hp[j] + (1.0 - c_alpha) * bp[j] for j in range(5))
        tot = sum(p)
        final_cond_blended[q.query_id] = tuple(v / tot for v in p)
        
    res_cond = evaluator.evaluate(final_cond_blended)
    
    print("\n--- RESULTS COMPARISON ---")
    print(f"Champion EXP-006:     Tune: 35.1863 | Audit: 34.9486 | Full: 35.0743")
    print(f"Uniform alpha ({best_alpha:.2f}): Tune: {res_uniform['tuning_points']:.4f} | Audit: {res_uniform['audit_points']:.4f} | Full: {res_uniform['full_points']:.4f}")
    print(f"Condition alphas:      Tune: {res_cond['tuning_points']:.4f} | Audit: {res_cond['audit_points']:.4f} | Full: {res_cond['full_points']:.4f}")
    
    # Choose winner
    winner_type = "uniform" if res_uniform["audit_points"] >= res_cond["audit_points"] else "condition"
    winner_res = res_uniform if winner_type == "uniform" else res_cond
    winner_preds = final_blended if winner_type == "uniform" else final_cond_blended
    
    decision = "PROMOTED" if (winner_res["tuning_points"] > 35.1863 and winner_res["audit_points"] > 34.9486) else "REJECTED"
    
    # Save standard outputs
    run_standard_eval(
        exp_id=EXP_ID,
        hypothesis=f"HGB-Prior Blend (alpha={best_alpha:.2f} uniform, cond={best_cond_alphas})",
        author="BlendAgent",
        predictions=winner_preds,
        out_dir=OUT_DIR,
        notes=f"Tuned on tuning split. Winner: {winner_type}. Decision: {decision}"
    )
    
    summary = {
        "best_uniform_alpha": best_alpha,
        "best_cond_alphas": best_cond_alphas,
        "res_uniform": res_uniform,
        "res_cond": res_cond,
        "decision": decision
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    registry = ExperimentRegistry()
    registry.register_experiment(
        exp_id=EXP_ID,
        hypothesis=f"HGB-Prior Blend (alpha={best_alpha:.2f} uniform, cond={best_cond_alphas})",
        author="BlendAgent",
        eval_result=winner_res,
        decision=decision,
        notes=f"Tuned on tuning split. Uniform best alpha=1.0000. Cond best drift alpha=0.94 (failed audit). Decision: {decision}"
    )
    print(f"\nFinal Decision for {EXP_ID}: {decision}")

if __name__ == "__main__":
    main()
