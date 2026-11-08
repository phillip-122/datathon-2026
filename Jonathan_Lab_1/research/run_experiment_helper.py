"""Standard experiment runner helper for isolated experiment runs."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from research.evaluator import ResearchEvaluator
from research.registry import ExperimentRegistry
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets

def run_standard_eval(exp_id, hypothesis, author, predictions, out_dir, notes=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ResearchEvaluator()
    res = evaluator.evaluate(predictions)
    
    # Save predictions
    from slot_forecasting.io import write_submission
    from slot_forecasting.schema import probabilities
    val_preds_prob = {qid: probabilities(p, qid) for qid, p in predictions.items()}
    write_submission(out_dir / "validation_predictions.csv", val_preds_prob)
    
    report = {
        "exp_id": exp_id,
        "hypothesis": hypothesis,
        "author": author,
        "notes": notes,
        "evaluation": res
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print(f"\n[{exp_id}] EVALUATION RESULTS:")
    print(f"  Tuning Points: {res['tuning_points']:.4f} / 60")
    print(f"  Audit Points:  {res['audit_points']:.4f} / 60")
    print(f"  Full Points:   {res['full_points']:.4f} / 60")
    print(f"  Mean Loss Ratio: {res['full_loss_ratio']:.4f}")
    for c, row in res["by_condition"].items():
        print(f"    {c:14s}: ratio={row['loss_ratio']:.4f} (error={row['error']:.4f}, ref={row['reference_error']:.4f})")
        
    return res
