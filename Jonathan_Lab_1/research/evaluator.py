"""Standardized, cutoff-safe evaluation harness for House Edge research.

Enforces:
1. Exact official contest_score scoring function.
2. Deterministic 50/50 tuning vs audit episode-grouped split.
3. Complete protection of test data: test data is never loaded or touched during research.
4. Detailed per-condition error ratios (stable, short_history, drift, shared_change).
"""

from collections import defaultdict
from pathlib import Path
import numpy as np

from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets
from slot_forecasting.schema import PAYOUTS, ValidationError, probabilities

DATA_PATH = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
if not DATA_PATH.exists():
    DATA_PATH = Path("../public")

class ResearchEvaluator:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_PATH
        self.val_dir = self.data_dir / "validation"
        
        # Load validation ground truth and reference once
        self.val_queries, self.val_history = load_inputs(self.val_dir)
        self.val_targets = read_targets(self.val_dir / "targets.csv")
        group_rows = read_rows(self.val_dir / "conditions.csv", ("query_id", "condition"))
        self.groups = {r["query_id"]: r["condition"] for r in group_rows}
        self.reference = read_submission(self.val_dir / "reference.csv")
        
        # Compute deterministic episode folds
        self.tuning_eps, self.audit_eps = episode_folds(self.val_targets, self.groups)
        
    def evaluate(self, predictions):
        """Evaluate a dictionary of predictions: query_id -> (p0, p1, p2, p5, p20).
        
        Returns dict with:
          - tuning_points: float (out of 60)
          - audit_points: float (out of 60)
          - full_points: float (out of 60)
          - tuning_loss_ratio: float
          - audit_loss_ratio: float
          - full_loss_ratio: float
          - by_condition: dict of condition -> loss_ratio
          - passed_checks: bool
        """
        # 1. Validation checks
        expected_ids = {t.query_id for t in self.val_targets}
        pred_ids = set(predictions.keys())
        if pred_ids != expected_ids:
            missing = expected_ids - pred_ids
            extra = pred_ids - expected_ids
            raise ValidationError(f"Prediction query_id mismatch. Missing: {len(missing)}, Extra: {len(extra)}")
            
        normalized_preds = {}
        for qid, p in predictions.items():
            if len(p) != 5:
                raise ValidationError(f"Query {qid} has {len(p)} probabilities, expected 5.")
            tot = sum(p)
            if abs(tot - 1.0) > 1e-4:
                raise ValidationError(f"Query {qid} probabilities sum to {tot}, expected 1.0.")
            normalized_preds[qid] = tuple(float(v / tot) for v in p)
            
        # 2. Score on tuning split
        tune_res = subset_score(normalized_preds, self.val_targets, self.groups, self.reference, self.tuning_eps, 60.0)
        
        # 3. Score on audit split
        audit_res = subset_score(normalized_preds, self.val_targets, self.groups, self.reference, self.audit_eps, 60.0)
        
        # 4. Score on full validation set
        full_res = contest_score(normalized_preds, self.val_targets, self.groups, self.reference, max_points=60.0)
        
        by_cond = {}
        for cond, row in full_res["by_condition"].items():
            by_cond[cond] = {
                "loss_ratio": row["loss_ratio"],
                "error": row["error"],
                "reference_error": row["reference_error"]
            }
            
        return {
            "tuning_points": tune_res["technical_points"],
            "tuning_loss_ratio": tune_res["mean_loss_ratio"],
            "audit_points": audit_res["technical_points"],
            "audit_loss_ratio": audit_res["mean_loss_ratio"],
            "full_points": full_res["technical_points"],
            "full_loss_ratio": full_res["mean_loss_ratio"],
            "by_condition": by_cond
        }

if __name__ == "__main__":
    # Test evaluator against reference predictions (should be exactly 30.0 points)
    evaluator = ResearchEvaluator()
    ref_preds = evaluator.reference
    res = evaluator.evaluate(ref_preds)
    print("Reference check:")
    print(f"Tuning points: {res['tuning_points']:.4f} (expected ~30.0)")
    print(f"Audit points:  {res['audit_points']:.4f} (expected ~30.0)")
    print(f"Full points:   {res['full_points']:.4f} (expected 30.0000)")
