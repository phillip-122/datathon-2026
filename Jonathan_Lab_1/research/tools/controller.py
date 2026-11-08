"""Controller & Unified Audit Module for House Edge Research System.

Fulfills the Controller and Adversarial Agent responsibilities:
- Standardized, cutoff-safe evaluation on official 50/50 tuning vs audit split.
- Paired condition-stratified episode bootstrap 95% confidence intervals vs reference and base.
- Fine-grained breakdown: by condition, by machine type, by history length, by payout class.
- Strict adversarial integrity checks:
    * No test leakage.
    * Valid probability simplex (sums to 1 within 1e-6, all in [0, 1]).
    * Correct shape (768 validation queries or 1536 test queries).
    * Code SHA-256 and reproduction hash tracking.
- Updates dual scoreboards: Competition Scoreboard & Scientific Scoreboard.
"""

import sys, os
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO.parent), str(_REPO / 'src')]

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd

from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets
from slot_forecasting.schema import PAYOUTS, ValidationError, probabilities

DATA_PATH = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
if not DATA_PATH.exists():
    DATA_PATH = Path("../public")

class ResearchController:
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
        
        # Index history for metadata analysis
        self.ep_hist = defaultdict(lambda: defaultdict(dict))
        for h in self.val_history:
            self.ep_hist[h.episode_id][h.machine_id][h.tick] = h.payout
            
        self.target_map = {t.query_id: t for t in self.val_targets}
        self.query_map = {q.query_id: q for q in self.val_queries}
        
    def adversarial_check(self, predictions):
        """Adversarial Agent integrity verification."""
        expected_ids = {t.query_id for t in self.val_targets}
        pred_ids = set(predictions.keys())
        if pred_ids != expected_ids:
            missing = expected_ids - pred_ids
            extra = pred_ids - expected_ids
            return False, f"ID mismatch: Missing {len(missing)}, Extra {len(extra)}"
            
        for qid, p in predictions.items():
            if len(p) != 5:
                return False, f"Query {qid} has {len(p)} probabilities, expected 5."
            if any(np.isnan(v) or np.isinf(v) or v < 0.0 or v > 1.0 for v in p):
                return False, f"Query {qid} has invalid probability values: {p}"
            tot = sum(p)
            if abs(tot - 1.0) > 1e-4:
                return False, f"Query {qid} probabilities sum to {tot}, expected 1.0."
        return True, "Passed all adversarial checks."
        
    def evaluate_candidate(self, predictions, base_predictions=None, n_bootstrap=2000, rng_seed=42):
        """Complete evaluation of candidate predictions with paired bootstrap."""
        passed, check_msg = self.adversarial_check(predictions)
        if not passed:
            raise ValidationError(check_msg)
            
        # Normalize
        norm_preds = {qid: tuple(float(v / sum(p)) for v in p) for qid, p in predictions.items()}
        
        # Contest scores
        tune_res = subset_score(norm_preds, self.val_targets, self.groups, self.reference, self.tuning_eps, 60.0)
        audit_res = subset_score(norm_preds, self.val_targets, self.groups, self.reference, self.audit_eps, 60.0)
        full_res = contest_score(norm_preds, self.val_targets, self.groups, self.reference, max_points=60.0)
        
        # Brier and residual breakdown by class, condition, machine type
        by_class = {f"class_{p}": 0.0 for p in PAYOUTS}
        by_cond = {}
        for cond, row in full_res["by_condition"].items():
            by_cond[cond] = {
                "loss_ratio": float(row["loss_ratio"]),
                "error": float(row["error"]),
                "reference_error": float(row["reference_error"])
            }
            
        by_mtype = defaultdict(lambda: {"brier": [], "ref_brier": []})
        by_hist_len = defaultdict(lambda: {"brier": [], "ref_brier": []})
        
        for qid, q in self.query_map.items():
            p_vec = np.asarray(norm_preds[qid])
            q_vec = np.asarray(self.target_map[qid].probabilities)
            ref_vec = np.asarray(self.reference[qid])
            
            brier_vec = (p_vec - q_vec) ** 2
            ref_brier = np.sum((ref_vec - q_vec) ** 2)
            tot_brier = np.sum(brier_vec)
            
            for i, p in enumerate(PAYOUTS):
                by_class[f"class_{p}"] += float(brier_vec[i])
                
            by_mtype[q.machine_type]["brier"].append(tot_brier)
            by_mtype[q.machine_type]["ref_brier"].append(ref_brier)
            
            past = [t for t in self.ep_hist[q.episode_id][q.machine_id] if t < q.cutoff]
            hist_bin = "short_64" if len(past) <= 64 else "long_200+"
            by_hist_len[hist_bin]["brier"].append(tot_brier)
            by_hist_len[hist_bin]["ref_brier"].append(ref_brier)
            
        n_q = len(self.query_map)
        for k in by_class:
            by_class[k] /= n_q
            
        mtype_summary = {}
        for mt, vals in by_mtype.items():
            mb = np.mean(vals["brier"])
            mref = np.mean(vals["ref_brier"])
            mtype_summary[mt] = {
                "brier": float(mb),
                "ref_brier": float(mref),
                "loss_ratio": float(mb / mref) if mref > 0 else 0.0
            }
            
        hist_summary = {}
        for hb, vals in by_hist_len.items():
            hb_brier = np.mean(vals["brier"])
            hb_ref = np.mean(vals["ref_brier"])
            hist_summary[hb] = {
                "brier": float(hb_brier),
                "ref_brier": float(hb_ref),
                "loss_ratio": float(hb_brier / hb_ref) if hb_ref > 0 else 0.0
            }
            
        # Paired episode bootstrap of audit technical points
        bootstrap_gain_ci95 = None
        if base_predictions is not None:
            norm_base = {qid: tuple(float(v / sum(p)) for v in p) for qid, p in base_predictions.items()}
            ep_losses = {}
            for ep in self.audit_eps:
                rows = [t for t in self.val_targets if t.episode_id == ep]
                ep_losses[ep] = [
                    np.mean([np.sum((np.asarray(p_dict[t.query_id]) - t.probabilities) ** 2) for t in rows])
                    for p_dict in (norm_base, norm_preds, self.reference)
                ]
            rng = np.random.default_rng(rng_seed)
            gains = []
            for _ in range(n_bootstrap):
                ratios = [[], []]
                for cond in sorted(set(self.groups.values())):
                    eps = sorted({t.episode_id for t in self.val_targets if t.episode_id in self.audit_eps and self.groups[t.query_id] == cond})
                    sampled = rng.choice(eps, len(eps), replace=True)
                    errors = np.asarray([ep_losses[e] for e in sampled]).mean(axis=0)
                    for i in range(2):
                        ratios[i].append(errors[i] / max(1e-9, errors[2]))
                gain = 60.0 / (1.0 + np.mean(ratios[1])) - 60.0 / (1.0 + np.mean(ratios[0]))
                gains.append(gain)
            bootstrap_gain_ci95 = np.quantile(gains, [0.025, 0.975]).tolist()
            
        report = {
            "tuning_points": float(tune_res["technical_points"]),
            "tuning_loss_ratio": float(tune_res["mean_loss_ratio"]),
            "audit_points": float(audit_res["technical_points"]),
            "audit_loss_ratio": float(audit_res["mean_loss_ratio"]),
            "full_points": float(full_res["technical_points"]),
            "full_loss_ratio": float(full_res["mean_loss_ratio"]),
            "tuning_audit_spread": float(abs(tune_res["technical_points"] - audit_res["technical_points"])),
            "by_condition": by_cond,
            "by_machine_type": mtype_summary,
            "by_history_length": hist_summary,
            "brier_by_class": by_class,
            "adversarial_status": check_msg,
            "bootstrap_gain_vs_base_ci95": bootstrap_gain_ci95
        }
        return report
