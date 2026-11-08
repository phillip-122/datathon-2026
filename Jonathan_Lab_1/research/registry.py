"""Experiment Registry and Champion Management for House Edge Research.

Tracks all experiments, their hypotheses, scores across tuning/audit/full splits,
condition breakdowns, decisions, and artifact paths.
"""

import json
from datetime import datetime
from pathlib import Path

REGISTRY_JSON_PATH = Path("research/registry.json")
REGISTRY_MD_PATH = Path("research/REGISTRY.md")

class ExperimentRegistry:
    def __init__(self):
        self.experiments = []
        self.champion = None
        self.load()

    def load(self):
        if REGISTRY_JSON_PATH.exists():
            data = json.loads(REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
            self.experiments = data.get("experiments", [])
            self.champion = data.get("champion", None)
        else:
            # Seed with baseline and current champion EXP-006
            self.experiments = [
                {
                    "id": "BASE-001",
                    "hypothesis": "Official pooled starter reference baseline",
                    "author": "organizers",
                    "tuning_points": 30.0000,
                    "audit_points": 30.0000,
                    "full_points": 30.0000,
                    "mean_loss_ratio": 1.000000,
                    "by_condition": {"stable": 1.0, "short_history": 1.0, "drift": 1.0, "shared_change": 1.0},
                    "decision": "BASELINE",
                    "notes": "Starter baseline reference"
                },
                {
                    "id": "EXP-006",
                    "hypothesis": "HistGradientBoosting with 211 features, multi-cutoff step 6, leaves=60, l2=2.0",
                    "author": "initial_iteration",
                    "tuning_points": 35.1863,
                    "audit_points": 34.9486,
                    "full_points": 35.0743,
                    "mean_loss_ratio": 0.710655,
                    "by_condition": {
                        "stable": 0.668450,
                        "short_history": 0.782616,
                        "drift": 0.758752,
                        "shared_change": 0.632801
                    },
                    "decision": "CHAMPION",
                    "notes": "Frozen champion v1 (runs/hgb_run/submission.csv, 35.0743 pts)"
                }
            ]
            self.champion = "EXP-006"
            self.save()

    def save(self):
        REGISTRY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(),
            "champion": self.champion,
            "experiments": self.experiments
        }
        REGISTRY_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._write_markdown()

    def register_experiment(self, exp_id, hypothesis, author, eval_result, decision, notes=""):
        # Check if already exists
        for e in self.experiments:
            if e["id"] == exp_id:
                e.update({
                    "hypothesis": hypothesis,
                    "author": author,
                    "tuning_points": eval_result["tuning_points"],
                    "audit_points": eval_result["audit_points"],
                    "full_points": eval_result["full_points"],
                    "mean_loss_ratio": eval_result["full_loss_ratio"],
                    "by_condition": {k: v["loss_ratio"] for k, v in eval_result["by_condition"].items()},
                    "decision": decision,
                    "notes": notes,
                    "timestamp": datetime.now().isoformat()
                })
                if decision == "CHAMPION":
                    self.champion = exp_id
                self.save()
                return

        entry = {
            "id": exp_id,
            "hypothesis": hypothesis,
            "author": author,
            "tuning_points": eval_result["tuning_points"],
            "audit_points": eval_result["audit_points"],
            "full_points": eval_result["full_points"],
            "mean_loss_ratio": eval_result["full_loss_ratio"],
            "by_condition": {k: v["loss_ratio"] for k, v in eval_result["by_condition"].items()},
            "decision": decision,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        self.experiments.append(entry)
        if decision == "CHAMPION":
            self.champion = exp_id
        self.save()

    def get_champion_scores(self):
        for e in self.experiments:
            if e["id"] == self.champion:
                return e
        return None

    def _write_markdown(self):
        lines = [
            "# House Edge Research Experiment Registry",
            "",
            f"**Current Champion:** `{self.champion}`",
            f"**Last Updated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "| Exp ID | Hypothesis | Author | Tuning Pts | Audit Pts | Full Pts | Mean Loss Ratio | Decision | Notes |",
            "|---|---|---|---|---|---|---|---|---|"
        ]
        for e in self.experiments:
            t_pts = f"{e['tuning_points']:.4f}" if e.get('tuning_points') is not None else "-"
            a_pts = f"{e['audit_points']:.4f}" if e.get('audit_points') is not None else "-"
            f_pts = f"{e['full_points']:.4f}" if e.get('full_points') is not None else "-"
            ratio = f"{e['mean_loss_ratio']:.4f}" if e.get('mean_loss_ratio') is not None else "-"
            lines.append(f"| {e['id']} | {e['hypothesis']} | {e['author']} | {t_pts} | {a_pts} | {f_pts} | {ratio} | **{e['decision']}** | {e.get('notes', '')} |")
        
        lines.append("")
        lines.append("## Condition Breakdown by Experiment")
        lines.append("")
        lines.append("| Exp ID | Stable Ratio | Short History Ratio | Drift Ratio | Shared Change Ratio |")
        lines.append("|---|---|---|---|---|")
        for e in self.experiments:
            bc = e.get("by_condition", {})
            st = f"{bc.get('stable', 0):.4f}" if "stable" in bc else "-"
            sh = f"{bc.get('short_history', 0):.4f}" if "short_history" in bc else "-"
            dr = f"{bc.get('drift', 0):.4f}" if "drift" in bc else "-"
            sc = f"{bc.get('shared_change', 0):.4f}" if "shared_change" in bc else "-"
            lines.append(f"| {e['id']} | {st} | {sh} | {dr} | {sc} |")
            
        REGISTRY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    reg = ExperimentRegistry()
    print("Registry initialized. Champion:", reg.champion)
