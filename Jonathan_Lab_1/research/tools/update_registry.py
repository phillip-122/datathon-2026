"""Unified Registry Updater maintaining Competition and Scientific Scoreboards."""

import json
from pathlib import Path

REGISTRY_PATH = Path("research/REGISTRY.md")

def update_registry(experiment_records, condition_records, scientific_records, current_champion="EXP-015"):
    lines = [
        "# House Edge Research Experiment Registry & Dual Scoreboards\n",
        f"**Active Submission Champion:** `{current_champion}`  ",
        "**Scoring Paradigm:** $$\\text{Technical Points} = \\frac{60}{1 + R}$$, where $$R = \\frac{\\text{MSE}}{\\text{Baseline MSE}}$$\n",
        "---\n",
        "## 1. Competition Scoreboard\n",
        "Tracks technical points, loss ratios, and generalization across strict episode splits.\n",
        "| Exp ID | Hypothesis / Candidate | Author | Tuning Pts | Audit Pts | Full Pts | Mean Loss Ratio ($R$) | Decision | Bootstrap Gain CI95 |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    
    for r in experiment_records:
        lines.append(f"| {r['id']} | {r['name']} | {r['author']} | {r['tuning']:.4f} | {r['audit']:.4f} | {r['full']:.4f} | {r['r']:.4f} | **{r['decision']}** | {r.get('ci', 'N/A')} |")
        
    lines.extend([
        "\n---",
        "## 2. Scientific Scoreboard: Discovered Physical & Stochastic Mechanisms\n",
        "Tracks the empirical discovery of underlying simulator structures and causal mechanisms.\n",
        "| Exp ID | Discovered Mechanism | Simulator / Physical Grounding | Target Weakness | Empirical Evidence |",
        "|---|---|---|---|---|"
    ])
    
    for s in scientific_records:
        lines.append(f"| {s['id']} | {s['mechanism']} | {s['grounding']} | {s['target']} | {s['evidence']} |")
        
    lines.extend([
        "\n---",
        "## 3. Condition Loss Ratio Breakdown\n",
        "Tracks error reduction ($R < 1.0$) across distinct environment conditions.\n",
        "| Exp ID | Stable ($R$) | Short History ($R$) | Drift ($R$) | Shared Change ($R$) | Notes |",
        "|---|---|---|---|---|---|"
    ])
    
    for c in condition_records:
        lines.append(f"| {c['id']} | {c['stable']:.4f} | {c['short_history']:.4f} | {c['drift']:.4f} | {c['shared_change']:.4f} | {c['notes']} |")
        
    REGISTRY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {REGISTRY_PATH}")

if __name__ == "__main__":
    # Test read or print
    print("Registry updater ready.")
