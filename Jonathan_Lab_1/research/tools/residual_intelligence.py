"""Experiment E: Residual Intelligence Diagnostic System for House Edge.

Analyzes prediction residuals on permitted tuning and audit partitions:
1. Payout class specific Brier loss (0, 1, 2, 5, 20).
2. Overprediction of rare payouts (e.g., 20x) and underprediction of zero payouts (0x).
3. Error concentration by condition (stable, short_history, drift, shared_change).
4. Error concentration by machine type (steady, balanced, jackpot).
5. Error concentration by history length / observation count (<20, 20-40, 40-60, 64).
6. Error correlation with drought lengths (streak of 0s) and recent-vs-historical divergence.
"""

import sys, os
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO.parent), str(_REPO / 'src')]

from collections import defaultdict
import json
import math
import numpy as np
import pandas as pd

from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets
from slot_forecasting.schema import PAYOUTS

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def run_diagnostics(val_dir, predictions_path, out_report_path=None):
    val_dir = Path(val_dir)
    queries, history = load_inputs(val_dir)
    targets = read_targets(val_dir / "targets.csv")
    group_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    conditions = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(val_dir / "reference.csv")
    preds = read_submission(predictions_path)
    
    tuning_eps, audit_eps = episode_folds(targets, conditions)
    
    # Map history by episode and machine
    ep_hist = defaultdict(lambda: defaultdict(dict))
    for h in history:
        ep_hist[h.episode_id][h.machine_id][h.tick] = h.payout

    target_map = {t.query_id: t for t in targets}
    query_map = {q.query_id: q for q in queries}
    
    records = []
    for qid, q in query_map.items():
        if qid not in preds or qid not in target_map:
            continue
        t = target_map[qid]
        p_vec = np.asarray(preds[qid], dtype=np.float64)
        q_vec = np.asarray(t.probabilities, dtype=np.float64)
        ref_vec = np.asarray(reference[qid], dtype=np.float64)
        
        # Brier error
        brier = np.sum((p_vec - q_vec) ** 2)
        ref_brier = np.sum((ref_vec - q_vec) ** 2)
        
        # Class-specific errors: (p_k - q_k)^2 and signed residual (p_k - q_k)
        class_brier = (p_vec - q_vec) ** 2
        signed_residual = p_vec - q_vec
        
        cond = conditions[qid]
        m_type = q.machine_type
        sig = q.signal
        ep_id = q.episode_id
        partition = "tuning" if ep_id in tuning_eps else ("audit" if ep_id in audit_eps else "other")
        
        m_seq = ep_hist[ep_id][q.machine_id]
        past = {tick: p for tick, p in m_seq.items() if tick < q.cutoff}
        n_obs = len(past)
        
        # Streak of 0s right before cutoff
        streak_0 = 0
        cur_t = q.cutoff - 1
        while cur_t in past and past[cur_t] == 0:
            streak_0 += 1
            cur_t -= 1
            
        # Recent vs historical win rate divergence
        recent_ticks = [past[tick] for tick in range(q.cutoff - 16, q.cutoff) if tick in past]
        older_ticks = [past[tick] for tick in range(0, q.cutoff - 16) if tick in past]
        
        rec_win_rate = (sum(1 for p in recent_ticks if p > 0) / len(recent_ticks)) if recent_ticks else 0.5
        old_win_rate = (sum(1 for p in older_ticks if p > 0) / len(older_ticks)) if older_ticks else 0.5
        win_divergence = rec_win_rate - old_win_rate
        
        rec = {
            "query_id": qid,
            "episode_id": ep_id,
            "partition": partition,
            "condition": cond,
            "machine_type": m_type,
            "signal": sig,
            "n_obs": n_obs,
            "streak_0": streak_0,
            "rec_win_rate": rec_win_rate,
            "old_win_rate": old_win_rate,
            "win_divergence": win_divergence,
            "brier": brier,
            "ref_brier": ref_brier,
            "loss_ratio": brier / max(1e-9, ref_brier),
            # Per class signed residuals
            "resid_0": signed_residual[0],
            "resid_1": signed_residual[1],
            "resid_2": signed_residual[2],
            "resid_5": signed_residual[3],
            "resid_20": signed_residual[4],
            # Per class brier
            "brier_0": class_brier[0],
            "brier_1": class_brier[1],
            "brier_2": class_brier[2],
            "brier_5": class_brier[3],
            "brier_20": class_brier[4],
            # Actual true probabilities
            "q_0": q_vec[0],
            "q_1": q_vec[1],
            "q_2": q_vec[2],
            "q_5": q_vec[3],
            "q_20": q_vec[4],
            # Predicted probabilities
            "p_0": p_vec[0],
            "p_1": p_vec[1],
            "p_2": p_vec[2],
            "p_5": p_vec[3],
            "p_20": p_vec[4],
        }
        records.append(rec)
        
    df = pd.DataFrame(records)
    
    # Analyze tuning partition
    tune_df = df[df["partition"] == "tuning"].copy()
    
    analysis = {}
    
    # 1. Overall Brier error by payout class
    analysis["brier_by_class_tuning"] = {
        f"class_{p}": float(tune_df[f"brier_{p}"].mean()) for p in PAYOUTS
    }
    analysis["mean_residual_by_class_tuning"] = {
        f"class_{p}": float(tune_df[f"resid_{p}"].mean()) for p in PAYOUTS
    }
    
    # 2. Performance by Condition
    cond_summary = {}
    for c, g in tune_df.groupby("condition"):
        cond_summary[c] = {
            "mean_brier": float(g["brier"].mean()),
            "mean_ref_brier": float(g["ref_brier"].mean()),
            "loss_ratio": float(g["brier"].mean() / g["ref_brier"].mean()),
            "technical_points": float(60.0 / (1.0 + (g["brier"].mean() / g["ref_brier"].mean()))),
            "resid_0": float(g["resid_0"].mean()),
            "resid_20": float(g["resid_20"].mean()),
        }
    analysis["by_condition_tuning"] = cond_summary
    
    # 3. Performance by Machine Type
    mtype_summary = {}
    for mt, g in tune_df.groupby("machine_type"):
        mtype_summary[mt] = {
            "n_queries": int(len(g)),
            "mean_brier": float(g["brier"].mean()),
            "mean_ref_brier": float(g["ref_brier"].mean()),
            "loss_ratio": float(g["brier"].mean() / g["ref_brier"].mean()),
            "resid_0": float(g["resid_0"].mean()),
            "resid_20": float(g["resid_20"].mean()),
            "brier_20": float(g["brier_20"].mean()),
        }
    analysis["by_machine_type_tuning"] = mtype_summary
    
    # 4. History Length bins
    tune_df["obs_bin"] = pd.cut(tune_df["n_obs"], bins=[-1, 25, 45, 63, 65], labels=["<25", "25-45", "46-63", "64_full"])
    obs_summary = {}
    for b, g in tune_df.groupby("obs_bin", observed=False):
        obs_summary[str(b)] = {
            "n_queries": int(len(g)),
            "mean_brier": float(g["brier"].mean()),
            "mean_ref_brier": float(g["ref_brier"].mean()),
            "loss_ratio": float(g["brier"].mean() / g["ref_brier"].mean()) if g["ref_brier"].mean() > 0 else 0.0,
        }
    analysis["by_history_length_tuning"] = obs_summary
    
    # 5. Drought analysis (streak of 0s)
    tune_df["drought_bin"] = pd.cut(tune_df["streak_0"], bins=[-1, 0, 3, 7, 50], labels=["0", "1-3", "4-7", "8+"])
    drought_summary = {}
    for db, g in tune_df.groupby("drought_bin", observed=False):
        drought_summary[str(db)] = {
            "n_queries": int(len(g)),
            "mean_brier": float(g["brier"].mean()),
            "resid_0": float(g["resid_0"].mean()),
            "resid_20": float(g["resid_20"].mean()),
        }
    analysis["by_drought_length_tuning"] = drought_summary
    
    # 6. Win rate divergence analysis (contradiction between recent and historical)
    tune_df["abs_divergence"] = tune_df["win_divergence"].abs()
    tune_df["div_bin"] = pd.cut(tune_df["abs_divergence"], bins=[-0.01, 0.1, 0.25, 1.0], labels=["low (<0.1)", "med (0.1-0.25)", "high (>0.25)"])
    div_summary = {}
    for dv, g in tune_df.groupby("div_bin", observed=False):
        div_summary[str(dv)] = {
            "n_queries": int(len(g)),
            "mean_brier": float(g["brier"].mean()),
            "loss_ratio": float(g["brier"].mean() / g["ref_brier"].mean()) if g["ref_brier"].mean() > 0 else 0.0,
        }
    analysis["by_divergence_tuning"] = div_summary
    
    # 7. Worst 10 Overpredictions of 20x and Underpredictions of 0x
    worst_over_20 = tune_df.sort_values(by="resid_20", ascending=False).head(10)[
        ["query_id", "condition", "machine_type", "n_obs", "streak_0", "p_20", "q_20", "resid_20", "brier"]
    ].to_dict(orient="records")
    worst_under_0 = tune_df.sort_values(by="resid_0", ascending=True).head(10)[
        ["query_id", "condition", "machine_type", "n_obs", "streak_0", "p_0", "q_0", "resid_0", "brier"]
    ].to_dict(orient="records")
    
    analysis["worst_overpredict_20x"] = worst_over_20
    analysis["worst_underpredict_0x"] = worst_under_0
    
    if out_report_path:
        with open(out_report_path, "w") as f:
            json.dump(analysis, f, indent=2)
            
    return analysis, df

if __name__ == "__main__":
    import sys
    pred_csv = sys.argv[1] if len(sys.argv) > 1 else "compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv"
    val_dir = sys.argv[2] if len(sys.argv) > 2 else "House_Edge_Lab_Starter_Pack/research_environment/public/validation"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "research/experiments/residual_intelligence_v4.json"
    
    print(f"Running Residual Intelligence Diagnostics on: {pred_csv}")
    analysis, _ = run_diagnostics(val_dir, pred_csv, out_json)
    
    print("\n--- Payout Class Brier Error & Bias (Tuning Partition) ---")
    for p in PAYOUTS:
        b = analysis["brier_by_class_tuning"][f"class_{p}"]
        bias = analysis["mean_residual_by_class_tuning"][f"class_{p}"]
        print(f"Payout {p:2d}: Brier={b:.6f}, Bias (pred - true)={bias:+.6f}")
        
    print("\n--- By Machine Type (Tuning Partition) ---")
    for mt, res in analysis["by_machine_type_tuning"].items():
        print(f"{mt:8s} (N={res['n_queries']:2d}): Brier={res['mean_brier']:.6f}, Loss Ratio={res['loss_ratio']:.4f}, Resid 0={res['resid_0']:+.4f}, Resid 20={res['resid_20']:+.4f}")
        
    print("\n--- By History Length (Tuning Partition) ---")
    for b, res in analysis["by_history_length_tuning"].items():
        print(f"Obs {b:8s} (N={res['n_queries']:2d}): Brier={res['mean_brier']:.6f}, Loss Ratio={res['loss_ratio']:.4f}")

    print("\n--- By Drought Length / Streak of 0s (Tuning Partition) ---")
    for db, res in analysis["by_drought_length_tuning"].items():
        print(f"Streak {db:8s} (N={res['n_queries']:2d}): Brier={res['mean_brier']:.6f}, Resid 0={res['resid_0']:+.4f}, Resid 20={res['resid_20']:+.4f}")

    print("\n--- By Divergence (Recent vs Older Win Rate) ---")
    for dv, res in analysis["by_divergence_tuning"].items():
        print(f"Divergence {dv:15s} (N={res['n_queries']:2d}): Brier={res['mean_brier']:.6f}, Loss Ratio={res['loss_ratio']:.4f}")
