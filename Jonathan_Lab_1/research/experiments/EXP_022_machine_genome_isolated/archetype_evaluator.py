"""EXP-022 Research Requirement 2 & 4: Archetype Classification Reliability & Threshold Evaluation.

Measures classification reliability and correlation against full-episode ground truth
at N in [10, 20, 30, 50, 100] observations across training history.
Compares fixed +/- 0.05 vs learned distribution quantiles and confidence-gated thresholds.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def compute_momentum_stats(seq, max_n=None):
    ticks = sorted(seq.keys())
    if max_n is not None:
        ticks = ticks[:max_n]
    win_after_win = []
    win_after_loss = []
    for i in range(1, len(ticks)):
        t_prev, t_curr = ticks[i-1], ticks[i]
        if t_curr == t_prev + 1:
            p_prev = 1 if seq[t_prev] > 0 else 0
            p_curr = 1 if seq[t_curr] > 0 else 0
            if p_prev == 1:
                win_after_win.append(p_curr)
            else:
                win_after_loss.append(p_curr)
    nw = len(win_after_win)
    nl = len(win_after_loss)
    if nw < 2 or nl < 2:
        return np.nan, np.nan, nw + nl, 0.5, 0.5
    # Laplace-smoothed rates with prior 0.4
    p_w_w = (sum(win_after_win) + 2.0 * 0.4) / (nw + 2.0)
    p_w_l = (sum(win_after_loss) + 2.0 * 0.4) / (nl + 2.0)
    mom = p_w_w - p_w_l
    # Standard error of the difference
    se = np.sqrt(max(1e-6, p_w_w * (1 - p_w_w) / nw + p_w_l * (1 - p_w_l) / nl))
    return mom, se, nw + nl, p_w_w, p_w_l

def evaluate_reliability(train_hist_path, out_path=None):
    df_hist = pd.read_csv(train_hist_path)
    grouped = defaultdict(dict)
    for _, r in df_hist.iterrows():
        grouped[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])

    # Compute full episode ground truth
    full_stats = {}
    for (ep, m), seq in grouped.items():
        if len(seq) >= 150:
            mom, se, tot, pww, pwl = compute_momentum_stats(seq)
            if not np.isnan(mom):
                full_stats[(ep, m)] = {"mom": mom, "se": se, "tot": tot, "pww": pww, "pwl": pwl}

    full_moms = [v["mom"] for v in full_stats.values()]
    q33 = float(np.percentile(full_moms, 33.33))
    q67 = float(np.percentile(full_moms, 66.67))

    results = {
        "dataset_machine_count": len(full_stats),
        "momentum_distribution": {
            "mean": float(np.mean(full_moms)),
            "std": float(np.std(full_moms)),
            "min": float(np.min(full_moms)),
            "q25": float(np.percentile(full_moms, 25)),
            "median": float(np.median(full_moms)),
            "q75": float(np.percentile(full_moms, 75)),
            "max": float(np.max(full_moms)),
            "learned_tertiles": {"q33": q33, "q67": q67}
        },
        "reliability_by_sample_size": {}
    }

    sample_sizes = [10, 20, 30, 50, 100]
    for n in sample_sizes:
        rows = []
        for (ep, m), full in full_stats.items():
            seq = grouped[(ep, m)]
            mom_n, se_n, tot_n, _, _ = compute_momentum_stats(seq, max_n=n)
            if np.isnan(mom_n):
                continue

            full_m = full["mom"]
            # Ground truth state using fixed +/- 0.05
            st_full_fixed = -1 if full_m < -0.05 else (1 if full_m > 0.05 else 0)
            # Ground truth state using learned tertiles
            st_full_learned = -1 if full_m < q33 else (1 if full_m > q67 else 0)

            # 1. Fixed +/- 0.05
            st_fixed = -1 if mom_n < -0.05 else (1 if mom_n > 0.05 else 0)

            # 2. Learned tertiles
            st_learned = -1 if mom_n < q33 else (1 if mom_n > q67 else 0)

            # 3. Confidence-gated (|mom| > 1.5 * se, otherwise 0 = uncertain)
            if abs(mom_n) > 1.5 * se_n:
                st_conf = 1 if mom_n > 0 else -1
            else:
                st_conf = 0

            # 4. Bayesian Shrunk momentum
            mom_shrunk = mom_n * (tot_n / (tot_n + 15.0))
            st_shrunk = -1 if mom_shrunk < -0.05 else (1 if mom_shrunk > 0.05 else 0)

            rows.append({
                "full_m": full_m,
                "mom_n": mom_n,
                "st_full_fixed": st_full_fixed,
                "st_full_learned": st_full_learned,
                "st_fixed": st_fixed,
                "st_learned": st_learned,
                "st_conf": st_conf,
                "st_shrunk": st_shrunk
            })

        df_n = pd.DataFrame(rows)
        corr = float(np.corrcoef(df_n["full_m"], df_n["mom_n"])[0, 1])

        # Metrics for fixed
        acc_fixed = float(np.mean(df_n["st_fixed"] == df_n["st_full_fixed"]))
        nz_fixed = df_n[df_n["st_fixed"] != 0]
        prec_fixed = float(np.mean(nz_fixed["st_fixed"] == nz_fixed["st_full_fixed"])) if len(nz_fixed) > 0 else 0.0

        # Metrics for learned
        acc_learned = float(np.mean(df_n["st_learned"] == df_n["st_full_learned"]))
        nz_learned = df_n[df_n["st_learned"] != 0]
        prec_learned = float(np.mean(nz_learned["st_learned"] == nz_learned["st_full_learned"])) if len(nz_learned) > 0 else 0.0

        # Metrics for confidence-gated
        acc_conf = float(np.mean(df_n["st_conf"] == df_n["st_full_fixed"]))
        nz_conf = df_n[df_n["st_conf"] != 0]
        prec_conf = float(np.mean(nz_conf["st_conf"] == nz_conf["st_full_fixed"])) if len(nz_conf) > 0 else 0.0
        cov_conf = float(len(nz_conf) / len(df_n))

        # Metrics for shrunk
        acc_shrunk = float(np.mean(df_n["st_shrunk"] == df_n["st_full_fixed"]))
        nz_shrunk = df_n[df_n["st_shrunk"] != 0]
        prec_shrunk = float(np.mean(nz_shrunk["st_shrunk"] == nz_shrunk["st_full_fixed"])) if len(nz_shrunk) > 0 else 0.0

        results["reliability_by_sample_size"][f"N_{n}"] = {
            "valid_machines": len(df_n),
            "correlation_with_full": corr,
            "fixed_pm05": {
                "overall_accuracy": acc_fixed,
                "precision_when_classified": prec_fixed,
                "coverage": float(len(nz_fixed) / len(df_n))
            },
            "learned_tertiles": {
                "overall_accuracy": acc_learned,
                "precision_when_classified": prec_learned,
                "coverage": float(len(nz_learned) / len(df_n))
            },
            "confidence_gated_uncertain_explicit": {
                "overall_accuracy": acc_conf,
                "precision_when_classified": prec_conf,
                "coverage": cov_conf,
                "uncertain_fraction": 1.0 - cov_conf
            },
            "bayesian_shrunk": {
                "overall_accuracy": acc_shrunk,
                "precision_when_classified": prec_shrunk,
                "coverage": float(len(nz_shrunk) / len(df_n))
            }
        }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Reliability analysis saved to: {out_path}")

    return results

if __name__ == "__main__":
    train_path = Path("House_Edge_Lab_Starter_Pack/research_environment/public/train/history.csv")
    if not train_path.exists():
        train_path = Path("../public/train/history.csv")
    out_file = Path("research/experiments/EXP_022_machine_genome_isolated/archetype_reliability_report.json")
    res = evaluate_reliability(train_path, out_file)
    print("\nSummary of Reliability vs Sample Size:")
    for k, v in res["reliability_by_sample_size"].items():
        print(f"  {k}: Correlation={v['correlation_with_full']:.4f} | "
              f"Fixed Precision={v['fixed_pm05']['precision_when_classified']*100:.1f}% | "
              f"Confidence Precision={v['confidence_gated_uncertain_explicit']['precision_when_classified']*100:.1f}% "
              f"(Uncertain={v['confidence_gated_uncertain_explicit']['uncertain_fraction']*100:.1f}%)")
