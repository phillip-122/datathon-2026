"""Investigate Pairwise Bankroll Coupling and Peer Win Rate Dispersion across 7 Peers.

Objective:
Determine whether cross-machine dispersion, consensus, or pairwise correlation across the 7 peer machines
in an episode explicitly separates coordinated episode-level transitions (e.g. shared_change, floor-wide drift)
from idiosyncratic machine noise (stable, short_history).
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

from slot_forecasting.io import load_inputs, read_rows
from slot_forecasting.schema import PAYOUTS

def analyze_peer_coupling():
    data_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    val_dir = data_dir / "validation"
    v_queries, v_history = load_inputs(val_dir)
    cond_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in cond_rows}
    
    # Group history by (episode_id, machine_id)
    grouped = defaultdict(dict)
    for r in v_history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
        
    ep_machines = defaultdict(list)
    for q in v_queries:
        if q.machine_id not in ep_machines[q.episode_id]:
            ep_machines[q.episode_id].append(q.machine_id)
            
    print(f"Loaded {len(v_queries)} validation queries across {len(ep_machines)} episodes.")
    
    # Metrics to collect per query by condition
    results_by_cond = defaultdict(lambda: defaultdict(list))
    
    for q in v_queries:
        cond = groups[q.query_id]
        cutoff = q.cutoff
        ep = q.episode_id
        m = q.machine_id
        
        my_seq = {t: p for t, p in grouped[(ep, m)].items() if t < cutoff}
        peers = [{t: p for t, p in grouped[(ep, pm)].items() if t < cutoff} for pm in ep_machines[ep] if pm != m]
        
        # 1. Immediate previous tick t-1 and t-2 peer win counts
        p_t1 = [p.get(cutoff - 1, -1) for p in peers]
        n_p_t1_obs = sum(1 for p in p_t1 if p >= 0)
        p_t1_wins = sum(1 for p in p_t1 if p > 0)
        p_t1_frac = (p_t1_wins / n_p_t1_obs) if n_p_t1_obs > 0 else 0.4
        
        p_t2 = [p.get(cutoff - 2, -1) for p in peers]
        n_p_t2_obs = sum(1 for p in p_t2 if p >= 0)
        p_t2_wins = sum(1 for p in p_t2 if p > 0)
        p_t2_frac = (p_t2_wins / n_p_t2_obs) if n_p_t2_obs > 0 else 0.4
        
        # 2. Peer win rates across different windows: W = 8, 16, 24, 48
        for win in (8, 16, 24, 48):
            wrs = []
            means = []
            for p_seq in peers:
                p_win = [p_seq[t] for t in range(cutoff - win, cutoff) if t in p_seq]
                if p_win:
                    wr = sum(1 for p in p_win if p > 0) / len(p_win)
                    wrs.append(wr)
                    means.append(sum(p_win) / len(p_win))
            if len(wrs) >= 4:
                results_by_cond[cond][f"peer_wr_mean_{win}"].append(float(np.mean(wrs)))
                results_by_cond[cond][f"peer_wr_std_{win}"].append(float(np.std(wrs)))
                results_by_cond[cond][f"peer_wr_range_{win}"].append(float(np.ptp(wrs)))
                results_by_cond[cond][f"peer_wr_skew_{win}"].append(float(stats.skew(wrs)))
                
        # 3. Peer velocity / slope dispersion (W=8 vs W=24)
        slopes = []
        for p_seq in peers:
            p8 = [p_seq[t] for t in range(cutoff - 8, cutoff) if t in p_seq]
            p24 = [p_seq[t] for t in range(cutoff - 24, cutoff) if t in p_seq]
            if len(p8) >= 2 and len(p24) >= 6:
                wr8 = sum(1 for p in p8 if p > 0) / len(p8)
                wr24 = sum(1 for p in p24 if p > 0) / len(p24)
                slopes.append(wr8 - wr24)
        if len(slopes) >= 4:
            results_by_cond[cond]["peer_slope_mean"].append(float(np.mean(slopes)))
            results_by_cond[cond]["peer_slope_std"].append(float(np.std(slopes)))
            pos = sum(1 for s in slopes if s > 0.05)
            neg = sum(1 for s in slopes if s < -0.05)
            consensus = max(pos, neg) / len(slopes)
            results_by_cond[cond]["peer_slope_consensus"].append(consensus)
            
        # 4. Pairwise co-occurrence / cross-correlation across machines
        # Find ticks common to multiple machines
        all_peer_ticks = set()
        for p in peers: all_peer_ticks.update(p.keys())
        common_ticks = sorted(set(my_seq.keys()) & all_peer_ticks)
        
        # Cross correlation of query machine with average peer payout at each tick
        peer_tick_payouts = defaultdict(list)
        for p in peers:
            for t, val in p.items():
                peer_tick_payouts[t].append(val)
        
        aligned_my = []
        aligned_peer_avg = []
        for t in common_ticks:
            if t in peer_tick_payouts and len(peer_tick_payouts[t]) >= 3:
                aligned_my.append(my_seq[t])
                aligned_peer_avg.append(np.mean(peer_tick_payouts[t]))
                
        if len(aligned_my) >= 12:
            s_my = np.std(aligned_my)
            s_peer = np.std(aligned_peer_avg)
            if s_my > 1e-4 and s_peer > 1e-4:
                corr = float(np.corrcoef(aligned_my, aligned_peer_avg)[0, 1])
                results_by_cond[cond]["my_vs_peer_corr"].append(corr if not np.isnan(corr) else 0.0)
                
        results_by_cond[cond]["p_t1_frac"].append(p_t1_frac)
        results_by_cond[cond]["p_t2_frac"].append(p_t2_frac)
        
    print("\n" + "="*80)
    print("PEER COUPLING & DISPERSION SUMMARY BY CONDITION")
    print("="*80)
    
    conditions = ["stable", "short_history", "drift", "shared_change"]
    metrics = [
        "p_t1_frac", "peer_wr_mean_16", "peer_wr_std_16", "peer_wr_range_16",
        "peer_wr_mean_24", "peer_wr_std_24", "peer_wr_range_24",
        "peer_slope_mean", "peer_slope_std", "peer_slope_consensus",
        "my_vs_peer_corr"
    ]
    
    header = f"{'Metric':<24s} | " + " | ".join(f"{c:<12s}" for c in conditions) + " | ANOVA F-stat | p-value"
    print(header)
    print("-" * len(header))
    
    findings = {}
    for m in metrics:
        vals = [results_by_cond[c][m] for c in conditions]
        means = [f"{np.mean(v):.4f}" if len(v) > 0 else "N/A" for v in vals]
        if all(len(v) > 2 for v in vals):
            f_val, p_val = stats.f_oneway(*vals)
            f_str = f"{f_val:.2f}"
            p_str = f"{p_val:.2e}"
        else:
            f_str, p_str = "N/A", "N/A"
        row_str = f"{m:<24s} | " + " | ".join(f"{mv:<12s}" for mv in means) + f" | {f_str:<12s} | {p_str}"
        print(row_str)
        findings[m] = {
            "means": {c: float(np.mean(results_by_cond[c][m])) for c in conditions if len(results_by_cond[c][m]) > 0},
            "f_stat": float(f_val) if f_str != "N/A" else None,
            "p_val": float(p_val) if p_str != "N/A" else None
        }
        
    return findings

if __name__ == "__main__":
    analyze_peer_coupling()
