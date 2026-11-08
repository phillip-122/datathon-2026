import pandas as pd
import numpy as np
from scipy import stats

targets = pd.read_csv("adaptation-participants/public/validation/targets.csv")
queries = pd.read_csv("adaptation-participants/public/validation/queries.csv")
conds = pd.read_csv("adaptation-participants/public/validation/conditions.csv")
df = targets.merge(queries, on="query_id").merge(conds, on="query_id")

df["w_high"] = df["p_2"] + df["p_5"] + df["p_20"]
df["r_1x"] = df["p_1"] / df["w_high"]

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

regimes = []
for idx, row in df.iterrows():
    mt = row['machine_type']
    r_low, r_high = REGIME_RATIOS[mt]
    reg = 0 if abs(row['r_1x'] - r_low) < abs(row['r_1x'] - r_high) else 1
    regimes.append(reg)
df['regime'] = regimes

print("=== DISTRIBUTION OF REGIME 1 MACHINES PER EPISODE (OUT OF 8) ===")
for cond, grp in df.groupby("condition"):
    ep_counts = grp.groupby("episode_id_x")["regime"].sum()
    n_eps = len(ep_counts)
    mean_k = ep_counts.mean()
    var_k = ep_counts.var()
    
    # Expected variance under independent Binomial(8, p)
    p = grp['regime'].mean()
    expected_var = 8 * p * (1 - p)
    
    # Binomial distribution counts
    expected_dist = [n_eps * stats.binom.pmf(k, 8, p) for k in range(9)]
    actual_dist = [ (ep_counts == k).sum() for k in range(9)]
    
    print(f"\nCondition: {cond.upper()} (Episodes: {n_eps}, Mean p(Regime 1) = {p:.3f})")
    print(f"  Mean Regime 1 count per ep: {mean_k:.2f} / 8")
    print(f"  Observed Variance: {var_k:.3f} vs Binomial Independent Variance: {expected_var:.3f}")
    print(f"  Variance Ratio (Dispersion): {var_k / expected_var:.2f}x")
    print(f"  Counts per episode (k=0..8):")
    for k in range(9):
        print(f"    k={k}: Observed={actual_dist[k]:2d} | Expected={expected_dist[k]:.1f}")
