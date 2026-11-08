import pandas as pd
import numpy as np

targets = pd.read_csv("adaptation-participants/public/validation/targets.csv")
queries = pd.read_csv("adaptation-participants/public/validation/queries.csv")
conds = pd.read_csv("adaptation-participants/public/validation/conditions.csv")

df = targets.merge(queries, on="query_id").merge(conds, on="query_id")

print("=== TARGET STATISTICS BY CONDITION ===")
for c, group in df.groupby("condition"):
    print(f"\nCondition: {c} (N={len(group)})")
    print(group[["p_0", "p_1", "p_2", "p_5", "p_20"]].describe().T[["mean", "std", "min", "50%", "max"]])
    
print("\n=== TARGET CORRELATIONS WITHIN EPISODE ===")
for c, group in df.groupby("condition"):
    # For each episode, what is the std of p_0 across the 8 machines?
    p0_std_within_ep = group.groupby("episode_id_x")["p_0"].std().mean()
    p0_mean_within_ep = group.groupby("episode_id_x")["p_0"].mean().std()
    print(f"Condition {c:15s}: Mean within-ep std(p_0) = {p0_std_within_ep:.4f}, Across-ep std of ep_mean(p_0) = {p0_mean_within_ep:.4f}")
