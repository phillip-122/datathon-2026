import pandas as pd
import numpy as np

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

for mt in ['steady', 'balanced', 'jackpot']:
    sub = df[df['machine_type'] == mt].copy()
    r_low, r_high = REGIME_RATIOS[mt]
    sub['regime'] = np.where(np.abs(sub['r_1x'] - r_low) < np.abs(sub['r_1x'] - r_high), 0, 1)
    
    print(f"=== Machine Type: {mt} ===")
    print("Mean p_0 by regime:")
    print(sub.groupby('regime')['p_0'].mean())
    print("Mean w_high by regime:")
    print(sub.groupby('regime')['w_high'].mean())
    print("Mean p_1 by regime:")
    print(sub.groupby('regime')['p_1'].mean())
    print("Total win probability (1 - p_0) by regime:")
    print(sub.groupby('regime')[['p_0']].apply(lambda g: 1 - g.mean()))
    print()
