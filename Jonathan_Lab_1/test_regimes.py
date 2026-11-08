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
    sub = df[df['machine_type'] == mt]
    r_low, r_high = REGIME_RATIOS[mt]
    dist_low = np.abs(sub['r_1x'] - r_low)
    dist_high = np.abs(sub['r_1x'] - r_high)
    regime = np.where(dist_low < dist_high, 0, 1)
    sub = sub.copy()
    sub['regime'] = regime
    sub['err'] = np.minimum(dist_low, dist_high)
    print(f"Machine Type: {mt:8s} | Count: {len(sub)}")
    print(f"  Regime 0 count: {(regime == 0).sum()} ({ (regime==0).mean()*100:.1f}%), Mean r_1x: {sub[sub['regime']==0]['r_1x'].mean():.6f} (target: {r_low:.6f})")
    print(f"  Regime 1 count: {(regime == 1).sum()} ({ (regime==1).mean()*100:.1f}%), Mean r_1x: {sub[sub['regime']==1]['r_1x'].mean():.6f} (target: {r_high:.6f})")
    print(f"  Max error from exact ratio: {sub['err'].max():.6e}")
    print(f"  Regime breakdown by condition:")
    print(sub.groupby(['condition', 'regime']).size().unstack(fill_value=0))
    print()
