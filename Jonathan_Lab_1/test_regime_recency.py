import pandas as pd
import numpy as np

targets = pd.read_csv("adaptation-participants/public/validation/targets.csv")
queries = pd.read_csv("adaptation-participants/public/validation/queries.csv")
conds = pd.read_csv("adaptation-participants/public/validation/conditions.csv")
hist = pd.read_csv("adaptation-participants/public/validation/history.csv")

df = targets.merge(queries, on="query_id").merge(conds, on="query_id")
df["w_high"] = df["p_2"] + df["p_5"] + df["p_20"]
df["r_1x"] = df["p_1"] / df["w_high"]

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

true_regimes = []
for idx, row in df.iterrows():
    mt = row['machine_type']
    r_low, r_high = REGIME_RATIOS[mt]
    true_regimes.append(0 if abs(row['r_1x'] - r_low) < abs(row['r_1x'] - r_high) else 1)
df['true_regime'] = true_regimes

# Let's test windows before cutoff: W in [16, 32, 48, 64, 96, 128, all]
# For each machine:
windows = [16, 32, 48, 64, 96, 128, 256]
results = {w: [] for w in windows}

for idx, row in df.iterrows():
    ep = row['episode_id_x']
    m = row['machine_id']
    cutoff = row['cutoff']
    mt = row['machine_type']
    r_low, r_high = REGIME_RATIOS[mt]
    log_mid = 0.5 * (np.log(r_low) + np.log(r_high))
    
    sub_h = hist[(hist['episode_id'] == ep) & (hist['machine_id'] == m)]
    sub_h = sub_h[sub_h['tick'] < cutoff]
    
    for w in windows:
        w_sub = sub_h[sub_h['tick'] >= (cutoff - w)]
        c1 = (w_sub['payout'] == 1).sum()
        chigh = (w_sub['payout'] >= 2).sum()
        r_hat = c1 / max(1, chigh)
        pred = 1 if np.log(max(1e-4, r_hat)) > log_mid else 0
        results[w].append(pred)

for w in windows:
    df[f'pred_{w}'] = results[w]
    acc = (df[f'pred_{w}'] == df['true_regime']).mean()
    print(f"Window {w:3d} ticks: Overall Accuracy = {acc*100:.2f}%")

print("\nBy Condition (Window 32 vs 64 vs 256):")
for cond, grp in df.groupby("condition"):
    a32 = (grp['pred_32'] == grp['true_regime']).mean()
    a64 = (grp['pred_64'] == grp['true_regime']).mean()
    a256 = (grp['pred_256'] == grp['true_regime']).mean()
    print(f"Condition: {cond.upper():15s} | W=32: {a32*100:.2f}% | W=64: {a64*100:.2f}% | W=256: {a256*100:.2f}%")
