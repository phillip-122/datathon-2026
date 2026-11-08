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

# Determine true regime at cutoff
true_regimes = []
for idx, row in df.iterrows():
    mt = row['machine_type']
    r_low, r_high = REGIME_RATIOS[mt]
    true_regimes.append(0 if abs(row['r_1x'] - r_low) < abs(row['r_1x'] - r_high) else 1)
df['true_regime'] = true_regimes

# Now compute empirical count ratio over ALL historical spins for each machine
all_counts = {}
early_counts = {} # ticks < 50
late_counts = {}  # cutoff - 32 <= tick < cutoff

for (ep, m), group in hist.groupby(["episode_id", "machine_id"]):
    c1_all = (group["payout"] == 1).sum()
    chigh_all = (group["payout"] >= 2).sum()
    all_counts[(ep, m)] = (c1_all, chigh_all)
    
    g_early = group[group["tick"] < 50]
    early_counts[(ep, m)] = ((g_early["payout"] == 1).sum(), (g_early["payout"] >= 2).sum())

df["c1_all"] = [all_counts.get((r.episode_id_x, r.machine_id), (0, 0))[0] for _, r in df.iterrows()]
df["chigh_all"] = [all_counts.get((r.episode_id_x, r.machine_id), (0, 0))[1] for _, r in df.iterrows()]
df["c1_early"] = [early_counts.get((r.episode_id_x, r.machine_id), (0, 0))[0] for _, r in df.iterrows()]
df["chigh_early"] = [early_counts.get((r.episode_id_x, r.machine_id), (0, 0))[1] for _, r in df.iterrows()]

# Predicted regime from all history:
pred_all = []
pred_early = []
for idx, row in df.iterrows():
    mt = row['machine_type']
    r_low, r_high = REGIME_RATIOS[mt]
    log_mid = 0.5 * (np.log(r_low) + np.log(r_high))
    
    # all history
    r_hat_all = row['c1_all'] / max(1, row['chigh_all'])
    pred_all.append(1 if np.log(max(1e-4, r_hat_all)) > log_mid else 0)
    
    # early history (ticks 0-49)
    r_hat_early = row['c1_early'] / max(1, row['chigh_early'])
    pred_early.append(1 if np.log(max(1e-4, r_hat_early)) > log_mid else 0)

df['pred_regime_all'] = pred_all
df['pred_regime_early'] = pred_early

acc_all = (df['pred_regime_all'] == df['true_regime']).mean()
acc_early = (df['pred_regime_early'] == df['true_regime']).mean()

print(f"Overall Accuracy of predicting Cutoff Regime using ALL history: {acc_all*100:.2f}%")
print(f"Overall Accuracy of predicting Cutoff Regime using FIRST 50 TICKS: {acc_early*100:.2f}%")
print()

for cond, grp in df.groupby("condition"):
    c_acc_all = (grp['pred_regime_all'] == grp['true_regime']).mean()
    c_acc_early = (grp['pred_regime_early'] == grp['true_regime']).mean()
    print(f"Condition: {cond.upper():15s} | All History Acc: {c_acc_all*100:.2f}% | First 50 Ticks Acc: {c_acc_early*100:.2f}%")
