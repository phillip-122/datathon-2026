import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

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

hist_map = {}
for (ep, m), group in hist.groupby(["episode_id", "machine_id"]):
    hist_map[(ep, m)] = group.sort_values("tick")

half_lives = [16, 32, 64, 96, 128, 180, 256, 10000]

res_by_cond = {c: [] for c in ["stable", "drift", "shared_change", "short_history"]}

for H in half_lives:
    probs = []
    for idx, row in df.iterrows():
        ep = row['episode_id_x']
        m = row['machine_id']
        cutoff = row['cutoff']
        mt = row['machine_type']
        r_low, r_high = REGIME_RATIOS[mt]
        
        gh = hist_map.get((ep, m), None)
        if gh is None or len(gh) == 0:
            probs.append(0.5)
            continue
            
        gh_sub = gh[gh['tick'] < cutoff]
        ages = cutoff - 1 - gh_sub['tick'].values
        weights = 2.0 ** (-ages / H)
        
        payouts = gh_sub['payout'].values
        w1 = weights[payouts == 1].sum()
        whigh = weights[np.isin(payouts, [2, 5, 20])].sum()
        
        p1_given_win_1 = r_high / (1.0 + r_high)
        p1_given_win_0 = r_low / (1.0 + r_low)
        
        ll_1 = w1 * np.log(p1_given_win_1) + whigh * np.log(1.0 - p1_given_win_1)
        ll_0 = w1 * np.log(p1_given_win_0) + whigh * np.log(1.0 - p1_given_win_0)
        
        log_odds = (ll_1 - ll_0) + 0.2007
        prob_1 = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20, 20)))
        probs.append(prob_1)
        
    df[f'prob_{H}'] = probs
    for cond in res_by_cond.keys():
        sub = df[df['condition'] == cond]
        auc = roc_auc_score(sub['true_regime'], sub[f'prob_{H}'])
        acc = ((sub[f'prob_{H}'] >= 0.5) == sub['true_regime']).mean()
        res_by_cond[cond].append((H, auc, acc))

for cond, vals in res_by_cond.items():
    print(f"\nCondition: {cond.upper()}")
    print(f"{'Half-Life (H)':>15s} | {'AUC':>10s} | {'Accuracy':>10s}")
    for H, auc, acc in vals:
        h_str = f"{H}" if H < 1000 else "Infinite"
        print(f"{h_str:>15s} | {auc:>10.4f} | {acc*100:>9.2f}%")
