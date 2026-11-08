import pandas as pd
import numpy as np

hist = pd.read_csv("adaptation-participants/public/validation/history.csv")
queries = pd.read_csv("adaptation-participants/public/validation/queries.csv")
conds = pd.read_csv("adaptation-participants/public/validation/conditions.csv")

q_meta = queries.merge(conds, on="query_id")
ep_cond = q_meta.groupby("episode_id")["condition"].first()
hist["is_win"] = (hist["payout"] > 0).astype(int)

results = []
for cond in ["stable", "drift", "shared_change", "short_history"]:
    eps = ep_cond[ep_cond == cond].index.tolist()
    t_stats = []
    deltas = []
    for ep in eps:
        ep_df = hist[hist["episode_id"] == ep]
        piv = ep_df.pivot(index="tick", columns="machine_id", values="is_win").dropna()
        floor_win = piv.mean(axis=1)
        ticks = floor_win.index.values
        if len(ticks) < 40: continue
        
        best_t_stat = 0
        best_delta = 0
        for t in range(15, len(ticks) - 15):
            s1 = floor_win.iloc[:t]
            s2 = floor_win.iloc[t:]
            diff = s2.mean() - s1.mean()
            var1 = s1.var() / len(s1)
            var2 = s2.var() / len(s2)
            denom = np.sqrt(var1 + var2)
            if denom > 1e-6:
                ts = abs(diff) / denom
                if ts > best_t_stat:
                    best_t_stat = ts
                    best_delta = abs(diff)
        t_stats.append(best_t_stat)
        deltas.append(best_delta)
        
    results.append({
        "condition": cond,
        "mean_max_t_stat": np.mean(t_stats),
        "median_max_t_stat": np.median(t_stats),
        "mean_delta_win_rate": np.mean(deltas),
        "pct_episodes_t_gt_4": (np.array(t_stats) >= 4.0).mean() * 100,
        "pct_episodes_t_gt_5": (np.array(t_stats) >= 5.0).mean() * 100
    })

print(pd.DataFrame(results).to_string(index=False))
