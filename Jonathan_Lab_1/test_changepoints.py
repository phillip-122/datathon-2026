import pandas as pd
import numpy as np

hist = pd.read_csv("adaptation-participants/public/validation/history.csv")
queries = pd.read_csv("adaptation-participants/public/validation/queries.csv")
conds = pd.read_csv("adaptation-participants/public/validation/conditions.csv")

q_meta = queries.merge(conds, on="query_id")
ep_cond = q_meta.groupby("episode_id")["condition"].first()
shared_eps = ep_cond[ep_cond == "shared_change"].index.tolist()

hist["is_win"] = (hist["payout"] > 0).astype(int)

# For each shared_change episode, find the optimal floor changepoint tick
changepoint_data = []

for ep in shared_eps:
    ep_df = hist[hist["episode_id"] == ep]
    piv = ep_df.pivot(index="tick", columns="machine_id", values="is_win").dropna()
    floor_win = piv.mean(axis=1)
    
    ticks = floor_win.index.values
    best_t = None
    best_t_stat = 0
    delta_win = 0
    
    for t in range(25, len(ticks) - 25):
        s1 = floor_win.iloc[:t]
        s2 = floor_win.iloc[t:]
        diff = s2.mean() - s1.mean()
        # Two-sample t-statistic
        var1 = s1.var() / len(s1)
        var2 = s2.var() / len(s2)
        denom = np.sqrt(var1 + var2)
        if denom > 1e-6:
            t_stat = abs(diff) / denom
            if t_stat > best_t_stat:
                best_t_stat = t_stat
                best_t = ticks[t]
                delta_win = diff
                
    changepoint_data.append({
        "episode_id": ep,
        "changepoint_tick": best_t,
        "delta_win_rate": delta_win,
        "t_statistic": best_t_stat,
        "floor_len": len(ticks)
    })

cp_df = pd.DataFrame(changepoint_data)
print("=== SHARED_CHANGE FLOOR CHANGEPOINTS (VALIDATION) ===")
print(cp_df.describe().round(3))
print("\nSample changepoint rows:")
print(cp_df.head(10).round(3))
