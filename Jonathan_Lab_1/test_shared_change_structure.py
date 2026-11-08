import pandas as pd
import numpy as np

def inspect_shared_change(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first()
    shared_eps = ep_cond[ep_cond == "shared_change"].index.tolist()
    
    print(f"Inspecting {len(shared_eps)} shared_change episodes in {data_dir}...")
    
    # Check if there is an abrupt shift in floor win rate or 1x rate at some tick
    for ep in shared_eps[:5]:
        ep_df = hist[hist["episode_id"] == ep]
        pivot_win = ep_df.pivot(index="tick", columns="machine_id", values="payout")
        floor_win = (pivot_win > 0).mean(axis=1)
        floor_p1 = (pivot_win == 1).mean(axis=1)
        floor_high = (pivot_win >= 2).mean(axis=1)
        
        # Test change point detection: find tick with max difference in mean before vs after
        ticks = floor_win.index.values
        best_t = None
        best_diff = 0
        for t in range(20, len(ticks) - 20):
            diff = abs(floor_win.iloc[:t].mean() - floor_win.iloc[t:].mean())
            if diff > best_diff:
                best_diff = diff
                best_t = ticks[t]
                
        # Also check rolling mean
        roll20 = floor_win.rolling(20, center=True).mean()
        min_roll = roll20.min()
        max_roll = roll20.max()
        
        print(f"Ep {ep}: Best change point tick={best_t}, Delta_WinRate={best_diff:.3f}, Roll20 range=[{min_roll:.3f}, {max_roll:.3f}]")

inspect_shared_change("adaptation-participants/public/validation")
inspect_shared_change("adaptation-participants/public/train")
