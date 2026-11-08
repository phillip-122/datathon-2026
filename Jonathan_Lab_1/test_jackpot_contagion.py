import pandas as pd
import numpy as np

def analyze_jackpot_contagion(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first().to_dict()
    hist["condition"] = hist["episode_id"].map(ep_cond)
    
    # Let's test payouts: 20x jackpot, and high wins (5x, 20x)
    hist["is_20x"] = (hist["payout"] == 20).astype(int)
    hist["is_5_or_20"] = (hist["payout"].isin([5, 20])).astype(int)
    
    for cond in ["stable", "drift", "shared_change", "short_history"]:
        c_hist = hist[hist["condition"] == cond]
        episodes = c_hist["episode_id"].unique()
        
        # When machine i hits 20x (jackpot) at tick t:
        # What is the probability that another machine hits 20x or 5x in window [t-3, t+3]?
        jackpot_clusters = 0
        total_jackpots = 0
        
        # Baseline probability of 20x across floor at any tick
        total_ticks = 0
        ticks_with_20x = 0
        ticks_with_multiple_20x = 0
        
        for ep in episodes:
            ep_df = c_hist[c_hist["episode_id"] == ep]
            piv = ep_df.pivot(index="tick", columns="machine_id", values="is_20x").fillna(0)
            
            floor_20x = piv.sum(axis=1)
            total_ticks += len(floor_20x)
            ticks_with_20x += (floor_20x >= 1).sum()
            ticks_with_multiple_20x += (floor_20x >= 2).sum()
            
        p_base_20x = ticks_with_20x / total_ticks if total_ticks > 0 else 0
        # Expected multiple 20x under binomial(8, p_indiv)
        # Average individual 20x rate:
        p_indiv = hist[hist["condition"] == cond]["is_20x"].mean()
        # Under independence: P(X >= 2) = 1 - (1-p)^8 - 8*p*(1-p)^7
        p_mult_expected = 1.0 - (1.0 - p_indiv)**8 - 8 * p_indiv * (1.0 - p_indiv)**7
        p_mult_observed = ticks_with_multiple_20x / total_ticks if total_ticks > 0 else 0
        
        print(f"Condition: {cond.upper():15s}")
        print(f"  Total ticks: {total_ticks}, Individual p_20x: {p_indiv:.5f}")
        print(f"  Observed multiple 20x ticks (>= 2 simultaneous jackpots): {ticks_with_multiple_20x} ({p_mult_observed:.6f})")
        print(f"  Expected multiple 20x ticks under independence:          {total_ticks * p_mult_expected:.2f} ({p_mult_expected:.6f})")
        print(f"  Simultaneous Jackpot Excess Ratio: {p_mult_observed / max(1e-9, p_mult_expected):.2f}x")
        print()

print("VALIDATION SET:")
analyze_jackpot_contagion("adaptation-participants/public/validation")
print("TRAINING SET:")
analyze_jackpot_contagion("adaptation-participants/public/train")
