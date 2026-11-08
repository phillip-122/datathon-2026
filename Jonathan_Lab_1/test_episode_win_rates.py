import pandas as pd
import numpy as np

for split in ["train", "validation"]:
    hist = pd.read_csv(f"adaptation-participants/public/{split}/history.csv")
    queries = pd.read_csv(f"adaptation-participants/public/{split}/queries.csv")
    conds = pd.read_csv(f"adaptation-participants/public/{split}/conditions.csv")
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first()
    
    hist["is_win"] = (hist["payout"] > 0).astype(int)
    # Average win rate per machine
    m_wr = hist.groupby(["episode_id", "machine_id"])["is_win"].mean().reset_index()
    m_wr["condition"] = m_wr["episode_id"].map(ep_cond)
    
    print(f"=== {split.upper()} WIN RATE VARIANCE DECOMPOSITION ===")
    for cond, grp in m_wr.groupby("condition"):
        within_ep_var = grp.groupby("episode_id")["is_win"].var().mean()
        across_ep_var = grp.groupby("episode_id")["is_win"].mean().var()
        icc = across_ep_var / (across_ep_var + within_ep_var)
        print(f"Condition: {cond.upper():15s} | Within-ep var: {within_ep_var:.6f} | Across-ep var: {across_ep_var:.6f} | ICC: {icc:.4f}")
    print()
