import pandas as pd
import numpy as np

def analyze_regime_shifts(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    # map (episode_id, machine_id) -> machine_type, condition
    m_info = q_meta.set_index(["episode_id", "machine_id"])[["machine_type", "condition"]].to_dict("index")
    
    REGIME_RATIOS = {
        'steady': (0.8352732428176273, 3.064865326422233),
        'balanced': (0.2723717096144427, 0.9994126064420332),
        'jackpot': (0.13738046756868819, 0.504089691845762),
    }

    results = []
    
    # Group history by episode and machine
    grouped = hist.groupby(["episode_id", "machine_id"])
    for (ep, m), group in grouped:
        if (ep, m) not in m_info: continue
        info = m_info[(ep, m)]
        mt = info["machine_type"]
        cond = info["condition"]
        
        group = group.sort_values("tick")
        n = len(group)
        if n < 40: continue
        
        # Split into two halves: t < n//2 vs t >= n//2
        h1 = group.iloc[: n//2]
        h2 = group.iloc[n//2 :]
        
        c1_h1 = (h1["payout"] == 1).sum()
        chigh_h1 = (h1["payout"] >= 2).sum()
        
        c1_h2 = (h2["payout"] == 1).sum()
        chigh_h2 = (h2["payout"] >= 2).sum()
        
        # Avoid division by zero with Laplace smoothing or minimum count
        r_low, r_high = REGIME_RATIOS[mt]
        r_h1 = c1_h1 / max(1, chigh_h1)
        r_h2 = c1_h2 / max(1, chigh_h2)
        
        # Log distance to mid-point
        log_r0 = np.log(r_low)
        log_r1 = np.log(r_high)
        log_mid = 0.5 * (log_r0 + log_r1)
        
        reg_h1 = 1 if np.log(max(0.01, r_h1)) > log_mid else 0
        reg_h2 = 1 if np.log(max(0.01, r_h2)) > log_mid else 0
        
        results.append({
            "episode_id": ep,
            "machine_id": m,
            "condition": cond,
            "machine_type": mt,
            "n_spins": n,
            "c1_h1": c1_h1, "chigh_h1": chigh_h1, "r_h1": r_h1, "reg_h1": reg_h1,
            "c1_h2": c1_h2, "chigh_h2": chigh_h2, "r_h2": r_h2, "reg_h2": reg_h2,
            "switched": 1 if reg_h1 != reg_h2 else 0
        })
        
    res_df = pd.DataFrame(results)
    print("=== PROBABILITY OF SWITCHING BETWEEN FIRST HALF AND SECOND HALF ===")
    print(res_df.groupby("condition")["switched"].agg(["count", "mean", "std"]))
    print("\nBy Condition and Machine Type:")
    print(res_df.groupby(["condition", "machine_type"])["switched"].mean().unstack())

print("--- TRAINING SET ---")
analyze_regime_shifts("adaptation-participants/public/train")

print("\n--- VALIDATION SET ---")
analyze_regime_shifts("adaptation-participants/public/validation")
