import pandas as pd
import numpy as np
from scipy import stats

def test_machine_constancy(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    m_info = q_meta.set_index(["episode_id", "machine_id"]).to_dict("index")
    
    p_values_by_cond = {"stable": [], "drift": [], "shared_change": [], "short_history": []}
    
    for (ep, m), group in hist.groupby(["episode_id", "machine_id"]):
        if (ep, m) not in m_info: continue
        info = m_info[(ep, m)]
        cond = info["condition"]
        group = group.sort_values("tick")
        
        # wins only
        wins = group[group["payout"] > 0]
        n_w = len(wins)
        if n_w < 20: continue
        
        half = n_w // 2
        w1 = wins.iloc[:half]
        w2 = wins.iloc[half:]
        
        c1_1 = (w1["payout"] == 1).sum()
        ch_1 = (w1["payout"] >= 2).sum()
        
        c1_2 = (w2["payout"] == 1).sum()
        ch_2 = (w2["payout"] >= 2).sum()
        
        odds, pval = stats.fisher_exact([[c1_1, ch_1], [c1_2, ch_2]])
        p_values_by_cond[cond].append(pval)
        
    print(f"\n=== FISHER EXACT TEST FOR REGIME SWITCH (FIRST HALF VS SECOND HALF) in {data_dir} ===")
    for cond, pvals in p_values_by_cond.items():
        pvals = np.array(pvals)
        frac_sig = (pvals < 0.05).mean()
        frac_sig_01 = (pvals < 0.01).mean()
        print(f"Condition: {cond.upper():15s} (N={len(pvals)} machines)")
        print(f"  Frac with p < 0.05: {frac_sig:.4f} (Under constancy null: 0.0500)")
        print(f"  Frac with p < 0.01: {frac_sig_01:.4f} (Under constancy null: 0.0100)")
        print(f"  Mean p-value:       {pvals.mean():.4f} (Under constancy null: 0.5000)")
        print(f"  Median p-value:     {np.median(pvals):.4f} (Under constancy null: 0.5000)")

test_machine_constancy("adaptation-participants/public/validation")
test_machine_constancy("adaptation-participants/public/train")
