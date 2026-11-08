import pandas as pd
import numpy as np
from scipy import stats

def analyze_peer_contagion(data_dir):
    print(f"=== ANALYZING PEER CONTAGION IN {data_dir} ===")
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    # Map query to condition and episode
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first().to_dict()
    ep_machines = q_meta.groupby("episode_id")["machine_id"].unique().to_dict()
    
    # Binary win (payout > 0) and high win (payout >= 2)
    hist["is_win"] = (hist["payout"] > 0).astype(int)
    hist["is_high_win"] = (hist["payout"] >= 2).astype(int)
    hist["is_1x"] = (hist["payout"] == 1).astype(int)
    hist["condition"] = hist["episode_id"].map(ep_cond)
    
    results = {}
    
    for cond in ["stable", "drift", "shared_change", "short_history"]:
        cond_hist = hist[hist["condition"] == cond]
        episodes = cond_hist["episode_id"].unique()
        
        contemp_corrs = []
        high_corrs = []
        lag1_corrs = []
        lag2_corrs = []
        lag3_corrs = []
        lag5_corrs = []
        lead1_corrs = []
        
        # Rolling window correlations (window=16)
        roll16_corrs = []
        roll8_corrs = []
        
        # Surge dispersion index: Var(sum_m W_m) / Sum(Var(W_m))
        dispersion_indices = []
        
        # Predictive surge probability: P(neighbor surge in [t+1, t+5] | machine i surges at t)
        # vs baseline P(neighbor surge in [t+1, t+5])
        surge_lift_list = []
        
        for ep in episodes:
            ep_df = cond_hist[cond_hist["episode_id"] == ep]
            pivot_win = ep_df.pivot(index="tick", columns="machine_id", values="is_win")
            pivot_high = ep_df.pivot(index="tick", columns="machine_id", values="is_high_win")
            
            # Drop ticks with missing values if any
            pivot_win = pivot_win.dropna()
            pivot_high = pivot_high.dropna()
            
            if len(pivot_win) < 20 or pivot_win.shape[1] < 2:
                continue
                
            # Pairwise contemporaneous correlation matrix
            corr_mat = pivot_win.corr().values
            n_m = corr_mat.shape[0]
            upper_tri = corr_mat[np.triu_indices(n_m, k=1)]
            contemp_corrs.extend(upper_tri[~np.isnan(upper_tri)])
            
            # High win correlation
            h_corr_mat = pivot_high.corr().values
            h_upper_tri = h_corr_mat[np.triu_indices(n_m, k=1)]
            high_corrs.extend(h_upper_tri[~np.isnan(h_upper_tri)])
            
            # Dispersion Index: Var(Total wins at tick t) / Sum(Var(each machine win))
            total_wins_t = pivot_win.sum(axis=1)
            var_total = total_wins_t.var()
            sum_var = pivot_win.var().sum()
            if sum_var > 1e-6:
                dispersion_indices.append(var_total / sum_var)
                
            # Rolling window correlation (smoothed win rate over 16 ticks)
            roll16 = pivot_win.rolling(window=16, min_periods=16).mean().dropna()
            if len(roll16) > 20:
                r16_mat = roll16.corr().values
                r16_upper = r16_mat[np.triu_indices(n_m, k=1)]
                roll16_corrs.extend(r16_upper[~np.isnan(r16_upper)])

            roll8 = pivot_win.rolling(window=8, min_periods=8).mean().dropna()
            if len(roll8) > 20:
                r8_mat = roll8.corr().values
                r8_upper = r8_mat[np.triu_indices(n_m, k=1)]
                roll8_corrs.extend(r8_upper[~np.isnan(r8_upper)])

            # Cross-correlation / Lagged correlation between distinct machines:
            # For each machine i and peer j != i: corr(W_i(t), W_j(t+lag))
            cols = pivot_win.columns
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i == j: continue
                    s_i = pivot_win[cols[i]].values
                    s_j = pivot_win[cols[j]].values
                    
                    # Lag 1: s_i(t) vs s_j(t+1)
                    c1 = np.corrcoef(s_i[:-1], s_j[1:])[0, 1]
                    if not np.isnan(c1): lag1_corrs.append(c1)
                    
                    # Lag 2
                    c2 = np.corrcoef(s_i[:-2], s_j[2:])[0, 1]
                    if not np.isnan(c2): lag2_corrs.append(c2)
                    
                    # Lag 3
                    c3 = np.corrcoef(s_i[:-3], s_j[3:])[0, 1]
                    if not np.isnan(c3): lag3_corrs.append(c3)

                    # Lag 5
                    c5 = np.corrcoef(s_i[:-5], s_j[5:])[0, 1]
                    if not np.isnan(c5): lag5_corrs.append(c5)

            # Win surge contagion:
            # Define surge on machine i as rolling 8-tick win count >= 5 (i.e. win rate >= 62.5% vs base ~39%)
            r8_counts = pivot_win.rolling(window=8, min_periods=8).sum()
            is_surge = (r8_counts >= 5).astype(int)
            # Baseline probability of peer surge in next 5 ticks
            # and conditional probability of peer surge given machine i surged
            for col in cols:
                peer_cols = [c for c in cols if c != col]
                peer_surge_any = (is_surge[peer_cols].sum(axis=1) > 0).astype(int)
                
                # Align time
                valid_mask = ~r8_counts[col].isna()
                my_surge = is_surge[col].values
                peer_surge = peer_surge_any.values
                
                # Check next 1-5 ticks:
                for t in range(8, len(my_surge) - 5):
                    # Did peer surge in [t+1, t+5]?
                    peer_next = 1 if peer_surge[t+1 : t+6].max() > 0 else 0
                    surge_lift_list.append({
                        "condition": cond,
                        "my_surge": my_surge[t],
                        "peer_next": peer_next
                    })
                    
        results[cond] = {
            "mean_contemp_corr": np.mean(contemp_corrs),
            "std_contemp_corr": np.std(contemp_corrs),
            "median_contemp_corr": np.median(contemp_corrs),
            "p95_contemp_corr": np.percentile(contemp_corrs, 95) if len(contemp_corrs)>0 else 0,
            "mean_high_corr": np.mean(high_corrs),
            "mean_roll8_corr": np.mean(roll8_corrs),
            "mean_roll16_corr": np.mean(roll16_corrs),
            "mean_dispersion": np.mean(dispersion_indices),
            "lag1_corr": np.mean(lag1_corrs),
            "lag2_corr": np.mean(lag2_corrs),
            "lag3_corr": np.mean(lag3_corrs),
            "lag5_corr": np.mean(lag5_corrs),
        }
        
        # Compute surge lift
        sdf = pd.DataFrame([x for x in surge_lift_list if x["condition"] == cond])
        if len(sdf) > 0:
            p_base = sdf["peer_next"].mean()
            p_given_surge = sdf[sdf["my_surge"] == 1]["peer_next"].mean() if (sdf["my_surge"] == 1).sum() > 0 else np.nan
            p_given_no_surge = sdf[sdf["my_surge"] == 0]["peer_next"].mean() if (sdf["my_surge"] == 0).sum() > 0 else np.nan
            results[cond]["base_peer_surge_prob"] = p_base
            results[cond]["p_peer_surge_given_surge"] = p_given_surge
            results[cond]["p_peer_surge_given_no_surge"] = p_given_no_surge
            results[cond]["surge_lift"] = (p_given_surge / p_base) if p_base > 0 else np.nan

    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    return res_df

print("TRAINING SET:")
train_res = analyze_peer_contagion("adaptation-participants/public/train")

print("\nVALIDATION SET:")
val_res = analyze_peer_contagion("adaptation-participants/public/validation")
