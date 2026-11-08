import pandas as pd
import numpy as np
from scipy import stats

def analyze_floor_contagion(data_dir):
    print(f"\n=======================================================")
    print(f" DEEP FLOOR SYNCHRONIZATION & CONTAGION: {data_dir}")
    print(f"=======================================================")
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first().to_dict()
    hist["condition"] = hist["episode_id"].map(ep_cond)
    
    hist["is_win"] = (hist["payout"] > 0).astype(int)
    hist["is_high"] = (hist["payout"] >= 2).astype(int)
    hist["is_1x"] = (hist["payout"] == 1).astype(int)
    
    for cond in ["stable", "drift", "shared_change", "short_history"]:
        c_hist = hist[hist["condition"] == cond]
        episodes = c_hist["episode_id"].unique()
        
        # Test 1: Contemporaneous peer win probability
        # P(peer wins at t | machine i wins at t) vs P(peer wins at t | machine i loses at t)
        p_peer_win_given_win = []
        p_peer_win_given_loss = []
        
        # Test 2: Lagged peer win probability
        # P(peer wins at t+1 | machine i wins at t) vs P(peer wins at t+1 | machine i loses at t)
        p_peer_win_lag1_given_win = []
        p_peer_win_lag1_given_loss = []
        
        # Test 3: Surge contagion
        # Define surge as rolling 5-tick win count >= 3 (60% win rate vs baseline 40%)
        # P(peer surges at [t+1, t+5] | machine i surges at t) vs P(peer surges at [t+1, t+5] | machine i does not surge at t)
        surge_given_surge = []
        surge_given_no_surge = []
        
        # Test 4: Contemporaneous correlation of rolling win rates (window 8 and 16)
        corr_r8_list = []
        corr_r16_list = []
        
        # Test 5: Floor aggregate variance vs independent sum of variances
        var_ratios = []
        
        for ep in episodes:
            ep_df = c_hist[c_hist["episode_id"] == ep]
            piv = ep_df.pivot(index="tick", columns="machine_id", values="is_win").dropna()
            if len(piv) < 20 or piv.shape[1] < 2:
                continue
                
            cols = piv.columns.tolist()
            
            # Contemporaneous peer win
            for c in cols:
                y = piv[c].values
                peer_mean = piv[[col for col in cols if col != c]].mean(axis=1).values
                
                win_mask = (y == 1)
                loss_mask = (y == 0)
                if win_mask.sum() > 0:
                    p_peer_win_given_win.append(peer_mean[win_mask].mean())
                if loss_mask.sum() > 0:
                    p_peer_win_given_loss.append(peer_mean[loss_mask].mean())
                    
                # Lag 1: y[t] predicts peer_mean[t+1]
                if len(y) > 1:
                    w1 = win_mask[:-1]
                    l1 = loss_mask[:-1]
                    p1 = peer_mean[1:]
                    if w1.sum() > 0:
                        p_peer_win_lag1_given_win.append(p1[w1].mean())
                    if l1.sum() > 0:
                        p_peer_win_lag1_given_loss.append(p1[l1].mean())
                        
            # Surge contagion: rolling 5 >= 3
            roll5 = (piv.rolling(5).sum() >= 3).astype(int).dropna()
            if len(roll5) > 10:
                for c in cols:
                    s_c = roll5[c].values
                    peer_cols = [col for col in cols if col != c]
                    # Any peer surges in [t+1, t+5]
                    peer_any_surge = (roll5[peer_cols].sum(axis=1) > 0).astype(int).values
                    
                    for t in range(len(s_c) - 5):
                        peer_next = 1 if peer_any_surge[t+1 : t+6].max() > 0 else 0
                        if s_c[t] == 1:
                            surge_given_surge.append(peer_next)
                        else:
                            surge_given_no_surge.append(peer_next)
                            
            # Rolling correlations
            if len(piv) >= 24:
                r8 = piv.rolling(8).mean().dropna().corr().values
                n_m = r8.shape[0]
                corr_r8_list.extend(r8[np.triu_indices(n_m, k=1)])
                r16 = piv.rolling(16).mean().dropna().corr().values
                corr_r16_list.extend(r16[np.triu_indices(n_m, k=1)])
                
            # Floor variance ratio
            tot = piv.sum(axis=1)
            v_tot = tot.var()
            sum_v = piv.var().sum()
            if sum_v > 1e-5:
                var_ratios.append(v_tot / sum_v)
                
        p_win_w = np.mean(p_peer_win_given_win)
        p_win_l = np.mean(p_peer_win_given_loss)
        p_lag_w = np.mean(p_peer_win_lag1_given_win)
        p_lag_l = np.mean(p_peer_win_lag1_given_loss)
        
        p_s_s = np.mean(surge_given_surge) if surge_given_surge else np.nan
        p_s_ns = np.mean(surge_given_no_surge) if surge_given_no_surge else np.nan
        
        print(f"Condition: {cond.upper():15s}")
        print(f"  Contemporaneous P(peer win | win):  {p_win_w:.4f} vs P(peer win | loss): {p_win_l:.4f} | Diff = {p_win_w - p_win_l:+.4f} (Lift = {p_win_w/p_win_l:.4f})")
        print(f"  Predictive Lag-1 P(peer win | win): {p_lag_w:.4f} vs P(peer win | loss): {p_lag_l:.4f} | Diff = {p_lag_w - p_lag_l:+.4f} (Lift = {p_lag_w/p_lag_l:.4f})")
        print(f"  Surge Contagion P(peer surge | surge): {p_s_s:.4f} vs P(peer surge | no surge): {p_s_ns:.4f} | Lift = {p_s_s/p_s_ns:.4f}")
        print(f"  Rolling Correlations: r_8 = {np.nanmean(corr_r8_list):+.4f}, r_16 = {np.nanmean(corr_r16_list):+.4f}")
        print(f"  Dispersion Ratio (Var(Floor) / Sum Var): {np.mean(var_ratios):.4f} (Under independence = 1.0000)")
        print()

analyze_floor_contagion("adaptation-participants/public/validation")
analyze_floor_contagion("adaptation-participants/public/train")
