import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

def run_study(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first().to_dict()
    
    hist["is_win"] = (hist["payout"] > 0).astype(float)
    hist["is_high_win"] = (hist["payout"] >= 2).astype(float)
    hist["is_1x"] = (hist["payout"] == 1).astype(float)
    hist["condition"] = hist["episode_id"].map(ep_cond)
    
    records = []
    
    for cond in ["stable", "drift", "shared_change", "short_history"]:
        cond_hist = hist[hist["condition"] == cond]
        episodes = cond_hist["episode_id"].unique()
        
        contemp_corrs = []
        high_corrs = []
        p1_corrs = []
        roll8_corrs = []
        roll16_corrs = []
        dispersion_ratios = []
        cross_lags = {k: [] for k in [1, 2, 3, 4, 5, 8]}
        
        # Win rate variance across machines at same tick
        floor_variance_ratios = []
        
        for ep in episodes:
            ep_df = cond_hist[cond_hist["episode_id"] == ep]
            pivot_win = ep_df.pivot(index="tick", columns="machine_id", values="is_win").dropna()
            pivot_high = ep_df.pivot(index="tick", columns="machine_id", values="is_high_win").dropna()
            pivot_1x = ep_df.pivot(index="tick", columns="machine_id", values="is_1x").dropna()
            
            if len(pivot_win) < 15 or pivot_win.shape[1] < 2:
                continue
                
            # Pairwise contemporaneous correlation
            cm = pivot_win.corr().values
            n = cm.shape[0]
            upper_tri = cm[np.triu_indices(n, k=1)]
            contemp_corrs.extend(upper_tri[~np.isnan(upper_tri)])
            
            cm_h = pivot_high.corr().values
            upper_h = cm_h[np.triu_indices(n, k=1)]
            high_corrs.extend(upper_h[~np.isnan(upper_h)])
            
            cm_1 = pivot_1x.corr().values
            upper_1 = cm_1[np.triu_indices(n, k=1)]
            p1_corrs.extend(upper_1[~np.isnan(upper_1)])
            
            # Dispersion: Var(floor aggregate) / Sum(individual variances)
            # Under independence, Var(Sum) == Sum(Var), so ratio == 1.0.
            # If machines are positively correlated, ratio > 1.0!
            # Ratio = 1 + (N-1) * mean_pairwise_covariance / mean_variance.
            var_sum = pivot_win.sum(axis=1).var()
            sum_var = pivot_win.var().sum()
            if sum_var > 1e-5:
                dispersion_ratios.append(var_sum / sum_var)
                
            # Rolling correlations
            if len(pivot_win) >= 24:
                r8 = pivot_win.rolling(8).mean().dropna().corr().values
                roll8_corrs.extend(r8[np.triu_indices(n, k=1)][~np.isnan(r8[np.triu_indices(n, k=1)])])
                r16 = pivot_win.rolling(16).mean().dropna().corr().values
                roll16_corrs.extend(r16[np.triu_indices(n, k=1)][~np.isnan(r16[np.triu_indices(n, k=1)])])
                
            # Cross lags
            cols = pivot_win.columns.tolist()
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i == j: continue
                    x = pivot_win[cols[i]].values
                    y = pivot_win[cols[j]].values
                    for lag in cross_lags.keys():
                        if len(x) > lag + 5:
                            c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                            if not np.isnan(c):
                                cross_lags[lag].append(c)

        records.append({
            "condition": cond,
            "mean_r_spin": np.mean(contemp_corrs),
            "median_r_spin": np.median(contemp_corrs),
            "mean_r_high": np.mean(high_corrs),
            "mean_r_1x": np.mean(p1_corrs),
            "mean_r_roll8": np.mean(roll8_corrs) if roll8_corrs else np.nan,
            "mean_r_roll16": np.mean(roll16_corrs) if roll16_corrs else np.nan,
            "dispersion_ratio": np.mean(dispersion_ratios),
            "lag1_cross_r": np.mean(cross_lags[1]),
            "lag2_cross_r": np.mean(cross_lags[2]),
            "lag3_cross_r": np.mean(cross_lags[3]),
            "lag5_cross_r": np.mean(cross_lags[5]),
            "lag8_cross_r": np.mean(cross_lags[8]),
        })
    return pd.DataFrame(records)

print("--- TRAIN PEER CORRELATIONS ---")
df_tr = run_study("adaptation-participants/public/train")
print(df_tr.to_string(index=False))

print("\n--- VALIDATION PEER CORRELATIONS ---")
df_val = run_study("adaptation-participants/public/validation")
print(df_val.to_string(index=False))
