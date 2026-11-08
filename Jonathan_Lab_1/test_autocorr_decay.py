import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def compute_regime_autocorrelation(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    q_meta = queries.merge(conds, on="query_id")
    ep_cond = q_meta.groupby("episode_id")["condition"].first().to_dict()
    hist["condition"] = hist["episode_id"].map(ep_cond)
    
    # Z_t = +1 if payout == 1, -1 if payout in [2, 5, 20], 0 if payout == 0
    # Also consider only win ticks: Z_w = payout == 1 (1 vs 0)
    
    print(f"\n=======================================================")
    print(f" REGIME AUTOCORRELATION & HALF-LIFE DECAY: {data_dir}")
    print(f"=======================================================")
    
    for cond in ["all", "stable", "drift", "shared_change", "short_history"]:
        if cond == "all":
            c_hist = hist
        else:
            c_hist = hist[hist["condition"] == cond]
            
        # We compute autocovariance over win events, and over raw ticks
        # For each machine:
        lags = list(range(1, 41))
        autocorr_by_lag = {lag: [] for lag in lags}
        
        # Also over tick time
        grouped = c_hist.groupby(["episode_id", "machine_id"])
        for (ep, m), group in grouped:
            group = group.sort_values("tick")
            p = group["payout"].values
            # Filter wins only:
            wins = p[p > 0]
            if len(wins) < 25: continue
            # z is 1 if 1x, 0 if high win (payout in 2,5,20)
            z = (wins == 1).astype(float)
            z = z - np.mean(z)
            var_z = np.var(z)
            if var_z < 1e-5: continue
            
            for lag in lags:
                if len(z) > lag + 5:
                    c = np.mean(z[:-lag] * z[lag:]) / var_z
                    autocorr_by_lag[lag].append(c)
                    
        mean_ac = [np.mean(autocorr_by_lag[lag]) for lag in lags]
        
        # Fit exponential decay: ac(k) = A * exp(-k / tau)
        # Half-life = tau * ln(2)
        def exp_decay(k, A, tau):
            return A * np.exp(-k / tau)
            
        try:
            popt, pcov = curve_fit(exp_decay, lags, mean_ac, p0=[0.5, 15.0], bounds=([0.0, 0.5], [1.0, 200.0]))
            A_fit, tau_fit = popt
            half_life_wins = tau_fit * np.log(2.0)
            # In raw ticks: since win rate is ~0.40, 1 win event occurs every ~2.5 ticks!
            # Therefore, raw tick half life = half_life_wins / win_rate
            mean_wr = (c_hist["payout"] > 0).mean()
            half_life_ticks = half_life_wins / mean_wr
            hazard_per_tick = np.log(2.0) / half_life_ticks
            
            print(f"Condition: {cond.upper():15s}")
            print(f"  Autocorr Lag 1: {mean_ac[0]:.4f}, Lag 2: {mean_ac[1]:.4f}, Lag 5: {mean_ac[4]:.4f}, Lag 10: {mean_ac[9]:.4f}")
            print(f"  Fitted Amplitude A = {A_fit:.4f}, Characteristic Scale tau = {tau_fit:.2f} wins")
            print(f"  Half-Life (in win events): {half_life_wins:.2f} wins")
            print(f"  Half-Life (in raw ticks):  {half_life_ticks:.2f} ticks (assuming win rate {mean_wr:.3f})")
            print(f"  Implied Transition Hazard (lambda): {hazard_per_tick:.5f} / tick")
            print(f"  Implied Mean Regime Duration (1/lambda): {1.0/hazard_per_tick:.1f} ticks")
            print()
        except Exception as e:
            print(f"Condition {cond}: Fit failed ({e})")

compute_regime_autocorrelation("adaptation-participants/public/validation")
compute_regime_autocorrelation("adaptation-participants/public/train")
