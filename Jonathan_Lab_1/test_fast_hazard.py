import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

def analyze_dataset(data_dir):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    m_info = q_meta.set_index(["episode_id", "machine_id"])[["machine_type", "condition"]].to_dict("index")
    
    # Pre-parse sequences
    sequences = []
    for (ep, m), group in hist.groupby(["episode_id", "machine_id"]):
        if (ep, m) not in m_info: continue
        info = m_info[(ep, m)]
        group = group.sort_values("tick")
        sequences.append({
            "ep": ep,
            "m": m,
            "mt": info["machine_type"],
            "cond": info["condition"],
            "payouts": group["payout"].values
        })
        
    print(f"Loaded {len(sequences)} sequences from {data_dir}")
    
    # For each sequence, build log emission difference:
    # Notice: when payout == 0, log-likelihood ratio log(P(0|S=1)/P(0|S=0)) ~ 0 (loss gives neutral evidence regarding 1x ratio)
    # When payout == 1: log(P(1|S=1)/P(1|S=0)) = log(r1/(1+r1)) - log(r0/(1+r0))
    # When payout in [2,5,20]: log(P(high|S=1)/P(high|S=0)) = log(1/(1+r1)) - log(1/(1+r0))
    # Let log_LR be the log emission odds: log(b_1(y_t) / b_0(y_t))
    # This reduces the 2-state HMM to an exact 1D log-odds filtering / forward recursion:
    # L_t = log_odds(S_t = 1 | y_{1:t})
    
    for s in sequences:
        mt = s["mt"]
        r0, r1 = REGIME_RATIOS[mt]
        ll_1 = np.log((r1 / (1 + r1)) / (r0 / (1 + r0)))
        ll_h = np.log((1 / (1 + r1)) / (1 / (1 + r0)))
        
        y = s["payouts"]
        log_evidence = np.zeros(len(y))
        log_evidence[y == 1] = ll_1
        log_evidence[np.isin(y, [2, 5, 20])] = ll_h
        s["log_evidence"] = log_evidence
        
    # Function to compute total log likelihood across sequences for given transition hazard lambda
    # Transition matrix: [[1 - lambda, lambda], [lambda, 1 - lambda]]
    def compute_total_log_lik(lam, seq_subset):
        if lam <= 1e-6 or lam >= 0.5: return -1e12
        p00 = 1.0 - lam
        p01 = lam
        log_A00 = np.log(p00)
        log_A01 = np.log(p01)
        
        total_ll = 0.0
        for s in seq_subset:
            lev = s["log_evidence"]
            T = len(lev)
            # Forward pass in log-space:
            # alpha[t, 0] = log P(y_{1:t}, S_t=0)
            # alpha[t, 1] = log P(y_{1:t}, S_t=1)
            # Start uniform: log(0.5) = -0.6931
            a0 = -0.693147
            a1 = -0.693147 + lev[0]
            
            for t in range(1, T):
                # prev to 0: logsumexp(a0 + log_A00, a1 + log_A01)
                m0 = max(a0 + log_A00, a1 + log_A01)
                next_a0 = m0 + np.log(np.exp(a0 + log_A00 - m0) + np.exp(a1 + log_A01 - m0))
                
                # prev to 1: logsumexp(a0 + log_A01, a1 + log_A00) + lev[t]
                m1 = max(a0 + log_A01, a1 + log_A00)
                next_a1 = m1 + np.log(np.exp(a0 + log_A01 - m1) + np.exp(a1 + log_A00 - m1)) + lev[t]
                
                a0 = next_a0
                a1 = next_a1
                
            m = max(a0, a1)
            total_ll += m + np.log(np.exp(a0 - m) + np.exp(a1 - m))
        return total_ll

    # Run for each condition
    results = []
    for cond in ["all", "stable", "drift", "shared_change", "short_history"]:
        if cond == "all":
            seq_sub = sequences
        else:
            seq_sub = [s for s in sequences if s["cond"] == cond]
            
        # Optimize lambda via Brent method on (-log_lik)
        res = minimize_scalar(lambda l: -compute_total_log_lik(l, seq_sub), bounds=(0.001, 0.15), method='bounded')
        opt_lam = res.x
        
        # Calculate mean duration, half-life
        mean_dur = 1.0 / opt_lam
        half_life = np.log(2.0) / opt_lam
        
        # Now run Viterbi with opt_lam to extract empirical dwell times
        log_A00 = np.log(1.0 - opt_lam)
        log_A01 = np.log(opt_lam)
        
        dwells_0 = []
        dwells_1 = []
        n_transitions = []
        
        for s in seq_sub:
            lev = s["log_evidence"]
            T = len(lev)
            V0 = np.zeros(T)
            V1 = np.zeros(T)
            ptr0 = np.zeros(T, dtype=int)
            ptr1 = np.zeros(T, dtype=int)
            
            V0[0] = -0.693147
            V1[0] = -0.693147 + lev[0]
            
            for t in range(1, T):
                # To state 0:
                if V0[t-1] + log_A00 >= V1[t-1] + log_A01:
                    V0[t] = V0[t-1] + log_A00
                    ptr0[t] = 0
                else:
                    V0[t] = V1[t-1] + log_A01
                    ptr0[t] = 1
                    
                # To state 1:
                if V0[t-1] + log_A01 >= V1[t-1] + log_A00:
                    V1[t] = V0[t-1] + log_A01 + lev[t]
                    ptr1[t] = 0
                else:
                    V1[t] = V1[t-1] + log_A00 + lev[t]
                    ptr1[t] = 1
                    
            # Backtrack
            path = np.zeros(T, dtype=int)
            path[T-1] = 0 if V0[T-1] >= V1[T-1] else 1
            for t in range(T - 2, -1, -1):
                path[t] = ptr0[t+1] if path[t+1] == 0 else ptr1[t+1]
                
            diffs = np.diff(path)
            n_transitions.append(np.count_nonzero(diffs))
            
            # Run lengths
            curr_s = path[0]
            curr_l = 1
            for st in path[1:]:
                if st == curr_s:
                    curr_l += 1
                else:
                    if curr_s == 0: dwells_0.append(curr_l)
                    else: dwells_1.append(curr_l)
                    curr_s = st
                    curr_l = 1
            if curr_s == 0: dwells_0.append(curr_l)
            else: dwells_1.append(curr_l)
            
        results.append({
            "condition": cond,
            "optimal_hazard": opt_lam,
            "mean_duration_model": mean_dur,
            "half_life_model": half_life,
            "viterbi_mean_dwell_0": np.mean(dwells_0) if dwells_0 else np.nan,
            "viterbi_median_dwell_0": np.median(dwells_0) if dwells_0 else np.nan,
            "viterbi_mean_dwell_1": np.mean(dwells_1) if dwells_1 else np.nan,
            "viterbi_median_dwell_1": np.median(dwells_1) if dwells_1 else np.nan,
            "mean_transitions_per_machine": np.mean(n_transitions),
            "pct_machines_with_zero_trans": (np.array(n_transitions) == 0).mean() * 100
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res

print("=== FITTING ADAPTATION VALIDATION SET ===")
val_res = analyze_dataset("adaptation-participants/public/validation")

print("\n=== FITTING ADAPTATION TRAINING SET ===")
tr_res = analyze_dataset("adaptation-participants/public/train")
