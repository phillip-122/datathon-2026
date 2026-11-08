import numpy as np
import pandas as pd
from scipy.special import logsumexp

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

def fit_hmm_for_split(data_dir):
    print(f"\n=======================================================")
    print(f" HMM REGIME DURATION & HAZARD ANALYSIS: {data_dir}")
    print(f"=======================================================")
    hist = pd.read_csv(f"{data_dir}/history.csv")
    queries = pd.read_csv(f"{data_dir}/queries.csv")
    conds = pd.read_csv(f"{data_dir}/conditions.csv")
    
    q_meta = queries.merge(conds, on="query_id")
    m_info = q_meta.set_index(["episode_id", "machine_id"])[["machine_type", "condition"]].to_dict("index")
    
    # We will run Baum-Welch on the sequences
    # For each machine sequence of payouts y_0, ..., y_{T-1}
    # Observations:
    # y = 0: loss
    # y = 1: 1x win
    # y in {2, 5, 20}: high win
    
    # Let's map payouts to discrete symbols: 0 -> loss (0), 1 -> 1x (1), >=2 -> high (2)
    hist["sym"] = hist["payout"].map(lambda p: 0 if p == 0 else (1 if p == 1 else 2))
    
    # We can fit HMM per condition, and overall
    all_dwell_0 = []
    all_dwell_1 = []
    
    condition_stats = {}
    
    for cond in ["all", "stable", "drift", "shared_change", "short_history"]:
        # Collect sequences
        seqs = []
        mt_list = []
        
        grouped = hist.groupby(["episode_id", "machine_id"])
        for (ep, m), group in grouped:
            if (ep, m) not in m_info: continue
            info = m_info[(ep, m)]
            if cond != "all" and info["condition"] != cond:
                continue
            group = group.sort_values("tick")
            seqs.append(group["sym"].values)
            mt_list.append(info["machine_type"])
            
        if not seqs: continue
        
        # Emission matrix for each machine type:
        # B[m_type, state, sym]: P(sym | state, m_type)
        # We can estimate P(loss) empirically, and use exact physics for P(1x | win) and P(high | win)
        # Alternatively, we can let Baum-Welch estimate transition probabilities A:
        # A = [[1 - lambda_01, lambda_01], [lambda_10, 1 - lambda_10]]
        
        # Let's initialize transition matrix
        # Initial guess: hazard ~ 0.01 per tick (mean duration ~ 100 ticks)
        p01 = 0.015
        p10 = 0.015
        pi = np.array([0.5, 0.5])
        
        # Precompute emission log-probs for each sequence given machine type
        # In validation, mean p_win is ~0.40, p_loss ~0.60
        # Let's compute empirical p_win per machine type from all data
        emp_win_rate = {}
        for mt in ['steady', 'balanced', 'jackpot']:
            sub_m = [seqs[i] for i in range(len(seqs)) if mt_list[i] == mt]
            if sub_m:
                all_obs = np.concatenate(sub_m)
                emp_win_rate[mt] = (all_obs > 0).mean()
            else:
                emp_win_rate[mt] = 0.40
                
        # For each sequence, build log_B: shape (T, 2)
        log_B_list = []
        for seq, mt in zip(seqs, mt_list):
            p_win = emp_win_rate[mt]
            p_loss = 1.0 - p_win
            r0, r1 = REGIME_RATIOS[mt]
            
            p_1_given_win_0 = r0 / (1.0 + r0)
            p_h_given_win_0 = 1.0 / (1.0 + r0)
            
            p_1_given_win_1 = r1 / (1.0 + r1)
            p_h_given_win_1 = 1.0 / (1.0 + r1)
            
            # Emissions: [loss, 1x, high]
            # State 0:
            b0 = np.array([p_loss, p_win * p_1_given_win_0, p_win * p_h_given_win_0])
            # State 1:
            b1 = np.array([p_loss, p_win * p_1_given_win_1, p_win * p_h_given_win_1])
            
            log_b0 = np.log(b0)
            log_b1 = np.log(b1)
            
            # (T, 2)
            log_B = np.column_stack([log_b0[seq], log_b1[seq]])
            log_B_list.append(log_B)
            
        # Baum-Welch (EM) iterations on A = [[1-p01, p01], [p10, 1-p10]]
        for it in range(30):
            total_trans_00 = 0.0
            total_trans_01 = 0.0
            total_trans_10 = 0.0
            total_trans_11 = 0.0
            
            log_A = np.log(np.array([[1.0 - p01, p01], [p10, 1.0 - p10]]))
            log_pi = np.log(pi)
            
            for log_B in log_B_list:
                T = len(log_B)
                # Forward pass
                log_alpha = np.zeros((T, 2))
                log_alpha[0] = log_pi + log_B[0]
                for t in range(1, T):
                    log_alpha[t] = log_B[t] + logsumexp(log_alpha[t-1][:, None] + log_A, axis=0)
                    
                # Backward pass
                log_beta = np.zeros((T, 2))
                log_beta[T-1] = 0.0
                for t in range(T - 2, -1, -1):
                    log_beta[t] = logsumexp(log_A + log_B[t+1] + log_beta[t+1], axis=1)
                    
                # Log likelihood
                log_lik = logsumexp(log_alpha[T-1])
                
                # Posterior transitions: xi[t, i, j]
                # log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j] - log_lik
                for t in range(T - 1):
                    log_xi = log_alpha[t][:, None] + log_A + log_B[t+1][None, :] + log_beta[t+1][None, :] - log_lik
                    xi = np.exp(log_xi)
                    total_trans_00 += xi[0, 0]
                    total_trans_01 += xi[0, 1]
                    total_trans_10 += xi[1, 0]
                    total_trans_11 += xi[1, 1]
                    
            # M-step: update transition hazards
            p01 = total_trans_01 / (total_trans_00 + total_trans_01)
            p10 = total_trans_10 / (total_trans_10 + total_trans_11)
            
        # Viterbi decoding to get most probable path and empirical dwell times
        dwell_0_cond = []
        dwell_1_cond = []
        num_transitions_list = []
        
        log_A = np.log(np.array([[1.0 - p01, p01], [p10, 1.0 - p10]]))
        log_pi = np.log(pi)
        
        for log_B in log_B_list:
            T = len(log_B)
            # Viterbi
            V = np.zeros((T, 2))
            ptr = np.zeros((T, 2), dtype=int)
            V[0] = log_pi + log_B[0]
            for t in range(1, T):
                trans_prob = V[t-1][:, None] + log_A
                ptr[t] = np.argmax(trans_prob, axis=0)
                V[t] = log_B[t] + np.max(trans_prob, axis=0)
                
            best_path = np.zeros(T, dtype=int)
            best_path[T-1] = np.argmax(V[T-1])
            for t in range(T - 2, -1, -1):
                best_path[t] = ptr[t+1, best_path[t+1]]
                
            # Count run lengths (dwell times)
            diffs = np.diff(best_path)
            num_transitions_list.append(np.count_nonzero(diffs))
            
            # Extract run lengths
            runs = []
            curr_state = best_path[0]
            curr_len = 1
            for s in best_path[1:]:
                if s == curr_state:
                    curr_len += 1
                else:
                    runs.append((curr_state, curr_len))
                    curr_state = s
                    curr_len = 1
            runs.append((curr_state, curr_len))
            
            for s, r_len in runs:
                if s == 0:
                    dwell_0_cond.append(r_len)
                else:
                    dwell_1_cond.append(r_len)
                    
        # Half-life: ln(2) / lambda
        hl_0 = np.log(2.0) / p01 if p01 > 0 else np.inf
        hl_1 = np.log(2.0) / p10 if p10 > 0 else np.inf
        
        # Expected duration: 1 / lambda
        dur_0 = 1.0 / p01 if p01 > 0 else np.inf
        dur_1 = 1.0 / p10 if p10 > 0 else np.inf
        
        condition_stats[cond] = {
            "n_machines": len(seqs),
            "hazard_0_to_1": p01,
            "hazard_1_to_0": p10,
            "mean_dur_0 (model)": dur_0,
            "mean_dur_1 (model)": dur_1,
            "half_life_0 (ticks)": hl_0,
            "half_life_1 (ticks)": hl_1,
            "mean_dwell_0 (viterbi)": np.mean(dwell_0_cond) if dwell_0_cond else np.nan,
            "median_dwell_0 (viterbi)": np.median(dwell_0_cond) if dwell_0_cond else np.nan,
            "mean_dwell_1 (viterbi)": np.mean(dwell_1_cond) if dwell_1_cond else np.nan,
            "median_dwell_1 (viterbi)": np.median(dwell_1_cond) if dwell_1_cond else np.nan,
            "mean_transitions_per_machine": np.mean(num_transitions_list)
        }
        
    res_df = pd.DataFrame(condition_stats).T
    print(res_df.to_string())
    return res_df

fit_hmm_for_split("adaptation-participants/public/train")
fit_hmm_for_split("adaptation-participants/public/validation")
