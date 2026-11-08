"""Cutoff-safe feature engineering with continuous drift trend slopes and jackpot hazard.
EXP-016: Temporal Drift & Volatility Specialist.
"""
from collections import defaultdict
from slot_forecasting.schema import PAYOUTS

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def build_advanced_features(query, sequence, peer_sequences, cutoff, base_forecaster=None):
    """Extract strictly cutoff-safe features available before `cutoff`."""
    feats = []
    
    # 1. Static features
    m_type = query.machine_type
    sig = int(query.signal)
    feats.append(1.0 if m_type == "steady" else 0.0)
    feats.append(1.0 if m_type == "balanced" else 0.0)
    feats.append(1.0 if m_type == "jackpot" else 0.0)
    feats.append(float(sig))
    feats.append(1.0 if sig == -1 else 0.0)
    feats.append(1.0 if sig == 0 else 0.0)
    feats.append(1.0 if sig == 1 else 0.0)
    
    # Interaction: type x signal (9 features)
    for t_name in ("steady", "balanced", "jackpot"):
        for s_val in (-1, 0, 1):
            feats.append(1.0 if (m_type == t_name and sig == s_val) else 0.0)
            
    # History length so far
    n_obs = len(sequence)
    feats.append(float(n_obs))
    feats.append(1.0 if n_obs == 64 else 0.0)
    
    # 2. Recency decay counts for half-lives [2, 4, 6, 12, 24, 48]
    for hl in (2.0, 4.0, 6.0, 12.0, 24.0, 48.0):
        counts = [0.0] * 5
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                w = 2.0 ** (-age / hl)
                counts[PAYOUT_TO_IDX[p]] += w
        tot = sum(counts) + 1e-6
        for c in counts:
            feats.append(c / tot)
        feats.append(sum(counts[1:]) / tot)
        
    # 3. Rolling window win rates, payout sums, and non-zero counts
    win_rates = {}
    for win in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        window_payouts = [sequence[t] for t in range(cutoff - win, cutoff) if t in sequence]
        if window_payouts:
            w_rate = sum(1 for p in window_payouts if p > 0) / len(window_payouts)
            m_payout = sum(window_payouts) / len(window_payouts)
            p20_count = sum(1 for p in window_payouts if p == 20)
            p5_count = sum(1 for p in window_payouts if p == 5)
            obs_ratio = len(window_payouts) / win
        else:
            w_rate = 0.5
            m_payout = 0.5
            p20_count = 0
            p5_count = 0
            obs_ratio = 0.0
        win_rates[win] = w_rate
        feats.extend([w_rate, m_payout, float(p20_count), float(p5_count), obs_ratio])
        
    # 4. Lag sequence features: lags 1 to 8
    lag_payouts = {}
    for lag in range(1, 9):
        p = sequence.get(cutoff - lag, -1)
        lag_payouts[lag] = p
        feats.append(float(p))
        for target_p in (0, 1, 2, 5, 20):
            feats.append(1.0 if p == target_p else 0.0)
            
    # Pair interaction features: (lag1, lag2), (lag1, lag4), (lag1, lag8), (lag2, lag4), (lag4, lag8)
    def cat(v):
        return -1 if v == -1 else (0 if v == 0 else (1 if v == 1 else 2))
    for l1, l2 in ((1, 2), (1, 4), (1, 8), (2, 4), (4, 8)):
        c1, c2 = cat(lag_payouts[l1]), cat(lag_payouts[l2])
        for v1 in (0, 1, 2):
            for v2 in (0, 1, 2):
                feats.append(1.0 if (c1 == v1 and c2 == v2) else 0.0)
                
    # Streaks
    streak_0 = 0
    t = cutoff - 1
    while t in sequence and sequence[t] == 0:
        streak_0 += 1
        t -= 1
    feats.append(float(streak_0))
    
    streak_win = 0
    t = cutoff - 1
    while t in sequence and sequence[t] > 0:
        streak_win += 1
        t -= 1
    feats.append(float(streak_win))
    
    # Ticks since last win (>0) and last big win (>=5)
    last_win_age = 50.0
    last_big_win_age = 100.0
    for tick in sorted(sequence.keys(), reverse=True):
        if sequence[tick] > 0 and last_win_age == 50.0:
            last_win_age = float(cutoff - 1 - tick)
        if sequence[tick] >= 5 and last_big_win_age == 100.0:
            last_big_win_age = float(cutoff - 1 - tick)
        if last_win_age < 50.0 and last_big_win_age < 100.0:
            break
    feats.extend([last_win_age, last_big_win_age])
    
    # 5. Peer machine features
    peer_win_rate_8 = 0.5
    peer_win_rate_24 = 0.5
    if peer_sequences:
        for p_win in (4, 8, 16, 24):
            peer_recent = []
            for p_seq in peer_sequences:
                for t in range(cutoff - p_win, cutoff):
                    if t in p_seq:
                        peer_recent.append(p_seq[t])
            if peer_recent:
                p_win_rate = sum(1 for p in peer_recent if p > 0) / len(peer_recent)
                p_mean_pay = sum(peer_recent) / len(peer_recent)
            else:
                p_win_rate = 0.5
                p_mean_pay = 0.5
            if p_win in (4, 8, 16):
                feats.extend([p_win_rate, p_mean_pay])
            if p_win == 8:
                peer_win_rate_8 = p_win_rate
            elif p_win == 24:
                peer_win_rate_24 = p_win_rate
            
        peer_t1 = [p_seq.get(cutoff - 1, -1) for p_seq in peer_sequences]
        peer_t1_wins = sum(1 for p in peer_t1 if p > 0)
        feats.append(float(peer_t1_wins))
    else:
        feats.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
        
    # 6. Baseline prior features
    if base_forecaster is not None:
        key = f"{m_type}|{sig}"
        context_prior = base_forecaster.context_priors.get(key, base_forecaster.global_prior)
        feats.extend(context_prior)
        pr_counts = [10.0 * v for v in context_prior]
        for tick, p in sequence.items():
            age = cutoff - 1 - tick
            if age >= 0:
                pr_counts[PAYOUT_TO_IDX[p]] += 2.0 ** (-age / 16.0)
        pr_tot = sum(pr_counts)
        feats.extend([v / pr_tot for v in pr_counts])
        
    # 7. Continuous Drift Trend Slopes and Jackpot Hazard (EXP-016)
    slope_16_48 = win_rates.get(16, 0.5) - win_rates.get(48, 0.5)
    slope_8_24 = win_rates.get(8, 0.5) - win_rates.get(24, 0.5)
    peer_slope = peer_win_rate_8 - peer_win_rate_24
    jackpot_hazard = (1.0 if m_type == "jackpot" else 0.0) * min(streak_0 / 30.0, 1.5)
    feats.extend([slope_16_48, slope_8_24, peer_slope, jackpot_hazard])
        
    return feats
