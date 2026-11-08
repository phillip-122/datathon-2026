"""Continuous Regime Recognition & Adaptive Recency Features (Experiments A & B).

Strictly cutoff-safe features extracting sequence and regime dynamics:
1. Recent payout entropy across multiple rolling horizons (8, 16, 32 ticks).
2. Short-versus-long volatility ratios and differences (8 vs 32, 16 vs 64).
3. Streak asymmetry: current win/loss streaks and ratio of win to loss sequence lengths.
4. Distribution divergence: Total Variation distance between recent (4, 8, 16) and older history.
5. Transition consistency: Empirical transition dynamics (loss -> win rebound, win -> win momentum)
   comparing recent 16 ticks to historical ticks.
6. Adaptive recency: EWMA difference vectors (2-8, 4-16, 6-48) and instability gating.
"""

import math
import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}

def compute_entropy(payout_list):
    """Shannon entropy of empirical payout distribution."""
    if not payout_list:
        return 1.6094  # log(5) uniform maximum entropy
    counts = [0] * 5
    for p in payout_list:
        counts[PAYOUT_TO_IDX.get(p, 0)] += 1
    total = len(payout_list)
    ent = 0.0
    for c in counts:
        if c > 0:
            pr = c / total
            ent -= pr * math.log(pr)
    return ent

def compute_empirical_dist(payout_list, smoothing=0.1):
    """Smoothed empirical probability distribution over 5 payouts."""
    counts = [smoothing] * 5
    for p in payout_list:
        counts[PAYOUT_TO_IDX.get(p, 0)] += 1.0
    total = sum(counts)
    return [c / total for c in counts]

def tv_divergence(dist_a, dist_b):
    """Total variation distance between two 5-class distributions."""
    return 0.5 * sum(abs(a - b) for a, b in zip(dist_a, dist_b))

def build_regime_features(sequence, cutoff):
    """Extract cutoff-safe continuous regime and adaptive memory descriptors."""
    past = {t: p for t, p in sequence.items() if t < cutoff}
    feats = []
    
    # 1. Recent Payout Entropy (8, 16, 32 ticks)
    for win in (8, 16, 32):
        window_p = [past[t] for t in range(cutoff - win, cutoff) if t in past]
        feats.append(compute_entropy(window_p))
        
    # 2. Short vs Long Volatility
    w8 = [past[t] for t in range(cutoff - 8, cutoff) if t in past]
    w16 = [past[t] for t in range(cutoff - 16, cutoff) if t in past]
    w32 = [past[t] for t in range(cutoff - 32, cutoff) if t in past]
    w64 = [past[t] for t in range(cutoff - 64, cutoff) if t in past]
    all_p = list(past.values())
    
    std8 = float(np.std(w8)) if len(w8) >= 2 else 0.5
    std16 = float(np.std(w16)) if len(w16) >= 2 else 0.5
    std32 = float(np.std(w32)) if len(w32) >= 2 else 0.5
    std64 = float(np.std(w64)) if len(w64) >= 2 else 0.5
    std_all = float(np.std(all_p)) if len(all_p) >= 2 else 0.5
    
    # Ratios and differences
    feats.extend([
        std8 / (std32 + 1e-4),
        std8 - std32,
        std16 / (std64 + 1e-4),
        std16 - std64,
        std16 / (std_all + 1e-4),
    ])
    
    # 3. Streak Asymmetry
    # Length of sequences of wins vs losses in past 48 ticks
    past_sorted = sorted([t for t in past if t >= cutoff - 48])
    win_streaks = []
    loss_streaks = []
    cur_streak = 0
    cur_is_win = None
    
    for t in past_sorted:
        is_w = past[t] > 0
        if cur_is_win is None:
            cur_is_win = is_w
            cur_streak = 1
        elif is_w == cur_is_win:
            cur_streak += 1
        else:
            if cur_is_win:
                win_streaks.append(cur_streak)
            else:
                loss_streaks.append(cur_streak)
            cur_is_win = is_w
            cur_streak = 1
    if cur_streak > 0 and cur_is_win is not None:
        if cur_is_win:
            win_streaks.append(cur_streak)
        else:
            loss_streaks.append(cur_streak)
            
    mean_w_streak = float(np.mean(win_streaks)) if win_streaks else 1.0
    mean_l_streak = float(np.mean(loss_streaks)) if loss_streaks else 1.0
    streak_asym_ratio = (mean_w_streak + 0.1) / (mean_l_streak + 0.1)
    max_l_streak = float(max(loss_streaks)) if loss_streaks else 0.0
    feats.extend([mean_w_streak, mean_l_streak, streak_asym_ratio, max_l_streak])
    
    # 4. Distribution Divergence: recent (4, 8, 16) vs older
    dist_older = compute_empirical_dist([past[t] for t in past if t < cutoff - 16])
    dist_all = compute_empirical_dist(all_p)
    
    dist_4 = compute_empirical_dist([past[t] for t in range(cutoff - 4, cutoff) if t in past])
    dist_8 = compute_empirical_dist([past[t] for t in range(cutoff - 8, cutoff) if t in past])
    dist_16 = compute_empirical_dist([past[t] for t in range(cutoff - 16, cutoff) if t in past])
    
    tv_4_older = tv_divergence(dist_4, dist_older)
    tv_8_older = tv_divergence(dist_8, dist_older)
    tv_16_older = tv_divergence(dist_16, dist_older)
    tv_8_all = tv_divergence(dist_8, dist_all)
    
    feats.extend([tv_4_older, tv_8_older, tv_16_older, tv_8_all])
    
    # 5. Transition Consistency: Rebound and Momentum
    # Estimate P(win_t | loss_t-1) and P(win_t | win_t-1)
    def transition_rates(ticks_subset):
        sorted_ticks = sorted(ticks_subset)
        loss_to_win = 0
        loss_total = 0
        win_to_win = 0
        win_total = 0
        for i in range(1, len(sorted_ticks)):
            t_prev, t_curr = sorted_ticks[i - 1], sorted_ticks[i]
            if t_curr == t_prev + 1:  # contiguous ticks
                if past[t_prev] == 0:
                    loss_total += 1
                    if past[t_curr] > 0:
                        loss_to_win += 1
                else:
                    win_total += 1
                    if past[t_curr] > 0:
                        win_to_win += 1
        p_rebound = (loss_to_win + 0.5) / (loss_total + 1.0)
        p_momentum = (win_to_win + 0.5) / (win_total + 1.0)
        return p_rebound, p_momentum
        
    recent_ticks = [t for t in past if t >= cutoff - 16]
    older_ticks = [t for t in past if t < cutoff - 16]
    
    r_reb, r_mom = transition_rates(recent_ticks)
    o_reb, o_mom = transition_rates(older_ticks)
    
    feats.extend([r_reb, r_mom, r_reb - o_reb, r_mom - o_mom])
    
    # 6. Adaptive Recency: EWMA Differences and Sigmoid Instability Gating
    # Compute EWMA distributions for half-lives 2, 8, 16, 48
    def ewma_dist(hl):
        counts = [0.0] * 5
        for t, p in past.items():
            age = cutoff - 1 - t
            if age >= 0:
                w = 2.0 ** (-age / hl)
                counts[PAYOUT_TO_IDX.get(p, 0)] += w
        tot = sum(counts) + 1e-6
        return [c / tot for c in counts]
        
    e2 = ewma_dist(2.0)
    e8 = ewma_dist(8.0)
    e16 = ewma_dist(16.0)
    e48 = ewma_dist(48.0)
    
    # Difference vectors (payout 0 and win rate)
    diff_2_8_p0 = e2[0] - e8[0]
    diff_2_8_win = sum(e2[1:]) - sum(e8[1:])
    diff_4_16_p0 = e2[0] - e16[0]
    diff_6_48_win = sum(e2[1:]) - sum(e48[1:])
    
    # Instability gating sigmoid feature
    instability_gate = 1.0 / (1.0 + math.exp(-10.0 * (tv_8_older - 0.25)))
    
    feats.extend([diff_2_8_p0, diff_2_8_win, diff_4_16_p0, diff_6_48_win, instability_gate])
    
    return feats
