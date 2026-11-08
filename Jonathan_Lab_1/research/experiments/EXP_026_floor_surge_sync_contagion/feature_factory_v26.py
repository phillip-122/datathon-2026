"""EXP-026 Feature Factory: Databricks Floor Dynamics (Surge, Sync & Contagion Engine).

Implements the 4 newly discovered empirical dynamics from Databricks Deep Dives:
1. Priority 1: Shared_Change Transient Surge Window (peaking at ticks 210-219, +55.7% payout lift).
2. Priority 2: Floor Synchronization Level (simultaneous peer wins at cutoff-1; 0->4 wins drops zero prob by 8.2 pp).
3. Priority 3: Universal Drought Length & Non-Linear Hazard (1-3: ~50%, 4-5: ~40%, 6+: <30%, 9+: <15% collapse).
4. Priority 4: Jackpot Contagion Dynamics (peer 5x/20x in last 3 ticks elevates win rate by +1.6 pp, shifting toward 1x/2x).
"""
from collections import Counter, defaultdict
import numpy as np

from slot_forecasting.schema import PAYOUTS
from lag_model.features import build_advanced_features as build_base_lag_features

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}
MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}

# Stable condition nominal empirical priors from 50,000+ spins
STABLE_BASE_RATES = {
    "steady": np.array([0.4220, 0.3565, 0.1751, 0.0455, 0.0009], dtype=np.float32),
    "balanced": np.array([0.5926, 0.1435, 0.1823, 0.0697, 0.0119], dtype=np.float32),
    "jackpot": np.array([0.7404, 0.0518, 0.0648, 0.1150, 0.0280], dtype=np.float32)
}

# Databricks Universal Drought Escape Hazard Table (Page 5)
DROUGHT_HAZARDS = {
    0: 0.00,
    1: 0.470,
    2: 0.450,
    3: 0.536,
    4: 0.372,
    5: 0.464,
    6: 0.318,
    7: 0.278,
    8: 0.168,
    9: 0.134,
    10: 0.066,
    11: 0.052,
    12: 0.045,
    13: 0.022,
    14: 0.018,
    15: 0.020
}

DROUGHT_ESCAPE_PAYOUTS = {
    0: 0.00,
    1: 2.48,
    2: 2.67,
    3: 2.92,
    4: 2.98,
    5: 3.72,
    6: 3.74,
    7: 2.69,
    8: 4.60,
    9: 3.87,
    10: 3.65,
    11: 5.22,
    12: 3.70,
    13: 6.33,
    14: 1.67,
    15: 3.94
}

def compute_cutoff_safe_archetype(sequence, cutoff):
    past = {t: p for t, p in sequence.items() if t < cutoff}
    ticks = sorted(past.keys())
    if len(ticks) < 4:
        return 0.0, 1.0, 0, 0, 0.4, 0.4
    
    win_after_win = []
    win_after_loss = []
    for i in range(1, len(ticks)):
        t_prev, t_curr = ticks[i-1], ticks[i]
        if t_curr == t_prev + 1:
            p_prev = 1 if past[t_prev] > 0 else 0
            p_curr = 1 if past[t_curr] > 0 else 0
            if p_prev == 1:
                win_after_win.append(p_curr)
            else:
                win_after_loss.append(p_curr)
                
    nw = len(win_after_win)
    nl = len(win_after_loss)
    if nw < 2 or nl < 2:
        return 0.0, 1.0, nw, nl, 0.4, 0.4
        
    p_w_w = (sum(win_after_win) + 2.0 * 0.4) / (nw + 2.0)
    p_w_l = (sum(win_after_loss) + 2.0 * 0.4) / (nl + 2.0)
    mom = p_w_w - p_w_l
    se = float(np.sqrt(max(1e-6, p_w_w * (1 - p_w_w) / nw + p_w_l * (1 - p_w_l) / nl)))
    return mom, se, nw, nl, p_w_w, p_w_l

def get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated", threshold=0.05):
    tot = nw + nl
    if mode == "confidence_gated":
        if tot < 20 or abs(mom) <= 1.5 * se: return 0
        return 1 if mom > 0 else -1
    return 0

def extract_soft_thresholded_slopes(past, peer_past, cutoff, kappa=1.50):
    p_16 = [past[t] for t in range(cutoff - 16, cutoff) if t in past]
    p_48 = [past[t] for t in range(cutoff - 48, cutoff) if t in past]
    wr_16 = sum(1 for p in p_16 if p > 0) / len(p_16) if p_16 else 0.5
    wr_48 = sum(1 for p in p_48 if p > 0) / len(p_48) if p_48 else 0.5
    slope_16_48 = wr_16 - wr_48
    
    p_8 = [past[t] for t in range(cutoff - 8, cutoff) if t in past]
    p_24 = [past[t] for t in range(cutoff - 24, cutoff) if t in past]
    wr_8 = sum(1 for p in p_8 if p > 0) / len(p_8) if p_8 else 0.5
    wr_24 = sum(1 for p in p_24 if p > 0) / len(p_24) if p_24 else 0.5
    slope_8_24 = wr_8 - wr_24
    
    peer_8 = []
    peer_24 = []
    for p_seq in peer_past:
        for t in range(cutoff - 8, cutoff):
            if t in p_seq: peer_8.append(p_seq[t])
        for t in range(cutoff - 24, cutoff):
            if t in p_seq: peer_24.append(p_seq[t])
    peer_wr_8 = sum(1 for p in peer_8 if p > 0) / len(peer_8) if peer_8 else 0.5
    peer_wr_24 = sum(1 for p in peer_24 if p > 0) / len(peer_24) if peer_24 else 0.5
    peer_slope = peer_wr_8 - peer_wr_24
    
    if kappa <= 0.0:
        return [float(slope_16_48), float(slope_8_24), float(peer_slope)]
        
    n_16 = max(1, len(p_16))
    n_48 = max(1, len(p_48))
    se_16_48 = np.sqrt(0.24 / n_16 + 0.24 / n_48)
    thresh_16_48 = kappa * se_16_48
    shrunk_16_48 = np.sign(slope_16_48) * max(0.0, abs(slope_16_48) - thresh_16_48) if abs(slope_16_48) > thresh_16_48 else 0.0

    n_8 = max(1, len(p_8))
    n_24 = max(1, len(p_24))
    se_8_24 = np.sqrt(0.24 / n_8 + 0.24 / n_24)
    thresh_8_24 = kappa * se_8_24
    shrunk_8_24 = np.sign(slope_8_24) * max(0.0, abs(slope_8_24) - thresh_8_24) if abs(slope_8_24) > thresh_8_24 else 0.0

    n_p8 = max(1, len(peer_8))
    n_p24 = max(1, len(peer_24))
    se_peer = np.sqrt(0.24 / n_p8 + 0.24 / n_p24)
    thresh_peer = kappa * se_peer
    shrunk_peer = np.sign(peer_slope) * max(0.0, abs(peer_slope) - thresh_peer) if abs(peer_slope) > thresh_peer else 0.0

    return [float(shrunk_16_48), float(shrunk_8_24), float(shrunk_peer)]

def extract_databricks_coupling_and_hazard_features(query, past, peer_past, cutoff):
    """Implement Databricks Discoveries: Priority 1 Coupling, Priority 2 Cold Freeze, Priority 3 Shrinkage."""
    m_type = query.machine_type
    
    # 1. Peer Payout Coupling in Last 40 Ticks
    peer_payouts_40 = []
    for s in peer_past:
        p40 = [s[t] for t in range(cutoff - 40, cutoff) if t in s]
        peer_payouts_40.append(sum(p40) if p40 else 40.0 * 1.045)
    total_peer_payout_40 = sum(peer_payouts_40)
    z_peer = (total_peer_payout_40 - 330.0) / 70.0
    
    if m_type in ("balanced", "jackpot"):
        coupling_feature = -float(z_peer ** 2)
        coupling_linear = 0.0
    else:
        coupling_feature = 0.0
        coupling_linear = float(z_peer)
        
    # 2. Jackpot Cold Streak Piecewise Hazard
    streak_0 = 0
    t = cutoff - 1
    while t in past and past[t] == 0:
        streak_0 += 1
        t -= 1
        
    if m_type == "jackpot":
        if streak_0 >= 9:
            jackpot_piecewise_hazard = -2.0
        elif 3 <= streak_0 <= 8:
            jackpot_piecewise_hazard = 1.0
        else:
            jackpot_piecewise_hazard = 0.0
    else:
        jackpot_piecewise_hazard = 0.0
        
    jackpot_linear_hazard = (1.0 if m_type == "jackpot" else 0.0) * min(streak_0 / 30.0, 1.5)
    
    # 3. Small-Sample Shrinkage (N=64)
    past_spins = [past[t] for t in past.keys() if t < cutoff]
    n_obs = len(past_spins)
    base_prior = STABLE_BASE_RATES[m_type]
    
    if n_obs > 0:
        counts = Counter(past_spins)
        emp_freq = np.array([counts[p] / n_obs for p in PAYOUTS], dtype=np.float32)
    else:
        emp_freq = base_prior
        
    alpha = float(n_obs / (n_obs + 150.0))
    shrunk_freq = alpha * emp_freq + (1.0 - alpha) * base_prior
    
    return [
        float(coupling_feature),
        float(coupling_linear),
        float(jackpot_piecewise_hazard),
        float(jackpot_linear_hazard),
        float(total_peer_payout_40 / (7.0 * 40.0))
    ] + [float(v) for v in shrunk_freq]

def extract_databricks_deep_dive_features(query, past, peer_past, cutoff):
    """Implement 4 newly reported Databricks Deep Dive Dynamics."""
    m_type = query.machine_type
    
    # -------------------------------------------------------------------------
    # Priority 1: Shared_Change Transient Surge Window (peaking ticks 210-219)
    # -------------------------------------------------------------------------
    in_surge_window = 1.0 if (210 <= cutoff <= 219) else 0.0
    surge_bell = float(np.exp(-0.5 * ((cutoff - 217.0) / 3.5) ** 2))
    surge_ramp = float(max(0.0, min(1.0, (cutoff - 200.0) / 17.0)) if cutoff <= 219 else max(0.0, 1.0 - (cutoff - 219.0) / 5.0))
    
    # Measure floor peer variance to estimate shared_change probability
    peer_wrs_24 = []
    for s in peer_past:
        wins_24 = sum(1 for t in range(cutoff - 24, cutoff) if t in s and s[t] > 0)
        tot_24 = sum(1 for t in range(cutoff - 24, cutoff) if t in s)
        peer_wrs_24.append(wins_24 / tot_24 if tot_24 > 0 else 0.4)
    peer_dispersion = float(np.std(peer_wrs_24)) if len(peer_wrs_24) > 1 else 0.0
    shared_change_surge_signal = surge_bell * (1.0 if peer_dispersion > 0.12 else 0.5)

    # -------------------------------------------------------------------------
    # Priority 2: Synchronization Level (simultaneous wins at cutoff-1)
    # -------------------------------------------------------------------------
    t_prev = cutoff - 1
    sync_level_prev = sum(1 for s in peer_past if s.get(t_prev, 0) > 0)
    
    sync_levels_3 = []
    for dt in range(1, 4):
        t_lag = cutoff - dt
        cnt = sum(1 for s in peer_past if s.get(t_lag, 0) > 0)
        sync_levels_3.append(cnt)
    sync_mean_3 = float(np.mean(sync_levels_3))
    
    # SQL sync buckets from Databricks report
    sync_bucket_high = 1.0 if sync_level_prev >= 6 else 0.0
    sync_bucket_med = 1.0 if (4 <= sync_level_prev <= 5) else 0.0
    sync_bucket_low = 1.0 if (2 <= sync_level_prev <= 3) else 0.0
    
    # -------------------------------------------------------------------------
    # Priority 3: Universal Drought Length & Non-Linear Hazard Curve
    # -------------------------------------------------------------------------
    drought_length = 0
    t = cutoff - 1
    while t in past and past[t] == 0:
        drought_length += 1
        t -= 1
        
    in_drought_zone = 1.0 if drought_length >= 3 else 0.0
    in_deep_drought = 1.0 if drought_length >= 6 else 0.0
    in_collapse_drought = 1.0 if drought_length >= 9 else 0.0
    
    clamped_k = min(15, drought_length)
    empirical_hazard = float(DROUGHT_HAZARDS.get(clamped_k, 0.02))
    expected_escape_payout = float(DROUGHT_ESCAPE_PAYOUTS.get(clamped_k, 3.5))

    # -------------------------------------------------------------------------
    # Priority 4: Jackpot Contagion Dynamics (peer 5x/20x in last 3 ticks)
    # -------------------------------------------------------------------------
    peer_jackpots_3 = 0
    jackpot_decay_weight = 0.0
    for s in peer_past:
        for dt in range(1, 4):
            t_lag = cutoff - dt
            if s.get(t_lag, 0) in (5, 20):
                peer_jackpots_3 += 1
                jackpot_decay_weight += float(0.5 ** (dt - 1))
                
    in_contagion_window = 1.0 if peer_jackpots_3 > 0 else 0.0
    
    return [
        # Priority 1: Surge window
        in_surge_window,
        surge_bell,
        surge_ramp,
        shared_change_surge_signal,
        # Priority 2: Sync level
        float(sync_level_prev / 7.0),
        float(sync_mean_3 / 7.0),
        sync_bucket_high,
        sync_bucket_med,
        sync_bucket_low,
        # Priority 3: Drought hazard
        float(min(drought_length, 25) / 25.0),
        in_drought_zone,
        in_deep_drought,
        in_collapse_drought,
        empirical_hazard,
        expected_escape_payout / 5.0,
        # Priority 4: Contagion
        float(min(peer_jackpots_3, 5)),
        in_contagion_window,
        jackpot_decay_weight
    ]

class ArchetypeConditionedTransitionTable:
    """Discrete transition table conditioned on machine archetype."""
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.global_prior = None
        self.mt_prior = {}
        self.pair_counts = defaultdict(Counter)
        self.archetype_pair_counts = defaultdict(lambda: defaultdict(Counter))
        
    def fit(self, queries, history):
        q_mtype = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
        grouped = defaultdict(dict)
        for r in history:
            grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
            
        global_counts = Counter()
        mt_counts = defaultdict(Counter)
        
        ep_archetypes = {}
        for (ep, m), seq in grouped.items():
            mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(seq, cutoff=99999)
            st = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
            ep_archetypes[(ep, m)] = st
            
        for (ep, m), seq in grouped.items():
            mt = q_mtype.get((ep, m))
            arch = ep_archetypes.get((ep, m), 0)
            ticks = sorted(seq.keys())
            for i, t in enumerate(ticks):
                outcome = seq[t]
                global_counts[outcome] += 1
                if mt: mt_counts[mt][outcome] += 1
                if i >= 2 and ticks[i-1] == t - 1 and ticks[i-2] == t - 2:
                    s2 = (seq[t-1], seq[t-2])
                    self.pair_counts[s2][outcome] += 1
                    self.archetype_pair_counts[arch][s2][outcome] += 1
                    
        tot = sum(global_counts.values())
        self.global_prior = np.array([global_counts[p] / tot for p in PAYOUTS], dtype=np.float32)
        for mt in MTYPES:
            mt_tot = sum(mt_counts[mt].values())
            self.mt_prior[mt] = np.array([mt_counts[mt][p] / mt_tot for p in PAYOUTS], dtype=np.float32)

    def get_features(self, query, past, cutoff, archetype_state=0):
        a = self.alpha
        prior = self.global_prior
        p1 = past.get(cutoff - 1)
        p2 = past.get(cutoff - 2)
        if p1 is not None and p2 is not None:
            s2 = (p1, p2)
            if archetype_state != 0:
                c2 = self.archetype_pair_counts[archetype_state].get(s2, {})
            else:
                c2 = self.pair_counts.get(s2, {})
            n2 = sum(c2.values())
            return [(c2.get(p, 0) + a * prior[i]) / (n2 + a) for i, p in enumerate(PAYOUTS)]
        else:
            return list(prior)

def extract_features_v26(query, past, peer_past, cutoff, base_forecaster, trans_table):
    # 1. Base lag features (275)
    feats = build_base_lag_features(query, past, peer_past, cutoff, base_forecaster)
    
    # 2. Archetype computation & discrete transitions
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    tot = nw + nl
    mom_shrunk = mom * (tot / (tot + 20.0))
    
    trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    feats += trans_feats
    
    # 3. Soft-thresholded differential drift slopes
    drift_feats = extract_soft_thresholded_slopes(past, peer_past, cutoff, kappa=1.50)
    feats += drift_feats
    
    # 4. Databricks Coupling, Hazard & Bayesian Shrinkage Features (10 features)
    db_feats_v25 = extract_databricks_coupling_and_hazard_features(query, past, peer_past, cutoff)
    feats += db_feats_v25
    
    # 5. Databricks Deep Dive Discoveries: Surge, Sync & Contagion (18 features)
    db_deep_dive_feats = extract_databricks_deep_dive_features(query, past, peer_past, cutoff)
    feats += db_deep_dive_feats
    
    # 6. Machine genome continuous & discrete representations
    genome_feats = [
        float(mom_shrunk),
        1.0 if arch_conf == -1 else 0.0,
        1.0 if arch_conf == 0 else 0.0,
        1.0 if arch_conf == 1 else 0.0
    ]
    feats += genome_feats
    return feats
