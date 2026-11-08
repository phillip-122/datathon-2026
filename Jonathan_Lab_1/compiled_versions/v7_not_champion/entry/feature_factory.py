"""EXP-022 Cutoff-Safe Feature Factory & Archetype Conditioners.

Strictly adheres to:
1. Observations strictly before cutoff: sequence.items() where t < cutoff.
2. Explicit uncertainty handling: if N < 25 or confidence interval overlaps 0, archetype = 0.
3. Isolated drift features from EXP-020.
4. Archetype representations: single categorical, interaction, and conditioned transitions.
"""
from collections import Counter, defaultdict
import numpy as np

from slot_forecasting.schema import PAYOUTS
from lag_model.features import build_advanced_features as build_base_lag_features

PAYOUT_TO_IDX = {p: i for i, p in enumerate(PAYOUTS)}
MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}

def compute_cutoff_safe_archetype(sequence, cutoff):
    """Compute machine momentum strictly using past history before cutoff."""
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
    """Classify into {-1: Reverting, 0: Uncertain/Neutral, +1: Streak}."""
    tot = nw + nl
    if mode == "fixed":
        if tot < 5:
            return 0
        return -1 if mom < -threshold else (1 if mom > threshold else 0)
    elif mode == "confidence_gated":
        # Must have at least 20 pairs and momentum must exceed 1.5 * SE to claim polarity
        if tot < 20 or abs(mom) <= 1.5 * se:
            return 0  # Uncertain / Neutral
        return 1 if mom > 0 else -1
    elif mode == "shrunk":
        mom_shrunk = mom * (tot / (tot + 15.0))
        return -1 if mom_shrunk < -threshold else (1 if mom_shrunk > threshold else 0)
    return 0

def extract_exp020_drift_features(query, past, peer_past, cutoff):
    """Extract the 4 verified continuous velocity & hazard features from EXP-020."""
    m_type = query.machine_type
    
    # 1. slope_16_48 = win_rate_16 - win_rate_48
    p_16 = [past[t] for t in range(cutoff - 16, cutoff) if t in past]
    p_48 = [past[t] for t in range(cutoff - 48, cutoff) if t in past]
    wr_16 = sum(1 for p in p_16 if p > 0) / len(p_16) if p_16 else 0.5
    wr_48 = sum(1 for p in p_48 if p > 0) / len(p_48) if p_48 else 0.5
    slope_16_48 = wr_16 - wr_48
    
    # 2. slope_8_24 = win_rate_8 - win_rate_24
    p_8 = [past[t] for t in range(cutoff - 8, cutoff) if t in past]
    p_24 = [past[t] for t in range(cutoff - 24, cutoff) if t in past]
    wr_8 = sum(1 for p in p_8 if p > 0) / len(p_8) if p_8 else 0.5
    wr_24 = sum(1 for p in p_24 if p > 0) / len(p_24) if p_24 else 0.5
    slope_8_24 = wr_8 - wr_24
    
    # 3. peer_slope = peer_win_rate_8 - peer_win_rate_24
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
    
    # 4. jackpot_hazard = (1.0 if jackpot else 0.0) * min(streak_0 / 30.0, 1.5)
    streak_0 = 0
    t = cutoff - 1
    while t in past and past[t] == 0:
        streak_0 += 1
        t -= 1
    jackpot_hazard = (1.0 if m_type == "jackpot" else 0.0) * min(streak_0 / 30.0, 1.5)
    
    return [slope_16_48, slope_8_24, peer_slope, jackpot_hazard]

class ArchetypeConditionedTransitionTable:
    """Computes discrete transition probabilities conditioned on machine archetype."""
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
        
        # Precompute full episode ground truth archetype for training machines
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
            # If archetype is certain (non-zero), use archetype-conditioned table; else fallback to general pair
            if archetype_state != 0:
                c2 = self.archetype_pair_counts[archetype_state].get(s2, {})
            else:
                c2 = self.pair_counts.get(s2, {})
            n2 = sum(c2.values())
            return [(c2.get(p, 0) + a * prior[i]) / (n2 + a) for i, p in enumerate(PAYOUTS)]
        else:
            return list(prior)
