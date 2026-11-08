"""EXP-024 Feature Factory: Soft-Thresholded Slopes, Bankroll Floor Coupling & Bayesian Genome Shrinkage."""
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
        if tot < 5: return 0
        return -1 if mom < -threshold else (1 if mom > threshold else 0)
    elif mode == "confidence_gated":
        if tot < 20 or abs(mom) <= 1.5 * se: return 0
        return 1 if mom > 0 else -1
    elif mode == "shrunk":
        mom_shrunk = mom * (tot / (tot + 15.0))
        return -1 if mom_shrunk < -threshold else (1 if mom_shrunk > threshold else 0)
    return 0

def extract_soft_thresholded_drift_features(query, past, peer_past, cutoff, kappa=1.50):
    m_type = query.machine_type
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
    
    streak_0 = 0
    t = cutoff - 1
    while t in past and past[t] == 0:
        streak_0 += 1
        t -= 1
    jackpot_hazard = (1.0 if m_type == "jackpot" else 0.0) * min(streak_0 / 30.0, 1.5)
    
    if kappa <= 0.0:
        return [slope_16_48, slope_8_24, peer_slope, jackpot_hazard]
        
    n_16 = max(1, sum(1 for t in range(cutoff - 16, cutoff) if t in past))
    n_48 = max(1, sum(1 for t in range(cutoff - 48, cutoff) if t in past))
    se_16_48 = np.sqrt(0.24 / n_16 + 0.24 / n_48)
    thresh_16_48 = kappa * se_16_48
    shrunk_16_48 = np.sign(slope_16_48) * max(0.0, abs(slope_16_48) - thresh_16_48) if abs(slope_16_48) > thresh_16_48 else 0.0

    n_8 = max(1, sum(1 for t in range(cutoff - 8, cutoff) if t in past))
    n_24 = max(1, sum(1 for t in range(cutoff - 24, cutoff) if t in past))
    se_8_24 = np.sqrt(0.24 / n_8 + 0.24 / n_24)
    thresh_8_24 = kappa * se_8_24
    shrunk_8_24 = np.sign(slope_8_24) * max(0.0, abs(slope_8_24) - thresh_8_24) if abs(slope_8_24) > thresh_8_24 else 0.0

    n_p8 = max(1, len(peer_8))
    n_p24 = max(1, len(peer_24))
    se_peer = np.sqrt(0.24 / n_p8 + 0.24 / n_p24)
    thresh_peer = kappa * se_peer
    shrunk_peer = np.sign(peer_slope) * max(0.0, abs(peer_slope) - thresh_peer) if abs(peer_slope) > thresh_peer else 0.0

    return [float(shrunk_16_48), float(shrunk_8_24), float(shrunk_peer), float(jackpot_hazard)]

def extract_bankroll_floor_coupling_features(peer_past, cutoff):
    """Compute floor dispersion metrics across the 7 peer machines."""
    peer_wrs = []
    peer_slopes = []
    peer_payouts_24 = []
    
    for p_seq in peer_past:
        p_24 = [p_seq[t] for t in range(cutoff - 24, cutoff) if t in p_seq]
        p_8 = [p_seq[t] for t in range(cutoff - 8, cutoff) if t in p_seq]
        
        if p_24:
            wr_24 = sum(1 for p in p_24 if p > 0) / len(p_24)
            peer_wrs.append(wr_24)
            peer_payouts_24.append(sum(p_24))
        else:
            peer_wrs.append(0.40)
            peer_payouts_24.append(10.0)
            
        if p_8 and p_24:
            wr_8 = sum(1 for p in p_8 if p > 0) / len(p_8)
            peer_slopes.append(wr_8 - wr_24)
        else:
            peer_slopes.append(0.0)
            
    peer_wr_std_24 = float(np.std(peer_wrs)) if len(peer_wrs) > 1 else 0.18
    peer_slope_std = float(np.std(peer_slopes)) if len(peer_slopes) > 1 else 0.10
    total_peer_payout_24 = float(sum(peer_payouts_24)) / (len(peer_past) * 24.0) if peer_past else 0.40
    
    return [peer_wr_std_24, peer_slope_std, total_peer_payout_24]

class StandardTransitionTable:
    """Standard 2-tick Markov transition table with Dirichlet alpha smoothing."""
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.global_prior = None
        self.mt_prior = {}
        self.pair_counts = defaultdict(Counter)
        
    def fit(self, queries, history):
        q_mtype = {(q.episode_id, q.machine_id): q.machine_type for q in queries}
        grouped = defaultdict(dict)
        for r in history:
            grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
            
        global_counts = Counter()
        mt_counts = defaultdict(Counter)
        
        for (ep, m), seq in grouped.items():
            mt = q_mtype.get((ep, m))
            ticks = sorted(seq.keys())
            for i, t in enumerate(ticks):
                outcome = seq[t]
                global_counts[outcome] += 1
                if mt: mt_counts[mt][outcome] += 1
                if i >= 2 and ticks[i-1] == t - 1 and ticks[i-2] == t - 2:
                    s2 = (seq[t-1], seq[t-2])
                    self.pair_counts[s2][outcome] += 1
                    
        tot = sum(global_counts.values())
        self.global_prior = np.array([global_counts[p] / tot for p in PAYOUTS], dtype=np.float32)
        for mt in MTYPES:
            mt_tot = sum(mt_counts[mt].values())
            self.mt_prior[mt] = np.array([mt_counts[mt][p] / mt_tot for p in PAYOUTS], dtype=np.float32)

    def get_features(self, query, past, cutoff):
        a = self.alpha
        prior = self.global_prior
        p1 = past.get(cutoff - 1)
        p2 = past.get(cutoff - 2)
        if p1 is not None and p2 is not None:
            s2 = (p1, p2)
            c2 = self.pair_counts.get(s2, {})
            n2 = sum(c2.values())
            return [(c2.get(p, 0) + a * prior[i]) / (n2 + a) for i, p in enumerate(PAYOUTS)]
        else:
            return list(prior)
