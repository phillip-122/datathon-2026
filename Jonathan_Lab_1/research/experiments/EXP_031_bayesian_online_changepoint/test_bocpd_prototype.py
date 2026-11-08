"""Prototype BOCPD test script on validation set."""
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

from research.tools.controller import ResearchController
from slot_forecasting.io import load_inputs, read_labels
from slot_forecasting.schema import probabilities

# 1. Physical Reel Constants
REEL_RATIOS = {
    "steady": (200.0 / 250.0, 49.0 / 250.0, 1.0 / 250.0),
    "balanced": (16.0 / 23.0, 6.0 / 23.0, 1.0 / 23.0),
    "jackpot": (7.0 / 19.0, 9.0 / 19.0, 3.0 / 19.0),
}

def to_category(payout):
    if payout == 0:
        return 0
    elif payout == 1:
        return 1
    else:
        return 2

class BOCPDCategoricalEngine:
    def __init__(self, hazard_lambda=50.0, prior_strength=10.0, use_signal=True):
        self.hazard_lambda = hazard_lambda
        self.prior_strength = prior_strength
        self.use_signal = use_signal
        self.priors = {}
        
    def fit_priors(self, queries, history):
        q_info = {(q.episode_id, q.machine_id): (q.machine_type, q.signal) for q in queries}
        counts = defaultdict(Counter)
        mt_counts = defaultdict(Counter)
        
        for r in history:
            info = q_info.get((r.episode_id, r.machine_id))
            if info:
                mt, sig = info
                cat = to_category(r.payout)
                counts[(mt, sig)][cat] += 1
                mt_counts[mt][cat] += 1
                
        for (mt, sig), c in counts.items():
            tot = sum(c.values())
            self.priors[(mt, sig)] = np.array([c[0]/tot, c[1]/tot, c[2]/tot], dtype=np.float64)
            
        for mt, c in mt_counts.items():
            tot = sum(c.values())
            self.priors[mt] = np.array([c[0]/tot, c[1]/tot, c[2]/tot], dtype=np.float64)
            
    def predict_query(self, query, sequence, cutoff):
        past = {t: p for t, p in sequence.items() if t < cutoff}
        ticks = sorted(past.keys())
        
        # Base prior
        if self.use_signal and (query.machine_type, query.signal) in self.priors:
            base_p = self.priors[(query.machine_type, query.signal)]
        else:
            base_p = self.priors.get(query.machine_type, np.array([0.5, 0.25, 0.25]))
            
        alpha_prior = self.prior_strength * base_p
        H_const = 1.0 / self.hazard_lambda
        
        if not ticks:
            p_cat = base_p
        else:
            # BOCPD forward recursion
            # R[r] is the joint probability P(r_t = r, x_{1:t})
            # counts[r, k] is the category counts for run length r
            R = np.array([1.0], dtype=np.float64)
            # Sufficient statistics: counts of [cat0, cat1, cat2] for each run length
            # counts array has shape (current_max_r + 1, 3)
            counts = np.zeros((1, 3), dtype=np.float64)
            
            for i, t in enumerate(ticks):
                x = to_category(past[t])
                
                # Predictive probability for observation x under each hypothetical run-length r
                # alpha_post = alpha_prior + counts[r]
                total_alpha = np.sum(alpha_prior) + np.sum(counts, axis=1) # shape (N_r,)
                pred_prob = (alpha_prior[x] + counts[:, x]) / total_alpha # shape (N_r,)
                
                # Growth probabilities: r_{t} = r_{t-1} + 1
                growth_probs = R * pred_prob * (1.0 - H_const)
                
                # Changepoint probability: r_t = 0
                cp_prob = np.sum(R * pred_prob * H_const)
                
                # New R distribution
                R_new = np.zeros(len(R) + 1, dtype=np.float64)
                R_new[0] = cp_prob
                R_new[1:] = growth_probs
                
                # Evidence (normalizer)
                evidence = np.sum(R_new)
                if evidence > 0:
                    R = R_new / evidence
                else:
                    R = np.zeros(len(R_new))
                    R[0] = 1.0
                    
                # New counts
                counts_new = np.zeros((len(R), 3), dtype=np.float64)
                counts_new[0, :] = 0.0 # reset at changepoint
                counts_new[1:, :] = counts + np.eye(3)[x]
                counts = counts_new
                
                # Prune negligible run-lengths to prevent numerical underflow and slow down
                if len(R) > 120:
                    keep = R > 1e-6
                    keep[0] = True # always keep changepoint
                    if np.sum(keep) < len(R):
                        R = R[keep]
                        counts = counts[keep]
                        R = R / np.sum(R)
                        
            # Predictive distribution for the next spin at cutoff
            # Average over posterior run-lengths
            total_alpha = np.sum(alpha_prior) + np.sum(counts, axis=1) # shape (N_r,)
            pred_cats_given_r = (alpha_prior[None, :] + counts) / total_alpha[:, None] # shape (N_r, 3)
            
            # Mixture: with probability (1 - H), run continues; with probability H, resets to prior
            p_cat = (1.0 - H_const) * np.sum(R[:, None] * pred_cats_given_r, axis=0) + H_const * base_p
            p_cat = p_cat / np.sum(p_cat)
            
        # Map 3 categories to 5 classes using hardcoded reel constants
        r2, r5, r20 = REEL_RATIOS[query.machine_type]
        p0 = p_cat[0]
        p1 = p_cat[1]
        p2plus = p_cat[2]
        p2 = p2plus * r2
        p5 = p2plus * r5
        p20 = p2plus * r20
        
        return (float(p0), float(p1), float(p2), float(p5), float(p20))

def main():
    ctrl = ResearchController()
    val_queries = ctrl.val_queries
    val_history = ctrl.val_history
    targets = ctrl.val_targets
    
    train_dir = ctrl.data_dir / "train"
    t_queries, t_history = load_inputs(train_dir)
    
    # Fit priors on train data
    for lmb in [15.0, 25.0, 40.0, 60.0, 100.0]:
        for alpha_0 in [2.0, 5.0, 10.0, 20.0]:
            engine = BOCPDCategoricalEngine(hazard_lambda=lmb, prior_strength=alpha_0, use_signal=True)
            engine.fit_priors(t_queries, t_history)
            
            grouped_val = defaultdict(dict)
            for r in val_history:
                grouped_val[(r.episode_id, r.machine_id)][r.tick] = r.payout
                
            preds = {}
            for q in val_queries:
                seq = grouped_val[(q.episode_id, q.machine_id)]
                p5 = engine.predict_query(q, seq, q.cutoff)
                preds[q.query_id] = p5
                
            res = ctrl.evaluate_candidate(preds)
            print(f"lambda={lmb:5.1f}, alpha_0={alpha_0:4.1f} | Full: {res['full_points']:.4f} pts (Tune: {res['tuning_points']:.4f}, Audit: {res['audit_points']:.4f}) | Loss Ratio: {res['full_loss_ratio']:.4f}")

if __name__ == "__main__":
    main()
