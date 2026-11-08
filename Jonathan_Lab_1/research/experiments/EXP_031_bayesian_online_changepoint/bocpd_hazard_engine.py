"""EXP-031: Clean-Sheet Bayesian Online Changepoint Detection & Dynamic Hazard Engine (BOCPD).

Scientific Methodology:
1. Pure online Bayesian filter based on Adams & MacKay (2007) forward recursion.
2. Zero reliance on tree ensembles or legacy models (HistGradientBoosting, XGBoost, v8, v10).
3. Exact 3-Macro Category formulation: {0: Loss, 1: Break-even, 2: High win (2, 5, 20)}.
4. Exact physical reel constant projection to 5 payout classes (mathematically lossless, std=1e-16).
5. Dynamic Hazard Engine:
   - Floor Flush Detector: Elevates hazard upon coordinated peer machine loss flushes.
   - Cold Streak Hazard: Accelerates changepoint posterior upon loss droughts.
   - Recency & Memory Pruning: O(R_max) complexity, execution time < 5 seconds.
6. Methodological Dual-Gate Protocol:
   - Temperature & bias calibration strictly fit on Tuning Fold (48 episodes).
   - Single-shot unbiased evaluation on Audit Fold (48 episodes).
   - 2000-repeat paired condition-stratified bootstrap against Champion v10 baseline.
"""
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

from research.tools.controller import ResearchController
from slot_forecasting.io import load_inputs, read_submission, write_submission
from slot_forecasting.schema import probabilities

# Physical Reel Invariants (verified std = 1e-16)
REEL_RATIOS = {
    "steady": (200.0 / 250.0, 49.0 / 250.0, 1.0 / 250.0),
    "balanced": (16.0 / 23.0, 6.0 / 23.0, 1.0 / 23.0),
    "jackpot": (7.0 / 19.0, 9.0 / 19.0, 3.0 / 19.0),
}
MTYPES = ["steady", "balanced", "jackpot"]

def to_category(payout):
    if payout == 0:
        return 0
    elif payout == 1:
        return 1
    else:
        return 2

class DynamicHazardBOCPD:
    def __init__(
        self,
        base_lambda=120.0,
        prior_strength=3.0,
        decay_gamma=1.00,
        floor_hazard_weight=2.0,
        streak_hazard_weight=1.5,
        streak_cold_threshold=4,
        max_run_length=120,
    ):
        self.base_lambda = base_lambda
        self.prior_strength = prior_strength
        self.decay_gamma = decay_gamma
        self.floor_hazard_weight = floor_hazard_weight
        self.streak_hazard_weight = streak_hazard_weight
        self.streak_cold_threshold = streak_cold_threshold
        self.max_run_length = max_run_length
        self.priors = {}
        self.mt_priors = {}
        self.trans_probs = {}

    def fit_priors(self, queries, history):
        """Fit empirical baseline priors and 1-step Markov transition dynamics on training data."""
        q_info = {(q.episode_id, q.machine_id): (q.machine_type, q.signal) for q in queries}
        counts = defaultdict(Counter)
        mt_counts = defaultdict(Counter)
        trans_counts = defaultdict(lambda: np.zeros((3, 3)))
        
        ep_seqs = defaultdict(dict)
        for r in history:
            info = q_info.get((r.episode_id, r.machine_id))
            if info:
                mt, sig = info
                cat = to_category(r.payout)
                counts[(mt, sig)][cat] += 1
                mt_counts[mt][cat] += 1
                ep_seqs[(r.episode_id, r.machine_id)][r.tick] = cat

        for (mt, sig), c in counts.items():
            tot = sum(c.values())
            self.priors[(mt, sig)] = np.array([c[0]/tot, c[1]/tot, c[2]/tot], dtype=np.float64)

        for mt, c in mt_counts.items():
            tot = sum(c.values())
            self.mt_priors[mt] = np.array([c[0]/tot, c[1]/tot, c[2]/tot], dtype=np.float64)

        for (ep, mid), seq in ep_seqs.items():
            mt, _ = q_info[(ep, mid)]
            ticks = sorted(seq.keys())
            for i in range(1, len(ticks)):
                t_prev, t_curr = ticks[i-1], ticks[i]
                if t_curr == t_prev + 1:
                    trans_counts[mt][seq[t_prev], seq[t_curr]] += 1

        for mt, mat in trans_counts.items():
            row_sums = mat.sum(axis=1, keepdims=True)
            self.trans_probs[mt] = np.where(row_sums > 0, mat / np.maximum(1e-9, row_sums), self.mt_priors[mt][None, :])

    def compute_dynamic_hazard(self, tick, current_streak_0, peer_histories, cutoff):
        """Calculate dynamic hazard probability incorporating floor flush and cold streak signals."""
        h_base = 1.0 / self.base_lambda
        
        # 1. Floor flush factor from peer machines
        floor_multiplier = 1.0
        if peer_histories:
            peer_losses = 0
            peer_active = 0
            for p_dict in peer_histories:
                p_val = p_dict.get(tick - 1)
                if p_val is not None:
                    peer_active += 1
                    if p_val == 0:
                        peer_losses += 1
            if peer_active >= 3:
                peer_loss_rate = peer_losses / peer_active
                if peer_loss_rate >= 0.70:
                    floor_multiplier += self.floor_hazard_weight * (peer_loss_rate - 0.5)

        # 2. Local streak hazard
        streak_multiplier = 1.0
        if current_streak_0 >= self.streak_cold_threshold:
            streak_multiplier += self.streak_hazard_weight * min(3.0, (current_streak_0 - self.streak_cold_threshold + 1) * 0.5)

        hazard = h_base * floor_multiplier * streak_multiplier
        return min(0.85, max(0.005, hazard))

    def predict_query(self, query, past, peer_past, cutoff):
        """Run online recursive BOCPD with dynamic hazard engine over observed sequence."""
        ticks = sorted([t for t, p in past.items() if t < cutoff])
        mt = query.machine_type
        sig = query.signal
        base_p = self.priors.get((mt, sig), self.mt_priors.get(mt, np.array([0.5, 0.25, 0.25])))
        alpha_prior = self.prior_strength * base_p
        
        if not ticks:
            p_cat = base_p
        else:
            # Adams & MacKay (2007) BOCPD Recursion
            # R[r] is the joint probability P(r_t = r, x_{1:t})
            # counts[r, k] is the category counts for run length r
            R = np.array([1.0], dtype=np.float64)
            counts = np.zeros((1, 3), dtype=np.float64)
            current_streak_0 = 0
            last_payout = None
            
            for i, t in enumerate(ticks):
                p_val = past[t]
                x = to_category(p_val)
                last_payout = p_val
                
                if x == 0:
                    current_streak_0 += 1
                else:
                    current_streak_0 = 0
                    
                # Dynamic hazard for this tick
                hazard = self.compute_dynamic_hazard(t, current_streak_0, peer_past, cutoff)
                
                # Predictive probability of observation x under each run-length hypothesis r
                total_alpha = np.sum(alpha_prior) + np.sum(counts, axis=1)
                pred_prob = (alpha_prior[x] + counts[:, x]) / np.maximum(1e-9, total_alpha)
                
                # Growth: r_t = r_{t-1} + 1
                growth_probs = R * pred_prob * (1.0 - hazard)
                # Changepoint: r_t = 0
                cp_prob = np.sum(R * pred_prob * hazard)
                
                # New run-length vector
                R_new = np.zeros(len(R) + 1, dtype=np.float64)
                R_new[0] = cp_prob
                R_new[1:] = growth_probs
                
                # Normalize evidence
                evidence = np.sum(R_new)
                if evidence > 0:
                    R = R_new / evidence
                else:
                    R = np.zeros(len(R_new))
                    R[0] = 1.0
                    
                # Sufficient statistics update with exponential discount gamma
                counts_new = np.zeros((len(R), 3), dtype=np.float64)
                counts_new[0, :] = 0.0
                counts_new[1:, :] = counts * self.decay_gamma + np.eye(3)[x]
                counts = counts_new
                
                # Prune negligible run-lengths
                if len(R) > self.max_run_length:
                    keep = R > 1e-5
                    keep[0] = True
                    if np.sum(keep) < len(R):
                        R = R[keep]
                        counts = counts[keep]
                        R = R / np.sum(R)

            # Predictive distribution for next spin at cutoff
            total_alpha = np.sum(alpha_prior) + np.sum(counts, axis=1)
            pred_cats_given_r = (alpha_prior[None, :] + counts) / np.maximum(1e-9, total_alpha[:, None])
            
            # Final hazard at cutoff
            hazard_cutoff = self.compute_dynamic_hazard(cutoff, current_streak_0, peer_past, cutoff)
            p_bocpd = (1.0 - hazard_cutoff) * np.sum(R[:, None] * pred_cats_given_r, axis=0) + hazard_cutoff * base_p
            p_bocpd = p_bocpd / np.sum(p_bocpd)
            
            # Modulate with 1-step Markov transition if last spin was recent (t == cutoff - 1)
            if ticks[-1] == cutoff - 1 and last_payout is not None and mt in self.trans_probs:
                last_cat = to_category(last_payout)
                trans_vec = self.trans_probs[mt][last_cat]
                p_cat = 0.70 * p_bocpd + 0.30 * trans_vec
                p_cat = p_cat / np.sum(p_cat)
            else:
                p_cat = p_bocpd
                
            # Cold lockout override: if trailing zero streak >= 8, enforce extreme lockout
            if current_streak_0 >= 8:
                lockout_p0 = 0.96 if current_streak_0 < 12 else (0.98 if current_streak_0 < 16 else 0.99)
                rem = (1.0 - lockout_p0)
                p_cat = np.array([
                    lockout_p0,
                    rem * (p_cat[1] / max(1e-6, p_cat[1] + p_cat[2])),
                    rem * (p_cat[2] / max(1e-6, p_cat[1] + p_cat[2]))
                ])
                p_cat = p_cat / np.sum(p_cat)

        # Exact physical reel constant projection to 5 classes
        r2, r5, r20 = REEL_RATIOS[mt]
        p0 = p_cat[0]
        p1 = p_cat[1]
        p2plus = p_cat[2]
        p2 = p2plus * r2
        p5 = p2plus * r5
        p20 = p2plus * r20
        return (float(p0), float(p1), float(p2), float(p5), float(p20))

def main():
    t_start = time.time()
    ctrl = ResearchController()
    val_queries = ctrl.val_queries
    val_history = ctrl.val_history
    targets = ctrl.val_targets
    target_ids = {t.query_id for t in targets}
    v10_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", target_ids)
    
    train_dir = ctrl.data_dir / "train"
    t_queries, t_history = load_inputs(train_dir)
    
    val_grouped = defaultdict(dict)
    for r in val_history:
        val_grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout

    eval_v10 = ctrl.evaluate_candidate(v10_preds)
    print("="*75)
    print("EXP-031: BAYESIAN ONLINE CHANGEPOINT DETECTION & DYNAMIC HAZARD ENGINE")
    print("="*75)
    print(f"Active Champion v10 Baseline: Full={eval_v10['full_points']:.4f} pts (Tune={eval_v10['tuning_points']:.4f}, Audit={eval_v10['audit_points']:.4f}) | Loss Ratio={eval_v10['full_loss_ratio']:.4f}")
    print(f"Condition Loss Ratios: stable={eval_v10['by_condition']['stable']['loss_ratio']:.4f}, short={eval_v10['by_condition']['short_history']['loss_ratio']:.4f}, drift={eval_v10['by_condition']['drift']['loss_ratio']:.4f}, shared={eval_v10['by_condition']['shared_change']['loss_ratio']:.4f}")
    
    # 1. Instantiate Clean-Sheet Dynamic Hazard BOCPD Engine
    engine = DynamicHazardBOCPD(
        base_lambda=120.0,
        prior_strength=3.0,
        decay_gamma=1.00,
        floor_hazard_weight=2.0,
        streak_hazard_weight=1.5,
        streak_cold_threshold=4,
        max_run_length=120,
    )
    engine.fit_priors(t_queries, t_history)
    
    # 2. Generate raw predictions
    print("\nExecuting forward recursive BOCPD over all queries...")
    raw_bocpd = np.array([
        engine.predict_query(
            q,
            val_grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in val_grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff
        )
        for q in val_queries
    ], dtype=np.float64)
    
    tune_indices = [i for i, t in enumerate(targets) if t.episode_id in ctrl.tuning_eps]
    audit_indices = [i for i, t in enumerate(targets) if t.episode_id in ctrl.audit_eps]
    
    # 3. Dual-Gate Calibration: Fit on Tuning Fold ONLY
    print("\nFitting hierarchical temperature & bias calibration strictly on Tuning Fold...")
    logP_full = np.log(np.maximum(1e-12, raw_bocpd))
    cal_bocpd = np.zeros_like(raw_bocpd)
    cal_params = {}
    
    for m in MTYPES:
        tune_m = [i for i in tune_indices if ctrl.query_map[targets[i].query_id].machine_type == m]
        all_m = [i for i in range(len(targets)) if ctrl.query_map[targets[i].query_id].machine_type == m]
        
        Y_m = np.array([targets[i].probabilities for i in tune_m])
        logP_m = logP_full[tune_m]
        
        def loss(p):
            inv_T = np.exp(p[0])
            b = p[1:6]
            logits = logP_m * inv_T + b[None, :]
            pred = softmax(logits, axis=1)
            return np.mean(np.sum((pred - Y_m)**2, axis=1))
            
        opt = minimize(loss, [0.0]*6, method='L-BFGS-B')
        inv_T = np.exp(opt.x[0])
        b = opt.x[1:6]
        cal_params[m] = {"inv_T": float(inv_T), "bias": [float(x) for x in b]}
        
        cal_bocpd[all_m] = softmax(logP_full[all_m] * inv_T + b[None, :], axis=1)
        print(f"  {m:10s}: inv_T={inv_T:.4f}, bias={[round(x, 4) for x in b]}")
        
    # Build dictionary of final predictions
    final_preds = {}
    for i, t in enumerate(targets):
        row = cal_bocpd[i]
        final_preds[t.query_id] = tuple(float(v / np.sum(row)) for v in row)
        
    # 4. Strict Evaluation via ResearchController (Dual-Gate + 2000 Bootstrap Repeats)
    print("\nEvaluating Candidate on ResearchController...")
    eval_res = ctrl.evaluate_candidate(final_preds, base_predictions=v10_preds, n_bootstrap=2000)
    
    t_end = time.time()
    exec_time = t_end - t_start
    print(f"Candidate BOCPD Full Points:   {eval_res['full_points']:.4f} / 60 (v10: {eval_v10['full_points']:.4f})")
    print(f"Candidate BOCPD Tuning Points: {eval_res['tuning_points']:.4f} / 60 (v10: {eval_v10['tuning_points']:.4f})")
    print(f"Candidate BOCPD Audit Points:  {eval_res['audit_points']:.4f} / 60 (v10: {eval_v10['audit_points']:.4f})")
    print(f"Candidate BOCPD Full Loss Ratio: {eval_res['full_loss_ratio']:.4f}")
    print(f"95% Paired Bootstrap CI vs v10: [{eval_res['bootstrap_gain_vs_base_ci95'][0]:.4f}, {eval_res['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
    print("Condition Breakdown:")
    for c, r in eval_res['by_condition'].items():
        v10_r = eval_v10['by_condition'][c]['loss_ratio']
        print(f"  {c:14s}: loss_ratio = {r['loss_ratio']:.4f} (v10: {v10_r:.4f}, Diff: {r['loss_ratio'] - v10_r:+.4f})")
    print(f"Execution Time: {exec_time:.2f} seconds")
    
    # 5. Save artifacts
    exp_dir = Path(__file__).resolve().parent
    pred_path = exp_dir / "predictions.csv"
    res_path = exp_dir / "results.json"
    
    out_preds = {qid: probabilities(p, qid) for qid, p in final_preds.items()}
    write_submission(pred_path, out_preds)
    
    results_payload = {
        "experiment": "EXP_031_bayesian_online_changepoint",
        "description": "Clean-sheet Bayesian Online Changepoint Detection (BOCPD) & Dynamic Hazard Engine",
        "full_points": eval_res["full_points"],
        "tuning_points": eval_res["tuning_points"],
        "audit_points": eval_res["audit_points"],
        "full_loss_ratio": eval_res["full_loss_ratio"],
        "tuning_audit_spread": eval_res["tuning_audit_spread"],
        "by_condition": eval_res["by_condition"],
        "by_machine_type": eval_res["by_machine_type"],
        "by_history_length": eval_res["by_history_length"],
        "brier_by_class": eval_res["brier_by_class"],
        "bootstrap_gain_vs_v10_ci95": eval_res["bootstrap_gain_vs_base_ci95"],
        "baseline_champion_v10": {
            "full_points": eval_v10["full_points"],
            "tuning_points": eval_v10["tuning_points"],
            "audit_points": eval_v10["audit_points"],
            "full_loss_ratio": eval_v10["full_loss_ratio"],
            "by_condition": {c: eval_v10["by_condition"][c]["loss_ratio"] for c in eval_v10["by_condition"]},
        },
        "official_frequency_baseline": {
            "full_points": 30.9023,
            "loss_ratio": 0.9416,
            "gain_full_points": eval_res["full_points"] - 30.9023,
        },
        "calibration_parameters": cal_params,
        "adversarial_status": eval_res["adversarial_status"],
        "execution_time_sec": exec_time,
    }
    
    with open(res_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nArtifacts saved successfully:\n  - Predictions: {pred_path}\n  - Results: {res_path}")

if __name__ == "__main__":
    main()
