"""EXP-012: Probability Calibration via Logit Temperature & Power Scaling.

Hypothesis: Gradient boosted decision trees minimize cross-entropy loss and can output
overconfident probabilities near the extreme boundaries (0.0 and 1.0).
Because the contest evaluation metric is the Brier score (mean squared error of probabilities),
temperature scaling of raw logits z -> softmax(z / T) or probability exponentiation
p -> (p^gamma) / sum(p^gamma) smooths extreme probability outputs, reducing squared error
and improving calibration on unseen episodes.

Protocol:
1. Load champion EXP-006 model and extract validation predictions and raw logits.
2. Grid search T in [0.7, 1.5] and gamma in [0.7, 1.3] on TUNING episodes only.
3. Select optimal calibration parameters on Tuning episodes.
4. Verify on untouched AUDIT episodes.
5. If both Tuning > 35.1863 and Audit > 34.9486, candidate is promoted.
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np

from research.evaluator import ResearchEvaluator
from research.run_experiment_helper import run_standard_eval
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import subset_score
from slot_forecasting.io import load_inputs, read_labels, read_submission
from src.features import build_advanced_features

EXP_ID = "EXP-012"
OUT_DIR = Path("research/experiments/EXP_012_temperature_calibration")

def softmax(logits, temperature=1.0):
    scaled = logits / temperature
    # Numerical stability
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

def power_scale(probs, gamma=1.0):
    p = probs ** gamma
    return p / np.sum(p, axis=1, keepdims=True)

def main():
    print(f"=== Starting {EXP_ID}: Temperature & Power Calibration ===")
    t_start = time.time()
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    val_q, val_h = evaluator.val_queries, evaluator.val_history
    val_q_list = list(val_q)
    
    # 1. Load champion base forecaster & HGB model
    model_dir = Path("artifacts/models/hgb_production")
    hgb = joblib.load(model_dir / "hgb_model.joblib")
    
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    
    # 2. Extract validation features
    val_hist = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        val_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X_val = []
    for q in val_q_list:
        ep_seqs = val_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_advanced_features(q, m_seq, peers, q.cutoff, base_forecaster)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    
    # Raw logits and raw probabilities
    raw_logits = hgb.decision_function(X_val)
    raw_probs = hgb.predict_proba(X_val)
    
    # Verify uncalibrated baseline matches EXP-006 exactly
    base_preds = {}
    for i, q in enumerate(val_q_list):
        p = raw_probs[i]
        tot = sum(p)
        base_preds[q.query_id] = tuple(float(v / tot) for v in p)
    res_base = evaluator.evaluate(base_preds)
    print(f"Verified uncalibrated base: Tune={res_base['tuning_points']:.4f}, Audit={res_base['audit_points']:.4f}, Full={res_base['full_points']:.4f}")
    
    # 3. Grid search Temperature on TUNING episodes
    print("\n--- Grid search Temperature T on Tuning episodes ---")
    best_t_pts = -1.0
    best_T = 1.0
    for T in np.linspace(0.70, 1.40, 71):
        cal_probs = softmax(raw_logits, temperature=T)
        preds = {}
        for i, q in enumerate(val_q_list):
            p = cal_probs[i]
            tot = sum(p)
            preds[q.query_id] = tuple(float(v / tot) for v in p)
        t_sc = subset_score(preds, evaluator.val_targets, evaluator.groups, evaluator.reference, evaluator.tuning_eps, 60.0)
        pts = t_sc["technical_points"]
        if pts > best_t_pts:
            best_t_pts = pts
            best_T = float(T)
            
    print(f"Optimal Temperature on Tuning: T = {best_T:.3f} -> Tuning points: {best_t_pts:.4f}")
    
    # 4. Grid search Power gamma on TUNING episodes
    print("--- Grid search Power gamma on Tuning episodes ---")
    best_g_pts = -1.0
    best_gamma = 1.0
    for g in np.linspace(0.70, 1.30, 61):
        cal_probs = power_scale(raw_probs, gamma=g)
        preds = {}
        for i, q in enumerate(val_q_list):
            p = cal_probs[i]
            tot = sum(p)
            preds[q.query_id] = tuple(float(v / tot) for v in p)
        t_sc = subset_score(preds, evaluator.val_targets, evaluator.groups, evaluator.reference, evaluator.tuning_eps, 60.0)
        pts = t_sc["technical_points"]
        if pts > best_g_pts:
            best_g_pts = pts
            best_gamma = float(g)
            
    print(f"Optimal Gamma on Tuning: gamma = {best_gamma:.3f} -> Tuning points: {best_g_pts:.4f}")
    
    # 5. Evaluate winning calibration on Audit and Full validation
    # Test best T
    cal_t_probs = softmax(raw_logits, temperature=best_T)
    preds_T = {q.query_id: tuple(float(v / sum(cal_t_probs[i])) for v in cal_t_probs[i]) for i, q in enumerate(val_q_list)}
    res_T = evaluator.evaluate(preds_T)
    
    # Test best gamma
    cal_g_probs = power_scale(raw_probs, gamma=best_gamma)
    preds_G = {q.query_id: tuple(float(v / sum(cal_g_probs[i])) for v in cal_g_probs[i]) for i, q in enumerate(val_q_list)}
    res_G = evaluator.evaluate(preds_G)
    
    print("\n--- RESULTS SUMMARY ---")
    print(f"Champion EXP-006: Tune: {res_base['tuning_points']:.4f} | Audit: {res_base['audit_points']:.4f} | Full: {res_base['full_points']:.4f}")
    print(f"Temperature T={best_T:.2f}: Tune: {res_T['tuning_points']:.4f} | Audit: {res_T['audit_points']:.4f} | Full: {res_T['full_points']:.4f}")
    print(f"Gamma gamma={best_gamma:.2f}: Tune: {res_G['tuning_points']:.4f} | Audit: {res_G['audit_points']:.4f} | Full: {res_G['full_points']:.4f}")
    
    winner_res = res_T if res_T["tuning_points"] >= res_G["tuning_points"] else res_G
    winner_preds = preds_T if res_T["tuning_points"] >= res_G["tuning_points"] else preds_G
    winner_desc = f"Temperature T={best_T:.3f}" if res_T["tuning_points"] >= res_G["tuning_points"] else f"Gamma gamma={best_gamma:.3f}"
    
    decision = "PROMOTED" if (winner_res["tuning_points"] > 35.1863 and winner_res["audit_points"] > 34.9486) else "REJECTED"
    
    run_standard_eval(
        exp_id=EXP_ID,
        hypothesis=f"Probability calibration via {winner_desc}",
        author="CalibrationSpecialist",
        predictions=winner_preds,
        out_dir=OUT_DIR,
        notes=f"Decision: {decision}. Total time: {time.time() - t_start:.1f}s"
    )
    print(f"\nFinal Decision for {EXP_ID}: {decision}")

if __name__ == "__main__":
    main()
