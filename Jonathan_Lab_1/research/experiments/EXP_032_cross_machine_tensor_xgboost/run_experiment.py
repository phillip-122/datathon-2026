"""EXP-032: Clean-Sheet Cross-Machine Interaction Tensor & 2-Variable XGBoost Architecture.

Completely independent of v10/v8/v6. 
Architecture:
1. Cross-Machine Delay-8 Interaction Tensor: 8 machines x 8 lags = 64 spatial-temporal inputs.
2. 2-Stage Binary Estimation:
   - Model A (Loss Hazard): Predicts p0 = P(Payout == 0)
   - Model B (Break-even Ratio): Predicts q1 = P(Payout == 1 | Payout > 0)
3. Physical Reel Invariant: Exactly projects W_high = (1 - p0) * (1 - q1) into [p2, p5, p20] using hardcoded reel odds.
4. Evaluated on Dual-Gate Protocol (Tuning vs Audit fold).
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_LAB = Path(".").resolve()
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB / "src")
]

from research.tools.controller import ResearchController
from slot_forecasting.io import load_inputs, read_labels, read_submission

REEL_RATIOS = {
    "steady":   np.array([200.0/250.0, 49.0/250.0, 1.0/250.0], dtype=np.float32),
    "balanced": np.array([16.0/23.0,   6.0/23.0,  1.0/23.0], dtype=np.float32),
    "jackpot":  np.array([7.0/19.0,    9.0/19.0,  3.0/19.0], dtype=np.float32)
}
MTYPES = ["steady", "balanced", "jackpot"]
MTYPE_MAP = {m: i for i, m in enumerate(MTYPES)}

def extract_tensor_features(q, ep_seqs, cutoff):
    # ep_seqs is a dict: machine_idx (0..7) -> {tick: payout}
    target_idx = int(q.machine_id[-1]) if q.machine_id[-1].isdigit() else 0
    target_seq = ep_seqs.get(target_idx, {})
    
    # 1. Target machine drought streak
    streak_0 = 0
    for t in range(cutoff - 1, -1, -1):
        if t in target_seq:
            if target_seq[t] == 0:
                streak_0 += 1
            else:
                break
                
    # 2. Cross-machine interaction tensor (8 machines x 8 lags)
    tensor = []
    for m in range(8):
        m_seq = ep_seqs.get(m, {})
        for lag in range(1, 9):
            t = cutoff - lag
            if t in m_seq:
                p = m_seq[t]
                # Categorize: 0, 1, or 2 (where 2 is 2, 5, 20)
                cat = 0 if p == 0 else (1 if p == 1 else 2)
                tensor.append(float(cat))
            else:
                tensor.append(-1.0)
                
    # 3. Floor aggregate statistics in last 8, 16 ticks
    wins_8 = 0
    total_8 = 0
    pay_8 = 0
    for m in range(8):
        m_seq = ep_seqs.get(m, {})
        for t in range(cutoff - 8, cutoff):
            if t in m_seq:
                p = m_seq[t]
                total_8 += 1
                pay_8 += p
                if p > 0:
                    wins_8 += 1
                    
    floor_wr_8 = wins_8 / max(1, total_8)
    floor_pay_8 = pay_8 / max(1, total_8)
    
    # 4. Target machine win rates
    sub_target = [target_seq[t] for t in range(cutoff - 32, cutoff) if t in target_seq]
    wr_target_16 = sum(1 for p in sub_target[-16:] if p > 0) / max(1, len(sub_target[-16:])) if sub_target else 0.5
    wr_target_32 = sum(1 for p in sub_target if p > 0) / max(1, len(sub_target)) if sub_target else 0.5
    
    # 5. One-hot machine type
    mtype_oh = [1.0 if q.machine_type == m else 0.0 for m in MTYPES]
    
    # Combined feature vector
    feats = [float(streak_0), floor_wr_8, floor_pay_8, wr_target_16, wr_target_32, float(q.signal)] + mtype_oh + tensor
    return feats

def main():
    start_time = time.time()
    ctrl = ResearchController()
    val_qs = ctrl.val_queries
    val_targets = ctrl.val_targets
    target_map = {t.query_id: t.probabilities for t in val_targets}
    v10_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", {t.query_id for t in val_targets})
    
    print("="*75)
    print("EXP-032: CLEAN-SHEET CROSS-MACHINE TENSOR & 2-VARIABLE XGBOOST")
    print("="*75)
    
    eval_v10 = ctrl.evaluate_candidate(v10_preds)
    print(f"Active Champion Baseline (v10): Full={eval_v10['full_points']:.4f} (Tune={eval_v10['tuning_points']:.4f}, Audit={eval_v10['audit_points']:.4f})")
    
    # Load training data
    print("\nLoading Training History...")
    train_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/train")
    train_qs, train_h = load_inputs(train_dir)
    
    # Group train history by episode and machine_idx
    train_ep_seqs = defaultdict(lambda: defaultdict(dict))
    for p in train_h:
        m_idx = int(p.machine_id[-1]) if p.machine_id[-1].isdigit() else 0
        train_ep_seqs[p.episode_id][m_idx][p.tick] = p.payout
        
    print(f"Building Training Examples from {len(train_qs)} episodes...")
    X_train = []
    y0_train = [] # is 0x
    y1_train = [] # is 1x (only for positive payouts)
    w1_indices = []
    
    for q in train_qs:
        ep_seqs = train_ep_seqs[q.episode_id]
        target_idx = int(q.machine_id[-1]) if q.machine_id[-1].isdigit() else 0
        target_seq = ep_seqs.get(target_idx, {})
        
        # Sample cutoffs every 6 ticks from 30 to q.cutoff
        for t in range(30, q.cutoff, 6):
            if t in target_seq:
                f = extract_tensor_features(q, ep_seqs, t)
                payout = target_seq[t]
                
                idx_example = len(X_train)
                X_train.append(f)
                y0_train.append(1 if payout == 0 else 0)
                
                if payout > 0:
                    y1_train.append(1 if payout == 1 else 0)
                    w1_indices.append(idx_example)
                    
    X_train = np.array(X_train, dtype=np.float32)
    y0_train = np.array(y0_train, dtype=np.int32)
    X1_train = X_train[w1_indices]
    y1_train = np.array(y1_train, dtype=np.int32)
    
    print(f"X_train shape: {X_train.shape} (Zero rate: {y0_train.mean():.3f})")
    print(f"X1_train shape: {X1_train.shape} (1x win rate among positive payouts: {y1_train.mean():.3f})")
    
    # Train Model A (p0 Loss Hazard)
    print("\nTraining Model A: Loss Hazard Estimator p0...")
    model_a = HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=5,
        learning_rate=0.06,
        l2_regularization=3.0,
        random_state=42
    )
    model_a.fit(X_train, y0_train)
    
    # Train Model B (Break-even Win Ratio q1)
    print("Training Model B: Break-even Ratio Estimator q1 = P(1x | Win)...")
    model_b = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        learning_rate=0.06,
        l2_regularization=3.0,
        random_state=42
    )
    model_b.fit(X1_train, y1_train)
    
    # Extract validation features
    print("\nGenerating Validation Predictions...")
    val_ep_seqs = defaultdict(lambda: defaultdict(dict))
    for p in ctrl.val_history:
        m_idx = int(p.machine_id[-1]) if p.machine_id[-1].isdigit() else 0
        val_ep_seqs[p.episode_id][m_idx][p.tick] = p.payout
        
    X_val = np.array([extract_tensor_features(q, val_ep_seqs[q.episode_id], q.cutoff) for q in val_qs], dtype=np.float32)
    
    # Predictions from Model A and Model B
    prob_0 = model_a.predict_proba(X_val)[:, 1]
    prob_1_cond = model_b.predict_proba(X_val)[:, 1]
    
    clean_sheet_preds = {}
    for i, q in enumerate(val_qs):
        p0 = float(prob_0[i])
        q1 = float(prob_1_cond[i])
        
        w_win = max(0.0, 1.0 - p0)
        p1 = w_win * q1
        w_high = max(0.0, w_win - p1)
        
        r = REEL_RATIOS[q.machine_type]
        p2 = w_high * r[0]
        p5 = w_high * r[1]
        p20 = w_high * r[2]
        
        vec = np.array([p0, p1, p2, p5, p20], dtype=np.float64)
        clean_sheet_preds[q.query_id] = tuple(vec / vec.sum())
        
    # Evaluate Clean-Sheet Model Alone:
    eval_clean = ctrl.evaluate_candidate(clean_sheet_preds, base_predictions=v10_preds, n_bootstrap=1000)
    print("\n" + "="*75)
    print("EXP-032 CLEAN-SHEET MODEL EVALUATION (Zero V10 code used):")
    print("="*75)
    print(f"Full Points:   {eval_clean['full_points']:.4f} / 60")
    print(f"Tuning Points: {eval_clean['tuning_points']:.4f} / 60")
    print(f"Audit Points:  {eval_clean['audit_points']:.4f} / 60")
    print(f"Loss Ratio R:  {eval_clean['full_loss_ratio']:.4f}")
    print("Conditions:")
    for c, r in eval_clean['by_condition'].items():
        print(f"  {c:14s}: loss_ratio = {r['loss_ratio']:.4f} (error = {r['error']:.5f})")
        
    # Evaluate Blend with V10
    print("\n" + "="*75)
    print("EXP-032 ENSEMBLE EVALUATION (Clean-Sheet + V10):")
    print("="*75)
    for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
        blend_preds = {}
        for q in val_qs:
            pv = np.array(v10_preds[q.query_id])
            pc = np.array(clean_sheet_preds[q.query_id])
            pb = (1.0 - alpha) * pv + alpha * pc
            blend_preds[q.query_id] = tuple(pb / pb.sum())
        res_blend = ctrl.evaluate_candidate(blend_preds, base_predictions=v10_preds, n_bootstrap=1000)
        print(f"Alpha(CleanSheet)={alpha:.2f} -> Full: {res_blend['full_points']:.4f} (Tune: {res_blend['tuning_points']:.4f}, Audit: {res_blend['audit_points']:.4f}) | CI vs v10: [{res_blend['bootstrap_gain_vs_base_ci95'][0]:.4f}, {res_blend['bootstrap_gain_vs_base_ci95'][1]:.4f}]")

if __name__ == "__main__":
    main()
