"""EXP-011: Specialist Models by Machine Type.

Hypothesis: In House Edge, the three machine types (steady, balanced, jackpot)
exhibit fundamentally different payout dynamics, variance regimes, and jackpot likelihoods.
A single global model with 60 leaves must compromise split capacity across all three types.
Training three specialist HistGradientBoosting models dedicated to each machine type
allows each model to specialize its tree splits and probability distributions,
improving calibration and reducing error across both tuning and audit sets.
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from research.evaluator import ResearchEvaluator
from research.run_experiment_helper import run_standard_eval
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels
from slot_forecasting.schema import PAYOUTS
from src.features import build_advanced_features, PAYOUT_TO_IDX

EXP_ID = "EXP-011"
OUT_DIR = Path("research/experiments/EXP_011_specialist")

def main():
    print(f"=== Starting {EXP_ID}: Specialist Models by Machine Type ===")
    t_start = time.time()
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q, val_h = evaluator.val_queries, evaluator.val_history
    
    base_forecaster = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base_forecaster.fit(train_q, train_h, train_labels)
    
    train_hist = defaultdict(lambda: defaultdict(dict))
    for h in train_h:
        train_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    # Group training examples by machine type
    m_types = ("steady", "balanced", "jackpot")
    X_train_by_type = {t: [] for t in m_types}
    y_train_by_type = {t: [] for t in m_types}
    
    print("Extracting specialist training data (step = 6 ticks)...")
    for q in train_q:
        ep_seqs = train_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        m_t = q.machine_type
        
        if q.query_id in train_labels:
            f = build_advanced_features(q, m_seq, peers, q.cutoff, base_forecaster)
            X_train_by_type[m_t].append(f)
            y_train_by_type[m_t].append(PAYOUT_TO_IDX[train_labels[q.query_id]])
            
        for t in range(30, q.cutoff, 6):
            if t in m_seq:
                sub_seq = {tick: p for tick, p in m_seq.items() if tick < t}
                if len(sub_seq) >= 8:
                    sub_peers = [{tick: p for tick, p in p_seq.items() if tick < t} for p_seq in peers]
                    f = build_advanced_features(q, sub_seq, sub_peers, t, base_forecaster)
                    X_train_by_type[m_t].append(f)
                    y_train_by_type[m_t].append(PAYOUT_TO_IDX[m_seq[t]])
                    
    for t in m_types:
        X_train_by_type[t] = np.array(X_train_by_type[t], dtype=np.float32)
        y_train_by_type[t] = np.array(y_train_by_type[t], dtype=np.int32)
        print(f"Type {t:10s}: {len(X_train_by_type[t])} samples, class counts: {np.bincount(y_train_by_type[t], minlength=5)}")
        
    # Extract validation features
    val_hist = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        val_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    val_q_list = list(val_q)
    val_feats_by_type = {t: [] for t in m_types}
    val_q_by_type = {t: [] for t in m_types}
    
    for q in val_q_list:
        ep_seqs = val_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_advanced_features(q, m_seq, peers, q.cutoff, base_forecaster)
        val_feats_by_type[q.machine_type].append(f)
        val_q_by_type[q.machine_type].append(q)
        
    # Train specialist models
    # Test hyperparameter capacity: 40 vs 50 vs 60 leaf nodes
    specialist_models = {}
    for t in m_types:
        print(f"Fitting specialist model for {t}...")
        # Note: Jackpot has rare 20x payouts, so min_samples_leaf=15 or 20 ensures rare class is kept
        leaf_nodes = 45 if t == "jackpot" else 50
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=leaf_nodes,
            min_samples_leaf=15 if t == "jackpot" else 20,
            l2_regularization=2.0,
            random_state=42
        )
        clf.fit(X_train_by_type[t], y_train_by_type[t])
        specialist_models[t] = clf
        
    # Generate predictions using specialist routing
    preds = {}
    for t in m_types:
        X_val_t = np.array(val_feats_by_type[t], dtype=np.float32)
        probs_t = specialist_models[t].predict_proba(X_val_t)
        # Check class ordering of specialist
        classes = specialist_models[t].classes_
        for i, q in enumerate(val_q_by_type[t]):
            raw = np.zeros(5, dtype=np.float64)
            for c_idx, c_val in enumerate(classes):
                raw[c_val] = probs_t[i, c_idx]
            tot = sum(raw)
            preds[q.query_id] = tuple(float(v / tot) for v in raw)
            
    res = evaluator.evaluate(preds)
    
    print("\n--- RESULTS COMPARISON ---")
    print(f"Champion EXP-006: Tune: 35.1863 | Audit: 34.9486 | Full: 35.0743")
    print(f"EXP-011:          Tune: {res['tuning_points']:.4f} | Audit: {res['audit_points']:.4f} | Full: {res['full_points']:.4f}")
    print(f"Mean loss ratio:  {res['full_loss_ratio']:.4f}")
    for c, row in res["by_condition"].items():
        print(f"  {c:14s}: ratio={row['loss_ratio']:.4f} (err={row['error']:.4f}, ref={row['reference_error']:.4f})")
        
    decision = "PROMOTED" if (res["tuning_points"] > 35.1863 and res["audit_points"] > 34.9486) else "REJECTED"
    
    run_standard_eval(
        exp_id=EXP_ID,
        hypothesis="Specialist HistGradientBoosting models dedicated to each machine type",
        author="ModelSpecialist",
        predictions=preds,
        out_dir=OUT_DIR,
        notes=f"Decision: {decision}. Total time: {time.time() - t_start:.1f}s"
    )
    print(f"\nFinal Decision for {EXP_ID}: {decision}")

if __name__ == "__main__":
    main()
