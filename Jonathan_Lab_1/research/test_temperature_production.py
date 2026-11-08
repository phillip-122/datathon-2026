"""Test production integration of temperature calibration T=1.06."""

import json
import math
from pathlib import Path
import joblib
import numpy as np

from research.evaluator import ResearchEvaluator
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from slot_forecasting.schema import probabilities
from src.features import build_advanced_features

def test_calibrated_production():
    evaluator = ResearchEvaluator()
    data_dir = evaluator.data_dir
    
    # Load model
    model_dir = Path("artifacts/models/hgb_production")
    hgb = joblib.load(model_dir / "hgb_model.joblib")
    
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(train_q, train_h, train_labels)
    
    # 1. Validation predictions
    val_q, val_h = evaluator.val_queries, evaluator.val_history
    val_q_list = list(val_q)
    
    from collections import defaultdict
    val_hist = defaultdict(lambda: defaultdict(dict))
    for h in val_h:
        val_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X_val = []
    for q in val_q_list:
        ep_seqs = val_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_advanced_features(q, m_seq, peers, q.cutoff, base)
        X_val.append(f)
    X_val = np.array(X_val, dtype=np.float32)
    
    # Calibrated logits with T = 1.06
    T = 1.06
    logits = hgb.decision_function(X_val)
    scaled = logits / T
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    
    val_preds = {}
    for i, q in enumerate(val_q_list):
        raw = probs[i]
        tot = sum(raw)
        val_preds[q.query_id] = tuple(float(v / tot) for v in raw)
        
    eval_res = evaluator.evaluate(val_preds)
    print("=== CALIBRATED PRODUCTION EVALUATION (T=1.06) ===")
    print(f"Tuning Points: {eval_res['tuning_points']:.4f}")
    print(f"Audit Points:  {eval_res['audit_points']:.4f}")
    print(f"Full Points:   {eval_res['full_points']:.4f}")
    print(f"Mean Loss Ratio: {eval_res['full_loss_ratio']:.4f}")
    for c, row in eval_res["by_condition"].items():
        print(f"  {c:14s}: ratio={row['loss_ratio']:.4f}")
        
    # 2. Test predictions
    test_q, test_h = load_inputs(data_dir / "test")
    test_q_list = list(test_q)
    test_hist = defaultdict(lambda: defaultdict(dict))
    for h in test_h:
        test_hist[h.episode_id][h.machine_id][h.tick] = h.payout
        
    X_test = []
    for q in test_q_list:
        ep_seqs = test_hist[q.episode_id]
        m_seq = ep_seqs[q.machine_id]
        peers = [ep_seqs[m] for m in ep_seqs if m != q.machine_id]
        f = build_advanced_features(q, m_seq, peers, q.cutoff, base)
        X_test.append(f)
    X_test = np.array(X_test, dtype=np.float32)
    
    test_logits = hgb.decision_function(X_test)
    test_scaled = test_logits / T
    test_scaled -= np.max(test_scaled, axis=1, keepdims=True)
    test_exp = np.exp(test_scaled)
    test_probs = test_exp / np.sum(test_exp, axis=1, keepdims=True)
    
    test_preds = {}
    for i, q in enumerate(test_q_list):
        raw = test_probs[i]
        tot = sum(raw)
        normalized = tuple(float(v / tot) for v in raw)
        test_preds[q.query_id] = probabilities(normalized, q.query_id)
        
    out_path = Path("research/experiments/EXP_012_temperature_calibration/test_submission.csv")
    write_submission(out_path, test_preds)
    print(f"\nWrote test submission to {out_path}")

if __name__ == "__main__":
    test_calibrated_production()
