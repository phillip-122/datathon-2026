"""EXP-017: Adaptive Regime Recognition & Adaptive Recency (Experiments A & B).

Hypotheses:
1. Adaptive Regime Detection (Experiment A): Sequence descriptors (recent payout entropy,
   short-vs-long volatility ratio, streak asymmetry, total variation divergence, and
   transition consistency) allow the model to distinguish between consistent, transitioning,
   and volatile machine behaviors without lookup-table memorization.
2. Adaptive Recency & Memory (Experiment B): Multi-horizon EWMA differential vectors
   and sigmoid instability gating inform the model when recency is informative vs noisy.
Target Conditions: Drift and Shared Change.
"""

import sys, os
from pathlib import Path
_LAB = Path(__file__).resolve().parents[2]
_REPO = _LAB.parent
sys.path[:0] = [str(_LAB), str(_LAB / 'House_Edge_Lab_Starter_Pack/research_environment'), str(_REPO), str(_LAB / 'src')]

from collections import defaultdict
import json
import time
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from research.features_regime import build_regime_features
from research.tools.controller import ResearchController
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_json, write_submission
from lag_model.features import build_advanced_features as base_features, PAYOUT_TO_IDX

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def build_combined_features(query, sequence, peer_sequences, cutoff, base_forecaster=None):
    """Combine base changing-lag features with 25 continuous regime & adaptive recency features."""
    base_f = base_features(query, sequence, peer_sequences, cutoff, base_forecaster)
    regime_f = build_regime_features(sequence, cutoff)
    return base_f + regime_f

def build_training_data(qs, history, labels, base_forecaster):
    grouped = sequences(qs, history)
    X = []
    y = []
    for q in qs:
        seq = grouped[(q.episode_id, q.machine_id)]
        peers = [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id]
        ticks = [t for t in range(30, q.cutoff, 6) if t in seq] + [q.cutoff]
        for t in ticks:
            past = {k: v for k, v in seq.items() if k < t}
            if len(past) < 8:
                continue
            earlier = [{k: v for k, v in s.items() if k < t} for s in peers]
            outcome = labels[q.query_id] if t == q.cutoff else seq[t]
            X.append(build_combined_features(q, past, earlier, t, base_forecaster))
            y.append(PAYOUT_TO_IDX[outcome])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def predict_ensemble(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_combined_features(
            q,
            grouped[(q.episode_id, q.machine_id)],
            [s for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base_forecaster
        ) for q in qs
    ], dtype=np.float32)
    
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    probs = softmax(logits / temp[:, None], axis=1)
    return {q.query_id: tuple(p / p.sum()) for q, p in zip(qs, probs)}

def mix_predictions(a, b, w):
    return {k: tuple(w * np.asarray(a[k]) + (1.0 - w) * np.asarray(b[k])) for k in a}

def main():
    val_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/validation")
    train_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public/train")
    
    t_queries, t_history = load_inputs(train_dir)
    t_labels = read_labels(train_dir / "labels.csv", t_queries)
    
    v_queries, v_history = load_inputs(val_dir)
    targets = read_targets(val_dir / "targets.csv")
    group_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(t_queries, t_history, t_labels)
    
    v4_path = Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv")
    v4_preds = read_submission(v4_path, {t.query_id for t in targets})
    
    v2_raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
    v2_raw = read_submission(v2_raw_path, {t.query_id for t in targets})
    
    ctrl = ResearchController()
    
    print("\n=======================================================")
    print("EXP-017: Extracting Regime Recognition Features...")
    print("=======================================================")
    t0 = time.time()
    X_tr, y_tr = build_training_data(t_queries, t_history, t_labels, base)
    print(f"Extracted {len(X_tr)} examples with {X_tr.shape[1]} features in {time.time() - t0:.1f}s")
    
    models = []
    for leaves, l2, s in [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]:
        t1 = time.time()
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_tr, y_tr)
        print(f"  Trained model with seed {s} (leaves={leaves}) in {time.time() - t1:.1f}s")
        models.append(m)
        
    print("\nPredicting on validation set...")
    preds = predict_ensemble(v_queries, v_history, base, models)
    
    results = {}
    print("\n--- Evaluating Regime Specialist and Blends vs Active Champion ---")
    for w in (0.0, 0.25, 0.50, 1.00):
        blended = mix_predictions(preds, v2_raw, w) if w > 0.0 else v2_raw
        rep = ctrl.evaluate_candidate(blended, base_predictions=v4_preds, n_bootstrap=1000)
        tag = f"regime_weight_{w:.2f}"
        results[tag] = rep
        print(f"[{tag:22s}] Tuning Pts: {rep['tuning_points']:.4f} | Audit Pts: {rep['audit_points']:.4f} | Full: {rep['full_points']:.4f}")
        print(f"  Drift:         Loss Ratio = {rep['by_condition']['drift']['loss_ratio']:.4f}")
        print(f"  Shared Change: Loss Ratio = {rep['by_condition']['shared_change']['loss_ratio']:.4f}")
        print(f"  Stable:        Loss Ratio = {rep['by_condition']['stable']['loss_ratio']:.4f}")
        print(f"  Short History: Loss Ratio = {rep['by_condition']['short_history']['loss_ratio']:.4f}")

    out_dir = Path("research/experiments/EXP_017_adaptive_regime_recognition")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", results)
    joblib.dump(models, out_dir / "regime_models.joblib", compress=3)
    print(f"\nArtifacts written to: {out_dir}")

if __name__ == "__main__":
    main()
