"""EXP-016: Observation-Pattern Augmentation (Experiment C).

Hypothesis:
Incomplete observations in 'short_history' (40 recent + 24 sparse older ticks)
cause substantial error degradation when models are trained exclusively on dense
contiguous histories. Synthesizing realistic observation masks during training
teaches the tree ensemble to reason with incomplete observations.

We test:
1. Base training (no augmentation).
2. Exact 40-recent / 24-sparse observation topology augmentation.
3. Multi-pattern missingness augmentation (40/24 + variable interval gaps).
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

from research.tools.controller import ResearchController
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_json, write_submission
from lag_model.features import build_advanced_features, PAYOUT_TO_IDX

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

def sequences(queries, history):
    grouped = defaultdict(dict)
    for r in history:
        grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    return grouped

def mask_exact_40_24(seq, cutoff, rng):
    """Reproduce exact short_history observation topology: 40 recent + 24 sparse older."""
    recent = {t: p for t, p in seq.items() if cutoff - 40 <= t < cutoff}
    older = sorted(t for t in seq if t < cutoff - 40)
    keep_older = rng.choice(older, min(24, len(older)), replace=False) if older else []
    return {**recent, **{int(t): seq[t] for t in keep_older}}

def mask_random_gaps(seq, cutoff, rng):
    """Mask out 1 or 2 random gaps in history to simulate dropouts."""
    past_ticks = sorted(t for t in seq if t < cutoff)
    if len(past_ticks) < 30:
        return {t: seq[t] for t in past_ticks}
    # choose gap start
    gap_len = rng.integers(8, 24)
    start_idx = rng.integers(5, max(6, len(past_ticks) - gap_len - 5))
    gap_ticks = set(past_ticks[start_idx : start_idx + gap_len])
    return {t: seq[t] for t in past_ticks if t not in gap_ticks}

def build_training_data(qs, history, labels, base_forecaster, pattern_type="none", rng_seed=42):
    grouped = sequences(qs, history)
    X = []
    y = []
    rng = np.random.default_rng(rng_seed)
    
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
            
            # Base example
            X.append(build_advanced_features(q, past, earlier, t, base_forecaster))
            y.append(PAYOUT_TO_IDX[outcome])
            
            # Pattern augmentation
            if pattern_type == "exact_40_24" and len(past) >= 64:
                m_past = mask_exact_40_24(past, t, rng)
                m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                X.append(build_advanced_features(q, m_past, m_earlier, t, base_forecaster))
                y.append(PAYOUT_TO_IDX[outcome])
            elif pattern_type == "multi_pattern" and len(past) >= 64:
                # 50% chance of 40/24 topology, 50% chance of random gap
                if rng.random() < 0.5:
                    m_past = mask_exact_40_24(past, t, rng)
                    m_earlier = [mask_exact_40_24(s, t, rng) for s in earlier]
                else:
                    m_past = mask_random_gaps(past, t, rng)
                    m_earlier = [mask_random_gaps(s, t, rng) for s in earlier]
                X.append(build_advanced_features(q, m_past, m_earlier, t, base_forecaster))
                y.append(PAYOUT_TO_IDX[outcome])
                
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def predict_ensemble(qs, h, base_forecaster, models):
    grouped = sequences(qs, h)
    X = np.asarray([
        build_advanced_features(
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
    
    # Load inputs
    t_queries, t_history = load_inputs(train_dir)
    t_labels = read_labels(train_dir / "labels.csv", t_queries)
    
    v_queries, v_history = load_inputs(val_dir)
    targets = read_targets(val_dir / "targets.csv")
    group_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(val_dir / "reference.csv")
    tuning, audit = episode_folds(targets, groups)
    
    # Load Champion v3 base forecaster and models
    champ_entry = Path("compiled_versions/v3_promoted_champion_exp013")
    # Base forecaster
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0)
    base.fit(t_queries, t_history, t_labels)
    
    # Validation predictions of Champion v4 for benchmarking
    v4_path = Path("compiled_versions/v4_promoted_champion_exp014/validation_predictions.csv")
    v4_preds = read_submission(v4_path, {t.query_id for t in targets})
    
    ctrl = ResearchController()
    
    results = {}
    
    for pattern_name in ("exact_40_24", "multi_pattern"):
        print(f"\n=======================================================")
        print(f"EXP-016: Training with Pattern Augmentation: {pattern_name}")
        print(f"=======================================================")
        start_t = time.time()
        X_tr, y_tr = build_training_data(t_queries, t_history, t_labels, base, pattern_type=pattern_name)
        feat_time = time.time() - start_t
        print(f"Extracted {len(X_tr)} examples ({X_tr.shape[1]} features) in {feat_time:.1f}s")
        
        models = []
        for leaves, l2, s in [(40, 4.0, 42), (60, 2.0, 101), (80, 1.5, 202)]:
            t0 = time.time()
            m = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
                min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
            )
            m.fit(X_tr, y_tr)
            print(f"  Trained seed {s} (leaves={leaves}) in {time.time() - t0:.1f}s")
            models.append(m)
            
        # Predict on validation set
        preds = predict_ensemble(v_queries, v_history, base, models)
        
        # Test 25% specialist blend with Champion v4 base
        v2_raw_path = Path("../Phillip_Lab/compiled_versions/v2_changing_lags/validation_predictions.csv")
        v2_raw = read_submission(v2_raw_path, {t.query_id for t in targets})
        
        for w in (0.0, 0.25, 0.50):
            blended = mix_predictions(preds, v2_raw, w) if w > 0.0 else v2_raw
            rep = ctrl.evaluate_candidate(blended, base_predictions=v4_preds, n_bootstrap=1000)
            tag = f"{pattern_name}_blend_{w:.2f}"
            results[tag] = rep
            print(f"[{tag:25s}] Tuning Pts: {rep['tuning_points']:.4f} | Audit Pts: {rep['audit_points']:.4f} | Full: {rep['full_points']:.4f}")
            print(f"  Short History: Loss Ratio = {rep['by_condition']['short_history']['loss_ratio']:.4f}")
            print(f"  Stable:        Loss Ratio = {rep['by_condition']['stable']['loss_ratio']:.4f}")
            print(f"  Drift:         Loss Ratio = {rep['by_condition']['drift']['loss_ratio']:.4f}")
            print(f"  Shared Change: Loss Ratio = {rep['by_condition']['shared_change']['loss_ratio']:.4f}")
            
    out_dir = Path("research/experiments/EXP_016_observation_augmentation")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", results)
    print(f"\nCompleted EXP-016. Results written to {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
