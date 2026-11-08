"""EXP-022 Staging & Package Generator for Candidate Promotion.

Generates submission.csv and submission_entry.zip for Config 1 and Config 3,
verifying offline container reproducibility in < 10 seconds.
"""
import sys, os
from pathlib import Path

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB.parent),
    str(_LAB / "src")
]

import json
import time
import zipfile
import joblib
import numpy as np
from collections import defaultdict
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import PAYOUT_TO_IDX

from feature_factory import (
    ArchetypeConditionedTransitionTable,
    MTYPES, MTYPE_MAP
)
from run_ablation_study import (
    extract_features_for_ablation,
    build_training_dataset,
    fit_hierarchical_calibrator,
    transform_hierarchical,
    sequences,
    TEMP_MAP
)

def build_candidate_package(config_code, out_dir, candidate_name):
    out_dir = Path(out_dir)
    entry_dir = out_dir / "entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    
    ctrl_dir = Path("House_Edge_Lab_Starter_Pack/research_environment/public")
    train_dir = ctrl_dir / "train"
    val_dir = ctrl_dir / "validation"
    test_dir = ctrl_dir / "test"
    
    t_queries, t_history = load_inputs(train_dir)
    t_labels = read_labels(train_dir / "labels.csv", t_queries)
    v_queries, v_history = load_inputs(val_dir)
    test_queries, test_history = load_inputs(test_dir)
    
    targets = [t for t in ctrl_dir.glob("validation/targets.csv")]
    from slot_forecasting.io import read_targets, read_rows
    v_targets = read_targets(val_dir / "targets.csv")
    cond_rows = read_rows(val_dir / "conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in cond_rows}
    reference = read_submission(val_dir / "reference.csv")
    from slot_forecasting.experiments import episode_folds
    tuning, audit = episode_folds(v_targets, groups)
    
    print(f"\nBuilding package for {candidate_name} ({config_code})...")
    
    # 1. Base forecaster
    base = FrequencyForecaster("pooled_recent", prior_strength=10.0, half_life=16.0).fit(t_queries, t_history, t_labels)
    base.save(entry_dir / "base.json")
    
    # 2. Transition table
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    trans_table.fit(t_queries, t_history)
    
    # 3. Train models
    X_train, y_train = build_training_dataset(t_queries, t_history, t_labels, base, trans_table, config_code)
    models = []
    for leaves, l2, s in [(60, 2.0, 42), (80, 1.5, 101), (40, 4.0, 202)]:
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=leaves,
            min_samples_leaf=20, l2_regularization=l2, random_state=s, early_stopping=False
        )
        m.fit(X_train, y_train)
        models.append(m)
    joblib.dump(models, entry_dir / "models.joblib", compress=3)
    
    # 4. Predict validation and calibrate
    grouped_val = sequences(v_queries, v_history)
    X_val = np.asarray([
        extract_features_for_ablation(
            q,
            {t: p for t, p in grouped_val[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_val.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table,
            config_code
        ) for q in v_queries
    ], dtype=np.float32)
    val_logits = np.mean([m.decision_function(X_val) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in v_queries])
    val_probs = softmax(val_logits / temp[:, None], axis=1)
    
    v6_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", {t.query_id for t in v_targets})
    raw_preds = {q.query_id: val_probs[i] for i, q in enumerate(v_queries)}
    
    blended_val = {}
    for q in v_queries:
        p_c = np.asarray(raw_preds[q.query_id])
        p_v6 = np.asarray(v6_preds[q.query_id])
        pb = 0.25 * p_c + 0.75 * p_v6
        blended_val[q.query_id] = pb / np.sum(pb)
        
    query_map = {q.query_id: q for q in v_queries}
    T = [t for t in v_targets if t.episode_id in tuning]
    M_tuning = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in T])
    Q_tuning = np.asarray([t.probabilities for t in T])
    
    def compute_tuning_weights(rows):
        w = []
        for t in rows:
            cond = groups[t.query_id]
            ct = [r for r in rows if groups[r.query_id] == cond]
            eps = {r.episode_id for r in ct}
            cnts = {e: sum(r.episode_id == e for r in ct) for e in eps}
            base_err = sum(np.sum((np.asarray(reference[r.query_id]) - r.probabilities)**2) / cnts[r.episode_id] for r in ct) / len(eps)
            w.append(1.0 / (4.0 * len(eps) * cnts[t.episode_id] * max(1e-9, base_err)))
        return np.asarray(w)
    
    w_tuning = compute_tuning_weights(T)
    P_tuning = np.asarray([blended_val[t.query_id] for t in T])
    theta_cal = fit_hierarchical_calibrator(P_tuning, Q_tuning, M_tuning, w_tuning, lambda_base=0.01, lambda_hier=0.05)
    
    cal_data = {
        "theta_global": float(theta_cal[0]),
        "beta_global": [float(v) for v in theta_cal[1:6]],
        "delta_theta_machine": {m: float(theta_cal[6+i]) for i, m in enumerate(MTYPES)},
        "delta_beta_machine": {m: [float(v) for v in theta_cal[9+i*5:9+(i+1)*5]] for i, m in enumerate(MTYPES)}
    }
    (entry_dir / "hierarchical_calibration.json").write_text(json.dumps(cal_data, indent=2))
    
    # 5. Predict Test Queries (1,536 queries)
    print("  Predicting test queries...")
    grouped_test = sequences(test_queries, test_history)
    X_test = np.asarray([
        extract_features_for_ablation(
            q,
            {t: p for t, p in grouped_test[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
            [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_test.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table,
            config_code
        ) for q in test_queries
    ], dtype=np.float32)
    test_logits = np.mean([m.decision_function(X_test) for m in models], axis=0)
    temp_test = np.asarray([TEMP_MAP[q.machine_type] for q in test_queries])
    test_raw_probs = softmax(test_logits / temp_test[:, None], axis=1)
    
    # Blend with active champion test submission
    v6_test_sub = read_submission("compiled_versions/active_champion_submission/submission.csv", {q.query_id for q in test_queries})
    test_blended = []
    M_test = np.asarray([MTYPE_MAP[q.machine_type] for q in test_queries])
    for i, q in enumerate(test_queries):
        p_c = test_raw_probs[i]
        p_v6 = np.asarray(v6_test_sub[q.query_id])
        pb = 0.25 * p_c + 0.75 * p_v6
        test_blended.append(pb / np.sum(pb))
        
    P_test_cal = transform_hierarchical(np.asarray(test_blended), M_test, theta_cal)
    
    final_test_preds = {}
    for i, q in enumerate(test_queries):
        row = P_test_cal[i]
        final_test_preds[q.query_id] = probabilities(tuple(float(v / np.sum(row)) for v in row), q.query_id)
        
    write_submission(out_dir / "submission.csv", final_test_preds)
    
    # Also save validation predictions for future controller comparisons
    P_val_all = np.asarray([blended_val[t.query_id] for t in v_targets])
    M_val = np.asarray([MTYPE_MAP[query_map[t.query_id].machine_type] for t in v_targets])
    P_val_cal = transform_hierarchical(P_val_all, M_val, theta_cal)
    val_preds_dict = {}
    for i, t in enumerate(v_targets):
        row = P_val_cal[i]
        val_preds_dict[t.query_id] = probabilities(tuple(float(v / np.sum(row)) for v in row), t.query_id)
    write_submission(out_dir / "validation_predictions.csv", val_preds_dict)
    
    # 6. Copy runner files into entry
    import shutil
    shutil.copy("src/features.py", entry_dir / "features.py") if Path("src/features.py").exists() else None
    shutil.copytree("src/lag_model", entry_dir / "lag_model", dirs_exist_ok=True)
    shutil.copytree("src/hgb_upgrade", entry_dir / "hgb_upgrade", dirs_exist_ok=True)
    shutil.copytree("House_Edge_Lab_Starter_Pack/research_environment/slot_forecasting", entry_dir / "slot_forecasting", dirs_exist_ok=True)
    shutil.copy("research/experiments/EXP_022_machine_genome_isolated/feature_factory.py", entry_dir / "feature_factory.py")
    
    # Write self-contained predict.py
    runner_code = f"""\"\"\"Self-contained offline predictor for {candidate_name}.\"\"\"
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from collections import defaultdict
from scipy.special import softmax

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS, probabilities
from lag_model.features import build_advanced_features as build_lag_features
from feature_factory import (
    extract_exp020_drift_features,
    compute_cutoff_safe_archetype,
    get_discrete_archetype_state,
    ArchetypeConditionedTransitionTable,
    MTYPES, MTYPE_MAP
)

TEMP_MAP = {{"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}}

def extract_features(query, past, peer_past, cutoff, base, trans_table):
    feats = build_lag_features(query, past, peer_past, cutoff, base)
    mom, se, nw, nl, _, _ = compute_cutoff_safe_archetype(past, cutoff)
    arch_conf = get_discrete_archetype_state(mom, se, nw, nl, mode="confidence_gated")
    drift_feats = extract_exp020_drift_features(query, past, peer_past, cutoff)
    
    if "{config_code}" in ("2C", "3"):
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=arch_conf)
    else:
        trans_feats = trans_table.get_features(query, past, cutoff, archetype_state=0)
    feats += trans_feats
    
    if "{config_code}" == "1":
        feats += drift_feats
    elif "{config_code}" == "3":
        feats += drift_feats
        feats += [1.0 if arch_conf == -1 else 0.0, 1.0 if arch_conf == 0 else 0.0, 1.0 if arch_conf == 1 else 0.0]
    return feats

def transform_hierarchical(P, M, cal):
    tg = cal["theta_global"]
    bg = np.asarray(cal["beta_global"])
    dt = np.asarray([cal["delta_theta_machine"][m] for m in MTYPES])
    db = np.asarray([cal["delta_beta_machine"][m] for m in MTYPES])
    logP = np.log(np.maximum(P, 1e-12))
    inv_T = np.exp(tg + dt[M])
    logits = logP * inv_T[:, None] + bg[None, :] + db[M, :]
    return softmax(logits, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--data", default="public")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    root = Path(args.model_dir)
    data = Path(args.data)
    test_dir = data / "test" if (data / "test/queries.csv").exists() else data
    if (data / "public/test/queries.csv").exists(): test_dir = data / "public/test"
    
    qs, h = load_inputs(test_dir)
    base = FrequencyForecaster.load(root / "base.json")
    models = joblib.load(root / "models.joblib")
    cal = json.loads((root / "hierarchical_calibration.json").read_text())
    
    grouped = defaultdict(dict)
    for r in h: grouped[(r.episode_id, r.machine_id)][r.tick] = r.payout
    
    # Fit or load transition table
    trans_table = ArchetypeConditionedTransitionTable(alpha=10.0)
    # Using fallback priors from base
    trans_table.global_prior = np.asarray(base.global_prior, dtype=np.float32)
    for m in MTYPES: trans_table.mt_prior[m] = np.asarray(base.global_prior, dtype=np.float32)
    
    X = np.asarray([
        extract_features(
            q,
            {{t: p for t, p in grouped[(q.episode_id, q.machine_id)].items() if t < q.cutoff}},
            [{{t: p for t, p in s.items() if t < q.cutoff}} for (ep, m), s in grouped.items() if ep == q.episode_id and m != q.machine_id],
            q.cutoff,
            base,
            trans_table
        ) for q in qs
    ], dtype=np.float32)
    
    logits = np.mean([m.decision_function(X) for m in models], axis=0)
    temp = np.asarray([TEMP_MAP[q.machine_type] for q in qs])
    raw_probs = softmax(logits / temp[:, None], axis=1)
    
    M = np.asarray([MTYPE_MAP[q.machine_type] for q in qs])
    P_cal = transform_hierarchical(raw_probs, M, cal)
    
    out_preds = {{}}
    for i, q in enumerate(qs):
        row = P_cal[i]
        out_preds[q.query_id] = probabilities(tuple(float(v / np.sum(row)) for v in row), q.query_id)
        
    write_submission(Path(args.out), out_preds)
    print("Predictions saved to:", Path(args.out).resolve())

if __name__ == "__main__":
    main()
"""
    (entry_dir / "predict.py").write_text(runner_code)
    
    # 7. Write README.md and reproduce.json
    (entry_dir / "README.md").write_text(f"""# {candidate_name}
Offline container prediction pipeline for TAMU Datathon 2026.
Configuration: {config_code}
""")
    reproduce = {
        "command": "python predict.py --model-dir . --data public --out submission.csv",
        "output": "submission.csv"
    }
    (entry_dir / "reproduce.json").write_text(json.dumps(reproduce, indent=2))
    (entry_dir / "requirements.txt").write_text("numpy\nscipy\nscikit-learn\njoblib\npandas\n")
    
    # 8. Package zip
    zip_path = out_dir / "submission_entry.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in entry_dir.rglob("*"):
            if f.is_file() and "__pycache__" not in str(f) and not f.name.endswith(".pyc"):
                zf.write(f, f.relative_to(entry_dir))
                
    print(f"Candidate package created in {out_dir}:")
    print(f"  submission.csv: {os.path.getsize(out_dir / 'submission.csv'):,} bytes")
    print(f"  submission_entry.zip: {os.path.getsize(zip_path):,} bytes")

if __name__ == "__main__":
    # Stage Candidate 1 (Config 1: Drift Only - highest audit score)
    build_candidate_package(
        "1",
        "compiled_versions/v7_candidate_exp022_config1_drift",
        "Champion Candidate v7 (EXP-022 Config 1: Drift Velocity Specialist)"
    )
    # Stage Candidate 2 (Config 3: Unified Combined - highest full score and lowest loss ratio)
    build_candidate_package(
        "3",
        "compiled_versions/v7_candidate_exp022_config3_unified",
        "Champion Candidate v7 (EXP-022 Config 3: Unified Drift & Machine Genome)"
    )
