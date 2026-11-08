import sys
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.special import softmax

_LAB = Path(".").resolve()
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB / "src"),
    str(_LAB / "compiled_versions/v8_promoted_champion_routed/entry"),
    str(_LAB / "research/experiments/EXP_025_short_history_subpop_triage")
]

from slot_forecasting.io import load_inputs, read_submission
from slot_forecasting.schema import probabilities
from research.tools.controller import ResearchController, subset_score
from feature_factory_v25 import extract_features_v25, MTYPES, MTYPE_MAP

TEMP_MAP = {"steady": 1.00, "balanced": 1.06, "jackpot": 1.12}

ctrl = ResearchController()
val_queries = ctrl.val_queries
val_history = ctrl.val_history
targets = ctrl.val_targets
groups = ctrl.groups
reference = ctrl.reference
tuning = ctrl.tuning_eps
audit = ctrl.audit_eps

grouped_val = {}
for r in val_history:
    grouped_val.setdefault((r.episode_id, r.machine_id), {})[r.tick] = r.payout
n_past_val = {q.query_id: len([t for t in grouped_val[(q.episode_id, q.machine_id)] if t < q.cutoff]) for q in val_queries}

# Load v8 baseline predictions
v8_preds = read_submission("compiled_versions/v8_promoted_champion_routed/validation_predictions.csv", {t.query_id for t in targets})
eval_v8 = ctrl.evaluate_candidate(v8_preds)

# Load v25 model and base
v25_dir = Path("research/experiments/EXP_025_short_history_subpop_triage")
models_v25 = joblib.load(v25_dir / "models.joblib")
base_v25 = joblib.load(v25_dir / "base.json") if (v25_dir / "base.json").suffix == ".joblib" else None
from slot_forecasting.baseline import FrequencyForecaster
base_v25 = FrequencyForecaster.load(v25_dir / "base.json")
trans_data = json.loads((v25_dir / "trans_table.json").read_text())
from feature_factory_v25 import ArchetypeConditionedTransitionTable
from collections import Counter
trans_table_v25 = ArchetypeConditionedTransitionTable(alpha=trans_data["alpha"])
trans_table_v25.global_prior = np.asarray(trans_data["global_prior"], dtype=np.float32)
for m in MTYPES:
    trans_table_v25.mt_prior[m] = np.asarray(trans_data["mt_prior"][m], dtype=np.float32)
for k_str, cnt in trans_data["pair_counts"].items():
    p1, p2 = [int(x) for x in k_str.split(",")]
    trans_table_v25.pair_counts[(p1, p2)] = Counter({int(out): c for out, c in cnt.items()})
for arch_str, p_dict in trans_data["archetype_pair_counts"].items():
    arch = int(arch_str)
    for k_str, cnt in p_dict.items():
        p1, p2 = [int(x) for x in k_str.split(",")]
        trans_table_v25.archetype_pair_counts[arch][(p1, p2)] = Counter({int(out): c for out, c in cnt.items()})

print("Extracting X_val for v25...")
X_val_v25 = np.asarray([
    extract_features_v25(
        q,
        {t: p for t, p in grouped_val[(q.episode_id, q.machine_id)].items() if t < q.cutoff},
        [{t: p for t, p in s.items() if t < q.cutoff} for (ep, m), s in grouped_val.items() if ep == q.episode_id and m != q.machine_id],
        q.cutoff,
        base_v25,
        trans_table_v25
    ) for q in val_queries
], dtype=np.float32)

temp = np.asarray([TEMP_MAP[q.machine_type] for q in val_queries])
logits_v25 = np.mean([m.decision_function(X_val_v25) for m in models_v25], axis=0)
raw_v25 = softmax(logits_v25 / temp[:, None], axis=1)

print("Baseline v8: Full={:.4f}, Tune={:.4f}, Audit={:.4f}, ShortHist R={:.4f}".format(
    eval_v8['full_points'], eval_v8['tuning_points'], eval_v8['audit_points'], eval_v8['by_condition']['short_history']['loss_ratio']
))
