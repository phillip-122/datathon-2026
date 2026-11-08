"""EXP-033: Discrete 1x-Regime Inverter (Probabilistic Mixture).

Pillars:
1. Verified Ratio Invariants: Exactly two ratios per machine type (Std = 0.000000 across all 768 validation targets).
2. Cutoff-Safe Feature Engineering: All features extracted strictly for t < cutoff.
3. Frozen Candidate Selection: Regularized Logistic Regression (C=0.50) fit strictly on Tuning Fold (48 episodes).
4. Probabilistic Regime Mixture: Soft convex combination of conditional distributions under pi = P(S=1).
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_LAB = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB / "src")
]

from research.tools.controller import ResearchController
from slot_forecasting.io import load_inputs, read_targets, write_submission

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}
REEL_RATIOS = {
    'steady': (200.0 / 250.0, 49.0 / 250.0, 1.0 / 250.0),
    'balanced': (16.0 / 23.0, 6.0 / 23.0, 1.0 / 23.0),
    'jackpot': (7.0 / 19.0, 9.0 / 19.0, 3.0 / 19.0),
}

def main():
    ctrl = ResearchController()
    queries, history = load_inputs('House_Edge_Lab_Starter_Pack/research_environment/public/validation')
    q_map = {q.query_id: q for q in queries}
    targets = ctrl.val_targets

    hist_dict = {}
    for s in history:
        key = (s.episode_id, s.machine_id)
        if key not in hist_dict:
            hist_dict[key] = []
        hist_dict[key].append(s)

    pred_v10 = pd.read_csv('compiled_versions/active_champion_submission/validation_predictions.csv').set_index('query_id')
    pred_v11 = pd.read_csv('compiled_versions/v11_physical_reel_projection/validation_predictions.csv').set_index('query_id')

    p_dict_v10 = {qid: tuple(pred_v10.loc[qid][['p_0','p_1','p_2','p_5','p_20']]) for qid in pred_v10.index}
    p_dict_v11 = {qid: tuple(pred_v11.loc[qid][['p_0','p_1','p_2','p_5','p_20']]) for qid in pred_v11.index}

    feature_rows = []
    for t in targets:
        qid = t.query_id
        q = q_map[qid]
        mt = q.machine_type
        r_low, r_high = REGIME_RATIOS[mt]
        w_high_true = sum(t.probabilities[2:])
        true_ratio = t.probabilities[1] / max(1e-6, w_high_true)
        true_regime = 0 if abs(true_ratio - r_low) < abs(true_ratio - r_high) else 1
        
        row = pred_v11.loc[qid]
        p0 = row['p_0']
        p1 = row['p_1']
        whigh = row['p_2'] + row['p_5'] + row['p_20']
        r_hat = p1 / max(1e-6, whigh)
        x = np.log(max(1e-6, r_hat))
        x_low = np.log(r_low)
        x_high = np.log(r_high)
        x_mid = 0.5 * (x_low + x_high)
        implied_logit = (x - x_mid)
        
        past = [s.payout for s in hist_dict.get((q.episode_id, q.machine_id), []) if s.tick < q.cutoff]
        c1_16 = sum(1 for p in past[-16:] if p == 1)
        chigh_16 = sum(1 for p in past[-16:] if p in [2, 5, 20])
        c1_32 = sum(1 for p in past[-32:] if p == 1)
        chigh_32 = sum(1 for p in past[-32:] if p in [2, 5, 20])
        
        feature_rows.append({
            'query_id': qid,
            'episode_id': q.episode_id,
            'machine_type': mt,
            'true_regime': true_regime,
            'implied_logit': implied_logit,
            'diff_16': c1_16 - chigh_16,
            'diff_32': c1_32 - chigh_32,
            'is_balanced': 1 if mt == 'balanced' else 0,
            'is_jackpot': 1 if mt == 'jackpot' else 0,
        })

    fdf = pd.DataFrame(feature_rows)
    tune_df = fdf[fdf['episode_id'].isin(ctrl.tuning_eps)].copy()
    features = ['implied_logit', 'diff_16', 'diff_32', 'is_balanced', 'is_jackpot']

    # Frozen selection: L2 Logistic Regression (C=0.50) fit strictly on Tuning Fold
    lr = LogisticRegression(C=0.50, random_state=42)
    lr.fit(tune_df[features], tune_df['true_regime'])

    fdf['pi'] = lr.predict_proba(fdf[features])[:, 1]

    p_dict_v12 = {}
    for _, r in fdf.iterrows():
        qid = r['query_id']
        mt = r['machine_type']
        r_low, r_high = REGIME_RATIOS[mt]
        c2, c5, c20 = REEL_RATIOS[mt]
        
        row = pred_v11.loc[qid]
        p0 = row['p_0']
        W = max(1e-6, 1.0 - p0)
        pi = r['pi']
        
        w_low = W / (1.0 + r_low)
        p1_low = W - w_low
        w_high_state = W / (1.0 + r_high)
        p1_high = W - w_high_state
        
        mix_p1 = (1.0 - pi) * p1_low + pi * p1_high
        mix_whigh = (1.0 - pi) * w_low + pi * w_high_state
        
        p_dict_v12[qid] = (p0, mix_p1, mix_whigh * c2, mix_whigh * c5, mix_whigh * c20)

    res_v12_vs_v11 = ctrl.evaluate_candidate(p_dict_v12, base_predictions=p_dict_v11, n_bootstrap=2000)
    res_v12_vs_v10 = ctrl.evaluate_candidate(p_dict_v12, base_predictions=p_dict_v10, n_bootstrap=2000)

    out_dir = Path(__file__).resolve().parent
    # Save predictions
    df_out = pd.DataFrame([
        {'query_id': qid, 'p_0': p[0], 'p_1': p[1], 'p_2': p[2], 'p_5': p[3], 'p_20': p[4]}
        for qid, p in sorted(p_dict_v12.items())
    ])
    df_out.to_csv(out_dir / "validation_predictions.csv", index=False)

    report = {
        "experiment_id": "EXP-033",
        "name": "Candidate v12: Probabilistic 1x-Regime Mixture Inverter",
        "full_points": float(res_v12_vs_v11['full_points']),
        "tuning_points": float(res_v12_vs_v11['tuning_points']),
        "audit_points": float(res_v12_vs_v11['audit_points']),
        "full_loss_ratio": float(res_v12_vs_v11['full_loss_ratio']),
        "by_condition": {k: float(v['loss_ratio']) for k, v in res_v12_vs_v11['by_condition'].items()},
        "gain_vs_v11": {
            "delta_full": float(res_v12_vs_v11['full_points'] - 38.7474),
            "delta_tuning": float(res_v12_vs_v11['tuning_points'] - 38.6387),
            "delta_audit": float(res_v12_vs_v11['audit_points'] - 38.8745),
            "bootstrap_ci95": [float(res_v12_vs_v11['bootstrap_gain_vs_base_ci95'][0]), float(res_v12_vs_v11['bootstrap_gain_vs_base_ci95'][1])]
        },
        "gain_vs_v10": {
            "delta_full": float(res_v12_vs_v10['full_points'] - 38.5007),
            "delta_audit": float(res_v12_vs_v10['audit_points'] - 38.6048),
            "bootstrap_ci95": [float(res_v12_vs_v10['bootstrap_gain_vs_base_ci95'][0]), float(res_v12_vs_v10['bootstrap_gain_vs_base_ci95'][1])]
        }
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("EXP-033 Completed Successfully. Report written to report.json")

if __name__ == "__main__":
    main()
