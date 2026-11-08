"""Candidate v11 Experiment: Discrete Regime Decoder & Methodological Counter-Edge.

Pillars:
1. Methodological Integrity: Fit strictly on Tuning Fold (48 episodes); evaluate Audit Fold (48 episodes) once.
2. Latent 5-State Emission Matrix: Derived from ground truth cluster archetypes.
3. Floor Bankroll & Delay-8 Coupling: Features include floor payout velocity and individual lags.
4. Drift Recency Window: Recency-weighted features for non-stationary episodes.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax

_LAB = Path(".").resolve()
sys.path[:0] = [
    str(_LAB),
    str(_LAB / "House_Edge_Lab_Starter_Pack/research_environment"),
    str(_LAB / "src")
]

from research.tools.controller import ResearchController
from slot_forecasting.io import read_submission

def main():
    ctrl = ResearchController()
    qs = ctrl.val_queries
    targets = ctrl.val_targets
    target_map = {t.query_id: t.probabilities for t in targets}
    groups = ctrl.groups
    v10_preds = read_submission("compiled_versions/active_champion_submission/validation_predictions.csv", {t.query_id for t in targets})
    
    print("="*75)
    print("CANDIDATE V11 EXPERIMENTAL SUITE: DISCRETE REGIME DECODER")
    print("="*75)
    
    # Baseline V10 score:
    eval_v10 = ctrl.evaluate_candidate(v10_preds)
    print(f"Active Champion v10 Baseline: Full={eval_v10['full_points']:.4f} pts (Tune={eval_v10['tuning_points']:.4f}, Audit={eval_v10['audit_points']:.4f})")
    print(f"Condition Loss Ratios: stable={eval_v10['by_condition']['stable']['loss_ratio']:.4f}, short={eval_v10['by_condition']['short_history']['loss_ratio']:.4f}, drift={eval_v10['by_condition']['drift']['loss_ratio']:.4f}, shared={eval_v10['by_condition']['shared_change']['loss_ratio']:.4f}")
    
    # 1. TUNING FOLD ONLY: Discover canonical emission archetypes
    tuning_targets = [t for t in targets if t.episode_id in ctrl.tuning_eps]
    audit_targets = [t for t in targets if t.episode_id in ctrl.audit_eps]
    
    print(f"\n[Split Verification] Tuning Episodes: {len(ctrl.tuning_eps)} (Queries: {len(tuning_targets)}) | Audit Episodes: {len(ctrl.audit_eps)} (Queries: {len(audit_targets)})")
    
    p_tuning = np.array([t.probabilities for t in tuning_targets])
    km = KMeans(n_clusters=5, random_state=42).fit(p_tuning)
    order = np.argsort(-km.cluster_centers_[:, 0])
    centers = km.cluster_centers_[order]
    label_remap = {old: new for new, old in enumerate(order)}
    y_tuning = np.array([label_remap[l] for l in km.labels_])
    
    print("\nCanonical Generator Archetypes (Learned on Tuning Fold ONLY):")
    for k in range(5):
        c = centers[k]
        cnt = np.sum(y_tuning == k)
        print(f"  State {k} (N={cnt:3d}, {cnt/len(y_tuning):.1%}): [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {c[3]:.3f}, {c[4]:.3f}]")
        
    # 2. Extract features
    hist_map = defaultdict(list)
    for p in ctrl.val_history:
        hist_map[(p.episode_id, p.machine_id)].append(p)
    for k in hist_map:
        hist_map[k].sort(key=lambda x: x.tick)
        
    def extract_features(q):
        sub_h = [p for p in hist_map.get((q.episode_id, q.machine_id), []) if p.tick < q.cutoff]
        n = len(sub_h)
        streak_0 = 0
        for p in reversed(sub_h):
            if p.payout == 0:
                streak_0 += 1
            else:
                break
        last_8 = [p.payout for p in sub_h[-8:]] if n >= 8 else [p.payout for p in sub_h]
        last_16 = [p.payout for p in sub_h[-16:]] if n >= 16 else [p.payout for p in sub_h]
        last_32 = [p.payout for p in sub_h[-32:]] if n >= 32 else [p.payout for p in sub_h]
        f_0_8 = sum(1 for p in last_8 if p == 0) / len(last_8) if last_8 else 0.5
        f_0_16 = sum(1 for p in last_16 if p == 0) / len(last_16) if last_16 else 0.5
        f_0_32 = sum(1 for p in last_32 if p == 0) / len(last_32) if last_32 else 0.5
        f_1_16 = sum(1 for p in last_16 if p == 1) / len(last_16) if last_16 else 0.2
        f_2_16 = sum(1 for p in last_16 if p == 2) / len(last_16) if last_16 else 0.2
        f_5_16 = sum(1 for p in last_16 if p == 5) / len(last_16) if last_16 else 0.05
        f_20_16 = sum(1 for p in last_16 if p == 20) / len(last_16) if last_16 else 0.01
        wr_8 = sum(1 for p in last_8 if p > 0) / len(last_8) if last_8 else 0.5
        wr_32 = sum(1 for p in last_32 if p > 0) / len(last_32) if last_32 else 0.5
        mom = wr_8 - wr_32
        lags = [sub_h[-k].payout if n >= k else -1 for k in range(1, 9)]
        cutoff = q.cutoff
        floor_plays_16 = [p for (ep, m), plist in hist_map.items() if ep == q.episode_id and m != q.machine_id for p in plist if cutoff - 16 <= p.tick < cutoff]
        floor_pay_16 = sum(p.payout for p in floor_plays_16) / max(1, len(floor_plays_16))
        floor_wins_16 = sum(1 for p in floor_plays_16 if p.payout > 0) / max(1, len(floor_plays_16))
        mtype_onehot = [1.0 if q.machine_type == m else 0.0 for m in ["steady", "balanced", "jackpot"]]
        return [streak_0, f_0_8, f_0_16, f_0_32, f_1_16, f_2_16, f_5_16, f_20_16, wr_8, wr_32, mom, floor_pay_16, floor_wins_16, float(q.signal)] + [float(x) for x in lags] + mtype_onehot

    X_tuning = np.array([extract_features(ctrl.query_map[t.query_id]) for t in tuning_targets])
    X_audit = np.array([extract_features(ctrl.query_map[t.query_id]) for t in audit_targets])
    
    # 3. Fit Classifier on Tuning Fold ONLY
    print("\nTraining State Classifier on Tuning Fold...")
    clf = LogisticRegression(max_iter=1000, C=0.2, random_state=42)
    clf.fit(X_tuning, y_tuning)
    
    probs_tuning = clf.predict_proba(X_tuning)
    mix_tuning = probs_tuning @ centers
    
    probs_audit = clf.predict_proba(X_audit)
    mix_audit = probs_audit @ centers
    
    # 4. Tune gamma and threshold STRICTLY ON TUNING FOLD (No Audit Gate Peeking!)
    print("\nTuning Gamma and Confidence Threshold on TUNING FOLD ONLY:")
    best_gamma = 0.0
    best_thresh = 0.0
    best_tuning_pts = eval_v10['tuning_points']
    
    for thresh in [0.0, 0.40, 0.50, 0.60]:
        for gamma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
            cand_preds = dict(v10_preds)
            for idx, t in enumerate(tuning_targets):
                p_v10 = np.array(v10_preds[t.query_id])
                p_probs = probs_tuning[idx]
                if np.max(p_probs) >= thresh:
                    blended = (1.0 - gamma) * p_v10 + gamma * mix_tuning[idx]
                    cand_preds[t.query_id] = tuple(blended / blended.sum())
            
            res_tune = ctrl.evaluate_candidate(cand_preds)
            pts = res_tune['tuning_points']
            if pts > best_tuning_pts:
                best_tuning_pts = pts
                best_gamma = gamma
                best_thresh = thresh
                print(f"  * New Best Tuning Config: Thresh={thresh:.2f}, Gamma={gamma:.2f} -> Tuning Points: {pts:.4f} (+{pts - eval_v10['tuning_points']:.4f})")
                
    print(f"\nSelected Best Config on Tuning Fold: Thresh={best_thresh:.2f}, Gamma={best_gamma:.2f} (Tuning Points: {best_tuning_pts:.4f})")
    
    # 5. SINGLE-SHOT UNBIASED AUDIT GATE EVALUATION
    # Now, and ONLY now, apply the exact chosen config to Audit episodes!
    print("\n" + "="*75)
    print("STEP 5: SINGLE-SHOT UNBIASED AUDIT GATE VERIFICATION (Zero Peeking)")
    print("="*75)
    
    final_preds = dict(v10_preds)
    # Apply to tuning
    for idx, t in enumerate(tuning_targets):
        p_v10 = np.array(v10_preds[t.query_id])
        if np.max(probs_tuning[idx]) >= best_thresh:
            blended = (1.0 - best_gamma) * p_v10 + best_gamma * mix_tuning[idx]
            final_preds[t.query_id] = tuple(blended / blended.sum())
            
    # Apply to audit
    for idx, t in enumerate(audit_targets):
        p_v10 = np.array(v10_preds[t.query_id])
        if np.max(probs_audit[idx]) >= best_thresh:
            blended = (1.0 - best_gamma) * p_v10 + best_gamma * mix_audit[idx]
            final_preds[t.query_id] = tuple(blended / blended.sum())
            
    eval_final = ctrl.evaluate_candidate(final_preds, base_predictions=v10_preds, n_bootstrap=2000)
    print(f"Candidate v11 Full Points:   {eval_final['full_points']:.4f} / 60 (v10: {eval_v10['full_points']:.4f})")
    print(f"Candidate v11 Tuning Points: {eval_final['tuning_points']:.4f} / 60 (v10: {eval_v10['tuning_points']:.4f})")
    print(f"Candidate v11 Audit Points:  {eval_final['audit_points']:.4f} / 60 (v10: {eval_v10['audit_points']:.4f})")
    print(f"95% Paired Bootstrap CI vs v10: [{eval_final['bootstrap_gain_vs_base_ci95'][0]:.4f}, {eval_final['bootstrap_gain_vs_base_ci95'][1]:.4f}]")
    print("Condition Breakdown:")
    for c, r in eval_final['by_condition'].items():
        v10_r = eval_v10['by_condition'][c]['loss_ratio']
        print(f"  {c:14s}: loss_ratio = {r['loss_ratio']:.4f} (v10: {v10_r:.4f}, Diff: {r['loss_ratio'] - v10_r:+.4f})")

if __name__ == "__main__":
    main()
