"""Verification and evaluation script for Short-History Triage in Round 2 Adaptation.

Applies:
1. Peer Bankroll Coupling (16-tick floor win rates)
2. Jackpot Piecewise Hazard Suppression (75% high payout freeze when streak >= 8)
3. Temperature calibration (T = 0.92)
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import softmax

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "adaptation-participants"))
sys.path.insert(0, str(_ROOT))

from slot_forecasting.contest import score_contest_files
from slot_forecasting.schema import probabilities
from slot_forecasting.io import write_submission


def evaluate():
    val_dir = _ROOT / "adaptation-participants" / "public" / "validation"
    q = pd.read_csv(val_dir / "queries.csv")
    c = pd.read_csv(val_dir / "conditions.csv")
    h = pd.read_csv(val_dir / "history.csv")
    
    cand_a2_csv = Path("/tmp/cand_a2_val.csv")
    if not cand_a2_csv.exists():
        # Generate cand_a2_val.csv from compiled version
        import subprocess
        subprocess.run([
            str(_ROOT / ".venv" / "bin" / "python"),
            str(_ROOT / "compiled_versions" / "adaptation_active_submission" / "entry" / "predict.py"),
            "--data", str(val_dir),
            "--out", str(cand_a2_csv)
        ], check=True)
        
    a2 = pd.read_csv(cand_a2_csv)
    p_cols = ["p_0", "p_1", "p_2", "p_5", "p_20"]
    a2_mat = a2[p_cols].values

    ep_h = defaultdict(lambda: defaultdict(dict))
    for _, r in h.iterrows():
        ep_h[r.episode_id][r.machine_id][r.tick] = r.payout

    c_dict = dict(zip(c["query_id"], c["condition"]))
    sh_indices = [i for i, row in q.iterrows() if c_dict[row.query_id] == "short_history"]

    # 1. Peer past 16 win rates
    peer_wr_16 = []
    for i in sh_indices:
        row = q.iloc[i]
        ep = row.episode_id
        mid = row.machine_id
        cutoff = row.cutoff
        peer_past = [
            seq[tick] for m, seq in ep_h[ep].items()
            if m != mid for tick in range(cutoff - 16, cutoff) if tick in seq
        ]
        p_wr = sum(1 for p in peer_past if p > 0) / len(peer_past) if peer_past else 0.4
        peer_wr_16.append(p_wr)

    peer_wr_16 = np.array(peer_wr_16)
    mean_peer_wr = peer_wr_16.mean()

    pred_mod = a2_mat.copy()

    # 1. Peer bankroll coupling (gamma = 0.80)
    gamma = 0.80
    for idx_sh, i in enumerate(sh_indices):
        p = pred_mod[i].copy()
        w_old = 1.0 - p[0]
        factor = max(0.05, 1.0 - gamma * (peer_wr_16[idx_sh] - mean_peer_wr))
        w_new = min(0.99, max(0.01, w_old * factor))
        p[0] = 1.0 - w_new
        win_sum = np.sum(p[1:])
        if win_sum > 0:
            p[1:] = p[1:] * (w_new / win_sum)
        pred_mod[i] = p / np.sum(p)

    # 2. Jackpot Piecewise Hazard Suppression (streak >= 8, 75% freeze)
    for idx_sh, i in enumerate(sh_indices):
        row = q.iloc[i]
        if row.machine_type == "jackpot":
            past = ep_h[row.episode_id][row.machine_id]
            cutoff = row.cutoff
            s0 = 0
            tick = cutoff - 1
            while tick in past and past[tick] == 0:
                s0 += 1
                tick -= 1
            if s0 >= 8:
                p = pred_mod[i].copy()
                old_whigh = p[2] + p[3] + p[4]
                new_whigh = old_whigh * 0.25
                p[2] *= 0.25
                p[3] *= 0.25
                p[4] *= 0.25
                p[0] += (old_whigh - new_whigh)
                pred_mod[i] = p / np.sum(p)

    # 3. Short-history temperature calibration (T = 0.92)
    for i in sh_indices:
        log_p = np.log(np.maximum(pred_mod[i], 1e-12))
        pred_mod[i] = softmax(log_p / 0.92)

    out_csv = Path("/tmp/cand_a2_triage_best.csv")
    out_preds = {row.query_id: probabilities(tuple(pred_mod[idx]), row.query_id) for idx, row in q.iterrows()}
    write_submission(out_csv, out_preds)

    res_orig = score_contest_files(
        cand_a2_csv,
        val_dir / "targets.csv",
        val_dir / "queries.csv",
        val_dir / "conditions.csv",
        val_dir / "reference.csv",
        max_points=20.0,
        version="temporal-adaptation-v2"
    )

    res_mod = score_contest_files(
        out_csv,
        val_dir / "targets.csv",
        val_dir / "queries.csv",
        val_dir / "conditions.csv",
        val_dir / "reference.csv",
        max_points=20.0,
        version="temporal-adaptation-v2"
    )

    print("=== FINAL CONTEST VERIFICATION ===")
    print(f"Original A2:      {res_orig['technical_points']:.6f} pts | Mean R: {res_orig['mean_loss_ratio']:.6f} | Short R: {res_orig['by_condition']['short_history']['loss_ratio']:.6f}")
    print(f"Enhanced Triage:  {res_mod['technical_points']:.6f} pts | Mean R: {res_mod['mean_loss_ratio']:.6f} | Short R: {res_mod['by_condition']['short_history']['loss_ratio']:.6f}")
    print(f"Net Gain:         +{res_mod['technical_points'] - res_orig['technical_points']:.6f} pts | Short R drop: {res_mod['by_condition']['short_history']['loss_ratio'] - res_orig['by_condition']['short_history']['loss_ratio']:.6f}")


if __name__ == "__main__":
    evaluate()
