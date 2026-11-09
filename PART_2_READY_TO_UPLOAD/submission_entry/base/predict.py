"""Self-contained offline predictor for Adaptation Round Champion (Candidate A4 Unified).

Architecture:
- High-Capacity Multi-Seed HGB: 3 seeds with max_leaf_nodes=150 fitted on Adaptation telemetry.
- Physical Reel Simplex Projection: Enforces zero-variance rational constants on (p2, p5, p20).
- Full-Horizon Discrete 1x-Regime Mixture Inverter: Full-horizon feature tracking (H=inf, AUC > 0.72) softly blending states.
- Short-History Specialist Triage: Peer bankroll coupling, jackpot piecewise hazard suppression, and polarization calibration for N <= 64 queries.
"""

import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from scipy.special import softmax

from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import probabilities


def main():
    parser = argparse.ArgumentParser(description="House Edge Adaptation Predictor")
    parser.add_argument("--data", required=True, help="Path to input data directory")
    parser.add_argument("--out", required=True, help="Destination path for output predictions CSV")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if (data_path / "test/queries.csv").exists():
        input_dir = data_path / "test"
    elif (data_path / "public/test/queries.csv").exists():
        input_dir = data_path / "public/test"
    elif (data_path / "queries.csv").exists():
        input_dir = data_path
    else:
        found = list(data_path.rglob("queries.csv"))
        if found:
            input_dir = found[0].parent
        else:
            raise FileNotFoundError(f"Could not locate queries.csv within {data_path}")

    print("=" * 70)
    print("TAMU Datathon 2026 - House Edge Adaptation Unified Champion Predictor (Candidate A4)")
    print("=" * 70)
    print(f"[1/4] Loading input data from: {input_dir} ...")
    queries, history = load_inputs(input_dir)
    print(f"      Loaded {len(queries)} queries and {len(history)} history records.")

    # 1. Load trained artifacts
    print("[2/4] Loading model weights & regime parameters ...")
    hgb = joblib.load(_ROOT / "hgb_models.joblib")
    with open(_ROOT / "regime_config.json", "r") as f:
        regime_cfg = json.load(f)

    coef = np.asarray(regime_cfg["coef"], dtype=np.float64)
    intercept = float(regime_cfg["intercept"])
    reel_ratios = regime_cfg["reel_ratios"]
    regime_ratios = regime_cfg["regime_ratios"]

    # 2. Run HGB Inference
    print("[3/4] Running multi-horizon tree ensemble inference ...")
    raw_preds = hgb.predict(queries, history)
    pred_mat = np.array([raw_preds[q.query_id] for q in queries], dtype=np.float64, copy=True)

    # 3. Physical Reel Simplex Projection
    print("[4/4] Applying Physical Reel Simplex, 1x-Inverter & Short-History Triage ...")
    for i, q in enumerate(queries):
        mt = q.machine_type
        c2, c5, c20 = reel_ratios[mt]
        whigh = pred_mat[i, 2] + pred_mat[i, 3] + pred_mat[i, 4]
        pred_mat[i, 2] = whigh * c2
        pred_mat[i, 3] = whigh * c5
        pred_mat[i, 4] = whigh * c20
    pred_mat = pred_mat / np.sum(pred_mat, axis=1, keepdims=True)

    # Group history by (episode_id, machine_id)
    grouped_h = defaultdict(lambda: defaultdict(dict))
    for s in history:
        grouped_h[s.episode_id][s.machine_id][s.tick] = s.payout

    # 1x-Regime Mixture Inversion (Full Horizon H=inf)
    inv_preds = []
    for i, q in enumerate(queries):
        mt = q.machine_type
        r_low, r_high = regime_ratios[mt]
        c2, c5, c20 = reel_ratios[mt]

        p0 = pred_mat[i, 0]
        p1 = pred_mat[i, 1]
        whigh = pred_mat[i, 2] + pred_mat[i, 3] + pred_mat[i, 4]
        r_hat = p1 / max(1e-6, whigh)
        x = np.log(max(1e-6, r_hat))
        x_low = np.log(r_low)
        x_high = np.log(r_high)
        implied_logit = x - 0.5 * (x_low + x_high)

        seq = grouped_h[q.episode_id][q.machine_id]
        past = [seq[tick] for tick in sorted(seq.keys()) if tick < q.cutoff]

        c1_full = sum(1 for p in past if p == 1)
        chigh_full = sum(1 for p in past if p in [2, 5, 20])
        c1_32 = sum(1 for p in past[-32:] if p == 1)
        chigh_32 = sum(1 for p in past[-32:] if p in [2, 5, 20])
        ratio_full = float(c1_full) / max(1.0, float(chigh_full))

        feats = np.array([
            implied_logit,
            float(c1_full - chigh_full),
            float(c1_32 - chigh_32),
            ratio_full,
            1.0 if mt == "balanced" else 0.0,
            1.0 if mt == "jackpot" else 0.0
        ], dtype=np.float64)

        logit = float(np.dot(feats, coef) + intercept)
        pi = 1.0 / (1.0 + np.exp(-logit))

        W = max(1e-6, 1.0 - p0)
        w_low = W / (1.0 + r_low)
        p1_low = W - w_low
        w_high_state = W / (1.0 + r_high)
        p1_high = W - w_high_state

        mix_p1 = (1.0 - pi) * p1_low + pi * p1_high
        mix_whigh = (1.0 - pi) * w_low + pi * w_high_state

        inv_preds.append([p0, mix_p1, mix_whigh * c2, mix_whigh * c5, mix_whigh * c20])

    final_mat = np.array(inv_preds, dtype=np.float64, copy=True)
    final_mat = final_mat / np.sum(final_mat, axis=1, keepdims=True)

    # 4. Short-History Specialist Triage (queries with len(past) <= 64)
    for i, q in enumerate(queries):
        past = grouped_h[q.episode_id][q.machine_id]
        past_ticks = [tick for tick in past if tick < q.cutoff]

        if len(past_ticks) <= 64:
            p = final_mat[i].copy()

            # A. Peer Bankroll Coupling (past 16 ticks)
            peer_ticks_16 = [
                seq[tick] for m, seq in grouped_h[q.episode_id].items()
                if m != q.machine_id
                for tick in range(q.cutoff - 16, q.cutoff) if tick in seq
            ]
            peer_wr_16 = sum(1 for x in peer_ticks_16 if x > 0) / len(peer_ticks_16) if peer_ticks_16 else 0.3968
            gamma = 0.80
            mean_peer_wr = 0.3968
            factor = max(0.05, 1.0 - gamma * (peer_wr_16 - mean_peer_wr))

            w_old = 1.0 - p[0]
            w_new = min(0.99, max(0.01, w_old * factor))
            p[0] = 1.0 - w_new
            win_sum = np.sum(p[1:])
            if win_sum > 0:
                p[1:] = p[1:] * (w_new / win_sum)

            # B. Jackpot Piecewise Hazard Suppression
            if q.machine_type == "jackpot":
                s0 = 0
                tick = q.cutoff - 1
                while tick in past and past[tick] == 0:
                    s0 += 1
                    tick -= 1
                if s0 >= 8:
                    old_whigh = p[2] + p[3] + p[4]
                    new_whigh = old_whigh * 0.25
                    p[2] *= 0.25
                    p[3] *= 0.25
                    p[4] *= 0.25
                    p[0] += (old_whigh - new_whigh)

            # C. Polarization Temperature Calibration (T = 0.92)
            log_p = np.log(np.maximum(p, 1e-12))
            p = softmax(log_p / 0.92)
            final_mat[i] = p

    final_mat = final_mat / np.sum(final_mat, axis=1, keepdims=True)

    # Format predictions dictionary
    out_preds = {
        q.query_id: probabilities(tuple(final_mat[i]), q.query_id)
        for i, q in enumerate(queries)
    }

    out_file = Path(args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_submission(out_file, out_preds)
    print(f"[SUCCESS] Forecast written: {len(out_preds)} queries saved to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
