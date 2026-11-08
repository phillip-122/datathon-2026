"""Self-contained offline predictor for Adaptation Round Champion.

Architecture:
- Model: 3-seed HistGradientBoosting multi-structure ensemble fitted on Adaptation telemetry.
- Physical Reel Simplex Projection: Enforces zero-variance rational constants on (p2, p5, p20).
- Discrete 1x-Regime Mixture Inverter: Predicts posterior regime pi = P(S=1 | features) via logistic regression, smoothly blending conditional states.
- Machine-Specific Temperature Scaling: T_steady=0.90, T_balanced=1.00, T_jackpot=0.95.
"""

import argparse
import json
from pathlib import Path
import sys

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
    print("TAMU Datathon 2026 - House Edge Adaptation Round Champion Predictor")
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
    temp_map = regime_cfg.get("temp_scaling", {"steady": 0.90, "balanced": 1.00, "jackpot": 0.95})

    # 2. Run HGB Inference
    print("[3/4] Running multi-horizon tree ensemble inference ...")
    raw_preds = hgb.predict(queries, history)
    pred_mat = np.array([raw_preds[q.query_id] for q in queries])

    # 3. Physical Reel Simplex Projection
    print("[4/4] Applying Physical Reel Simplex & 1x-Regime Inverter ...")
    for i, q in enumerate(queries):
        mt = q.machine_type
        c2, c5, c20 = reel_ratios[mt]
        whigh = pred_mat[i, 2] + pred_mat[i, 3] + pred_mat[i, 4]
        pred_mat[i, 2] = whigh * c2
        pred_mat[i, 3] = whigh * c5
        pred_mat[i, 4] = whigh * c20
    pred_mat = pred_mat / np.sum(pred_mat, axis=1, keepdims=True)

    # Group history by (episode_id, machine_id)
    grouped_h = {}
    for s in history:
        grouped_h.setdefault((s.episode_id, s.machine_id), []).append(s)

    # 1x-Regime Mixture Inversion
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
        x_mid = 0.5 * (x_low + x_high)
        implied_logit = x - x_mid

        past = [s.payout for s in grouped_h.get((q.episode_id, q.machine_id), []) if s.tick < q.cutoff]
        c1_16 = sum(1 for p in past[-16:] if p == 1)
        chigh_16 = sum(1 for p in past[-16:] if p in [2, 5, 20])
        c1_32 = sum(1 for p in past[-32:] if p == 1)
        chigh_32 = sum(1 for p in past[-32:] if p in [2, 5, 20])

        feats = np.array([
            implied_logit,
            float(c1_16 - chigh_16),
            float(c1_32 - chigh_32),
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

    inv_mat = np.array(inv_preds)
    inv_mat = inv_mat / np.sum(inv_mat, axis=1, keepdims=True)

    # Machine-Specific Temperature Scaling
    mt_arr = np.array([q.machine_type for q in queries])
    t_vec = np.where(mt_arr == "steady", temp_map.get("steady", 0.90),
            np.where(mt_arr == "balanced", temp_map.get("balanced", 1.00), temp_map.get("jackpot", 0.95)))
    log_p = np.log(np.maximum(inv_mat, 1e-12))
    final_mat = softmax(log_p / t_vec[:, None], axis=1)

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
