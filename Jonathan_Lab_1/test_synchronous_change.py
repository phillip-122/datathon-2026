import pandas as pd
import numpy as np

def test_ep_sync(data_dir, ep_id):
    hist = pd.read_csv(f"{data_dir}/history.csv")
    ep_df = hist[hist["episode_id"] == ep_id]
    pivot = ep_df.pivot(index="tick", columns="machine_id", values="payout")
    
    print(f"\n--- Episode {ep_id} Machine Breakdown ---")
    change_points = {}
    for m in pivot.columns:
        s = (pivot[m] > 0).astype(float)
        best_t = None
        best_diff = 0
        direction = 0
        for t in range(20, len(s) - 20):
            diff = s.iloc[t:].mean() - s.iloc[:t].mean()
            if abs(diff) > best_diff:
                best_diff = abs(diff)
                best_t = t
                direction = np.sign(diff)
        change_points[m] = (best_t, best_diff, direction)
        print(f"Machine {m[-4:]}: changepoint t={best_t:3d}, delta={direction*best_diff:+.3f}")
        
    ts = [cp[0] for cp in change_points.values()]
    print(f"Spread of change points across 8 machines: min={min(ts)}, max={max(ts)}, std={np.std(ts):.2f}")

test_ep_sync("adaptation-participants/public/validation", "adaptation-validation-e00010")
test_ep_sync("adaptation-participants/public/validation", "adaptation-validation-e00011")
test_ep_sync("adaptation-participants/public/train", "adaptation-train-e00006")
