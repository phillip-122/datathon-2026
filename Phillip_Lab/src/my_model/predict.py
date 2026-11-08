"""Reproduce predictions from saved XGBoost and baseline models."""
import argparse
import json
from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, write_submission
from .model import HouseEdgeForecaster


def blend(tree, baseline, weight):
    return {key: tuple(weight * a + (1 - weight) * b for a, b in zip(tree[key], baseline[key]))
            for key in tree}


def predict_saved(model_dir, data, output):
    root = Path(model_dir)
    config = json.loads((root / "selection.json").read_text())
    q, h = load_inputs(Path(data) / "test")
    baseline = FrequencyForecaster.load(root / "baseline.json").predict(q, h)
    trees = HouseEdgeForecaster.load(root / "model.ubj").predict(q, h)
    write_submission(output, blend(trees, baseline, config["xgboost_weight"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    predict_saved(args.model_dir, args.data, args.out)
