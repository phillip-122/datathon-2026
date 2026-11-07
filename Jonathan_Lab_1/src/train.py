"""Train and save the production HGB forecaster."""
import argparse
import json
import time
from pathlib import Path

from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_submission
from src.model import HGBForecaster

def train_and_evaluate(data_dir, output_dir, model_artifact_dir):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    model_artifact_dir = Path(model_artifact_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print("=== TRAINING PRODUCTION MODEL ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model artifact directory: {model_artifact_dir}")
    
    # 1. Load data
    train_q, train_h = load_inputs(data_dir / "train")
    train_labels = read_labels(data_dir / "train/labels.csv", train_q)
    val_q, val_h = load_inputs(data_dir / "validation")
    val_targets = read_targets(data_dir / "validation/targets.csv")
    group_rows = read_rows(data_dir / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    reference = read_submission(data_dir / "validation/reference.csv")
    test_q, test_h = load_inputs(data_dir / "test")
    
    # 2. Fit model on training data
    print("Fitting HGBForecaster...")
    model = HGBForecaster(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=60,
        min_samples_leaf=25,
        l2_regularization=2.0,
        step_ticks=6,
        random_state=42
    )
    t0 = time.time()
    model.fit(train_q, train_h, train_labels)
    fit_duration = time.time() - t0
    print(f"Model fitted in {fit_duration:.2f}s")
    
    # 3. Save model artifacts
    print(f"Saving model artifacts to {model_artifact_dir}...")
    model.save(model_artifact_dir)
    
    # 4. Predict on validation
    print("Generating validation predictions...")
    val_preds = model.predict(val_q, val_h)
    val_csv = output_dir / "validation_predictions.csv"
    write_submission(val_csv, val_preds)
    
    # 5. Score validation predictions
    val_score = contest_score(val_preds, val_targets, groups, reference, max_points=60.0)
    print(f"\n==========================================")
    print(f"VALIDATION TECHNICAL POINTS: {val_score['technical_points']:.4f} / 60")
    print(f"Mean loss ratio: {val_score['mean_loss_ratio']:.4f}")
    for c, row in val_score["by_condition"].items():
        print(f"  {c:14s}: loss_ratio={row['loss_ratio']:.4f} (error={row['error']:.5f})")
    print(f"==========================================\n")
    
    # 6. Predict on test
    print("Generating test submission predictions...")
    test_preds = model.predict(test_q, test_h)
    test_csv = output_dir / "submission.csv"
    write_submission(test_csv, test_preds)
    print(f"Wrote submission to {test_csv} ({len(test_preds)} queries)")
    
    # 7. Write run report
    report = {
        "model": "HGBForecaster",
        "technical_points": val_score["technical_points"],
        "mean_loss_ratio": val_score["mean_loss_ratio"],
        "by_condition": val_score["by_condition"],
        "fit_seconds": fit_duration,
        "total_seconds": time.time() - start_time
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Run completed in {report['total_seconds']:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="House_Edge_Lab_Starter_Pack/research_environment/public")
    parser.add_argument("--out", default="runs/hgb_run")
    parser.add_argument("--artifacts", default="artifacts/models/hgb_production")
    args = parser.parse_args()
    train_and_evaluate(args.data, args.out, args.artifacts)

if __name__ == "__main__":
    main()
