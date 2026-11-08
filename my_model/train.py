"""Train on training data; choose on tuning episodes; audit the winner once."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import sys
import numpy as np
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import (load_inputs, read_labels, read_targets, read_rows,
                                 read_submission, write_submission, write_json)
from slot_forecasting.rounds import round_metadata
from slot_forecasting.submission_contract import package_code
from .features import FEATURE_NAMES, HistoryFeatures
from .model import HouseEdgeForecaster
from .predict import blend, predict_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="public")
    parser.add_argument("--out", default="house-edge-output")
    parser.add_argument("--max-rounds", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(2, 11)))
    args = parser.parse_args()
    if any(d < 1 or d > 16 for d in args.depths):
        parser.error("Depths must be between 1 and 16.")
    if args.max_rounds < 1 or args.checkpoint_every < 1:
        parser.error("Round counts must be positive.")
    public, out = Path(args.data).resolve(), Path(args.out).resolve()
    if out.exists():
        parser.error("Use a fresh output directory to preserve previous experiments.")
    if out == public or public in out.parents or out in public.parents:
        parser.error("Output must be separate from the input data.")
    metadata = round_metadata(public)
    train_q, train_h = load_inputs(public / "train")
    labels = read_labels(public / "train/labels.csv", train_q)
    val_q, val_h = load_inputs(public / "validation")
    targets = read_targets(public / "validation/targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(public / "validation/conditions.csv", ("query_id", "condition"))}
    reference = read_submission(public / "validation/reference.csv")
    tuning, audit = episode_folds(targets, groups)
    baseline = FrequencyForecaster("pooled_recent").fit(train_q, train_h, labels)
    base_prediction = baseline.predict(val_q, val_h)
    comparisons, models, predictions = {}, {}, {}
    # Train each trajectory once, then score prefixes using tuning episodes only.
    checkpoints = sorted(set(range(args.checkpoint_every, args.max_rounds + 1,
                                   args.checkpoint_every)) | {args.max_rounds})
    import xgboost as xgb
    validation_matrix = xgb.DMatrix(HistoryFeatures(val_q, val_h).matrix(val_q),
                                   feature_names=FEATURE_NAMES, nthread=2)
    for depth in sorted(set(args.depths)):
        print(f"Training depth {depth} through {args.max_rounds} rounds...", flush=True)
        full = HouseEdgeForecaster(depth=depth, rounds=args.max_rounds).fit(train_q, train_h, labels)
        for rounds in checkpoints:
            name = f"depth_{depth}_rounds_{rounds}"
            model = HouseEdgeForecaster(depth=depth, rounds=rounds)
            model.booster = full.booster[:rounds]
            model.training_rows = full.training_rows
            models[name] = model
            values = model.booster.predict(validation_matrix).astype(np.float64)
            values /= values.sum(axis=1, keepdims=True)
            predictions[name] = {q.query_id: tuple(p) for q, p in zip(val_q, values)}
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                key = f"{name}_weight_{weight}"
                result = subset_score(blend(predictions[name], base_prediction, weight), targets,
                                      groups, reference, tuning, metadata["max_points"])
                comparisons[key] = dict(model=name, rounds=rounds, xgboost_weight=weight, tuning=result)
            best = max(comparisons[f"{name}_weight_{w}"]["tuning"]["technical_points"]
                       for w in (0.0, 0.25, 0.5, 0.75, 1.0))
            print(f"  {rounds} rounds: best tuning points {best:.3f}", flush=True)
    chosen = min(comparisons, key=lambda k: (comparisons[k]["tuning"]["mean_loss_ratio"], k))
    selection = comparisons[chosen]
    winner = models[selection["model"]]
    prediction = blend(predictions[selection["model"]], base_prediction, selection["xgboost_weight"])
    audit_result = subset_score(prediction, targets, groups, reference, audit, metadata["max_points"])
    base_audit = subset_score(base_prediction, targets, groups, reference, audit, metadata["max_points"])
    out.mkdir(parents=True)
    # A small readable sample connects raw query/history data to model inputs.
    import csv
    preview = HistoryFeatures(train_q, train_h)
    with (out / "training-feature-preview.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", *FEATURE_NAMES, "observed_payout"])
        for q in train_q[:20]:
            writer.writerow([q.query_id, *preview.row(q), labels[q.query_id]])
    winner.save(out / "model.ubj")
    baseline.save(out / "baseline.json")
    write_json(out / "selection.json", selection)
    write_submission(out / "xgboost_validation_predictions.csv", predictions[selection["model"]])
    write_submission(out / "validation_predictions.csv", prediction)
    reloaded = HouseEdgeForecaster.load(out / "model.ubj").predict(val_q, val_h)
    difference = max(abs(a-b) for k in reloaded for a,b in zip(reloaded[k], predictions[selection["model"]][k]))
    if difference > 1e-8:
        raise RuntimeError("Saved model failed reproduction check.")
    predict_saved(out, public, out / "submission.csv")
    # Also verify serialization, complete test IDs, and reproducibility on test inputs.
    test_q, test_h = load_inputs(public / "test")
    tree_test = HouseEdgeForecaster.load(out / "model.ubj").predict(test_q, test_h)
    write_submission(out / "xgboost_submission.csv", tree_test)
    expected = blend(tree_test,
                     baseline.predict(test_q, test_h), selection["xgboost_weight"])
    actual = read_submission(out / "submission.csv", {q.query_id for q in test_q})
    test_difference = max(abs(a-b) for k in expected for a,b in zip(expected[k], actual[k]))
    if test_difference > 1e-8:
        raise RuntimeError("Test CSV failed reproduction check.")
    inputs = [public / n for n in ("metadata.json", "train/queries.csv", "train/history.csv", "train/labels.csv",
        "validation/queries.csv", "validation/history.csv", "validation/targets.csv", "validation/conditions.csv",
        "validation/reference.csv", "test/queries.csv", "test/history.csv")]
    report = dict(chosen=chosen, selection=selection, comparison=comparisons, audit=audit_result,
        baseline_audit=base_audit, tuning_episodes=sorted(tuning), audit_episodes=sorted(audit),
        settings=winner.settings, boosting_rounds=winner.rounds, training_rows=winner.training_rows,
        features=FEATURE_NAMES, python=sys.version,
        packages={p: importlib.metadata.version(p) for p in ("xgboost", "numpy", "scipy")},
        feature_importance_gain=winner.booster.get_score(importance_type="gain"),
        source_sha256={str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in (Path(__file__).parent, Path(__file__).parent.parent / "slot_forecasting")
                       for p in folder.glob("*.py")},
        input_sha256={str(p.relative_to(public)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        saved_model_max_difference=difference, test_csv_max_difference=test_difference,
        selection_rule="Lowest tuning loss ratio; audit scored only after selection. Audit is no longer fresh after this run.")
    write_json(out / "training-report.json", report)
    # Stage a compact offline prediction entry, excluding data and the virtual environment.
    entry = out / "entry"
    entry.mkdir()
    for package in ("my_model", "slot_forecasting"):
        destination = entry / package
        destination.mkdir()
        for source in (Path(__file__).resolve().parent.parent / package).glob("*.py"):
            shutil.copy2(source, destination / source.name)
    for name in ("model.ubj", "baseline.json", "selection.json", "training-report.json"):
        shutil.copy2(out / name, entry / name)
    shutil.copy2(Path(__file__).parent / "requirements.txt", entry / "requirements.txt")
    shutil.copy2(Path(__file__).parent / "README.md", entry / "README.md")
    write_json(entry / "reproduce.json", dict(image="python", command=["python", "-m", "my_model.predict",
        "--model-dir", ".", "--data", "/data", "--out", "/output/submission.csv"]))
    package_code(entry, out / "entry.zip")
    print(f"Selected: {chosen}\nAudit: {audit_result['technical_points']:.3f}/{metadata['max_points']}\n"
          f"Baseline audit: {base_audit['technical_points']:.3f}\nFiles: {out}", flush=True)


if __name__ == "__main__":
    main()
