"""Offline competition inference script for House Edge."""
import argparse
import sys
from pathlib import Path

# Add current folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slot_forecasting.io import load_inputs, write_submission
from src.model import HGBForecaster

def main():
    parser = argparse.ArgumentParser(description="House Edge Prediction Entry Point")
    parser.add_argument("--data", required=True, help="Path to public data folder")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    
    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    
    # Check if data_path contains 'test' subfolder or is test itself
    if (data_path / "test").is_dir():
        input_dir = data_path / "test"
    elif (data_path / "queries.csv").is_file():
        input_dir = data_path
    elif (data_path / "public" / "test").is_dir():
        input_dir = data_path / "public" / "test"
    else:
        raise FileNotFoundError(f"Cannot find queries.csv under {data_path}")
        
    print(f"Loading test inputs from {input_dir}...")
    queries, history = load_inputs(input_dir)
    print(f"Loaded {len(queries)} queries and {len(history)} history observations.")
    
    model_dir = Path(__file__).resolve().parent / "artifacts"
    print(f"Loading trained model from {model_dir}...")
    model = HGBForecaster.load(model_dir)
    
    print("Generating predictions...")
    predictions = model.predict(queries, history)
    
    print(f"Writing {len(predictions)} predictions to {out_path}...")
    write_submission(out_path, predictions)
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
