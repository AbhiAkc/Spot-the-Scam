# ✅ retrain.py — CLI utility to retrain the model on demand
import os
from app.pipeline import train_model
import argparse

# -----------------------------
# CLI argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Retrain the Spot the Scam model")
parser.add_argument("--data", type=str, default=os.path.join("data", "train.csv"), help="Path to training CSV")
args = parser.parse_args()

# -----------------------------
# Run training function
# -----------------------------
print(f"🔁 Starting model retraining with data: {args.data}")
train_model(data_path=args.data)
print("✅ Model retraining complete!")