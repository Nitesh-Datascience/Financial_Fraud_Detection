"""Batch inference with the saved training preprocessing and validation threshold."""

import argparse, joblib, pandas as pd
from src.data import ROOT, CONFIG, features
from src.models import score

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(CONFIG), required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    a = p.parse_args()
    b = joblib.load(ROOT / "artifacts" / a.dataset / "best_model.joblib")
    df = pd.read_csv(a.input)
    s = score(b["model"], features(df, a.dataset), b["kind"])
    df["risk_score"] = s
    df["flagged_for_review"] = (s >= b["threshold"]).astype(int)
    df.to_csv(a.output, index=False)
    print(f"Saved {len(df)} predictions to {a.output}")
