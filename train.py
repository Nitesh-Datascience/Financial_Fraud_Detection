"""Run python train.py --dataset all. Test is untouched until model/threshold selection."""

import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "2")
import argparse, json, time, warnings, hashlib, sqlite3
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline
from src.data import ROOT, CONFIG, load_data, features
from src.models import preprocessor, candidates, score


def evaluate(y, s, t):
    pred = s >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        average_precision=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(fbeta_score(y, pred, beta=1, zero_division=0)),
        f2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f)])


def train(key, cap=20000):
    out = ROOT / "artifacts" / key
    out.mkdir(parents=True, exist_ok=True)
    df, audit = load_data(key)
    x = features(df, key)
    y = df.label
    indices = np.arange(len(df))
    if key == "synthetic":
        a, b = next(
            GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(
                x, y, groups=df.User_ID
            )
        )
        v, t = next(
            GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43).split(
                x.iloc[b], y.iloc[b], groups=df.User_ID.iloc[b]
            )
        )
        v, t = b[v], b[t]
        split_method = "60/20/20 approximate user-group split (no overlapping User_ID)"
    else:
        a, b = train_test_split(indices, test_size=0.4, stratify=y, random_state=42)
        v, t = train_test_split(b, test_size=0.5, stratify=y.iloc[b], random_state=43)
        split_method = (
            "60/20/20 stratified random benchmark; not a future-time backtest"
        )
    fit = a
    if cap and len(a) > cap:
        fit, _ = train_test_split(
            a, train_size=cap, stratify=y.iloc[a], random_state=44
        )
    splits = pd.DataFrame({"row_id": indices, "split": "unused"})
    for name, idx in [("train", a), ("validation", v), ("test", t)]:
        splits.loc[idx, "split"] = name
    splits["used_for_fit"] = False
    splits.loc[fit, "used_for_fit"] = True
    splits.to_csv(out / "split_manifest.csv", index=False)
    rows = []
    fitted = {}
    thresholds = {}
    for name, (kind, est) in candidates().items():
        start = time.time()
        ii = fit
        # Kernel methods have quadratic memory/time: explicit train-only stratified cap.
        if name in ["SVR", "One-Class SVM"] and len(ii) > 5000:
            ii, _ = train_test_split(
                ii, train_size=5000, stratify=y.iloc[ii], random_state=45
            )
        model = make_pipeline(preprocessor(x), est)
        kwargs = {}
        if name == "Deep Neural Network":
            kwargs["mlpclassifier__sample_weight"] = compute_sample_weight(
                "balanced", y.iloc[ii]
            )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            if name == "One-Class SVM":
                ii = ii[y.iloc[ii].to_numpy() == 0]
            model.fit(x.iloc[ii], y.iloc[ii], **kwargs)
        s = score(model, x.iloc[v], kind)
        th = threshold(y.iloc[v], s)
        row = {
            "model": name,
            "kind": kind,
            "fit_rows": len(ii),
            "threshold": th,
            "seconds": round(time.time() - start, 2),
            "warnings": " | ".join(sorted(set(str(w.message) for w in ws))),
        }
        row.update(
            {"val_" + k: value for k, value in evaluate(y.iloc[v], s, th).items()}
        )
        rows.append(row)
        fitted[name] = (model, kind)
        thresholds[name] = th
        print(key, name, "val AP", round(row["val_average_precision"], 4), flush=True)
    # Primary ranking by validation average precision, not by test or accuracy.
    eligible = [
        r for r in rows if r["kind"] == "classifier" and r["model"] != "Dummy baseline"
    ]
    winner = max(eligible, key=lambda r: (r["val_average_precision"], r["val_f2"]))[
        "model"
    ]
    for row in rows:
        model, kind = fitted[row["model"]]
        s = score(model, x.iloc[t], kind)
        row.update(
            {
                "test_" + k: value
                for k, value in evaluate(y.iloc[t], s, row["threshold"]).items()
            }
        )
    metrics = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    model, kind = fitted[winner]
    scores = score(model, x.iloc[t], kind)
    scored = df.iloc[t].copy()
    scored["row_id"] = t
    scored["risk_score"] = scores
    scored["flagged"] = (scores >= thresholds[winner]).astype(int)
    scored.to_csv(out / "test_predictions.csv", index=False)
    bundle = {
        "model": model,
        "kind": kind,
        "name": winner,
        "threshold": thresholds[winner],
        "dataset": key,
        "features": x.columns.tolist(),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, out / "best_model.joblib", compress=3)
    # Persist a separate full-data ETL table, never used for reported test metrics.
    with sqlite3.connect(ROOT / "artifacts" / "fraud_detection.db") as con:
        df.to_sql(key + "_transactions", con, if_exists="replace", index=False)
    audit.update(
        {
            "split_method": split_method,
            "train_rows": len(a),
            "validation_rows": len(v),
            "test_rows": len(t),
            "fit_cap": cap,
            "fit_rows": len(fit),
            "selected_model": winner,
            "selected_threshold": thresholds[winner],
            "feature_columns": x.columns.tolist(),
            "excluded_columns": [c for c in df if c not in x],
            "split_fraud_counts": {
                n: int(y.iloc[ix].sum())
                for n, ix in [("train", a), ("validation", v), ("test", t)]
            },
            "source_sha256": hashlib.sha256(
                (ROOT / "data" / CONFIG[key]["file"]).read_bytes()
            ).hexdigest(),
            "sklearn_version": sklearn.__version__,
        }
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print("FINISHED", key, winner, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["all", *CONFIG], default="all")
    p.add_argument(
        "--max-train",
        type=int,
        default=20000,
        help="Per-model train cap; 0 uses all training rows except kernel caps.",
    )
    args = p.parse_args()
    for key in CONFIG if args.dataset == "all" else [args.dataset]:
        train(key, args.max_train)
