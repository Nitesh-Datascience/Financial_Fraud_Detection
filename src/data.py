"""Explicit per-source schemas: labels, IDs and post-event fields never enter X."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "bank": dict(
        file="bank_transactions.xlsx",
        target="isFraud",
        date="Date of transaction",
        amount="amount",
        category="type",
        location="branch",
        numeric=["amount", "oldbalanceOrg", "oldbalanceDest"],
        categorical=["type", "branch", "Acct type", "Time of day"],
    ),
    "synthetic": dict(
        file="synthetic_transactions.csv",
        target="Fraud_Label",
        date="Date",
        amount="Transaction_Amount",
        category="Transaction_Type",
        location="Location",
        numeric=["Transaction_Amount", "Card_Age"],
        categorical=[
            "Transaction_Type",
            "Device_Type",
            "Location",
            "Merchant_Category",
            "Card_Type",
        ],
    ),
    "credit_card": dict(
        file="credit_card_transactions.csv",
        target="IsFraud",
        date="TransactionDate",
        amount="Amount",
        category="TransactionType",
        location="Location",
        numeric=["Amount"],
        categorical=["TransactionType", "Location"],
    ),
}


def load_data(key):
    c = CONFIG[key]
    p = ROOT / "data" / c["file"]
    raw = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
    labels = (
        raw[c["target"]].map({"Safe": 0, "Fraud": 1})
        if key == "bank"
        else pd.to_numeric(raw[c["target"]], errors="coerce")
    )
    valid = labels.isin([0, 1])
    df = raw.loc[valid].copy()
    df["label"] = labels.loc[valid].astype(int)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    audit = {
        "source": c["file"],
        "raw_rows": len(raw),
        "excluded_unreviewed": int((~valid).sum()),
        "duplicate_rows_removed": duplicates,
        "rows": len(df),
        "fraud_count": int(df.label.sum()),
        "fraud_rate": float(df.label.mean()),
        "missing_cells_raw": int(raw.isna().sum().sum()),
    }
    return df, audit


def features(df, key):
    c = CONFIG[key]
    required = c["numeric"] + c["categorical"] + [c["date"]]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    x = pd.DataFrame(index=df.index)
    for col in c["numeric"]:
        x[col] = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if (df[col].notna() & x[col].isna()).any():
            raise ValueError(f"{col} contains invalid numbers")
        if (x[col].dropna() < 0).any():
            raise ValueError(f"{col} must be non-negative")
    for col in c["categorical"]:
        x[col] = df[col].fillna("Unknown").astype(str)
    fmt = {"bank": None, "credit_card": "%d-%m-%Y", "synthetic": "%d %B %Y"}[key]
    dates = pd.to_datetime(df[c["date"]], format=fmt, errors="coerce")
    if (df[c["date"]].notna() & dates.isna()).any():
        raise ValueError("Invalid date; use the supplied dataset date format.")
    x["month"] = dates.dt.month
    x["weekday"] = dates.dt.dayofweek
    x["log_amount"] = np.log1p(x[c["amount"]])
    return x
