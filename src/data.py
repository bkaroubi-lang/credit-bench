from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# IEEE-CIS categorical columns (per Kaggle data description).
# id_12..id_38 are categorical; id_01..id_11 are numeric.
CATEGORICAL_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)],
    "DeviceType", "DeviceInfo",
    *[f"id_{i:02d}" for i in range(12, 39)],
]

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


def _load_pair(trans_path: Path, ident_path: Path) -> pl.DataFrame:
    trans = pl.read_csv(trans_path, infer_schema_length=10_000, null_values=["", "NaN"])
    ident = pl.read_csv(ident_path, infer_schema_length=10_000, null_values=["", "NaN"])
    # The public test set uses "id-01" hyphenated names; normalize to "id_01".
    ident = ident.rename({c: c.replace("-", "_") for c in ident.columns if c.startswith("id-")})
    return trans.join(ident, on=ID_COL, how="left").sort(TIME_COL)


def load_train(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return _load_pair(raw_dir / "train_transaction.csv", raw_dir / "train_identity.csv")


def load_test(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    # Kaggle public test files: test_transaction.csv / test_identity.csv
    return _load_pair(raw_dir / "test_transaction.csv", raw_dir / "test_identity.csv")


def to_pandas_features(df: pl.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Return (X, y_or_None, time). Drops target and ID from X. Keeps NaNs as NaN."""
    pdf = df.to_pandas()
    y = pdf[TARGET].to_numpy().astype(np.int8) if TARGET in pdf.columns else None
    time = pdf[TIME_COL].to_numpy().astype(np.int64)
    drop = [c for c in (TARGET, ID_COL) if c in pdf.columns]
    X = pdf.drop(columns=drop)
    return X, y, time


def existing_categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_COLS if c in X.columns]
