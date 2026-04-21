"""
src/data_pipeline.py
--------------------
Ingestion and cleaning for the Telco Churn dataset.

Industry concept: Raw data is treated as IMMUTABLE. Every transformation
is reproducible — no manual edits to CSVs. This module reads raw, cleans,
and writes to data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and do a quick sanity check."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    assert "customerID" in df.columns, "Expected 'customerID' column"
    assert "Churn" in df.columns, "Expected 'Churn' target column"
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps. Returns a clean DataFrame.

    Industry concept: NEVER modify in place without logging what changed.
    Document every decision — future you (and code reviewers) need to know
    WHY a decision was made, not just what it was.
    """
    df = df.copy()

    # --- 1. Fix TotalCharges: it comes in as object due to spaces ---
    # Decision: spaces mean the customer is new (tenure=0). Fill with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_null = df["TotalCharges"].isna().sum()
    if n_null:
        logger.info(f"Filling {n_null} TotalCharges NaN with 0 (new customers, tenure=0)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --- 2. Encode binary target ---
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    churn_rate = df["Churn"].mean()
    logger.info(f"Class distribution — Churn rate: {churn_rate:.1%}")

    # --- 3. Drop the ID column (not a predictor) ---
    df = df.drop(columns=["customerID"])

    # --- 4. Standardise column names ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # --- 5. Fix SeniorCitizen (0/1 int → consistent Yes/No string for
    #         uniform encoding downstream) ---
    df["seniorcitizen"] = df["seniorcitizen"].map({0: "No", 1: "Yes"})

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Three-way split: train / validation / test.

    Industry concept: Keep a HELD-OUT test set that is NEVER used during
    model development or hyperparameter tuning. This is the only honest
    measure of generalisation. Validation set is used for tuning.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First cut: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second cut: train vs validation (relative to temp size)
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        f"Split — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    logger.info(
        f"Churn rates — Train: {y_train.mean():.1%} | Val: {y_val.mean():.1%} | Test: {y_test.mean():.1%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(config: dict) -> None:
    """End-to-end data preparation, writes to disk."""
    raw_path = config["data"]["raw_path"]
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # Save splits for reproducibility
    for name, data in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        data.to_csv(processed_dir / f"{name}.csv", index=False)
        logger.info(f"Saved {name}.csv")

    logger.info("✅ Data pipeline complete")


if __name__ == "__main__":
    cfg = load_config()
    run_pipeline(cfg)
