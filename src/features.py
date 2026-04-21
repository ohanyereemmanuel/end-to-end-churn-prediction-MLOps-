"""
src/features.py
---------------
Feature engineering and preprocessing pipeline.

Industry concept: sklearn Pipelines are the industry standard.
They prevent DATA LEAKAGE (fitting on test data), make deployment
trivial (save one object, get preprocessing + model), and make
your code reviewable and testable.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────
# Column definitions (centralize these!)
# ─────────────────────────────────────────

NUMERIC_COLS = [
    "tenure",
    "monthlycharges",
    "totalcharges",
]

BINARY_COLS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "paperlessbilling",
]

MULTI_CAT_COLS = [
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod",
]


# ─────────────────────────────────────────
# Custom transformer: feature engineering
# ─────────────────────────────────────────

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that creates domain-informed features.

    Industry concept: wrap custom logic in a TransformerMixin so it plugs
    into sklearn Pipelines and benefits from fit/transform separation.
    The fit() method here is a no-op because these are deterministic
    transformations, but the pattern matters — you might need fit() if
    you compute statistics from the training set (e.g. mean encoding).
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self  # stateless transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Feature 1: Charge per month of tenure
        # Intuition: high monthly charges relative to short tenure → likely to churn
        df["charge_per_tenure"] = np.where(
            df["tenure"] > 0,
            df["monthlycharges"] / df["tenure"],
            df["monthlycharges"],  # new customers
        )

        # Feature 2: Tenure bucket (ordinal)
        # Intuition: churn risk is non-linear with tenure
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
        ).astype(str)

        # Feature 3: Total add-on services count
        addon_cols = [
            "onlinesecurity", "onlinebackup", "deviceprotection",
            "techsupport", "streamingtv", "streamingmovies",
        ]
        for col in addon_cols:
            if col in df.columns:
                df[f"{col}_bin"] = (df[col] == "Yes").astype(int)

        bin_cols = [c for c in df.columns if c.endswith("_bin")]
        df["total_addons"] = df[bin_cols].sum(axis=1)

        # Feature 4: High-value customer flag
        df["is_high_value"] = (df["monthlycharges"] > 70).astype(int)

        # Feature 5: Month-to-month flag (most predictive per domain knowledge)
        if "contract" in df.columns:
            df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)

        return df

    def get_feature_names_out(self, input_features=None):
        """Required by sklearn ≥ 1.0 for pipeline feature name tracking."""
        new_features = [
            "charge_per_tenure", "tenure_bucket", "total_addons",
            "is_high_value", "is_month_to_month",
        ]
        if input_features is not None:
            return list(input_features) + new_features
        return new_features


# ─────────────────────────────────────────
# Full preprocessing pipeline builder
# ─────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """
    Returns a sklearn Pipeline with:
      1. Custom feature engineering
      2. Column-specific preprocessing (numeric scaling, categorical encoding)

    Industry concept: ColumnTransformer lets you apply different preprocessing
    to different column types in a single, clean, leak-proof object.
    """

    # After feature engineering, we'll have these additional numeric columns
    engineered_numeric = ["charge_per_tenure", "total_addons", "is_high_value", "is_month_to_month"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # The ColumnTransformer will also handle the engineered string column
    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS + engineered_numeric),
            ("cat", categorical_pipeline, BINARY_COLS + MULTI_CAT_COLS + ["tenure_bucket"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", ct),
    ])


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Extract feature names after fitting the preprocessor."""
    ct = preprocessor.named_steps["preprocessor"]
    return list(ct.get_feature_names_out())
