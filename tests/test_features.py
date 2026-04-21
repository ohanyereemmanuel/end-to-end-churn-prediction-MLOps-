"""
tests/test_features.py
-----------------------
Unit tests for the feature engineering pipeline.

Industry concept: Untested ML code is a liability. Tests catch regressions
(when a code change silently breaks something). In production, tests run in
CI/CD (GitHub Actions) on every pull request — if they fail, the PR is blocked.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import ChurnFeatureEngineer, build_preprocessor


@pytest.fixture
def sample_customer():
    """Minimal valid customer record for testing."""
    return pd.DataFrame([{
        "tenure": 3,
        "monthlycharges": 85.5,
        "totalcharges": 256.5,
        "gender": "Male",
        "seniorcitizen": "No",
        "partner": "No",
        "dependents": "No",
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
    }])


@pytest.fixture
def sample_batch(sample_customer):
    """Small batch for testing batch operations."""
    return pd.concat([sample_customer] * 10, ignore_index=True)


class TestChurnFeatureEngineer:

    def test_engineer_adds_expected_columns(self, sample_customer):
        eng = ChurnFeatureEngineer()
        result = eng.fit_transform(sample_customer)
        assert "charge_per_tenure" in result.columns
        assert "tenure_bucket" in result.columns
        assert "total_addons" in result.columns
        assert "is_high_value" in result.columns
        assert "is_month_to_month" in result.columns

    def test_charge_per_tenure_positive(self, sample_customer):
        eng = ChurnFeatureEngineer()
        result = eng.fit_transform(sample_customer)
        assert (result["charge_per_tenure"] > 0).all()

    def test_zero_tenure_does_not_divide_by_zero(self):
        df = pd.DataFrame([{"tenure": 0, "monthlycharges": 50.0,
                             "totalcharges": 0.0, "contract": "Month-to-month",
                             "onlinesecurity": "No", "onlinebackup": "No",
                             "deviceprotection": "No", "techsupport": "No",
                             "streamingtv": "No", "streamingmovies": "No"}])
        eng = ChurnFeatureEngineer()
        result = eng.fit_transform(df)
        assert np.isfinite(result["charge_per_tenure"].iloc[0])

    def test_total_addons_range(self, sample_customer):
        eng = ChurnFeatureEngineer()
        result = eng.fit_transform(sample_customer)
        assert 0 <= result["total_addons"].iloc[0] <= 6

    def test_month_to_month_flag(self, sample_customer):
        eng = ChurnFeatureEngineer()
        result = eng.fit_transform(sample_customer)
        # sample_customer has Month-to-month contract
        assert result["is_month_to_month"].iloc[0] == 1

    def test_is_stateless(self, sample_customer, sample_batch):
        eng = ChurnFeatureEngineer()
        eng.fit(sample_customer)
        result1 = eng.transform(sample_customer)
        result2 = eng.transform(sample_batch)
        # Should produce same result regardless of batch size
        assert result1["charge_per_tenure"].iloc[0] == result2["charge_per_tenure"].iloc[0]


class TestPreprocessorPipeline:

    def test_pipeline_produces_numeric_output(self, sample_customer):
        pipeline = build_preprocessor()
        result = pipeline.fit_transform(sample_customer)
        assert result.dtype in [np.float32, np.float64]

    def test_pipeline_no_nan_output(self, sample_customer):
        pipeline = build_preprocessor()
        result = pipeline.fit_transform(sample_customer)
        assert not np.isnan(result).any()

    def test_pipeline_consistent_output_shape(self, sample_customer, sample_batch):
        pipeline = build_preprocessor()
        pipeline.fit(sample_batch)  # fit on batch
        single_out = pipeline.transform(sample_customer)
        batch_out = pipeline.transform(sample_batch)
        # Same number of features
        assert single_out.shape[1] == batch_out.shape[1]

    def test_pipeline_fit_transform_matches_separate(self, sample_batch):
        """Verify fit_transform == fit then transform (no data leakage)."""
        p1 = build_preprocessor()
        p2 = build_preprocessor()
        out1 = p1.fit_transform(sample_batch)
        p2.fit(sample_batch)
        out2 = p2.transform(sample_batch)
        np.testing.assert_array_almost_equal(out1, out2)
