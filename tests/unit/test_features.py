"""
Unit tests for src/fraud_detection/features/build_features.py.

Each transformer is tested in isolation with minimal synthetic data,
then the full Pipeline is tested end-to-end via the session fixture from
conftest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.build_features import (
    AmountFeatureExtractor,
    EmailRiskEncoder,
    FinalEncoder,
    HighMissingDropper,
    TimeFeatureExtractor,
    build_pipeline,
    temporal_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Minimal raw DataFrame for transformer unit tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "TransactionDT": np.arange(n) * 3_600 + 86_400,
        "TransactionAmt": rng.uniform(1.0, 2_000.0, n),
        "isFraud": (rng.random(n) < 0.15).astype(int),
        "ProductCD": rng.choice(["W", "H", "C"], n),
        "card1": rng.uniform(100, 20_000, n).astype(float),
        "C1": rng.uniform(0, 20, n).astype(float),
        "D1": rng.uniform(0, 500, n).astype(float),
        "M1": rng.choice(["T", "F", None], n),
        "M4": rng.choice(["M0", "M1", "M2", None], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "V1": rng.uniform(0, 1, n).astype(float),
    })


# ---------------------------------------------------------------------------
# HighMissingDropper
# ---------------------------------------------------------------------------


class TestHighMissingDropper:
    def test_drops_column_above_threshold(self):
        n = 100
        df = pd.DataFrame({
            "keep_me": np.ones(n),
            "drop_me": np.where(np.arange(n) < 15, 1.0, np.nan),  # 85% missing
        })
        dropper = HighMissingDropper(threshold=0.7)
        dropper.fit(df)
        out = dropper.transform(df)
        assert "drop_me" not in out.columns
        assert "keep_me" in out.columns

    def test_preserves_column_below_threshold(self):
        n = 100
        df = pd.DataFrame({
            "sparse": np.where(np.arange(n) < 50, 1.0, np.nan),  # 50% missing — under 70%
        })
        dropper = HighMissingDropper(threshold=0.7)
        dropper.fit(df)
        out = dropper.transform(df)
        assert "sparse" in out.columns

    def test_fit_records_cols_to_drop(self):
        df = pd.DataFrame({
            "a": np.ones(10),
            "b": np.array([1.0] + [np.nan] * 9),  # 90% missing
        })
        dropper = HighMissingDropper(threshold=0.7)
        dropper.fit(df)
        assert "b" in dropper.cols_to_drop_
        assert "a" not in dropper.cols_to_drop_

    def test_transform_ignores_new_columns_not_seen_at_fit(self):
        df_fit = pd.DataFrame({"a": np.ones(10), "b": np.full(10, np.nan)})
        dropper = HighMissingDropper(threshold=0.7).fit(df_fit)
        df_new = pd.DataFrame({"a": np.ones(5), "c": np.ones(5)})
        out = dropper.transform(df_new)
        assert "c" in out.columns  # new col kept since it wasn't in cols_to_drop_

    def test_threshold_of_zero_drops_all_with_any_missing(self):
        df = pd.DataFrame({"x": [1.0, np.nan], "y": [1.0, 1.0]})
        dropper = HighMissingDropper(threshold=0.0).fit(df)
        out = dropper.transform(df)
        assert "x" not in out.columns
        assert "y" in out.columns


# ---------------------------------------------------------------------------
# TimeFeatureExtractor
# ---------------------------------------------------------------------------


class TestTimeFeatureExtractor:
    _SECS_PER_DAY = 86_400
    _SECS_PER_HOUR = 3_600

    @pytest.fixture
    def simple_df(self):
        """24 rows, one per hour starting at midnight (hour 0)."""
        return pd.DataFrame({
            "TransactionID": range(24),
            "TransactionDT": np.arange(24) * self._SECS_PER_HOUR,
            "TransactionAmt": 100.0,
        })

    def test_creates_expected_columns(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        for col in ("hour_of_day", "day_of_week", "is_weekend", "time_in_dataset"):
            assert col in out.columns, f"Missing: {col}"

    def test_drops_transaction_dt(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert "TransactionDT" not in out.columns

    def test_drops_transaction_id(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert "TransactionID" not in out.columns

    def test_hour_of_day_range(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert out["hour_of_day"].between(0, 23).all()

    def test_hour_of_day_values(self, simple_df):
        """Row i has TransactionDT = i * 3600, so hour_of_day should equal i."""
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        pd.testing.assert_series_equal(
            out["hour_of_day"].astype(int),
            pd.Series(range(24), name="hour_of_day"),
        )

    def test_day_of_week_range(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert out["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self, simple_df):
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_time_in_dataset_bounds_on_training_data(self, simple_df):
        """Training data min → 0.0, max → 1.0."""
        tte = TimeFeatureExtractor().fit(simple_df)
        out = tte.transform(simple_df)
        assert out["time_in_dataset"].min() == pytest.approx(0.0, abs=1e-6)
        assert out["time_in_dataset"].max() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AmountFeatureExtractor
# ---------------------------------------------------------------------------


class TestAmountFeatureExtractor:
    @pytest.fixture
    def amount_df(self):
        return pd.DataFrame({
            "TransactionAmt": [5.0, 25.0, 100.0, 500.0, 2_000.0],
        })

    def test_creates_log_feature(self, amount_df):
        out = AmountFeatureExtractor().fit(amount_df).transform(amount_df)
        assert "amt_log" in out.columns

    def test_log_values_match_log1p(self, amount_df):
        out = AmountFeatureExtractor().fit(amount_df).transform(amount_df)
        expected = np.log1p(amount_df["TransactionAmt"].values).astype(np.float32)
        np.testing.assert_allclose(out["amt_log"].values, expected, rtol=1e-5)

    def test_creates_bin_feature(self, amount_df):
        out = AmountFeatureExtractor().fit(amount_df).transform(amount_df)
        assert "amt_bin" in out.columns

    def test_bin_boundaries(self, amount_df):
        """
        Bins: [0, 10] → 0, (10, 50] → 1, (50, 200] → 2, (200, 1000] → 3, >1000 → 4
        """
        out = AmountFeatureExtractor().fit(amount_df).transform(amount_df)
        bins = out["amt_bin"].astype(int).tolist()
        assert bins == [0, 1, 2, 3, 4]

    def test_bin_values_are_0_to_4(self):
        df = pd.DataFrame({"TransactionAmt": np.linspace(0.5, 5_000, 100)})
        out = AmountFeatureExtractor().fit(df).transform(df)
        assert out["amt_bin"].isin([0, 1, 2, 3, 4]).all()

    def test_original_amount_column_preserved(self, amount_df):
        out = AmountFeatureExtractor().fit(amount_df).transform(amount_df)
        assert "TransactionAmt" in out.columns


# ---------------------------------------------------------------------------
# EmailRiskEncoder
# ---------------------------------------------------------------------------


class TestEmailRiskEncoder:
    @pytest.fixture
    def email_df(self):
        return pd.DataFrame({
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "gmail.com", None],
            "other_col": [1, 2, 3, 4, 5],
        })

    @pytest.fixture
    def email_y(self):
        return pd.Series([1, 0, 0, 1, 0])

    def test_creates_risk_column(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        out = enc.transform(email_df.copy())
        assert "P_emaildomain_risk" in out.columns

    def test_drops_original_domain_column(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        out = enc.transform(email_df.copy())
        assert "P_emaildomain" not in out.columns

    def test_risk_values_are_in_0_1(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        out = enc.transform(email_df.copy())
        assert (out["P_emaildomain_risk"] >= 0.0).all()
        assert (out["P_emaildomain_risk"] <= 1.0).all()

    def test_nan_domain_receives_global_rate(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        out = enc.transform(email_df.copy())
        global_rate = enc.global_rate_
        nan_risk = out.loc[email_df["P_emaildomain"].isna(), "P_emaildomain_risk"]
        assert nan_risk.iloc[0] == pytest.approx(global_rate, abs=1e-5)

    def test_unseen_domain_receives_global_rate(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        df_new = pd.DataFrame({"P_emaildomain": ["unknown.com"], "other_col": [9]})
        out = enc.transform(df_new)
        assert out["P_emaildomain_risk"].iloc[0] == pytest.approx(enc.global_rate_, abs=1e-5)

    def test_preserves_unrelated_columns(self, email_df, email_y):
        enc = EmailRiskEncoder().fit(email_df, email_y)
        out = enc.transform(email_df.copy())
        assert "other_col" in out.columns

    def test_smoothing_blends_toward_global_rate(self):
        """High smoothing should pull all domain rates toward the global rate."""
        df = pd.DataFrame({"P_emaildomain": ["a.com"] * 2 + ["b.com"] * 2})
        y  = pd.Series([1, 1, 0, 0])

        enc_low  = EmailRiskEncoder(smoothing=0.01).fit(df, y)
        enc_high = EmailRiskEncoder(smoothing=1000).fit(df, y)

        out_low  = enc_low.transform(df.copy())
        out_high = enc_high.transform(df.copy())

        # High smoothing should make a.com and b.com rates closer together
        spread_low  = out_low["P_emaildomain_risk"].max() - out_low["P_emaildomain_risk"].min()
        spread_high = out_high["P_emaildomain_risk"].max() - out_high["P_emaildomain_risk"].min()
        assert spread_high < spread_low


# ---------------------------------------------------------------------------
# FinalEncoder
# ---------------------------------------------------------------------------


class TestFinalEncoder:
    @pytest.fixture
    def m_flag_df(self):
        return pd.DataFrame({
            "M1": ["T", "F", None, "T"],
            "M2": ["F", None, "T", "F"],
            "C1": [1.0, 2.0, np.nan, 4.0],
            "V1": [0.5, 0.3, 0.8, np.nan],
        })

    def test_m_binary_t_maps_to_1(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        assert (out.loc[m_flag_df["M1"] == "T", "M1"] == 1).all()

    def test_m_binary_f_maps_to_0(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        assert (out.loc[m_flag_df["M1"] == "F", "M1"] == 0).all()

    def test_m_binary_nan_maps_to_minus1(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        nan_mask = m_flag_df["M1"].isna()
        assert (out.loc[nan_mask, "M1"] == -1).all()

    def test_numeric_nan_is_imputed(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        assert out["C1"].notna().all()
        assert out["V1"].notna().all()

    def test_numeric_imputed_with_median(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        # Median of [1, 2, 4] (non-NaN) = 2.0 — NaN row should get that
        nan_row = m_flag_df["C1"].isna()
        expected_median = np.median([1.0, 2.0, 4.0])
        assert out.loc[nan_row, "C1"].iloc[0] == pytest.approx(expected_median, abs=1e-4)

    def test_categorical_unknown_encodes_to_minus1(self):
        df_fit = pd.DataFrame({"cat": ["a", "b", "a"]})
        df_new = pd.DataFrame({"cat": ["z"]})  # unseen value
        enc = FinalEncoder().fit(df_fit)
        out = enc.transform(df_new)
        assert out["cat"].iloc[0] == -1.0

    def test_output_is_numeric(self, m_flag_df):
        enc = FinalEncoder().fit(m_flag_df)
        out = enc.transform(m_flag_df.copy())
        for col in out.columns:
            assert pd.api.types.is_numeric_dtype(out[col]), f"{col} is not numeric"


# ---------------------------------------------------------------------------
# Full Pipeline (build_pipeline)
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """
    End-to-end pipeline tests using the session-scoped fitted fixture from conftest.
    """

    def test_output_is_dataframe(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert isinstance(X_train_t, pd.DataFrame)

    def test_transaction_dt_dropped(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "TransactionDT" not in X_train_t.columns

    def test_transaction_id_dropped(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "TransactionID" not in X_train_t.columns

    def test_target_not_in_output(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "isFraud" not in X_train_t.columns

    def test_time_features_created(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        for col in ("hour_of_day", "day_of_week", "is_weekend", "time_in_dataset"):
            assert col in X_train_t.columns, f"Missing engineered col: {col}"

    def test_amount_features_created(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "amt_log" in X_train_t.columns
        assert "amt_bin" in X_train_t.columns

    def test_email_risk_feature_created(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "P_emaildomain_risk" in X_train_t.columns
        assert "P_emaildomain" not in X_train_t.columns

    def test_high_missing_column_dropped(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "mostly_missing" not in X_train_t.columns

    def test_low_missing_column_kept(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "rarely_missing" in X_train_t.columns

    def test_no_nan_in_numeric_output(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        numeric_cols = X_train_t.select_dtypes(include=np.number).columns
        assert X_train_t[numeric_cols].isna().sum().sum() == 0, (
            "Pipeline left NaN values in numeric columns"
        )

    def test_all_output_numeric(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        non_numeric = [
            c for c in X_train_t.columns
            if not pd.api.types.is_numeric_dtype(X_train_t[c])
        ]
        assert non_numeric == [], f"Non-numeric columns in output: {non_numeric}"

    def test_val_transform_same_columns_as_train(self, fitted_pipeline_and_data):
        _, X_train_t, _, X_val_t, _ = fitted_pipeline_and_data
        assert list(X_train_t.columns) == list(X_val_t.columns)

    def test_amt_log_equals_log1p_of_amount(self, fitted_pipeline_and_data):
        """
        AmountFeatureExtractor keeps TransactionAmt and adds amt_log = log1p(amt).
        Verify the invariant holds across every row of the transformed training data.
        """
        _, X_train_t, *_ = fitted_pipeline_and_data
        assert "amt_log" in X_train_t.columns
        assert "TransactionAmt" in X_train_t.columns
        expected = np.log1p(X_train_t["TransactionAmt"].values).astype(np.float32)
        np.testing.assert_allclose(X_train_t["amt_log"].values, expected, rtol=1e-4)

    def test_output_row_count_matches_input(self, fitted_pipeline_and_data):
        pipeline, X_train_t, y_train, *_ = fitted_pipeline_and_data
        assert len(X_train_t) == len(y_train)

    def test_output_column_count_is_reasonable(self, fitted_pipeline_and_data):
        _, X_train_t, *_ = fitted_pipeline_and_data
        # Should have more columns than raw input (engineered features added)
        # but fewer than the total raw + identity columns
        assert X_train_t.shape[1] > 5


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    @pytest.fixture
    def df_100(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "TransactionDT": np.arange(100) * 3_600,
            "isFraud": (rng.random(100) < 0.1).astype(int),
            "value": rng.random(100),
        })

    def test_splits_sum_to_total(self, df_100):
        train, val, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        assert len(train) + len(val) + len(test) == len(df_100)

    def test_test_fraction_approximately_correct(self, df_100):
        _, _, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        assert abs(len(test) / len(df_100) - 0.2) < 0.02

    def test_val_fraction_approximately_correct(self, df_100):
        _, val, _ = temporal_split(df_100, test_size=0.2, val_size=0.15)
        assert abs(len(val) / len(df_100) - 0.15) < 0.02

    def test_train_has_earliest_transactions(self, df_100):
        train, val, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        assert train["TransactionDT"].max() <= val["TransactionDT"].min()

    def test_val_before_test(self, df_100):
        _, val, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        assert val["TransactionDT"].max() <= test["TransactionDT"].min()

    def test_no_overlap_between_splits(self, df_100):
        train, val, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_output_is_sorted_by_time(self, df_100):
        train, val, test = temporal_split(df_100, test_size=0.2, val_size=0.15)
        for split in (train, val, test):
            assert split["TransactionDT"].is_monotonic_increasing
