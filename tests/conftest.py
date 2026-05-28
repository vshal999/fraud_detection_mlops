"""
Shared pytest fixtures for the fraud-detection test suite.

Session-scoped fixtures are built once per test run; fitting pipelines and
loading models are too expensive to repeat per test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Raw synthetic data builders
# ---------------------------------------------------------------------------


def _make_transactions(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic transaction table that mimics the IEEE-CIS raw schema.

    Includes:
      - All named columns the feature pipeline touches
      - Two diagnostic columns: one >70% missing (HighMissingDropper drops it),
        one <20% missing (should survive)
      - ~10% fraud rate so stratified splits always contain both classes
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)

    n_fraud = max(2, int(n * 0.1))
    fraud_mask = np.zeros(n, dtype=int)
    fraud_mask[:n_fraud] = 1
    rng.shuffle(fraud_mask)

    # Helper to introduce NaN with a given rate
    def _nullable(arr, nan_rate: float):
        out = arr.astype(object)
        out[rng.random(n) < nan_rate] = None
        return out

    df = pd.DataFrame({
        # Identifiers
        "TransactionID": ids,
        "TransactionDT": ids * 3_600,          # 1-hour steps — easy to reason about
        "TransactionAmt": rng.uniform(1.0, 2_000.0, n),
        "isFraud": fraud_mask,

        # Categorical product / card
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
        "card1": rng.uniform(100, 20_000, n).astype(float),
        "card2": rng.uniform(100, 600, n).astype(float),
        "card3": _nullable(rng.uniform(100, 200, n), 0.1),
        "card4": rng.choice(["visa", "mastercard", "discover", None], n),
        "card5": rng.uniform(100, 250, n).astype(float),
        "card6": rng.choice(["credit", "debit", "charge card", None], n),

        # Address / distance
        "addr1": rng.uniform(100, 500, n).astype(float),
        "addr2": rng.uniform(10,  100, n).astype(float),
        "dist1": _nullable(rng.uniform(0, 10_000, n), 0.3),
        "dist2": _nullable(rng.uniform(0, 10_000, n), 0.6),

        # Email domains (P mostly present, R mostly missing → dropper removes R)
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n, p=[0.5, 0.25, 0.15, 0.10]),
        "R_emaildomain": _nullable(rng.choice(["gmail.com", "yahoo.com"], n), 0.85),

        # Count features C1–C14
        **{f"C{i}": rng.uniform(0, 20, n).astype(float) for i in range(1, 15)},

        # Timedelta features D1–D15 (with some missingness)
        **{f"D{i}": _nullable(rng.uniform(0, 500, n), 0.2) for i in range(1, 16)},

        # Match flags M1–M9 (string T/F/None)
        **{f"M{i}": rng.choice(["T", "F", None], n) for i in [1, 2, 3, 5, 6, 7, 8, 9]},
        "M4": rng.choice(["M0", "M1", "M2", None], n),

        # Anonymous V-features (just a few for tests)
        "V1": _nullable(rng.uniform(0, 1, n), 0.05),
        "V2": rng.uniform(0, 6, n).astype(float),
        "V3": rng.uniform(0, 6, n).astype(float),

        # Diagnostic: >70% missing → HighMissingDropper should drop it
        "mostly_missing": _nullable(np.ones(n), 0.85),
        # Diagnostic: <20% missing → should survive
        "rarely_missing": _nullable(np.ones(n), 0.15),
    })
    return df


def _make_identity(transaction_ids: list[int], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(transaction_ids)
    return pd.DataFrame({
        "TransactionID": transaction_ids,
        "DeviceType": rng.choice(["desktop", "mobile", None], n),
        "DeviceInfo": rng.choice(["Windows", "iOS", "MacOS", None], n),
        "id_01": rng.uniform(-10, 0, n).astype(float),
        "id_02": rng.uniform(0, 500_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Session-scoped shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_transactions_df() -> pd.DataFrame:
    return _make_transactions(60)


@pytest.fixture(scope="session")
def raw_identity_df(raw_transactions_df: pd.DataFrame) -> pd.DataFrame:
    tids = raw_transactions_df["TransactionID"].tolist()[:35]
    return _make_identity(tids)


@pytest.fixture(scope="session")
def merged_df(raw_transactions_df: pd.DataFrame, raw_identity_df: pd.DataFrame) -> pd.DataFrame:
    from fraud_detection.data.load import merge_tables
    return merge_tables(raw_transactions_df, raw_identity_df)


@pytest.fixture(scope="session")
def fitted_pipeline_and_data(merged_df: pd.DataFrame):
    """
    Return (pipeline, X_train_t, y_train, X_val_t, y_val) fitted on synthetic data.

    Pipeline is fitted on the training split only — mirrors the real training
    workflow so transformers have realistic fitted state for downstream tests.
    """
    from fraud_detection.features.build_features import build_pipeline, temporal_split

    _TARGET = "isFraud"
    train, val, _test = temporal_split(merged_df, test_size=0.2, val_size=0.2)

    X_train = train.drop(columns=[_TARGET])
    y_train = train[_TARGET]
    X_val   = val.drop(columns=[_TARGET])
    y_val   = val[_TARGET]

    pipeline = build_pipeline(missing_threshold=0.7)
    X_train_t = pipeline.fit_transform(X_train, y_train)
    X_val_t   = pipeline.transform(X_val)

    return pipeline, X_train_t, y_train, X_val_t, y_val


# ---------------------------------------------------------------------------
# API test client (real models, session-scoped)
# ---------------------------------------------------------------------------

def _models_present() -> bool:
    return (
        Path("models/best_model.joblib").exists()
        and Path("models/isolation_forest.joblib").exists()
        and Path("data/processed/feature_pipeline.joblib").exists()
        and Path("data/processed/X_val.parquet").exists()
        and Path("data/processed/X_test.parquet").exists()
    )


@pytest.fixture(scope="session")
def api_client():
    """
    Session-scoped TestClient that loads real trained models.

    Skipped automatically when model artefacts are absent — lets the
    feature-pipeline and data-loader tests still run in a fresh environment.
    """
    if not _models_present():
        pytest.skip(
            "Trained model artefacts not found. "
            "Run the full training pipeline before the API tests."
        )

    from starlette.testclient import TestClient
    from fraud_detection.api.app import app

    with TestClient(app) as client:
        yield client
