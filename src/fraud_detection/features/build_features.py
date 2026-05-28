"""
Feature engineering pipeline for the IEEE-CIS fraud detection dataset.

Pipeline steps (in order):
  1. HighMissingDropper   — drop cols with > threshold missing (fit on train only)
  2. TimeFeatureExtractor — hour_of_day, day_of_week, is_weekend, time_in_dataset
                            from TransactionDT; then drop TransactionID + TransactionDT
  3. AmountFeatureExtractor — amt_log (log1p), amt_bin (5-band ordinal)
  4. EmailRiskEncoder     — smoothed target-encoding of P_emaildomain → risk score;
                            R_emaildomain is already dropped by step 1 (76.8% missing)
  5. FinalEncoder         — M-flags → 1/0/−1; other strings → OrdinalEncoder;
                            numerics → SimpleImputer(median)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_ID_COLS = ["TransactionID", "TransactionDT"]
_TARGET = "isFraud"

# M1–M3, M5–M9 encode T / F strings; M4 encodes M0 / M1 / M2 strings
_M_BINARY_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
_M4_COL = "M4"

_EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]


# ---------------------------------------------------------------------------
# Step 1 — High-missing dropper
# ---------------------------------------------------------------------------


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose fraction of missing values exceeds *threshold*."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "HighMissingDropper":
        missing_rates = X.isnull().mean()
        self.cols_to_drop_: list[str] = (
            missing_rates[missing_rates > self.threshold].index.tolist()
        )
        logger.info(
            "HighMissingDropper: will drop %d / %d columns (>%.0f%% missing)",
            len(self.cols_to_drop_),
            X.shape[1],
            self.threshold * 100,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.cols_to_drop_ if c in X.columns]
        return X.drop(columns=drop)


# ---------------------------------------------------------------------------
# Step 2 — Time feature extractor
# ---------------------------------------------------------------------------


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Derive cyclic time features from TransactionDT (seconds from unknown epoch).

    New columns added:
        hour_of_day       int   0–23
        day_of_week       int   0–6 (relative to dataset start day)
        is_weekend        int8  1 if day_of_week >= 5, else 0
        time_in_dataset   f32   0.0 (earliest) → 1.0 (latest) in training window
    """

    _SECS_PER_DAY = 86_400

    def fit(self, X: pd.DataFrame, y=None) -> "TimeFeatureExtractor":
        self.dt_min_: float = float(X["TransactionDT"].min())
        self.dt_max_: float = float(X["TransactionDT"].max())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = X["TransactionDT"]

        X["hour_of_day"] = ((dt % self._SECS_PER_DAY) // 3600).astype(np.int8)
        X["day_of_week"] = ((dt // self._SECS_PER_DAY) % 7).astype(np.int8)
        X["is_weekend"] = ((dt // self._SECS_PER_DAY) % 7 >= 5).astype(np.int8)
        span = max(self.dt_max_ - self.dt_min_, 1.0)
        X["time_in_dataset"] = ((dt - self.dt_min_) / span).astype(np.float32)

        return X.drop(columns=[c for c in _ID_COLS if c in X.columns])


# ---------------------------------------------------------------------------
# Step 3 — Amount feature extractor
# ---------------------------------------------------------------------------


class AmountFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Enrich TransactionAmt:
        amt_log   log1p-transformed amount (compresses the heavy right tail)
        amt_bin   ordinal band 0–4: micro(<$10) / small($10–50) / medium($50–200)
                                    / large($200–1 k) / very-large(>$1 k)
    """

    _BINS = [0.0, 10.0, 50.0, 200.0, 1_000.0, float("inf")]
    _LABELS = list(range(5))

    def fit(self, X: pd.DataFrame, y=None) -> "AmountFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        amt = X["TransactionAmt"]
        X["amt_log"] = np.log1p(amt).astype(np.float32)
        X["amt_bin"] = (
            pd.cut(amt, bins=self._BINS, labels=self._LABELS, include_lowest=True)
            .astype(np.int8)
        )
        return X


# ---------------------------------------------------------------------------
# Step 4 — Email domain risk encoder (target encoding)
# ---------------------------------------------------------------------------


class EmailRiskEncoder(BaseEstimator, TransformerMixin):
    """
    Additive-smoothed target-encoding of email domain columns.

    For each surviving email domain column a new ``<col>_risk`` float column
    is added containing the per-domain fraud rate blended toward the global
    rate.  Unseen domains and NaN receive the global rate.  The original
    column is dropped.

    Parameters
    ----------
    smoothing : float
        Pseudo-count added to both numerator and denominator when blending
        toward the global rate.  Higher values → more shrinkage.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EmailRiskEncoder":
        self.global_rate_: float = float(np.asarray(y).mean())
        self.domain_rates_: dict[str, dict[str, float]] = {}

        for col in _EMAIL_COLS:
            if col not in X.columns:
                continue
            frame = pd.DataFrame(
                {"domain": X[col].values, "label": np.asarray(y)}
            )
            agg = frame.groupby("domain")["label"].agg(["sum", "count"])
            smoothed = (agg["sum"] + self.smoothing * self.global_rate_) / (
                agg["count"] + self.smoothing
            )
            self.domain_rates_[col] = smoothed.to_dict()
            logger.info(
                "EmailRiskEncoder: %d domains in %-20s  global_rate=%.4f",
                len(smoothed),
                col,
                self.global_rate_,
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in _EMAIL_COLS:
            if col not in X.columns:
                continue
            X[f"{col}_risk"] = (
                X[col]
                .map(self.domain_rates_.get(col, {}))
                .fillna(self.global_rate_)
                .astype(np.float32)
            )
            X = X.drop(columns=[col])
        return X


# ---------------------------------------------------------------------------
# Step 5 — Final encoder
# ---------------------------------------------------------------------------


def _is_str_col(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


class FinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode and impute remaining columns, detecting types at fit time:

    - M-flag cols (T / F / NaN)  → map to  1 / 0 / −1  (int8)
    - M4 col    (M0 / M1 / M2)   → OrdinalEncoder → int8
    - Other string cols           → OrdinalEncoder, unknown → −1
    - Numeric cols                → SimpleImputer(strategy='median')
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FinalEncoder":
        self.m_binary_cols_: list[str] = [c for c in _M_BINARY_COLS if c in X.columns]
        self.m4_present_: bool = _M4_COL in X.columns

        str_cols = [c for c in X.columns if _is_str_col(X[c])]
        self.cat_cols_: list[str] = [
            c for c in str_cols
            if c not in self.m_binary_cols_ and c != _M4_COL
        ]
        self.numeric_cols_: list[str] = [
            c for c in X.select_dtypes(include=np.number).columns
        ]

        if self.cat_cols_:
            self.cat_enc_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,  # NaN → -1, not np.nan
            ).fit(X[self.cat_cols_])

        if self.m4_present_:
            self.m4_enc_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,  # M4 is 47.7% missing
            ).fit(X[[_M4_COL]])

        if self.numeric_cols_:
            self.num_imputer_ = SimpleImputer(strategy="median").fit(
                X[self.numeric_cols_]
            )

        logger.info(
            "FinalEncoder: %d M-binary, M4=%s, %d categorical, %d numeric",
            len(self.m_binary_cols_),
            self.m4_present_,
            len(self.cat_cols_),
            len(self.numeric_cols_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.m_binary_cols_:
            X[col] = X[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)

        if self.m4_present_:
            X[_M4_COL] = (
                self.m4_enc_.transform(X[[_M4_COL]]).ravel().astype(np.int8)
            )

        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_enc_.transform(
                X[self.cat_cols_]
            ).astype(np.float32)

        if self.numeric_cols_:
            imputed = self.num_imputer_.transform(X[self.numeric_cols_])
            X[self.numeric_cols_] = imputed.astype(np.float32)

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(
            self.m_binary_cols_
            + ([_M4_COL] if self.m4_present_ else [])
            + self.cat_cols_
            + self.numeric_cols_
        )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_pipeline(missing_threshold: float = 0.7) -> Pipeline:
    """Return a complete, unfitted feature-engineering Pipeline."""
    return Pipeline(
        [
            ("drop_high_missing", HighMissingDropper(threshold=missing_threshold)),
            ("time_features",     TimeFeatureExtractor()),
            ("amount_features",   AmountFeatureExtractor()),
            ("email_risk",        EmailRiskEncoder()),
            ("encode",            FinalEncoder()),
        ]
    )


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into (train, val, test) in TransactionDT order.

    The latest *test_size* fraction → test; next *val_size* → val;
    remainder → train.  This mirrors a walk-forward evaluation strategy.
    """
    df_s = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_s)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))

    train = df_s.iloc[:val_start].copy()
    val   = df_s.iloc[val_start:test_start].copy()
    test  = df_s.iloc[test_start:].copy()

    logger.info(
        "Temporal split → train %d (%.0f%%)  val %d (%.0f%%)  test %d (%.0f%%)",
        len(train), 100 * len(train) / n,
        len(val),   100 * len(val)   / n,
        len(test),  100 * len(test)  / n,
    )
    for name, split in [("train", train), ("val", val), ("test", test)]:
        logger.info(
            "  %-5s  fraud rate: %.3f%%  DT range: [%d, %d]",
            name,
            100 * split[_TARGET].mean(),
            int(split["TransactionDT"].min()),
            int(split["TransactionDT"].max()),
        )
    return train, val, test


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(config_path: str | Path = "config/config.yaml") -> dict[str, Path]:
    """
    Orchestrate the full feature engineering job:

    1. Load merged parquet from data/interim/
    2. Temporal split into train / val / test
    3. Fit the pipeline on train only (no leakage)
    4. Transform all three splits
    5. Save X_{split}.parquet, y_{split}.parquet, and feature_pipeline.joblib
       to data/processed/

    Returns a dict mapping output names to their paths.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg  = cfg["data"]
    split_cfg = cfg["split"]
    feat_cfg  = cfg["features"]

    # --- Load ---
    interim_path = Path(data_cfg["interim_dir"]) / "merged.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Merged parquet not found at {interim_path}. "
            "Run 'python -m fraud_detection.data.load' first."
        )
    logger.info("Loading merged data from %s", interim_path)
    df = pd.read_parquet(interim_path)
    logger.info("Loaded: %d rows × %d cols", *df.shape)

    # --- Split ---
    train, val, test = temporal_split(
        df,
        test_size=split_cfg["test_size"],
        val_size=split_cfg["val_size"],
    )

    def _split_Xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return frame.drop(columns=[_TARGET]), frame[_TARGET]

    X_train, y_train = _split_Xy(train)
    X_val,   y_val   = _split_Xy(val)
    X_test,  y_test  = _split_Xy(test)

    # --- Fit pipeline on training data only ---
    pipeline = build_pipeline(
        missing_threshold=feat_cfg.get("missing_threshold", 0.7)
    )
    logger.info("Fitting pipeline on %d training rows …", len(X_train))
    X_train_t = pipeline.fit_transform(X_train, y_train)
    logger.info(
        "Pipeline complete: %d → %d features",
        X_train.shape[1],
        X_train_t.shape[1],
    )

    X_val_t  = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    # --- Save ---
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for name, X_t, y_s in [
        ("train", X_train_t, y_train),
        ("val",   X_val_t,   y_val),
        ("test",  X_test_t,  y_test),
    ]:
        x_path = processed_dir / f"X_{name}.parquet"
        y_path = processed_dir / f"y_{name}.parquet"
        X_t.to_parquet(x_path, index=False)
        y_s.reset_index(drop=True).to_frame().to_parquet(y_path, index=False)
        out[f"X_{name}"] = x_path
        out[f"y_{name}"] = y_path
        logger.info(
            "  %-5s  X: %d×%d  fraud rate: %.3f%%  → %s",
            name, *X_t.shape, 100 * y_s.mean(), x_path,
        )

    pipeline_path = processed_dir / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    out["pipeline"] = pipeline_path
    logger.info("Saved fitted pipeline → %s", pipeline_path)

    return out


# ---------------------------------------------------------------------------
# Feature summary helper
# ---------------------------------------------------------------------------


def print_feature_summary(pipeline: Pipeline, X_sample: pd.DataFrame) -> None:
    """Print a human-readable breakdown of the output features."""
    enc: FinalEncoder = pipeline.named_steps["encode"]
    dropper: HighMissingDropper = pipeline.named_steps["drop_high_missing"]

    print(f"\n{'='*60}")
    print("Feature pipeline summary")
    print(f"{'='*60}")
    print(f"  Columns dropped (>70% missing) : {len(dropper.cols_to_drop_)}")
    print(f"  New time features              : hour_of_day, day_of_week, is_weekend, time_in_dataset")
    print(f"  New amount features            : amt_log, amt_bin")
    print(f"  Email risk features            : P_emaildomain_risk")
    print(f"  M-binary encoded cols          : {len(enc.m_binary_cols_)}")
    print(f"  M4 present                     : {enc.m4_present_}")
    print(f"  Other categorical cols         : {len(enc.cat_cols_)}  {enc.cat_cols_}")
    print(f"  Numeric cols (imputed)         : {len(enc.numeric_cols_)}")
    print(f"  Total output features          : {X_sample.shape[1]}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths = run()
    pipeline = joblib.load(paths["pipeline"])
    X_train = pd.read_parquet(paths["X_train"])
    print_feature_summary(pipeline, X_train)

    print("Output files:")
    for name, path in paths.items():
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  {name:12s}  {size_mb:6.1f} MB  {path}")
