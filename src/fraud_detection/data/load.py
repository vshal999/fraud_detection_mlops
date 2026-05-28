"""Load, merge, and validate the raw IEEE-CIS fraud dataset."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_TRANSACTION_REQUIRED = {"TransactionID", "TransactionDT", "isFraud"}
_IDENTITY_REQUIRED = {"TransactionID"}


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    """Load YAML config from *config_path*."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_tables(
    transaction_path: str | Path,
    identity_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read transaction and identity CSVs from disk."""
    logger.info("Reading transaction table: %s", transaction_path)
    transactions = pd.read_csv(transaction_path, low_memory=False)

    logger.info("Reading identity table: %s", identity_path)
    identity = pd.read_csv(identity_path, low_memory=False)

    logger.info("Transaction table: %d rows × %d cols", *transactions.shape)
    logger.info("Identity table: %d rows × %d cols", *identity.shape)

    return transactions, identity


def _validate_raw_tables(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> None:
    """Raise ValueError if either raw table fails basic schema checks."""
    missing_t = _TRANSACTION_REQUIRED - set(transactions.columns)
    if missing_t:
        raise ValueError(f"Transaction table missing required columns: {missing_t}")

    missing_i = _IDENTITY_REQUIRED - set(identity.columns)
    if missing_i:
        raise ValueError(f"Identity table missing required columns: {missing_i}")

    n_dupes_t = transactions["TransactionID"].duplicated().sum()
    if n_dupes_t:
        raise ValueError(
            f"Transaction table has {n_dupes_t} duplicate TransactionIDs"
        )

    # Duplicate identity rows would fan-out the merge
    n_dupes_i = identity["TransactionID"].duplicated().sum()
    if n_dupes_i:
        raise ValueError(
            f"Identity table has {n_dupes_i} duplicate TransactionIDs"
        )


def merge_tables(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join identity features onto transactions via TransactionID."""
    merged = transactions.merge(
        identity, on="TransactionID", how="left", indicator=True
    ).copy()  # defragment to avoid PerformanceWarning on 434-col frame

    n_total = len(merged)
    n_with_identity = (merged["_merge"] == "both").sum()
    merged = merged.drop(columns=["_merge"])

    logger.info(
        "Merge complete: %d transactions, %d (%.1f%%) have identity records",
        n_total,
        n_with_identity,
        100 * n_with_identity / n_total,
    )
    return merged


def validate(df: pd.DataFrame) -> None:
    """Validate the merged DataFrame; raise on hard errors, warn on soft ones."""
    # Hard: required columns present
    missing_cols = _TRANSACTION_REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Merged DataFrame missing required columns: {missing_cols}")

    # Hard: no duplicate rows from a bad merge
    n_dupes = df["TransactionID"].duplicated().sum()
    if n_dupes:
        raise ValueError(
            f"Merged DataFrame has {n_dupes} duplicate TransactionIDs — "
            "identity table likely had duplicate keys"
        )

    # Hard: target is binary
    target_values = set(df["isFraud"].dropna().unique())
    unexpected = target_values - {0, 1}
    if unexpected:
        raise ValueError(f"isFraud contains unexpected values: {unexpected}")

    # Soft: fraud rate should be ~3.5 %
    fraud_rate = df["isFraud"].mean()
    if not (0.02 <= fraud_rate <= 0.06):
        logger.warning(
            "Fraud rate %.3f%% is outside the expected 2–6%% range",
            100 * fraud_rate,
        )

    # Diagnostics
    n_fraud = int(df["isFraud"].sum())
    logger.info("Shape: %d rows × %d cols", *df.shape)
    logger.info(
        "Fraud rate: %.3f%% (%d / %d)", 100 * fraud_rate, n_fraud, len(df)
    )
    logger.info(
        "TransactionDT range: [%d, %d]",
        int(df["TransactionDT"].min()),
        int(df["TransactionDT"].max()),
    )

    high_missing = (df.isnull().mean() > 0.5).sum()
    logger.info("%d features have >50%% missing values", high_missing)


def load_dataset(
    config_path: str | Path = "config/config.yaml",
) -> pd.DataFrame:
    """
    Load, merge, and validate the IEEE-CIS fraud dataset.

    Returns a single merged DataFrame (transactions left-joined with identity).
    """
    config = load_config(config_path)
    data_cfg = config["data"]

    raw_dir = Path(data_cfg["raw_dir"])
    transaction_path = raw_dir / data_cfg["transaction_file"]
    identity_path = raw_dir / data_cfg["identity_file"]

    transactions, identity = load_raw_tables(transaction_path, identity_path)
    _validate_raw_tables(transactions, identity)

    merged = merge_tables(transactions, identity)
    validate(merged)

    return merged


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = load_dataset()

    interim_path = Path("data/interim/merged.parquet")
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_path, index=False)
    logger.info("Saved merged dataset to %s", interim_path)

    print(f"\nDataset ready: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Fraud rate: {df['isFraud'].mean():.3%}")
    print(f"Saved to: {interim_path}")
