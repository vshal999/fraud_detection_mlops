"""Unit tests for fraud_detection.data.load."""

import textwrap

import pandas as pd
import pytest

from fraud_detection.data.load import (
    _validate_raw_tables,
    load_dataset,
    merge_tables,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transactions(n: int = 5, fraud_indices: list[int] | None = None) -> pd.DataFrame:
    fraud_indices = fraud_indices or [0]
    is_fraud = [1 if i in fraud_indices else 0 for i in range(n)]
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "TransactionDT": range(86400, 86400 + n * 3600, 3600),
            "TransactionAmt": [100.0 + i for i in range(n)],
            "isFraud": is_fraud,
        }
    )


def _make_identity(transaction_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "id_01": [-5.0] * len(transaction_ids),
            "DeviceType": ["mobile"] * len(transaction_ids),
        }
    )


# ---------------------------------------------------------------------------
# merge_tables
# ---------------------------------------------------------------------------

class TestMergeTables:
    def test_all_transactions_preserved(self):
        transactions = _make_transactions(5)
        identity = _make_identity([1, 2])  # only 2 of 5 have identity
        merged = merge_tables(transactions, identity)
        assert len(merged) == 5

    def test_identity_columns_present(self):
        transactions = _make_transactions(3)
        identity = _make_identity([1, 2, 3])
        merged = merge_tables(transactions, identity)
        assert "id_01" in merged.columns
        assert "DeviceType" in merged.columns

    def test_missing_identity_rows_are_nan(self):
        transactions = _make_transactions(3)
        identity = _make_identity([1])  # only TransactionID=1 has identity
        merged = merge_tables(transactions, identity)
        no_identity = merged[merged["TransactionID"] != 1]
        assert no_identity["id_01"].isna().all()

    def test_no_indicator_column_in_output(self):
        transactions = _make_transactions(3)
        identity = _make_identity([1, 2])
        merged = merge_tables(transactions, identity)
        assert "_merge" not in merged.columns

    def test_no_duplicate_rows(self):
        transactions = _make_transactions(4)
        identity = _make_identity([1, 2])
        merged = merge_tables(transactions, identity)
        assert merged["TransactionID"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# _validate_raw_tables
# ---------------------------------------------------------------------------

class TestValidateRawTables:
    def test_valid_tables_pass(self):
        _validate_raw_tables(_make_transactions(), _make_identity([1]))

    def test_missing_transaction_column_raises(self):
        transactions = _make_transactions().drop(columns=["isFraud"])
        with pytest.raises(ValueError, match="Transaction table missing"):
            _validate_raw_tables(transactions, _make_identity([1]))

    def test_missing_identity_column_raises(self):
        identity = _make_identity([1]).drop(columns=["TransactionID"])
        with pytest.raises(ValueError, match="Identity table missing"):
            _validate_raw_tables(_make_transactions(), identity)

    def test_duplicate_transaction_id_raises(self):
        transactions = pd.concat(
            [_make_transactions(3), _make_transactions(2)], ignore_index=True
        )
        with pytest.raises(ValueError, match="Transaction table has"):
            _validate_raw_tables(transactions, _make_identity([1]))

    def test_duplicate_identity_id_raises(self):
        identity = pd.concat(
            [_make_identity([1]), _make_identity([1])], ignore_index=True
        )
        with pytest.raises(ValueError, match="Identity table has"):
            _validate_raw_tables(_make_transactions(), identity)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_merged_passes(self):
        transactions = _make_transactions(5, fraud_indices=[0])
        merged = merge_tables(transactions, _make_identity([1, 2]))
        validate(merged)  # should not raise

    def test_missing_required_column_raises(self):
        transactions = _make_transactions(5)
        merged = merge_tables(transactions, _make_identity([1]))
        merged = merged.drop(columns=["isFraud"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate(merged)

    def test_invalid_target_values_raise(self):
        transactions = _make_transactions(3)
        merged = merge_tables(transactions, _make_identity([1]))
        merged["isFraud"] = 2  # invalid
        with pytest.raises(ValueError, match="unexpected values"):
            validate(merged)

    def test_duplicate_transaction_id_raises(self):
        transactions = _make_transactions(3)
        merged = merge_tables(transactions, _make_identity([1]))
        merged = pd.concat([merged, merged.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate TransactionIDs"):
            validate(merged)

    def test_unusual_fraud_rate_logs_warning(self, caplog):
        import logging
        transactions = _make_transactions(
            10, fraud_indices=list(range(10))  # 100% fraud — clearly unusual
        )
        merged = merge_tables(transactions, _make_identity([1]))
        with caplog.at_level(logging.WARNING, logger="fraud_detection.data.load"):
            validate(merged)
        assert any("outside" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_dataset (integration — uses tmp filesystem)
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_dataset_end_to_end(self, tmp_path):
        # Write minimal CSVs
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        transactions = _make_transactions(10, fraud_indices=[0])
        identity = _make_identity([1, 2, 3])
        transactions.to_csv(raw_dir / "train_transaction.csv", index=False)
        identity.to_csv(raw_dir / "train_identity.csv", index=False)

        # Write minimal config pointing to tmp dirs
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(f"""\
                data:
                  raw_dir: "{raw_dir.as_posix()}"
                  interim_dir: "{(tmp_path / 'interim').as_posix()}"
                  processed_dir: "{(tmp_path / 'processed').as_posix()}"
                  transaction_file: "train_transaction.csv"
                  identity_file: "train_identity.csv"
            """)
        )

        df = load_dataset(config_path)

        assert len(df) == 10
        assert "id_01" in df.columns
        assert "isFraud" in df.columns
        assert df["TransactionID"].duplicated().sum() == 0

    def test_load_dataset_missing_file_raises(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(f"""\
                data:
                  raw_dir: "{tmp_path.as_posix()}"
                  interim_dir: "{tmp_path.as_posix()}"
                  processed_dir: "{tmp_path.as_posix()}"
                  transaction_file: "missing.csv"
                  identity_file: "also_missing.csv"
            """)
        )
        with pytest.raises(FileNotFoundError):
            load_dataset(config_path)
