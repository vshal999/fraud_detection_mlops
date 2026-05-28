"""Inspect processed parquet files before building the training module."""
import pandas as pd
import numpy as np

X_train = pd.read_parquet("data/processed/X_train.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
X_val   = pd.read_parquet("data/processed/X_val.parquet")
y_val   = pd.read_parquet("data/processed/y_val.parquet").squeeze()

print("=== SHAPES ===")
print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
print(f"  X_val   {X_val.shape}   y_val   {y_val.shape}")

print("\n=== DTYPES ===")
print(X_train.dtypes.value_counts())

print("\n=== INT8 COLUMNS ===")
int8_cols = X_train.select_dtypes(include="int8").columns.tolist()
print(int8_cols)

print("\n=== FLOAT32 SAMPLE COLUMNS (first 10) ===")
f32_cols = X_train.select_dtypes(include="float32").columns.tolist()
print(f32_cols[:10], "...")

print("\n=== ANY MISSING? ===")
print(f"  X_train: {X_train.isnull().sum().sum()}")
print(f"  X_val:   {X_val.isnull().sum().sum()}")

print("\n=== FRAUD RATES ===")
print(f"  train: {y_train.mean():.4f}  val: {y_val.mean():.4f}")

print("\n=== VALUE RANGES (int8 cols) ===")
for c in int8_cols:
    vals = sorted(X_train[c].unique().tolist())
    print(f"  {c}: {vals}")

print("\n=== COLUMN LIST (first 30) ===")
print(X_train.columns.tolist()[:30])
print("...")
print(X_train.columns.tolist()[-10:])

# Check if any column has near-zero variance (could cause IForest issues)
print("\n=== LOW VARIANCE COLUMNS (std < 0.01) ===")
stds = X_train.astype(float).std()
low_var = stds[stds < 0.01]
print(f"  {len(low_var)} columns: {low_var.index.tolist()[:10]}")
