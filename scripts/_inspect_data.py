import pandas as pd
import numpy as np

df = pd.read_parquet("data/interim/merged.parquet")

print("=== SHAPE ===")
print(df.shape)

print("\n=== OBJECT COLUMNS ===")
obj_cols = df.select_dtypes(include="object").columns.tolist()
print(obj_cols)

print("\n=== CARDINALITY OF OBJECT COLS ===")
for c in obj_cols:
    print(f"  {c:25s}: {df[c].nunique():5d} unique, {df[c].isna().mean()*100:.1f}% missing")

print("\n=== M COLUMNS UNIQUE VALUES ===")
m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
for c in m_cols:
    vals = sorted(df[c].dropna().unique().tolist()[:8])
    print(f"  {c}: {vals}")

print("\n=== TransactionDT ===")
print(f"  min={df['TransactionDT'].min()}, max={df['TransactionDT'].max()}")
print(f"  span_days={(df['TransactionDT'].max()-df['TransactionDT'].min())/86400:.1f}")

print("\n=== TransactionAmt percentiles ===")
print(df["TransactionAmt"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))

print("\n=== MISSING RATE BY GROUP ===")
groups = {
    "V": [c for c in df.columns if c.startswith("V")],
    "C": [c for c in df.columns if c.startswith("C") and c[1:].isdigit()],
    "D": [c for c in df.columns if c.startswith("D") and c[1:].isdigit()],
    "M": [c for c in df.columns if c.startswith("M") and c[1:].isdigit()],
    "id": [c for c in df.columns if c.startswith("id_")],
    "card": [c for c in df.columns if c.startswith("card")],
}
for g, cols in groups.items():
    miss = df[cols].isnull().mean()
    print(f"  {g}: min={miss.min():.2f} max={miss.max():.2f} n_above_70pct={(miss>0.7).sum()}")

miss_all = df.isnull().mean()
print(f"\n=== COLS SURVIVING 70% THRESHOLD: {(miss_all <= 0.7).sum()} / {len(df.columns)} ===")

print("\n=== EMAIL DOMAINS ===")
for col in ["P_emaildomain", "R_emaildomain"]:
    top = df[col].value_counts().head(8)
    print(f"  {col}:")
    for v, n in top.items():
        fraud_r = df.loc[df[col] == v, "isFraud"].mean()
        print(f"    {v:35s}: n={n:7,}  fraud_rate={fraud_r:.3f}")

print("\n=== ProductCD ===")
for v, n in df["ProductCD"].value_counts().items():
    print(f"  {v}: n={n:6,} fraud={df.loc[df['ProductCD']==v,'isFraud'].mean():.3f}")

print("\n=== card4 / card6 ===")
for col in ["card4", "card6"]:
    print(f"  {col}: {df[col].value_counts().to_dict()}")

print("\n=== DeviceType ===")
print(df["DeviceType"].value_counts())

print("\n=== C-columns missing ===")
c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
for c in c_cols:
    print(f"  {c}: missing={df[c].isna().mean()*100:.1f}%  mean={df[c].mean():.2f}")

print("\n=== D-columns missing ===")
d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
for c in d_cols:
    print(f"  {c}: missing={df[c].isna().mean()*100:.1f}%")
