"""Re-run threshold search on saved model without re-training."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

import joblib
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from fraud_detection.models.train import (
    evaluate_lgbm,
    find_optimal_threshold,
    plot_lgbm_results,
)

booster  = joblib.load("models/best_model.joblib")
proc_dir = Path("data/processed")
splits_X = {n: pd.read_parquet(proc_dir / f"X_{n}.parquet") for n in ("train", "val", "test")}
splits_y = {n: pd.read_parquet(proc_dir / f"y_{n}.parquet").squeeze() for n in ("train", "val", "test")}

threshold, val_f1 = find_optimal_threshold(booster, splits_X["val"], splits_y["val"])
metrics, probas   = evaluate_lgbm(booster, splits_X, splits_y, threshold)
plot_lgbm_results(splits_X, splits_y, metrics, probas, threshold, Path("reports/figures"))

W, SEP, DIV = 72, "=" * 72, "-" * 72
col_w   = 13
splits  = ["train", "val", "test"]

def hrow(label, *vals):
    cells = "".join(f"{v:>{col_w}}" for v in vals)
    pad   = W - len(cells) - 2
    return f"  {label:<{pad}}{cells}"

print(f"\n{SEP}")
print(f"{'  LIGHTGBM -- UPDATED THRESHOLD RESULTS':^{W}}")
print(SEP)
print(f"  Classification threshold: {threshold:.2f}   (extended search 0.01-0.95)")
print(f"  {DIV}")
print(hrow("", *splits))
print(f"  {DIV}")
for key, label in [
    ("auroc",     "AUROC"),
    ("ap",        "Avg Precision (AP)"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1"),
]:
    print(hrow(label, *[f"{metrics[s][key]:.4f}" for s in splits]))

print(f"\n{SEP}")
print(f"  Confusion matrix  (threshold={threshold:.2f})")
print(f"  {DIV}")
print(hrow("", *splits))
print(f"  {DIV}")
for key, label in [
    ("tp", "True Positives  (fraud caught)"),
    ("fn", "False Negatives (fraud missed)"),
    ("fp", "False Positives (legit flagged)"),
    ("tn", "True Negatives  (legit cleared)"),
]:
    vals  = [f"{metrics[s][key]:>{col_w}}" for s in splits]
    cells = "".join(vals)
    pad   = W - len(cells) - 2
    print(f"  {label:<{pad}}{cells}")

fraud_total_val  = int(splits_y["val"].sum())
fraud_total_test = int(splits_y["test"].sum())
print(f"\n  Val  fraud catch rate : {metrics['val']['tp']  / fraud_total_val  * 100:.1f}%  ({metrics['val']['tp']}/{fraud_total_val})")
print(f"  Test fraud catch rate : {metrics['test']['tp'] / fraud_total_test * 100:.1f}%  ({metrics['test']['tp']}/{fraud_total_test})")
print(f"\n{SEP}\n")
