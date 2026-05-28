"""
Evaluation module — comprehensive test-set metrics and SHAP analysis.

Figures saved to reports/figures/:
  eval_roc_pr.png           - ROC + PR curves (test set)
  eval_confusion_matrix.png - annotated confusion matrix at optimal threshold
  eval_threshold_curve.png  - F1 / Precision / Recall vs threshold (val set)
  eval_shap_overview.png    - SHAP beeswarm + mean |SHAP| bar chart (top-20)
  eval_shap_waterfall_N.png - per-transaction waterfall for top-5 flagged fraud cases

Threshold is re-derived from the val set (never from test) to avoid leakage.

Usage:
    python -m fraud_detection.evaluation.metrics
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

_LGBM_FILENAME = "best_model.joblib"
_SHAP_SAMPLE_N = 5_000   # rows sampled for SHAP (full test set is slow)
_WATERFALL_N   = 5       # flagged transactions to explain individually


# ==============================================================================
# Threshold optimisation (val set only — test stays blind)
# ==============================================================================


def _find_optimal_threshold(
    booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float]:
    """Grid-search threshold 0.01–0.95 for best F1 on the val set."""
    probas  = booster.predict(X_val.values.astype(np.float32))
    y_true  = y_val.values
    best_f1 = -1.0
    best_t  = 0.5

    for t in np.arange(0.01, 0.96, 0.01):
        f1 = f1_score(y_true, (probas >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t  = float(round(t, 2))

    logger.info("Optimal threshold=%.2f  val_F1=%.4f", best_t, best_f1)
    return best_t, best_f1


# ==============================================================================
# Metrics computation
# ==============================================================================


def compute_metrics(
    booster,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    """
    Return (metrics_dict, probas) for one split at the given threshold.

    metrics_dict keys: auroc, pr_auc, precision, recall, f1,
                       tn, fp, fn, tp, threshold.
    """
    probas = booster.predict(X.values.astype(np.float32))
    preds  = (probas >= threshold).astype(int)
    y_np   = y.values

    auroc  = roc_auc_score(y_np, probas)
    pr_auc = average_precision_score(y_np, probas)
    prec   = precision_score(y_np, preds, zero_division=0)
    rec    = recall_score(y_np, preds, zero_division=0)
    f1     = f1_score(y_np, preds, zero_division=0)
    cm     = confusion_matrix(y_np, preds)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(cm[0, 0])))

    return dict(
        auroc=auroc, pr_auc=pr_auc,
        precision=float(prec), recall=float(rec), f1=float(f1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        threshold=threshold,
    ), probas


# ==============================================================================
# Figure 1 — ROC + PR curves (test set)
# ==============================================================================


def plot_roc_pr_curves(
    y_test: pd.Series,
    probas_test: np.ndarray,
    metrics: dict,
    out_dir: Path,
) -> Path:
    """
    Two-panel figure: ROC curve (left) + PR curve (right) on the test set.

    PR curve is the primary signal under 27:1 imbalance — a random classifier
    achieves AP ~= fraud_rate ~= 0.035; the LightGBM model should reach ~0.60+.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test Set — ROC & Precision-Recall Curves",
                 fontsize=13, fontweight="bold")

    y_np       = y_test.values
    auroc      = metrics["auroc"]
    pr_auc_val = metrics["pr_auc"]
    fraud_rate = float(y_np.mean())

    # --- ROC ---
    ax = axes[0]
    fpr, tpr, roc_thresholds = roc_curve(y_np, probas_test)
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2,
            label=f"LightGBM  AUC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random  AUC = 0.50")
    ax.fill_between(fpr, tpr, alpha=0.07, color="#1565C0")

    # annotate the operating point (at optimal threshold)
    idx = int(np.searchsorted(roc_thresholds[::-1], metrics["threshold"]))
    idx = min(idx, len(fpr) - 1)
    ax.scatter(fpr[idx], tpr[idx], s=80, color="#E53935", zorder=5,
               label=f"t={metrics['threshold']:.2f}  TPR={tpr[idx]:.2f}  FPR={fpr[idx]:.2f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    # --- PR ---
    ax = axes[1]
    prec_vals, rec_vals, pr_thresholds = precision_recall_curve(y_np, probas_test)
    ax.plot(rec_vals, prec_vals, color="#6A1B9A", linewidth=2,
            label=f"LightGBM  AP = {pr_auc_val:.4f}")
    ax.axhline(fraud_rate, color="k", linestyle="--", linewidth=0.8,
               label=f"Random  AP ≈ {fraud_rate:.3f}")
    ax.fill_between(rec_vals, prec_vals, fraud_rate,
                    where=(prec_vals >= fraud_rate), alpha=0.07, color="#6A1B9A")

    # annotate operating point
    if len(pr_thresholds) > 0:
        pr_idx = int(np.searchsorted(pr_thresholds, metrics["threshold"]))
        pr_idx = min(pr_idx, len(rec_vals) - 2)
        ax.scatter(rec_vals[pr_idx], prec_vals[pr_idx], s=80, color="#E53935", zorder=5,
                   label=f"t={metrics['threshold']:.2f}  P={prec_vals[pr_idx]:.2f}  R={rec_vals[pr_idx]:.2f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "eval_roc_pr.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Figure 2 — Confusion matrix (test set)
# ==============================================================================


def plot_confusion_matrix_figure(
    y_test: pd.Series,
    probas_test: np.ndarray,
    metrics: dict,
    out_dir: Path,
) -> Path:
    """
    Annotated confusion matrix on the test set.

    Each cell shows absolute count AND row-normalised percentage.
    Sidebar annotations: TPR (recall), FNR, FPR, TNR.
    """
    threshold = metrics["threshold"]
    preds     = (probas_test >= threshold).astype(int)
    y_np      = y_test.values
    cm        = confusion_matrix(y_np, preds)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"Confusion Matrix — Test Set  (threshold = {threshold:.2f})",
        fontsize=12, fontweight="bold",
    )

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes    = ["Legit (0)", "Fraud (1)"]
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm    = cm / row_totals.clip(min=1)

    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = "white" if count > thresh_val else "black"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)

    # Rate annotations on the right
    n_legit = int(tn + fp)
    n_fraud = int(tp + fn)
    tpr     = tp / max(n_fraud, 1)
    fnr     = fn / max(n_fraud, 1)
    fpr_val = fp / max(n_legit, 1)
    tnr     = tn / max(n_legit, 1)

    info = (
        f"TPR (Recall)  = {tpr:.3f}\n"
        f"FNR (Miss)    = {fnr:.3f}\n"
        f"FPR           = {fpr_val:.3f}\n"
        f"TNR (Spec.)   = {tnr:.3f}\n"
        f"Precision     = {metrics['precision']:.3f}\n"
        f"F1            = {metrics['f1']:.3f}"
    )
    ax.text(2.35, 0.5, info, transform=ax.transData,
            fontsize=9, va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3E5F5", alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / "eval_confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Figure 3 — Threshold optimisation curve (val set)
# ==============================================================================


def plot_threshold_curve(
    y_val: pd.Series,
    probas_val: np.ndarray,
    optimal_threshold: float,
    out_dir: Path,
) -> Path:
    """
    F1 / Precision / Recall vs classification threshold on the val set.

    Shows why the default 0.5 is wrong for heavily imbalanced fraud detection
    and where the optimal F1 threshold lands.
    """
    thresholds = np.arange(0.01, 0.96, 0.005)
    y_np       = y_val.values

    f1_vals   = [f1_score(y_np, (probas_val >= t).astype(int), zero_division=0)
                 for t in thresholds]
    prec_vals = [precision_score(y_np, (probas_val >= t).astype(int), zero_division=0)
                 for t in thresholds]
    rec_vals  = [recall_score(y_np, (probas_val >= t).astype(int), zero_division=0)
                 for t in thresholds]

    best_idx = int(np.argmax(f1_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Threshold Optimisation — Validation Set",
                 fontsize=12, fontweight="bold")

    ax.plot(thresholds, f1_vals,   color="#6A1B9A", linewidth=2,   label="F1")
    ax.plot(thresholds, prec_vals, color="#1565C0", linewidth=1.5,
            linestyle="--", label="Precision")
    ax.plot(thresholds, rec_vals,  color="#E53935", linewidth=1.5,
            linestyle="--", label="Recall")

    ax.axvline(optimal_threshold, color="black", linestyle=":",
               linewidth=1.5, label=f"Optimal  t = {optimal_threshold:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1.0,
               alpha=0.7, label="Default  t = 0.50")

    ax.scatter(thresholds[best_idx], f1_vals[best_idx],
               s=80, zorder=5, color="#6A1B9A",
               label=f"Best F1 = {f1_vals[best_idx]:.4f}")

    ax.set_xlabel("Classification threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlim(0, 0.96)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "eval_threshold_curve.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Figure 4 — SHAP overview (beeswarm + bar chart)
# ==============================================================================


def plot_shap_overview(
    booster,
    X_test: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, np.ndarray, float | np.ndarray]:
    """
    Compute SHAP values and produce a two-panel overview figure:

      Left  — beeswarm plot showing feature impact distribution for top-20 features.
               Each dot is one transaction. Colour = feature value (red=high, blue=low).
               X-axis = SHAP value (impact on log-odds of fraud).
      Right — mean |SHAP| bar chart (top-20 by mean absolute impact).
               Cleaner than beeswarm for ranking; beeswarm shows the direction + spread.

    Returns (out_path, shap_values_array, expected_value) so the waterfall
    plots can reuse the already-computed TreeExplainer without re-running it.
    """
    rng    = np.random.default_rng(42)
    n      = min(_SHAP_SAMPLE_N, len(X_test))
    idx    = rng.choice(len(X_test), size=n, replace=False)
    X_samp = X_test.iloc[idx].reset_index(drop=True)
    X_np   = X_samp.values.astype(np.float32)

    logger.info(
        "Computing SHAP values on %d-row sample  (%d features) ...",
        n, X_np.shape[1],
    )
    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_np)

    # Some SHAP versions return list[neg, pos] for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    feature_names = X_test.columns.tolist()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top20_idx     = np.argsort(mean_abs_shap)[-20:]

    # --- beeswarm panel ---
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("SHAP Analysis — Top-20 Feature Impact (test set sample)",
                 fontsize=13, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(
        shap_values[:, top20_idx],
        X_samp.iloc[:, top20_idx],
        feature_names=[feature_names[i] for i in top20_idx],
        show=False,
        max_display=20,
        plot_size=None,
    )
    axes[0].set_title("Beeswarm — Feature Impact Distribution",
                      fontweight="bold", pad=10)
    axes[0].spines[["top", "right"]].set_visible(False)

    # --- bar chart panel ---
    ax = axes[1]
    sorted_top = top20_idx[np.argsort(mean_abs_shap[top20_idx])]
    ax.barh(
        [feature_names[i] for i in sorted_top],
        mean_abs_shap[sorted_top],
        color="#1565C0", edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)", fontsize=9)
    ax.set_title("Mean Absolute SHAP — Feature Ranking",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "eval_shap_overview.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)

    return out_path, shap_values, expected_value, X_samp


# ==============================================================================
# Figure 5 — SHAP waterfall plots for flagged fraud transactions
# ==============================================================================


def plot_shap_waterfalls(
    booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probas_test: np.ndarray,
    threshold: float,
    out_dir: Path,
    n_cases: int = _WATERFALL_N,
) -> list[Path]:
    """
    SHAP waterfall plots for the top-N highest-confidence flagged fraud transactions.

    Selects transactions where:
      - predicted fraud (proba >= threshold)
      - true label = 1 (actual fraud — so we're explaining correct catches)
      - highest predicted probability among that group

    Each waterfall shows how each feature pushed the model's output above/below
    the base rate (expected log-odds). Red bars = pushed toward fraud; blue = away.
    """
    preds_bool = probas_test >= threshold
    y_np       = y_test.values

    # True positives sorted by confidence (highest proba first)
    tp_mask = preds_bool & (y_np == 1)
    if tp_mask.sum() == 0:
        logger.warning("No true-positive fraud cases found — skipping waterfall plots.")
        return []

    tp_indices = np.where(tp_mask)[0]
    tp_probas  = probas_test[tp_indices]
    order      = tp_indices[np.argsort(-tp_probas)]
    top_n_idx  = order[:n_cases]

    X_cases = X_test.iloc[top_n_idx].reset_index(drop=True)
    X_np    = X_cases.values.astype(np.float32)

    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_np)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    feature_names = X_test.columns.tolist()
    out_paths: list[Path] = []

    for rank, (local_i, global_i) in enumerate(zip(range(len(top_n_idx)), top_n_idx)):
        proba = probas_test[global_i]
        explanation = shap.Explanation(
            values       = shap_values[local_i],
            base_values  = float(expected_value),
            data         = X_np[local_i],
            feature_names= feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=15, show=False)

        fig.suptitle(
            f"SHAP Waterfall — Flagged Transaction #{rank + 1}  "
            f"(P(fraud) = {proba:.3f})",
            fontsize=11, fontweight="bold", y=1.01,
        )

        out_path = out_dir / f"eval_shap_waterfall_{rank}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=130)
        plt.close(fig)
        logger.info("Saved -> %s", out_path)
        out_paths.append(out_path)

    return out_paths


# ==============================================================================
# Console report
# ==============================================================================


def _print_metrics_table(
    metrics: dict,
    val_f1: float,
    threshold: float,
    fraud_rate: float,
) -> None:
    """
    ASCII summary table.

    Sections:
      [1] Test set metrics vs simple baselines
      [2] Confusion matrix breakdown on test
      [3] Business impact summary
    """
    W   = 64
    SEP = "=" * W
    DIV = "-" * W

    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    n_fraud = tp + fn
    n_legit = tn + fp
    n_total = tn + fp + fn + tp

    print(f"\n{SEP}")
    print(f"{'  EVALUATION REPORT — TEST SET':^{W}}")
    print(SEP)

    # [1] Metrics
    print("  [1] Discriminative performance")
    print(f"  {DIV}")
    rows = [
        ("ROC-AUC",            f"{metrics['auroc']:.4f}", "0.5000 (random)"),
        ("PR-AUC",             f"{metrics['pr_auc']:.4f}", f"{fraud_rate:.4f} (random)"),
        ("Precision",          f"{metrics['precision']:.4f}", "--"),
        ("Recall (TPR)",       f"{metrics['recall']:.4f}",  "--"),
        ("F1",                 f"{metrics['f1']:.4f}",      "--"),
        ("Val F1 (threshold)", f"{val_f1:.4f}",             f"threshold={threshold:.2f}"),
    ]
    col1, col2, col3 = 24, 12, 22
    for label, val, baseline in rows:
        print(f"    {label:<{col1}}{val:>{col2}}    {baseline}")

    # [2] Confusion matrix
    print(f"\n{SEP}")
    print(f"  [2] Confusion matrix  (threshold = {threshold:.2f})")
    print(f"  {DIV}")
    tpr = tp / max(n_fraud, 1)
    fnr = fn / max(n_fraud, 1)
    fpr = fp / max(n_legit, 1)
    tnr = tn / max(n_legit, 1)
    fdr = fp / max(tp + fp, 1)
    print(f"    {'True  Positives (fraud caught)':<34} {tp:>7,}  ({tpr*100:.1f}% of fraud)")
    print(f"    {'False Negatives (fraud missed)':<34} {fn:>7,}  ({fnr*100:.1f}% of fraud)")
    print(f"    {'False Positives (legit flagged)':<34} {fp:>7,}  ({fpr*100:.2f}% of legit)")
    print(f"    {'True  Negatives (legit cleared)':<34} {tn:>7,}  ({tnr*100:.2f}% of legit)")
    print(f"  {DIV}")
    print(f"    {'False Discovery Rate (FDR)':<34} {fdr:.4f}")

    # [3] Business impact
    print(f"\n{SEP}")
    print("  [3] Business impact summary")
    print(f"  {DIV}")
    review_queue = tp + fp
    review_pct   = review_queue / max(n_total, 1) * 100
    fraud_caught_pct = tpr * 100
    print(f"    Transactions in test set     {n_total:>10,}")
    print(f"    Actual fraud                 {n_fraud:>10,}  ({fraud_rate*100:.2f}%)")
    print(f"    Flagged for review           {review_queue:>10,}  ({review_pct:.2f}% of all txns)")
    print(f"    Fraud caught                 {tp:>10,}  ({fraud_caught_pct:.1f}% recall)")
    print(f"    Fraud missed                 {fn:>10,}")
    print(f"    Legit incorrectly flagged    {fp:>10,}")
    print(f"\n{SEP}\n")


# ==============================================================================
# Main orchestrator
# ==============================================================================


def run(config_path: str | Path = "config/config.yaml") -> dict[str, Path]:
    """
    Full evaluation pipeline:

    1. Load model (models/best_model.joblib) and processed parquet splits.
    2. Re-derive optimal F1 threshold on val set.
    3. Compute metrics on test set.
    4. Generate all five figure groups.
    5. Print ASCII summary table.

    Returns dict mapping figure names to their file paths.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir    = Path("models")
    figures_dir   = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = models_dir / _LGBM_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    booster = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # Load splits
    splits_X: dict[str, pd.DataFrame] = {}
    splits_y: dict[str, pd.Series]    = {}
    for name in ("val", "test"):
        x_path = processed_dir / f"X_{name}.parquet"
        y_path = processed_dir / f"y_{name}.parquet"
        if not x_path.exists():
            raise FileNotFoundError(
                f"{x_path} not found. Run build_features.py and train.py first."
            )
        splits_X[name] = pd.read_parquet(x_path)
        splits_y[name] = pd.read_parquet(y_path).squeeze()
        logger.info(
            "  %-5s  %d x %d  fraud=%.3f%%",
            name, *splits_X[name].shape, 100 * splits_y[name].mean(),
        )

    # Threshold from val (test stays blind until final evaluation below)
    threshold, val_f1 = _find_optimal_threshold(
        booster, splits_X["val"], splits_y["val"]
    )

    # Evaluate on test
    logger.info("Evaluating on test set ...")
    metrics, probas_test = compute_metrics(
        booster, splits_X["test"], splits_y["test"], threshold
    )
    logger.info(
        "Test  AUROC=%.4f  PR-AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f",
        metrics["auroc"], metrics["pr_auc"],
        metrics["f1"], metrics["precision"], metrics["recall"],
    )

    # Val probas for threshold curve
    _, probas_val = compute_metrics(
        booster, splits_X["val"], splits_y["val"], threshold
    )

    fraud_rate = float(splits_y["test"].mean())
    out: dict[str, Path] = {}

    logger.info("Generating figures ...")

    out["roc_pr"] = plot_roc_pr_curves(
        splits_y["test"], probas_test, metrics, figures_dir,
    )

    out["confusion_matrix"] = plot_confusion_matrix_figure(
        splits_y["test"], probas_test, metrics, figures_dir,
    )

    out["threshold_curve"] = plot_threshold_curve(
        splits_y["val"], probas_val, threshold, figures_dir,
    )

    shap_path, *_ = plot_shap_overview(booster, splits_X["test"], figures_dir)
    out["shap_overview"] = shap_path

    waterfall_paths = plot_shap_waterfalls(
        booster,
        splits_X["test"],
        splits_y["test"],
        probas_test,
        threshold,
        figures_dir,
        n_cases=_WATERFALL_N,
    )
    for i, p in enumerate(waterfall_paths):
        out[f"shap_waterfall_{i}"] = p

    _print_metrics_table(metrics, val_f1, threshold, fraud_rate)

    return out


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()


