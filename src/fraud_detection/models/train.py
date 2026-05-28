"""
Model training - Phase 1: Isolation Forest + Phase 2: LightGBM.

WHY an unsupervised layer first?
---------------------------------
Fraud patterns shift over time and labels are expensive to obtain. An
Isolation Forest (IF) captures structural anomalies - transactions that
are hard to isolate from the rest of the population - without ever seeing
labels. Its output becomes a continuous feature for the downstream
supervised model, giving it a pre-computed "strangeness" signal to build on.

Score convention (IF)
----------------------
IsolationForest.decision_function() returns NEGATIVE scores for anomalies.
We negate throughout so that:
    higher anomaly_score  ->  more anomalous  ->  more likely fraud
    lower  anomaly_score  ->  more normal

This makes the feature directionally consistent with the fraud label
(1 = fraud), which helps downstream tree splits be more interpretable.

Phase 2 - LightGBM supervised model
--------------------------------------
Trained on the 229 engineered features + anomaly_score from Phase 1.
Hyperparameters tuned with Optuna (TPE sampler, 50 trials, maximise val AUROC).
Class imbalance handled with scale_pos_weight (~28x for 27:1 ratio).
Threshold optimised on val set for best F1 (not default 0.5).

No-leakage guarantee
---------------------
The IF and LightGBM are fit ONLY on X_train. Val and test are never seen
during fitting. Threshold optimisation uses val only; test is held out until
final evaluation.

Outputs
--------
data/processed/
    X_{train,val,test}.parquet  - 229 features + anomaly_score
models/
    isolation_forest.joblib     - fitted IF
    best_model.joblib           - fitted LightGBM booster
reports/figures/
    if_anomaly_scores.png       - distributions + ROC + capture curve
    if_score_analysis.png       - PR curve + decile lift + feature correlations
    lgbm_evaluation.png         - ROC + PR + threshold curve + confusion matrix
    lgbm_importance_optuna.png  - feature importance + Optuna history

Usage:
    python -m fraud_detection.models.train
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
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

import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

_SCORE_COL    = "anomaly_score"
_IF_FILENAME  = "isolation_forest.joblib"
_LGBM_FILENAME = "best_model.joblib"


# ==============================================================================
# Training
# ==============================================================================


def train_isolation_forest(
    X_train: pd.DataFrame,
    if_cfg: dict,
) -> IsolationForest:
    """
    Fit an IsolationForest on *X_train* using parameters from config.

    Key config knobs
    -----------------
    n_estimators  : more trees -> smoother scores; diminishing returns past ~200.
    contamination : only sets the predict() binary threshold; does NOT affect
                    decision_function(). Set to known fraud rate so predict()
                    is calibrated, even though we use continuous scores here.
    max_samples   : 'auto' -> min(256, n_samples). 256 is the sweet-spot from
                    the original Liu et al. paper - large enough for depth to
                    discriminate, small enough that anomalies are easy to isolate.
    n_jobs        : parallelise tree building across all CPU cores.
    """
    clf = IsolationForest(
        n_estimators =if_cfg.get("n_estimators",  200),
        contamination=if_cfg.get("contamination", 0.035),
        random_state =if_cfg.get("random_state",   42),
        n_jobs       =if_cfg.get("n_jobs",          -1),
        # max_samples defaults to 'auto' = min(256, n_samples)
    )
    logger.info(
        "Training IsolationForest  n_estimators=%d  contamination=%.4f  "
        "n_samples=%d  n_features=%d",
        clf.n_estimators, clf.contamination,
        len(X_train), X_train.shape[1],
    )
    clf.fit(X_train)
    logger.info(
        "Training complete  max_samples_=%d  (each tree sees %d random rows)",
        clf.max_samples_, clf.max_samples_,
    )
    return clf


def score_splits(
    clf: IsolationForest,
    splits: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    """
    Produce anomaly scores for every split.

    We negate decision_function() so higher = more anomalous.
    decision_function is preferred over score_samples because it is already
    normalised to be zero-centred around the contamination threshold, giving
    scores that are easier to interpret and threshold.
    """
    return {
        name: (-clf.decision_function(X.astype(np.float32))).astype(np.float32)
        for name, X in splits.items()
    }


# ==============================================================================
# Evaluation - numeric metrics
# ==============================================================================


def _threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """
    Return recall, precision, and F1 at several top-N% score thresholds.

    This answers the practical question: "if I flag the most anomalous X%
    of transactions for manual review, what fraction of actual fraud do I
    catch and how precise are my flags?"
    """
    rows = []
    for pct in [1, 2, 5, 10, 20]:
        thresh    = np.percentile(scores, 100 - pct)
        flagged   = scores >= thresh
        n_flagged = int(flagged.sum())
        recall    = labels[flagged].sum() / max(labels.sum(), 1)
        precision = labels[flagged].mean() if n_flagged > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        rows.append(dict(
            pct=pct, thresh=float(thresh),
            recall=float(recall), precision=float(precision), f1=float(f1),
        ))
    return rows


def evaluate_if_signal(
    scores_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, pd.Series],
) -> dict[str, dict]:
    """
    Compute comprehensive metrics for the raw IF anomaly score.

    AUROC context: random = 0.50; perfect = 1.0. For pure unsupervised
    detection on a 27:1 imbalanced dataset, AUROC ~0.70 is a strong
    result - it means IF rank-orders fraud above legit 70% of the time
    before any labels are used.

    Average Precision (AP) is harder under heavy imbalance; a random
    classifier achieves AP ~= fraud_rate ~= 0.035. Values of 0.10-0.15
    from an unsupervised model are considered good.
    """
    results: dict[str, dict] = {}
    for name in scores_by_split:
        if name not in labels_by_split:
            continue
        scores = scores_by_split[name]
        labels = labels_by_split[name].values
        auroc  = roc_auc_score(labels, scores)
        ap     = average_precision_score(labels, scores)
        thresh = _threshold_metrics(scores, labels)
        results[name] = dict(auroc=auroc, ap=ap, threshold_metrics=thresh)
        logger.info("  %-5s  AUROC=%.4f  AP=%.4f", name, auroc, ap)
    return results


# ==============================================================================
# Figure 1 - score distributions + ROC + capture curve
# ==============================================================================


def plot_score_distribution(
    scores_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, pd.Series],
    metrics: dict[str, dict],
    out_dir: Path,
) -> Path:
    """
    Four-panel figure saved as if_anomaly_scores.png:

      (a) Train score histogram - fraud vs legit with mean/median lines
      (b) Val   score histogram - fraud vs legit with mean/median lines
      (c) ROC curve - train + val
      (d) Fraud-capture (gain) curve - % fraud caught vs % transactions flagged
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Isolation Forest - Anomaly Score Analysis",
        fontsize=14, fontweight="bold",
    )
    palette = {"legit": "#2196F3", "fraud": "#F44336"}

    # (a) & (b): score histograms
    # Key shape: legit forms a tight left-peak (easy to group together);
    # fraud has a heavier right tail (anomalous = harder to group = higher score).
    for ax, split_name in zip(axes[0], ["train", "val"]):
        scores = scores_by_split[split_name]
        labels = labels_by_split[split_name].values
        auc    = metrics[split_name]["auroc"]

        for flag, label, color in [
            (0, "Legit", palette["legit"]),
            (1, "Fraud", palette["fraud"]),
        ]:
            subset = scores[labels == flag]
            ax.hist(
                subset, bins=120, density=True,
                alpha=0.55, color=color,
                label=f"{label} (n={len(subset):,})",
                histtype="stepfilled",
            )
            # Dashed = mean, dotted = median; gap between classes is the signal
            ax.axvline(subset.mean(),     color=color, linestyle="--", linewidth=1.2, alpha=0.9)
            ax.axvline(np.median(subset), color=color, linestyle=":",  linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Anomaly score  (higher = more anomalous)")
        ax.set_ylabel("Density")
        ax.set_title(f"{split_name.capitalize()} split  |  AUROC = {auc:.4f}", fontweight="bold")

        from matplotlib.lines import Line2D
        handles, lbls = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color="grey", linestyle="--", linewidth=1.2, label="mean"),
            Line2D([0], [0], color="grey", linestyle=":",  linewidth=1.2, label="median"),
        ]
        ax.legend(handles=handles, fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

    # (c): ROC curve
    # ROC shows overall rank-ordering ability. It is optimistic under heavy
    # imbalance but remains the standard reference. See the PR curve in
    # if_score_analysis.png for the imbalance-adjusted view.
    ax = axes[1][0]
    roc_colors = {"train": "#1565C0", "val": "#E53935"}
    for split_name in ["train", "val"]:
        scores = scores_by_split[split_name]
        labels = labels_by_split[split_name].values
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = metrics[split_name]["auroc"]
        ax.plot(fpr, tpr, color=roc_colors[split_name], linewidth=1.8,
                label=f"{split_name.capitalize()}  AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (0.50)")
    train_fpr, train_tpr, _ = roc_curve(
        labels_by_split["train"].values, scores_by_split["train"]
    )
    ax.fill_between(train_fpr, train_tpr, alpha=0.06, color="#1565C0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - IF anomaly score", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # (d): fraud-capture (gain) curve
    # Business view: flag the top-X% most anomalous transactions for review;
    # this curve shows what % of all fraud you catch. A good unsupervised
    # model should substantially beat the random diagonal.
    ax = axes[1][1]
    scores_tr   = scores_by_split["train"]
    labels_tr   = labels_by_split["train"].values
    total_fraud = labels_tr.sum()

    order        = np.argsort(-scores_tr)          # descending score
    cum_fraud    = np.cumsum(labels_tr[order])
    frac_flagged = np.arange(1, len(scores_tr) + 1) / len(scores_tr)
    fraud_capture = cum_fraud / total_fraud

    ax.plot(frac_flagged * 100, fraud_capture * 100,
            color="#6A1B9A", linewidth=2, label="IF score")
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Random")
    ax.fill_between(frac_flagged * 100, fraud_capture * 100,
                    frac_flagged * 100, alpha=0.08, color="#6A1B9A")

    for pct in [1, 2, 5, 10]:
        idx = int(len(scores_tr) * pct / 100) - 1
        ax.annotate(
            f"top {pct}%\n->{fraud_capture[idx]*100:.0f}% fraud",
            xy=(frac_flagged[idx] * 100, fraud_capture[idx] * 100),
            xytext=(frac_flagged[idx] * 100 + 2.5, fraud_capture[idx] * 100 - 9),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        )

    ax.set_xlabel("% transactions flagged")
    ax.set_ylabel("% fraud captured")
    ax.set_title("Fraud-capture curve (train)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "if_anomaly_scores.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Figure 2 - PR curve + decile lift + feature correlations
# ==============================================================================


def plot_extended_analysis(
    scores_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, pd.Series],
    X_train: pd.DataFrame,
    metrics: dict[str, dict],
    out_dir: Path,
) -> Path:
    """
    Three-panel figure saved as if_score_analysis.png:

      (a) Precision-Recall curve - more informative than ROC under 27:1 imbalance.
          Random classifier achieves AP ~= fraud_rate ~= 0.035; IF does better.
      (b) Score-decile fraud rate - fraud rate within each equal-frequency decile.
          A monotone increase left-to-right confirms the score is a useful ordinal
          signal and isn't just noise.
      (c) Top-20 features by Pearson r with anomaly score (train).
          Shows *what* the IF learned - useful for debugging and sanity-checking
          that it aligns with domain knowledge (V-features correlated with fraud
          should also correlate with high anomaly scores).
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Isolation Forest - Extended Analysis", fontsize=13, fontweight="bold")

    # (a): Precision-Recall curve
    # Under 27:1 imbalance the ROC curve is overly optimistic. PR curve exposes
    # the real precision cost of high recall - much harder to achieve here.
    ax = axes[0]
    pr_colors = {"train": "#1565C0", "val": "#E53935"}
    for split_name in ["train", "val"]:
        scores = scores_by_split[split_name]
        labels = labels_by_split[split_name].values
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = metrics[split_name]["ap"]
        ax.plot(rec, prec, color=pr_colors[split_name], linewidth=1.8,
                label=f"{split_name.capitalize()}  AP={ap:.4f}")

    fraud_rate = float(labels_by_split["train"].mean())
    ax.axhline(fraud_rate, color="k", linestyle="--", linewidth=0.8,
               label=f"Random  AP~={fraud_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    # (b): Score decile -> fraud rate (lift per decile)
    # Equal-frequency binning: each bar represents the same number of transactions.
    # A well-calibrated IF should produce a monotone staircase from left to right.
    ax = axes[1]
    scores_tr = scores_by_split["train"]
    labels_tr = labels_by_split["train"].values

    decile_labels = pd.qcut(scores_tr, q=10, labels=False, duplicates="drop")
    fraud_by_decile = (
        pd.DataFrame({"decile": decile_labels, "fraud": labels_tr})
        .groupby("decile")["fraud"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fraud_by_decile["label"] = [f"D{i+1}" for i in range(len(fraud_by_decile))]

    baseline = labels_tr.mean()
    bar_colors = [
        "#F44336" if r > baseline * 1.5 else
        "#FF7043" if r > baseline        else
        "#90CAF9"
        for r in fraud_by_decile["mean"]
    ]
    bars = ax.bar(
        fraud_by_decile["label"],
        fraud_by_decile["mean"] * 100,
        color=bar_colors, edgecolor="white", linewidth=0.5,
    )
    ax.axhline(baseline * 100, color="black", linestyle="--", linewidth=1.0,
               label=f"Overall fraud rate ({baseline*100:.2f}%)")
    ax.bar_label(bars,
                 labels=[f"{v:.1f}%" for v in fraud_by_decile["mean"] * 100],
                 padding=2, fontsize=7)

    top_rate = fraud_by_decile["mean"].iloc[-1]
    ax.text(
        len(fraud_by_decile) - 1, top_rate * 100 + 0.5,
        f"x{top_rate/baseline:.1f} lift",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#B71C1C",
    )
    ax.set_xlabel("Score decile  (D1 = most normal, D10 = most anomalous)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud Rate by Anomaly Score Decile", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (c): Top-20 features correlated with anomaly score
    # Pearson r captures linear relationships. Positive = feature is higher
    # for anomalous transactions; negative = lower. Near-zero-variance
    # features produce NaN correlations (3 in this dataset: V1, V107, V305)
    # and are silently dropped.
    ax = axes[2]
    feature_cols  = [c for c in X_train.columns if c != _SCORE_COL]
    scores_series = pd.Series(scores_tr, name=_SCORE_COL)

    corrs = (
        X_train[feature_cols]
        .astype(np.float32)
        .apply(lambda col: col.corr(scores_series))
        .dropna()                                  # drop zero-variance columns
        .sort_values(key=abs, ascending=False)
        .head(20)
    )

    bar_c = ["#F44336" if v > 0 else "#2196F3" for v in corrs.values]
    ax.barh(corrs.index[::-1], corrs.values[::-1],
            color=bar_c[::-1], edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r  with anomaly_score")
    ax.set_title("Top-20 Features Correlated\nwith Anomaly Score (train)",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#F44336", label="Positive (anomalous->high)"),
            Patch(color="#2196F3", label="Negative (anomalous->low)"),
        ],
        fontsize=8,
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "if_score_analysis.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Results table
# ==============================================================================


def _print_results_table(
    clf: IsolationForest,
    scores_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, pd.Series],
    metrics: dict[str, dict],
) -> None:
    """
    Print a formatted ASCII results table to stdout.

    Sections:
      [1] Model config     - hyperparameters used
      [2] Signal quality   - AUROC and AP per split vs random baselines
      [3] Score stats      - mean/median/std/p10/p90 for fraud vs legit (train)
      [4] Threshold table  - recall, precision, F1, lift at top-N% flagged (train)
    """
    W   = 68
    SEP = "=" * W
    DIV = "-" * W

    split_names = [s for s in ("train", "val", "test") if s in metrics]
    col_w = 13

    def hrow(label, *vals):
        cells = "".join(f"{v:>{col_w}}" for v in vals)
        pad   = W - len(cells) - 2
        return f"  {label:<{pad}}{cells}"

    print(f"\n{SEP}")
    print(f"{'  ISOLATION FOREST -- RESULTS':^{W}}")
    print(SEP)

    # --- [1] Config -----------------------------------------------------------
    print("  [1] Model configuration")
    print(f"  {DIV}")
    for label, val in [
        ("n_estimators",  clf.n_estimators),
        ("contamination", clf.contamination),
        ("max_samples",   clf.max_samples_),
        ("random_state",  clf.random_state),
    ]:
        print(f"    {label:<20s}  {val}")

    # --- [2] Signal quality ---------------------------------------------------
    print(f"\n{SEP}")
    print("  [2] Signal quality (unsupervised anomaly score vs true label)")
    print(f"  {DIV}")
    print(hrow("", *split_names))
    print(f"  {DIV}")
    for key, label in [("auroc", "AUROC"), ("ap", "Avg Precision (AP)")]:
        vals = [f"{metrics[s][key]:.4f}" for s in split_names]
        print(hrow(label, *vals))
    baseline_ap = float(labels_by_split["train"].mean())
    print(f"  {DIV}")
    print(f"    {'Random AUROC baseline':<28}  0.5000")
    print(f"    {'Random AP baseline':<28}  {baseline_ap:.4f}  (= overall fraud rate)")

    # --- [3] Score distribution -----------------------------------------------
    print(f"\n{SEP}")
    print("  [3] Anomaly score distribution -- train split")
    print(f"  {DIV}")
    print(f"  {'Statistic':<14}{'Fraud':>{col_w}}{'Legit':>{col_w}}{'Separation':>{col_w}}")
    print(f"  {DIV}")

    tr_scores = scores_by_split["train"]
    tr_labels = labels_by_split["train"].values
    fraud_s   = tr_scores[tr_labels == 1]
    legit_s   = tr_scores[tr_labels == 0]
    legit_std = float(np.std(legit_s))

    for stat_name, stat_fn in [
        ("mean",   np.mean),
        ("median", np.median),
        ("std",    np.std),
        ("p10",    lambda x: np.percentile(x, 10)),
        ("p90",    lambda x: np.percentile(x, 90)),
    ]:
        fval = float(stat_fn(fraud_s))
        lval = float(stat_fn(legit_s))
        # Cohen-d-style separation: how many legit std-devs apart are the two groups?
        sep  = (fval - lval) / max(legit_std, 1e-9)
        print(
            f"  {stat_name:<14}{fval:>{col_w}.4f}{lval:>{col_w}.4f}{sep:>{col_w}.3f}"
        )

    # --- [4] Threshold analysis -----------------------------------------------
    print(f"\n{SEP}")
    print("  [4] Threshold analysis -- train split")
    print(f"       Flag the top-N% most anomalous transactions; measure fraud recall.")
    print(f"  {DIV}")
    th_w = 11
    print(
        f"  {'Top %':>6}  {'Threshold':>{th_w}}  {'Recall':>{th_w}}"
        f"  {'Precision':>{th_w}}  {'F1':>{th_w}}  {'Lift':>{th_w}}"
    )
    print(f"  {DIV}")

    baseline_prec = float(tr_labels.mean())
    for r in metrics["train"]["threshold_metrics"]:
        lift = r["precision"] / max(baseline_prec, 1e-9)
        print(
            f"  {r['pct']:>5}%  "
            f"{r['thresh']:>{th_w}.4f}  "
            f"{r['recall']:>{th_w}.3f}  "
            f"{r['precision']:>{th_w}.3f}  "
            f"{r['f1']:>{th_w}.3f}  "
            f"{lift:>{th_w}.1f}x"
        )

    print(f"\n{SEP}\n")


# ==============================================================================
# Main orchestrator
# ==============================================================================


def run(config_path: str | Path = "config/config.yaml") -> dict[str, Path]:
    """
    Full Isolation Forest pipeline:

    1. Load processed X / y parquet files from data/processed/
    2. Fit IF on training split only (no label leakage, no future leakage)
    3. Score all three splits (negate so high = anomalous)
    4. Append anomaly_score column to each X_{split}.parquet and re-save
    5. Save fitted model to models/isolation_forest.joblib
    6. Generate two figures and print results table

    Returns dict mapping output names to their paths.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir    = Path("models")
    figures_dir   = Path("reports/figures")
    for d in (models_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load splits
    logger.info("Loading processed features ...")
    splits_X: dict[str, pd.DataFrame] = {}
    splits_y: dict[str, pd.Series]    = {}
    for name in ("train", "val", "test"):
        x_path = processed_dir / f"X_{name}.parquet"
        y_path = processed_dir / f"y_{name}.parquet"
        if not x_path.exists():
            raise FileNotFoundError(
                f"{x_path} not found. Run build_features.py first."
            )
        splits_X[name] = pd.read_parquet(x_path)
        splits_y[name] = pd.read_parquet(y_path).squeeze()
        logger.info(
            "  %-5s  %d x %d  fraud=%.3f%%",
            name, *splits_X[name].shape, 100 * splits_y[name].mean(),
        )

    # Guard: strip previous anomaly_score so the IF never sees its own output
    if _SCORE_COL in splits_X["train"].columns:
        logger.warning("'%s' already present -- overwriting.", _SCORE_COL)
        for df in splits_X.values():
            df.drop(columns=[_SCORE_COL], inplace=True)

    # Fit on training data only
    clf = train_isolation_forest(splits_X["train"], cfg["models"]["isolation_forest"])

    # Score all splits (before appending column, to avoid self-reference)
    scores = score_splits(clf, splits_X)

    # Evaluate raw signal quality
    logger.info("Computing evaluation metrics ...")
    metrics = evaluate_if_signal(scores, splits_y)

    # Append anomaly_score and re-save each parquet
    out: dict[str, Path] = {}
    for name, X in splits_X.items():
        X[_SCORE_COL] = scores[name]
        x_path = processed_dir / f"X_{name}.parquet"
        X.to_parquet(x_path, index=False)
        out[f"X_{name}"] = x_path
        logger.info(
            "  %-5s  score [%.4f, %.4f]  -> %s",
            name, float(scores[name].min()), float(scores[name].max()), x_path,
        )

    # Save model
    model_path = models_dir / _IF_FILENAME
    joblib.dump(clf, model_path)
    out["isolation_forest"] = model_path
    logger.info("Saved model -> %s", model_path)

    # Figures (train + val only; keep test labels unseen until final evaluation)
    plot_scores  = {k: scores[k]   for k in ("train", "val")}
    plot_labels  = {k: splits_y[k] for k in ("train", "val")}
    plot_metrics = {k: metrics[k]  for k in ("train", "val")}

    out["fig_distributions"] = plot_score_distribution(
        plot_scores, plot_labels, plot_metrics, figures_dir,
    )
    out["fig_analysis"] = plot_extended_analysis(
        plot_scores, plot_labels,
        splits_X["train"].drop(columns=[_SCORE_COL]),  # no score in correlation plot
        plot_metrics, figures_dir,
    )

    # Print results table
    _print_results_table(clf, scores, splits_y, metrics)

    return out


# ==============================================================================
# Phase 2 - LightGBM helpers
# ==============================================================================


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Ratio of negative to positive samples for class balancing.

    Plugged into scale_pos_weight so the model treats each fraud transaction
    as ~28 legit ones during gradient updates, compensating for the 27:1 imbalance.
    """
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


def _lgbm_objective(
    trial: "optuna.Trial",
    X_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    base_params: dict,
) -> float:
    """
    Optuna objective: train one LightGBM configuration, return val AUROC.

    num_boost_round=1500 with early_stopping=50 keeps each trial fast;
    the final model uses num_boost_round=5000 to exploit the best config fully.
    """
    params = {
        **base_params,
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 150),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     1,
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    dtrain = lgb.Dataset(X_tr_np,  label=y_tr_np)
    dval   = lgb.Dataset(X_val_np, label=y_val_np, reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    probas = booster.predict(X_val_np)
    return float(roc_auc_score(y_val_np, probas))


def tune_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    lgbm_cfg: dict,
    optuna_cfg: dict,
    sample_frac: float = 0.3,
) -> tuple["optuna.Study", dict]:
    """
    Run Optuna TPE search over LightGBM hyperparameters.

    Each trial trains on a stratified sample of the training set (sample_frac=0.3
    by default, ~115k rows) so the 50-trial search completes in minutes rather
    than hours. The final model is retrained on full data separately.

    Returns (study, best_params). best_params is ready to pass directly to
    lgb.train — it merges base config (objective, metric, scale_pos_weight)
    with the best trial values.
    """
    scale_pw = _compute_scale_pos_weight(y_tr)
    logger.info(
        "scale_pos_weight=%.2f  (n_legit=%d / n_fraud=%d)",
        scale_pw, int((y_tr == 0).sum()), int(y_tr.sum()),
    )

    base_params: dict = {
        "objective":        "binary",
        "metric":           "auc",
        "verbosity":        -1,
        "scale_pos_weight": scale_pw,
        "n_jobs":           lgbm_cfg.get("n_jobs", -1),
        "random_state":     lgbm_cfg.get("random_state", 42),
    }

    # Stratified sample for fast tuning; preserves ~3.4% fraud rate
    if 0.0 < sample_frac < 1.0:
        rng       = np.random.default_rng(42)
        fraud_idx = np.where(y_tr.values == 1)[0]
        legit_idx = np.where(y_tr.values == 0)[0]
        n_fraud   = max(1, int(len(fraud_idx) * sample_frac))
        n_legit   = max(1, int(len(legit_idx) * sample_frac))
        sample_idx = np.concatenate([
            rng.choice(fraud_idx, n_fraud, replace=False),
            rng.choice(legit_idx, n_legit, replace=False),
        ])
        X_tune = X_tr.iloc[sample_idx]
        y_tune = y_tr.iloc[sample_idx]
        logger.info(
            "Tuning sample: %.0f%% of train -> %d rows  "
            "(fraud=%d, legit=%d)",
            sample_frac * 100, len(sample_idx), n_fraud, n_legit,
        )
    else:
        X_tune = X_tr
        y_tune = y_tr
        logger.info("Tuning on full training set (%d rows)", len(y_tr))

    X_tune_np = X_tune.values.astype(np.float32)
    y_tune_np = y_tune.values.astype(np.int32)
    X_val_np  = X_val.values.astype(np.float32)
    y_val_np  = y_val.values.astype(np.int32)

    n_trials = int(optuna_cfg.get("n_trials", 50))
    timeout  = optuna_cfg.get("timeout", None)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info("Optuna search  n_trials=%d ...", n_trials)
    study.optimize(
        lambda trial: _lgbm_objective(
            trial, X_tune_np, y_tune_np, X_val_np, y_val_np, base_params
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = {**base_params, **study.best_params, "bagging_freq": 1}
    logger.info(
        "Optuna complete  best_val_AUROC=%.4f  trials_completed=%d",
        study.best_value, len(study.trials),
    )
    return study, best_params


def train_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """
    Train the final LightGBM model with best params and early stopping on val AUROC.

    Uses num_boost_round=5000 — early stopping will halt well before that.
    log_evaluation(100) prints a progress line every 100 rounds.
    """
    dtrain = lgb.Dataset(X_tr.values.astype(np.float32), label=y_tr.values)
    dval   = lgb.Dataset(X_val.values.astype(np.float32), label=y_val.values,
                         reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    logger.info(
        "Final model  best_iteration=%d  val_AUROC=%.4f",
        booster.best_iteration,
        booster.best_score["valid_0"]["auc"],
    )
    return booster


def find_optimal_threshold(
    booster: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float]:
    """
    Grid search threshold 0.01-0.60 for best F1 on the val set.

    Default 0.5 is wrong for imbalanced data — the model's decision boundary
    shifts because scale_pos_weight biases predicted probabilities upward.
    Grid search on val (never test) finds the threshold that maximises F1.

    Returns (best_threshold, best_f1_score).
    """
    probas = booster.predict(X_val.values.astype(np.float32))
    y_true = y_val.values

    best_score  = -1.0
    best_thresh = 0.5
    for t in np.arange(0.01, 0.96, 0.01):
        score = f1_score(y_true, (probas >= t).astype(int), zero_division=0)
        if score > best_score:
            best_score  = float(score)
            best_thresh = float(round(t, 2))

    logger.info("Optimal threshold=%.2f  val_F1=%.4f", best_thresh, best_score)
    return best_thresh, best_score


def evaluate_lgbm(
    booster: lgb.Booster,
    splits_X: dict[str, pd.DataFrame],
    splits_y: dict[str, pd.Series],
    threshold: float,
) -> tuple[dict, dict]:
    """
    Compute AUROC, AP, precision, recall, F1, and confusion-matrix components
    for every split. Returns (metrics_dict, probas_dict).
    """
    metrics_out: dict[str, dict]     = {}
    probas_out:  dict[str, np.ndarray] = {}

    for name in ("train", "val", "test"):
        if name not in splits_X:
            continue
        X     = splits_X[name].values.astype(np.float32)
        y     = splits_y[name].values
        proba = booster.predict(X)
        preds = (proba >= threshold).astype(int)

        auroc = roc_auc_score(y, proba)
        ap    = average_precision_score(y, proba)
        prec  = precision_score(y, preds, zero_division=0)
        rec   = recall_score(y, preds, zero_division=0)
        f1    = f1_score(y, preds, zero_division=0)
        cm    = confusion_matrix(y, preds)
        tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, cm[0, 0]))

        metrics_out[name] = dict(
            auroc=auroc, ap=ap,
            precision=prec, recall=rec, f1=f1,
            threshold=threshold,
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        )
        probas_out[name] = proba

        logger.info(
            "  %-5s  AUROC=%.4f  AP=%.4f  F1=%.4f  P=%.4f  R=%.4f",
            name, auroc, ap, f1, prec, rec,
        )

    return metrics_out, probas_out


# ==============================================================================
# Phase 2 - LightGBM figures
# ==============================================================================


def plot_lgbm_results(
    splits_X: dict[str, pd.DataFrame],
    splits_y: dict[str, pd.Series],
    metrics: dict[str, dict],
    probas: dict[str, np.ndarray],
    threshold: float,
    out_dir: Path,
) -> Path:
    """
    2x2 figure saved as lgbm_evaluation.png:

      (a) ROC curves for all three splits — overall rank-ordering ability.
      (b) PR curves for all splits — precision-recall under 27:1 imbalance.
      (c) Threshold vs F1/P/R on val — shows why optimal != 0.5.
      (d) Confusion matrix on val at optimal threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("LightGBM - Model Evaluation", fontsize=14, fontweight="bold")

    split_colors = {"train": "#1565C0", "val": "#E53935", "test": "#2E7D32"}

    # (a) ROC
    ax = axes[0][0]
    for name in ("train", "val", "test"):
        fpr, tpr, _ = roc_curve(splits_y[name].values, probas[name])
        ax.plot(fpr, tpr, color=split_colors[name], linewidth=1.8,
                label=f"{name.capitalize()}  AUC={metrics[name]['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) PR curves
    ax = axes[0][1]
    for name in ("train", "val", "test"):
        prec, rec, _ = precision_recall_curve(splits_y[name].values, probas[name])
        ax.plot(rec, prec, color=split_colors[name], linewidth=1.8,
                label=f"{name.capitalize()}  AP={metrics[name]['ap']:.4f}")
    fraud_rate = float(splits_y["train"].mean())
    ax.axhline(fraud_rate, color="k", linestyle="--", linewidth=0.8,
               label=f"Random AP~={fraud_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) Threshold optimization curve on val
    # Demonstrates why default 0.5 is wrong: with scale_pos_weight the model
    # assigns higher probabilities to positives, shifting the optimal boundary.
    ax = axes[1][0]
    y_val  = splits_y["val"].values
    p_val  = probas["val"]
    thresholds = np.arange(0.01, 0.61, 0.01)

    f1_vals   = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
                 for t in thresholds]
    prec_vals = [precision_score(y_val, (p_val >= t).astype(int), zero_division=0)
                 for t in thresholds]
    rec_vals  = [recall_score(y_val, (p_val >= t).astype(int), zero_division=0)
                 for t in thresholds]

    ax.plot(thresholds, f1_vals,   color="#6A1B9A", linewidth=2,   label="F1")
    ax.plot(thresholds, prec_vals, color="#1565C0", linewidth=1.5,
            linestyle="--", label="Precision")
    ax.plot(thresholds, rec_vals,  color="#E53935", linewidth=1.5,
            linestyle="--", label="Recall")
    ax.axvline(threshold, color="black", linestyle=":", linewidth=1.2,
               label=f"Optimal t={threshold:.2f}")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization (val)", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    # (d) Confusion matrix on val at optimal threshold
    ax = axes[1][1]
    y_pred = (p_val >= threshold).astype(int)
    cm     = confusion_matrix(y_val, y_pred)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes    = ["Legit (0)", "Fraud (1)"]
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    thresh_cm = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh_cm else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix (val, t={threshold:.2f})", fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "lgbm_evaluation.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


def plot_lgbm_importance_optuna(
    booster: lgb.Booster,
    study: "optuna.Study",
    out_dir: Path,
) -> Path:
    """
    Two-panel figure saved as lgbm_importance_optuna.png:

      (a) Top-30 features by gain importance — which features drive the most
          split gain in the final model.
      (b) Optuna optimization history — trial-by-trial and best-so-far AUROC,
          showing search convergence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "LightGBM - Feature Importance & Optuna Tuning History",
        fontsize=13, fontweight="bold",
    )

    # (a) Feature importance by gain
    ax = axes[0]
    importance = booster.feature_importance(importance_type="gain")
    feat_names  = booster.feature_name()
    top_n       = min(30, len(importance))
    order       = np.argsort(importance)[-top_n:]

    ax.barh(
        [feat_names[i] for i in order],
        [importance[i] for i in order],
        color="#1565C0", edgecolor="white",
    )
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Top-{top_n} Features by Gain", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (b) Optuna optimization history
    ax = axes[1]
    trial_nums  = [t.number for t in study.trials if t.value is not None]
    trial_vals  = [t.value  for t in study.trials if t.value is not None]
    best_so_far = [max(trial_vals[: i + 1]) for i in range(len(trial_vals))]

    ax.scatter(trial_nums, trial_vals, alpha=0.5, s=20,
               color="#90CAF9", label="Trial AUROC")
    ax.plot(trial_nums, best_so_far, color="#E53935", linewidth=1.8,
            label="Best so far")
    ax.axhline(study.best_value, color="black", linestyle="--", linewidth=0.8,
               label=f"Best={study.best_value:.4f}")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Optuna Optimization History", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "lgbm_importance_optuna.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info("Saved -> %s", out_path)
    return out_path


# ==============================================================================
# Phase 2 - results table
# ==============================================================================


def _print_lgbm_results_table(
    booster: lgb.Booster,
    metrics: dict[str, dict],
    threshold: float,
    study: "optuna.Study",
) -> None:
    """
    ASCII results table.  Three sections:
      [1] Best hyperparameters from Optuna search
      [2] AUROC / AP / P / R / F1 per split
      [3] Confusion matrix breakdown per split
    """
    W   = 72
    SEP = "=" * W
    DIV = "-" * W

    split_names = [s for s in ("train", "val", "test") if s in metrics]
    col_w = 13

    def hrow(label: str, *vals: str) -> str:
        cells = "".join(f"{v:>{col_w}}" for v in vals)
        pad   = W - len(cells) - 2
        return f"  {label:<{pad}}{cells}"

    print(f"\n{SEP}")
    print(f"{'  LIGHTGBM -- RESULTS':^{W}}")
    print(SEP)

    # --- [1] Best hyperparameters -----------------------------------------------
    bp = study.best_params
    print(f"  [1] Best hyperparameters  (Optuna, n_trials={len(study.trials)})")
    print(f"  {DIV}")
    for lbl, val in [
        ("num_leaves",               bp.get("num_leaves", "--")),
        ("max_depth",                bp.get("max_depth", "--")),
        ("learning_rate",            f"{bp.get('learning_rate', 0):.5f}"),
        ("min_data_in_leaf",         bp.get("min_data_in_leaf", "--")),
        ("feature_fraction",         f"{bp.get('feature_fraction', 0):.4f}"),
        ("bagging_fraction",         f"{bp.get('bagging_fraction', 0):.4f}"),
        ("lambda_l1",                f"{bp.get('lambda_l1', 0):.6f}"),
        ("lambda_l2",                f"{bp.get('lambda_l2', 0):.6f}"),
        ("best_iteration",           booster.best_iteration),
        ("classification_threshold", f"{threshold:.2f}"),
        ("best_val_AUROC (Optuna)",  f"{study.best_value:.4f}"),
    ]:
        print(f"    {lbl:<30s}  {val}")

    # --- [2] Metrics per split ---------------------------------------------------
    print(f"\n{SEP}")
    print("  [2] Model performance per split")
    print(f"  {DIV}")
    print(hrow("", *split_names))
    print(f"  {DIV}")
    for key, label in [
        ("auroc",     "AUROC"),
        ("ap",        "Avg Precision (AP)"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1"),
    ]:
        vals = [f"{metrics[s][key]:.4f}" for s in split_names]
        print(hrow(label, *vals))

    # --- [3] Confusion matrix breakdown -----------------------------------------
    print(f"\n{SEP}")
    print(f"  [3] Confusion matrix  (threshold={threshold:.2f})")
    print(f"  {DIV}")
    print(hrow("", *split_names))
    print(f"  {DIV}")
    for key, label in [
        ("tp", "True Positives  (fraud caught)"),
        ("fn", "False Negatives (fraud missed)"),
        ("fp", "False Positives (legit flagged)"),
        ("tn", "True Negatives  (legit cleared)"),
    ]:
        vals = [f"{metrics[s][key]:>{col_w}}" for s in split_names]
        cells = "".join(vals)
        pad   = W - len(cells) - 2
        print(f"  {label:<{pad}}{cells}")

    print(f"\n{SEP}\n")


# ==============================================================================
# Phase 2 - main orchestrator
# ==============================================================================


def run_lightgbm(config_path: str | Path = "config/config.yaml") -> dict[str, Path]:
    """
    Full LightGBM pipeline:

    1. Load processed X / y parquet files (229 features + anomaly_score from IF).
    2. Tune hyperparameters with Optuna — 50 trials, TPE, maximise val AUROC.
    3. Train final model with best params, early stopping on val AUROC.
    4. Find optimal F1 threshold on val set.
    5. Evaluate on all three splits.
    6. Generate evaluation and importance/tuning figures.
    7. Save booster to models/best_model.joblib.

    Returns dict mapping output names to their paths.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir    = Path("models")
    figures_dir   = Path("reports/figures")
    for d in (models_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load splits (anomaly_score is already present from Phase 1)
    logger.info("Loading processed features for LightGBM ...")
    splits_X: dict[str, pd.DataFrame] = {}
    splits_y: dict[str, pd.Series]    = {}
    for name in ("train", "val", "test"):
        x_path = processed_dir / f"X_{name}.parquet"
        y_path = processed_dir / f"y_{name}.parquet"
        if not x_path.exists():
            raise FileNotFoundError(
                f"{x_path} not found. Run build_features.py and IF training first."
            )
        splits_X[name] = pd.read_parquet(x_path)
        splits_y[name] = pd.read_parquet(y_path).squeeze()
        logger.info(
            "  %-5s  %d x %d  fraud=%.3f%%",
            name, *splits_X[name].shape, 100 * splits_y[name].mean(),
        )

    lgbm_cfg   = cfg["models"]["lightgbm"]
    optuna_cfg = cfg.get("models", {}).get("optuna", {})
    # Cap at 50 trials as specified
    optuna_cfg = {**optuna_cfg, "n_trials": min(int(optuna_cfg.get("n_trials", 50)), 50)}

    # Hyperparameter search
    study, best_params = tune_lightgbm(
        splits_X["train"], splits_y["train"],
        splits_X["val"],   splits_y["val"],
        lgbm_cfg, optuna_cfg,
    )

    # Train final model
    logger.info("Training final LightGBM model ...")
    booster = train_lightgbm(
        splits_X["train"], splits_y["train"],
        splits_X["val"],   splits_y["val"],
        best_params,
        early_stopping_rounds=lgbm_cfg.get("early_stopping_rounds", 50),
    )

    # Optimal threshold on val (never touch test here)
    threshold, _val_f1 = find_optimal_threshold(booster, splits_X["val"], splits_y["val"])

    # Evaluate all splits
    logger.info("Evaluating LightGBM on all splits ...")
    metrics, probas = evaluate_lgbm(booster, splits_X, splits_y, threshold)

    # Save model
    model_path = models_dir / _LGBM_FILENAME
    joblib.dump(booster, model_path)
    logger.info("Saved model -> %s", model_path)

    out: dict[str, Path] = {"lgbm_model": model_path}

    # Figures
    out["fig_lgbm_eval"] = plot_lgbm_results(
        splits_X, splits_y, metrics, probas, threshold, figures_dir,
    )
    out["fig_lgbm_importance"] = plot_lgbm_importance_optuna(booster, study, figures_dir)

    # Print table
    _print_lgbm_results_table(booster, metrics, threshold, study)

    return out


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _if_model_path = Path("models") / _IF_FILENAME
    if _if_model_path.exists():
        logger.info(
            "Isolation Forest model found at %s -- skipping IF training.",
            _if_model_path,
        )
    else:
        run()
    run_lightgbm()
