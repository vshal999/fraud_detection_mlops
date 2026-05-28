"""
End-to-end fraud detection pipeline.

Stages (in order):
  1. Data      — load and merge raw CSVs → data/interim/merged.parquet
  2. Features  — engineer 229 features, split train/val/test → data/processed/
  3. Phase 1   — fit Isolation Forest, append anomaly_score
  4. Phase 2   — tune + train LightGBM with Optuna
  5. Evaluate  — final metrics and SHAP figures on test set

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --skip-data     # skip if merged.parquet exists
    python scripts/run_pipeline.py --skip-features # skip if processed/ exists
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.data.load import load_dataset
from fraud_detection.features.build_features import run as run_features
from fraud_detection.models.train import run as run_isolation_forest
from fraud_detection.models.train import run_lightgbm
from fraud_detection.evaluation.metrics import run as run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_BANNER = "=" * 64


def _stage(name: str) -> None:
    logger.info(_BANNER)
    logger.info("  STAGE: %s", name)
    logger.info(_BANNER)


def main(config_path: str = "config/config.yaml",
         skip_data: bool = False,
         skip_features: bool = False) -> None:
    t_total = time.time()

    # ------------------------------------------------------------------
    # Stage 1 — Data loading
    # ------------------------------------------------------------------
    interim_path = Path("data/interim/merged.parquet")
    if skip_data and interim_path.exists():
        logger.info("Skipping data stage (--skip-data, %s exists)", interim_path)
    else:
        _stage("1 / 5  —  Data loading & merging")
        t0 = time.time()
        df = load_dataset(config_path)
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(interim_path, index=False)
        logger.info("Saved merged dataset → %s  (%.1fs)", interim_path, time.time() - t0)
        logger.info("Shape: %d rows × %d cols  |  fraud rate: %.3f%%",
                    df.shape[0], df.shape[1], df["isFraud"].mean() * 100)

    # ------------------------------------------------------------------
    # Stage 2 — Feature engineering
    # ------------------------------------------------------------------
    processed_dir = Path("data/processed")
    if skip_features and (processed_dir / "X_train.parquet").exists():
        logger.info("Skipping features stage (--skip-features, processed/ exists)")
    else:
        _stage("2 / 5  —  Feature engineering")
        t0 = time.time()
        feat_outputs = run_features(config_path)
        logger.info(
            "Feature engineering complete  (%.1fs)  outputs: %s",
            time.time() - t0,
            {k: str(v) for k, v in feat_outputs.items()},
        )

    # ------------------------------------------------------------------
    # Stage 3 — Isolation Forest
    # ------------------------------------------------------------------
    _stage("3 / 5  —  Isolation Forest (unsupervised anomaly detection)")
    t0 = time.time()
    if_outputs = run_isolation_forest(config_path)
    logger.info("Isolation Forest complete  (%.1fs)", time.time() - t0)

    # ------------------------------------------------------------------
    # Stage 4 — LightGBM
    # ------------------------------------------------------------------
    _stage("4 / 5  —  LightGBM + Optuna hyperparameter tuning")
    t0 = time.time()
    lgbm_outputs = run_lightgbm(config_path)
    logger.info("LightGBM training complete  (%.1fs)", time.time() - t0)

    # ------------------------------------------------------------------
    # Stage 5 — Evaluation
    # ------------------------------------------------------------------
    _stage("5 / 5  —  Final evaluation & SHAP analysis")
    t0 = time.time()
    eval_outputs = run_evaluation(config_path)
    logger.info("Evaluation complete  (%.1fs)", time.time() - t0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(_BANNER)
    logger.info("  Pipeline complete  —  total time: %.1f min", (time.time() - t_total) / 60)
    logger.info(_BANNER)
    logger.info("Key outputs:")
    logger.info("  Models   : models/isolation_forest.joblib")
    logger.info("           : models/best_model.joblib")
    logger.info("  Figures  : reports/figures/")
    logger.info("  Data     : data/processed/X_{{train,val,test}}.parquet")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  uvicorn fraud_detection.api.app:app --reload --port 8000")
    logger.info("  streamlit run src/fraud_detection/dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full fraud detection pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data loading if data/interim/merged.parquet already exists")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering if data/processed/ already exists")
    args = parser.parse_args()

    main(
        config_path=args.config,
        skip_data=args.skip_data,
        skip_features=args.skip_features,
    )
