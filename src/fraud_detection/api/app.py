"""
FastAPI fraud detection inference service.

Endpoints:
  GET  /health   — model load status, threshold, feature count, uptime.
  POST /predict  — score a single transaction; return fraud probability,
                   classification decision, threshold used, and top-3 SHAP
                   risk factors explaining the score.

Startup loads all artefacts once into module-level state:
  • feature_pipeline.joblib  — sklearn Pipeline (229 features)
  • isolation_forest.joblib  — IsolationForest (adds anomaly_score → 230)
  • best_model.joblib        — LightGBM Booster
  • shap.TreeExplainer       — pre-built from booster

Usage:
    uvicorn fraud_detection.api.app:app --reload
    uvicorn fraud_detection.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engineered column sets — not present in raw input
# ---------------------------------------------------------------------------

_ENGINEERED_COLS = frozenset({
    "amt_log", "amt_bin",
    "hour_of_day", "day_of_week", "is_weekend", "time_in_dataset",
})
_RISK_SUFFIX = "_risk"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TransactionFeatures(BaseModel):
    """
    Raw transaction features accepted by the /predict endpoint.

    Required: TransactionDT and TransactionAmt — all others are optional.
    Anonymous V1–V339 and identity id_01–id_38 fields are accepted as
    extra fields and passed through to the feature pipeline unchanged.
    """

    model_config = ConfigDict(extra="allow")

    # ---- core ---------------------------------------------------------------
    TransactionDT: int = Field(
        ...,
        description="Seconds elapsed since a reference epoch (from the dataset)",
    )
    TransactionAmt: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Transaction amount in USD (must be positive, max $1 M)",
    )
    ProductCD: str | None = Field(None, description="Product code: W, H, C, S, or R")

    # ---- card ---------------------------------------------------------------
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = Field(None, description="Card network: visa / mastercard / discover / american express")
    card5: float | None = None
    card6: str | None = Field(None, description="Card type: credit / debit / debit or credit / charge card")

    # ---- address / distance -------------------------------------------------
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None

    # ---- email domains ------------------------------------------------------
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    # ---- count features C1–C14 ----------------------------------------------
    C1:  float | None = None
    C2:  float | None = None
    C3:  float | None = None
    C4:  float | None = None
    C5:  float | None = None
    C6:  float | None = None
    C7:  float | None = None
    C8:  float | None = None
    C9:  float | None = None
    C10: float | None = None
    C11: float | None = None
    C12: float | None = None
    C13: float | None = None
    C14: float | None = None

    # ---- timedelta features D1–D15 ------------------------------------------
    D1:  float | None = None
    D2:  float | None = None
    D3:  float | None = None
    D4:  float | None = None
    D5:  float | None = None
    D6:  float | None = None
    D7:  float | None = None
    D8:  float | None = None
    D9:  float | None = None
    D10: float | None = None
    D11: float | None = None
    D12: float | None = None
    D13: float | None = None
    D14: float | None = None
    D15: float | None = None

    # ---- match flags M1–M9 --------------------------------------------------
    M1: str | None = Field(None, description="'T' or 'F'")
    M2: str | None = Field(None, description="'T' or 'F'")
    M3: str | None = Field(None, description="'T' or 'F'")
    M4: str | None = Field(None, description="'M0', 'M1', or 'M2'")
    M5: str | None = Field(None, description="'T' or 'F'")
    M6: str | None = Field(None, description="'T' or 'F'")
    M7: str | None = Field(None, description="'T' or 'F'")
    M8: str | None = Field(None, description="'T' or 'F'")
    M9: str | None = Field(None, description="'T' or 'F'")

    # ---- device (identity table features) -----------------------------------
    DeviceType: str | None = None
    DeviceInfo: str | None = None

    # V1–V339 and id_01–id_38 are accepted via extra="allow"

    @field_validator("M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9")
    @classmethod
    def _check_m_binary(cls, v: str | None) -> str | None:
        if v is not None and v not in ("T", "F"):
            raise ValueError("must be 'T' or 'F'")
        return v

    @field_validator("M4")
    @classmethod
    def _check_m4(cls, v: str | None) -> str | None:
        if v is not None and v not in ("M0", "M1", "M2"):
            raise ValueError("must be 'M0', 'M1', or 'M2'")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Return a 1-row DataFrame including all extra fields (V1–V339, id_*)."""
        return pd.DataFrame([self.model_dump()])


class RiskFactor(BaseModel):
    """One SHAP-derived risk factor driving the fraud score."""

    feature: str = Field(..., description="Feature name as seen by the model")
    shap_value: float = Field(..., description="SHAP contribution (log-odds); positive = increases fraud risk")
    feature_value: float | None = Field(None, description="Processed feature value fed to the model")
    direction: Literal["increases_risk", "decreases_risk"]


class PredictResponse(BaseModel):
    """Fraud scoring result."""

    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold: float = Field(..., description="Classification threshold used")
    top_risk_factors: list[RiskFactor] = Field(..., description="Top 3 features by |SHAP|")
    model_version: str = "1.0"


class HealthResponse(BaseModel):
    """Service health metadata."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    threshold: float | None
    feature_count: int | None
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Artefact state
# ---------------------------------------------------------------------------


@dataclass
class _ModelState:
    pipeline: Pipeline
    isolation_forest: Any           # IsolationForest
    booster: Any                    # lgb.Booster
    shap_explainer: Any             # shap.TreeExplainer
    shap_expected_value: float
    threshold: float
    feature_names: list[str]        # 230 ordered output feature names
    raw_col_template: list[str]     # raw input column names (pipeline entry)
    loaded_at: float = field(default_factory=time.time)


_state: _ModelState | None = None


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------


def _build_raw_col_template(pipeline: Pipeline) -> list[str]:
    """
    Reconstruct the list of raw input column names the pipeline expects.

    The pipeline was fitted on the raw merged DataFrame. We reverse-engineer
    the expected input columns from the fitted state of each step rather than
    requiring the raw data files to be present at runtime.

    Logic:
      FinalEncoder.numeric_cols_  — includes engineered cols (remove them)
                                  — includes email *_risk cols (map back to originals)
      FinalEncoder.cat_cols_      — other string columns (post email-encoding)
      FinalEncoder.m_binary_cols_ — M1–M3, M5–M9 match flags
      m4_present_                 — M4 match flag
      TransactionDT               — consumed by TimeFeatureExtractor
    """
    encoder = pipeline.named_steps["encode"]

    raw_numeric: list[str] = []
    for col in encoder.numeric_cols_:
        if col in _ENGINEERED_COLS:
            continue
        if col.endswith(_RISK_SUFFIX):
            # e.g. P_emaildomain_risk → P_emaildomain
            raw_numeric.append(col[: -len(_RISK_SUFFIX)])
        else:
            raw_numeric.append(col)

    raw_m = list(encoder.m_binary_cols_)
    if encoder.m4_present_:
        raw_m.append("M4")

    # TransactionDT is consumed by TimeFeatureExtractor (then dropped)
    all_raw = ["TransactionDT"] + raw_numeric + list(encoder.cat_cols_) + raw_m

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in all_raw:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    logger.info("Raw column template: %d columns", len(unique))
    return unique


def _derive_threshold(booster, processed_dir: Path) -> float:
    """
    Recompute the optimal F1 threshold from the val split, mirroring train.py.

    This means the API always uses a threshold consistent with how the model
    was evaluated — no magic constants in config or code.
    """
    from sklearn.metrics import f1_score as _f1

    x_path = processed_dir / "X_val.parquet"
    y_path = processed_dir / "y_val.parquet"

    if not (x_path.exists() and y_path.exists()):
        logger.warning(
            "Val split not found; falling back to threshold=0.5. "
            "Run the full pipeline to restore correct threshold."
        )
        return 0.5

    X_val = pd.read_parquet(x_path).values.astype(np.float32)
    y_val = pd.read_parquet(y_path).squeeze().values

    probas    = booster.predict(X_val)
    best_f1   = -1.0
    best_t    = 0.5

    for t in np.arange(0.01, 0.96, 0.01):
        f1 = _f1(y_val, (probas >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t  = float(round(t, 2))

    logger.info("Threshold (re-derived from val): %.2f  (val F1=%.4f)", best_t, best_f1)
    return best_t


def _patch_main_for_unpickling() -> None:
    """
    Expose the pipeline's custom transformer classes in __main__.

    build_features.py was run as a script entry point, so joblib pickled its
    classes with __module__ = "__main__". At API load time __main__ is this
    module, so the attribute lookup fails. Patching fixes the mismatch without
    re-running training.
    """
    import sys
    from fraud_detection.features import build_features as _bf

    _main = sys.modules["__main__"]
    for _name in (
        "HighMissingDropper",
        "TimeFeatureExtractor",
        "AmountFeatureExtractor",
        "EmailRiskEncoder",
        "FinalEncoder",
    ):
        if not hasattr(_main, _name):
            setattr(_main, _name, getattr(_bf, _name))


def load_artifacts(config_path: str | Path = "config/config.yaml") -> _ModelState:
    """
    Load all inference artefacts from disk into a _ModelState.

    Called once at server startup; all objects are read-only at runtime
    so concurrent requests are safe without locks.
    """
    _patch_main_for_unpickling()

    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir    = Path("models")

    def _require(path: Path) -> Path:
        if not path.exists():
            raise RuntimeError(
                f"Required artefact not found: {path}. "
                "Run the full training pipeline first."
            )
        return path

    pipeline_path = _require(processed_dir / "feature_pipeline.joblib")
    if_path       = _require(models_dir    / "isolation_forest.joblib")
    lgbm_path     = _require(models_dir    / "best_model.joblib")

    logger.info("Loading feature pipeline ...")
    pipeline = joblib.load(pipeline_path)

    logger.info("Loading IsolationForest ...")
    isolation_forest = joblib.load(if_path)

    logger.info("Loading LightGBM booster ...")
    booster = joblib.load(lgbm_path)

    # Feature names from the processed test split schema (fast — reads metadata only)
    test_schema_path = processed_dir / "X_test.parquet"
    if test_schema_path.exists():
        feature_names = pq.read_schema(test_schema_path).names
        logger.info("Feature schema: %d features", len(feature_names))
    else:
        raise RuntimeError(
            f"Cannot determine feature schema: {test_schema_path} not found. "
            "Run build_features.py and train.py first."
        )

    raw_col_template = _build_raw_col_template(pipeline)

    logger.info("Building SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(booster)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        ev_arr = list(expected_value)
        # Binary LightGBM: some SHAP versions return [neg, pos], others return [pos]
        expected_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
    else:
        expected_value = float(expected_value)

    threshold = _derive_threshold(booster, processed_dir)

    logger.info(
        "All artefacts loaded. threshold=%.2f  features=%d  raw_cols=%d",
        threshold, len(feature_names), len(raw_col_template),
    )

    return _ModelState(
        pipeline          = pipeline,
        isolation_forest  = isolation_forest,
        booster           = booster,
        shap_explainer    = explainer,
        shap_expected_value = expected_value,
        threshold         = threshold,
        feature_names     = list(feature_names),
        raw_col_template  = raw_col_template,
    )


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------


def _score_transaction(
    state: _ModelState,
    features: TransactionFeatures,
) -> PredictResponse:
    """
    Run one transaction through the full inference pipeline.

    Steps:
      1. Build 1-row DataFrame aligned to the raw column template.
      2. Feature engineering via the fitted sklearn Pipeline (229 features).
      3. IsolationForest anomaly score (negated so high = anomalous).
      4. Append anomaly_score → 230 features, reorder to training column order.
      5. LightGBM predict_proba.
      6. SHAP TreeExplainer for the single row → top-3 risk factors.
    """
    # 1. Build aligned raw DataFrame in one shot (avoids pandas fragmentation)
    provided   = features.to_dataframe().iloc[0].to_dict()
    aligned    = {col: provided.get(col, np.nan) for col in state.raw_col_template}
    raw_df     = pd.DataFrame([aligned])

    # 2. Feature engineering (pipeline was fitted; transform only)
    X_eng: pd.DataFrame = state.pipeline.transform(raw_df)

    # 3. IF anomaly score — reorder to match training column order via feature_names_in_
    if hasattr(state.isolation_forest, "feature_names_in_"):
        if_cols = list(state.isolation_forest.feature_names_in_)
        X_for_if = X_eng[if_cols].astype(np.float32)
    else:
        X_for_if = X_eng.astype(np.float32)
    anomaly_score = float(-state.isolation_forest.decision_function(X_for_if)[0])

    # 4. Assemble the 230-feature vector in the exact training column order
    extra: dict[str, float] = {}
    if "anomaly_score" not in X_eng.columns:
        extra["anomaly_score"] = anomaly_score
    missing_out = [c for c in state.feature_names if c not in X_eng.columns and c not in extra]
    if missing_out:
        logger.warning("Output cols missing after pipeline; padding with NaN: %s", missing_out)
        for c in missing_out:
            extra[c] = np.nan

    X_final_df = pd.concat(
        [X_eng, pd.DataFrame([extra])] if extra else [X_eng],
        axis=1,
    )[state.feature_names]
    X_final = X_final_df.values.astype(np.float32)  # shape (1, 230)

    # 5. LightGBM probability
    proba = float(state.booster.predict(X_final)[0])
    is_fraud = bool(proba >= state.threshold)

    # 6. SHAP for this single row
    shap_vals = state.shap_explainer.shap_values(X_final)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    sv_row = shap_vals[0]  # shape: (230,)

    top3_idx = np.argsort(np.abs(sv_row))[-3:][::-1]
    risk_factors = []
    for i in top3_idx:
        fval = float(X_final[0, i])
        risk_factors.append(
            RiskFactor(
                feature       = state.feature_names[i],
                shap_value    = round(float(sv_row[i]), 6),
                feature_value = None if np.isnan(fval) else round(fval, 6),
                direction     = "increases_risk" if sv_row[i] > 0 else "decreases_risk",
            )
        )

    return PredictResponse(
        fraud_probability = round(proba, 6),
        is_fraud          = is_fraud,
        threshold         = state.threshold,
        top_risk_factors  = risk_factors,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _state
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Loading inference artefacts ...")
    try:
        _state = load_artifacts()
        logger.info("Startup complete — service ready.")
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        _state = None
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Fraud Detection API",
    description = (
        "IEEE-CIS fraud detection pipeline — IsolationForest + LightGBM + SHAP.\n\n"
        "**POST /predict** — score a transaction and get the top risk factors.\n"
        "**GET  /health**  — service status."
    ),
    version     = "1.0.0",
    lifespan    = _lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health() -> HealthResponse:
    """Return model load status and basic metadata."""
    if _state is None:
        return HealthResponse(
            status        = "degraded",
            model_loaded  = False,
            threshold     = None,
            feature_count = None,
            uptime_seconds= 0.0,
        )
    return HealthResponse(
        status        = "healthy",
        model_loaded  = True,
        threshold     = _state.threshold,
        feature_count = len(_state.feature_names),
        uptime_seconds= round(time.time() - _state.loaded_at, 1),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score a transaction for fraud",
    responses={
        200: {"description": "Fraud scoring result with SHAP risk factors"},
        422: {"description": "Validation error — check field constraints"},
        503: {"description": "Models not loaded; retry after server startup"},
    },
)
def predict(features: TransactionFeatures) -> PredictResponse:
    """
    Score a single transaction.

    **Required fields:** `TransactionDT`, `TransactionAmt`.
    All other named fields are optional; V1–V339 and id_01–id_38 can be
    passed as extra fields and will be forwarded to the feature pipeline.

    **Response:**
    - `fraud_probability` — calibrated fraud score in [0, 1]
    - `is_fraud` — True if probability ≥ threshold
    - `threshold` — classification threshold (optimised for F1 on val set)
    - `top_risk_factors` — top 3 features by |SHAP| driving this score
    """
    if _state is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. The service may still be starting up.",
        )
    try:
        return _score_transaction(_state, features)
    except Exception as exc:
        logger.exception("Prediction failed for incoming request")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        ) from exc
