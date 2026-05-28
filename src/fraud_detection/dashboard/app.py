"""
Streamlit fraud-detection dashboard.

Connects to the FastAPI inference service (default http://localhost:8000).
Lets users input transaction details, then displays:
  - Fraud probability gauge
  - Colour-coded verdict card
  - Top 3 SHAP risk factors
  - SHAP waterfall chart (log-odds space)
  - Sidebar with model info and health status
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approx. training-period midpoint so TimeFeatureExtractor gets a sensible value
_DEFAULT_DT: int = 5_184_000  # ≈ day 60

# log(p_fraud / (1 - p_fraud)) at the training base rate of 3.4 %
_BASE_LOGIT: float = float(np.log(0.034 / (1.0 - 0.034)))  # ≈ -3.35

_EVAL_METRICS: dict[str, str] = {
    "AUROC": "0.8733",
    "PR-AUC": "0.4612",
    "F1": "0.4651",
    "Precision": "0.5640",
    "Recall": "0.3957",
}

_PRODUCT_CODES = ["W", "H", "C", "S", "R"]
_CARD_NETWORKS = ["visa", "mastercard", "discover", "american express"]
_CARD_TYPES    = ["credit", "debit", "charge card"]
_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "protonmail.com", "other",
]
_M_FLAG_OPTS = ["T", "F"]
_M4_OPTS     = ["M0", "M1", "M2"]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _health(api_url: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _predict(api_url: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return (result_dict, error_message). Exactly one of them is None."""
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        if r.status_code == 422:
            detail = r.json().get("detail", r.text)
            if isinstance(detail, list):
                msgs = [f"{e.get('loc', ['?'])[-1]}: {e.get('msg', '')}" for e in detail]
                return None, "; ".join(msgs)
            return None, str(detail)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API — is `uvicorn fraud_detection.api.app:app` running?"
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def _gauge_fig(probability: float, threshold: float) -> go.Figure:
    pct = probability * 100
    thr = threshold  * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar":  {"color": "#222", "thickness": 0.25},
            "steps": [
                {"range": [0,  30],  "color": "#2ecc71"},
                {"range": [30, 60],  "color": "#f39c12"},
                {"range": [60, 100], "color": "#e74c3c"},
            ],
            "threshold": {
                "line":      {"color": "#1a1a2e", "width": 4},
                "thickness": 0.85,
                "value":     thr,
            },
        },
        title={"text": "Fraud Probability", "font": {"size": 18}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig



def _waterfall_fig(risk_factors: list[dict], probability: float) -> go.Figure:
    """SHAP waterfall in log-odds space. Top-3 named; residual lumped as 'Other'."""
    pred_logit = math.log(probability / (1.0 - probability)) if 0 < probability < 1 else _BASE_LOGIT
    top3_sum   = sum(rf["shap_value"] for rf in risk_factors)
    residual   = pred_logit - _BASE_LOGIT - top3_sum

    names  = [rf["feature"] for rf in risk_factors] + ["Other features", "Prediction"]
    values = [rf["shap_value"] for rf in risk_factors] + [residual, 0.0]
    measures = ["relative"] * (len(risk_factors) + 1) + ["total"]

    colors = []
    for rf in risk_factors:
        colors.append("#e74c3c" if rf["shap_value"] > 0 else "#2ecc71")
    colors.append("#f39c12" if residual >= 0 else "#2ecc71")
    colors.append("#1a1a2e")

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=names,
        y=values,
        base=_BASE_LOGIT,
        connector={"line": {"color": "#aaa", "dash": "dot", "width": 1}},
        decreasing={"marker": {"color": "#2ecc71"}},
        increasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#1a1a2e"}},
        textposition="outside",
        text=[f"{v:+.3f}" for v in values[:-1]] + [f"{pred_logit:.3f}"],
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#888",
        annotation_text="Decision boundary (log-odds = 0)",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title="SHAP Feature Contributions (log-odds)",
        yaxis_title="Log-odds contribution",
        xaxis_title=None,
        height=380,
        margin=dict(t=60, b=30, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML card helpers
# ---------------------------------------------------------------------------

def _verdict_html(is_fraud: bool, probability: float, threshold: float) -> str:
    if is_fraud:
        bg, border, text_color, emoji, label = "#fde8e8", "#e74c3c", "#7b1212", "🚨", "FRAUD DETECTED"
        detail = f"Probability {probability:.1%} exceeds threshold {threshold:.2f}"
    else:
        bg, border, text_color, emoji, label = "#e8fde8", "#2ecc71", "#145214", "✅", "LEGITIMATE"
        detail = f"Probability {probability:.1%} is below threshold {threshold:.2f}"

    return f"""
    <div style="
        background:{bg}; border-left:6px solid {border};
        border-radius:8px; padding:18px 24px; margin:8px 0;
    ">
        <p style="font-size:1.6rem; font-weight:700; margin:0; color:{text_color};">{emoji} {label}</p>
        <p style="font-size:0.95rem; color:#333; margin:4px 0 0;">{detail}</p>
    </div>
    """


def _risk_card_html(rank: int, rf: dict) -> str:
    arrow  = "▲" if rf["direction"] == "increases_risk" else "▼"
    colour = "#e74c3c" if rf["direction"] == "increases_risk" else "#2ecc71"
    val_str = f"{rf['feature_value']:.4g}" if rf["feature_value"] is not None else "N/A"
    return f"""
    <div style="
        border:1px solid #ddd; border-radius:8px; padding:12px 16px;
        margin:4px 0; background:#fafafa;
    ">
        <span style="color:#999; font-size:0.8rem;">#{rank}</span>
        <span style="font-weight:600; margin-left:8px;">{rf['feature']}</span>
        <span style="float:right; color:{colour}; font-weight:700; font-size:1.1rem;">
            {arrow} {rf['shap_value']:+.4f}
        </span>
        <br/>
        <span style="color:#666; font-size:0.85rem;">
            Value: {val_str} &nbsp;|&nbsp;
            {rf['direction'].replace('_', ' ').title()}
        </span>
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, dict | None]:
    with st.sidebar:
        st.title("⚙️ Configuration")

        api_url = st.text_input(
            "FastAPI URL",
            value="http://localhost:8000",
            help="Base URL of the running inference API.",
        )

        st.divider()
        st.subheader("API Health")
        health = _health(api_url)
        if health is None:
            st.error("API unreachable")
        else:
            cols = st.columns(2)
            cols[0].metric("Status",    health.get("status", "?").upper())
            cols[1].metric("Threshold", f"{health.get('threshold', 0):.2f}")
            cols[0].metric("Features",  health.get("feature_count", "?"))
            cols[1].metric("Uptime",    f"{health.get('uptime_seconds', 0):.0f}s")

        st.divider()
        st.subheader("Model Performance (test set)")
        for metric, value in _EVAL_METRICS.items():
            st.metric(metric, value)

        st.divider()
        st.subheader("Feature Pipeline")
        st.markdown("""
        **229 engineered features + anomaly score = 230 total**

        Transformers (in order):
        1. `HighMissingDropper` — drops cols > 70 % null
        2. `TimeFeatureExtractor` — hour, day-of-week, time-in-dataset
        3. `AmountFeatureExtractor` — log, bins, velocity ratios
        4. `EmailRiskEncoder` — fraud-rate per domain
        5. `FinalEncoder` — OHE categoricals, M-flag booleans,
           median-impute numerics, float32 cast

        **+ IsolationForest anomaly score** (unsupervised layer)
        """)

    return api_url.rstrip("/"), health


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

def _input_form(health: dict | None) -> dict[str, Any] | None:
    """Render the transaction input form. Returns payload dict or None."""
    threshold_hint = f"  (model threshold: {health['threshold']:.2f})" if health else ""
    st.subheader(f"Transaction Details{threshold_hint}")

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            amt = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01, max_value=999_999.0,
                value=150.0, step=10.0,
            )
            product = st.selectbox("Product Code", _PRODUCT_CODES, index=0)
            email = st.selectbox("Payer Email Domain", _EMAIL_DOMAINS, index=0)

        with col2:
            card4 = st.selectbox("Card Network", _CARD_NETWORKS, index=0)
            card6 = st.selectbox("Card Type", _CARD_TYPES, index=0)
            m1 = st.selectbox("M1 (name match)", _M_FLAG_OPTS, index=0)
            m4 = st.selectbox("M4 (address match)", _M4_OPTS, index=0)

        with col3:
            c1 = st.number_input("C1 (card count)", min_value=0.0, value=1.0, step=1.0)
            d1 = st.number_input("D1 (days since last txn)", min_value=0.0, value=30.0, step=1.0)
            card1 = st.number_input("Card1 (card ID proxy)", min_value=0.0, value=10_000.0, step=100.0)

        with st.expander("Additional features (optional)"):
            ecol1, ecol2, ecol3 = st.columns(3)
            addr1 = ecol1.number_input("addr1", value=300.0, step=10.0)
            addr2 = ecol2.number_input("addr2", value=87.0,  step=1.0)
            c2    = ecol3.number_input("C2", value=1.0, step=1.0)
            d10   = ecol1.number_input("D10 (days since acct)", value=100.0, step=1.0)
            m2    = ecol2.selectbox("M2", _M_FLAG_OPTS, index=0)
            m6    = ecol3.selectbox("M6", _M_FLAG_OPTS, index=0)

        submitted = st.form_submit_button("Analyse Transaction", use_container_width=True, type="primary")

    if not submitted:
        return None

    payload: dict[str, Any] = {
        "TransactionDT":  _DEFAULT_DT,
        "TransactionAmt": amt,
        "ProductCD":      product,
        "card1":          card1,
        "card4":          card4,
        "card6":          card6,
        "addr1":          addr1,
        "addr2":          addr2,
        "P_emaildomain":  email if email != "other" else None,
        "C1":             c1,
        "C2":             c2,
        "D1":             d1,
        "D10":            d10,
        "M1":             m1,
        "M2":             m2,
        "M4":             m4,
        "M6":             m6,
    }
    # Remove None values so they remain genuinely absent (API fills with NaN)
    return {k: v for k, v in payload.items() if v is not None}


# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------

def _render_results(result: dict[str, Any]) -> None:
    prob      = result["fraud_probability"]
    is_fraud  = result["is_fraud"]
    threshold = result["threshold"]
    factors   = result["top_risk_factors"]

    # Row 1: gauge + verdict
    row1_left, row1_right = st.columns([1, 1])
    with row1_left:
        st.plotly_chart(_gauge_fig(prob, threshold), use_container_width=True)
    with row1_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical centering spacer
        st.markdown(_verdict_html(is_fraud, prob, threshold), unsafe_allow_html=True)
        st.caption(f"Model version: {result.get('model_version', 'N/A')}")

    st.divider()

    # Row 2: risk factors + waterfall
    row2_left, row2_right = st.columns([1, 1.4])
    with row2_left:
        st.subheader("Top 3 Risk Factors")
        for i, rf in enumerate(factors, start=1):
            st.markdown(_risk_card_html(i, rf), unsafe_allow_html=True)

    with row2_right:
        st.plotly_chart(_waterfall_fig(factors, prob), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔍 Fraud Detection Dashboard")
    st.caption(
        "Powered by LightGBM + IsolationForest hybrid model · "
        "IEEE-CIS Fraud Detection Dataset"
    )

    api_url, health = _sidebar()

    if health is None:
        st.warning(
            "API is not reachable. Start the server with:\n\n"
            "```\nuvicorn fraud_detection.api.app:app --reload\n```\n\n"
            "Then refresh this page.",
            icon="⚠️",
        )

    st.divider()
    payload = _input_form(health)

    if payload is not None:
        with st.spinner("Scoring transaction…"):
            result, error = _predict(api_url, payload)

        if error:
            st.error(f"Prediction failed: {error}")
        else:
            st.divider()
            _render_results(result)


if __name__ == "__main__":
    main()
