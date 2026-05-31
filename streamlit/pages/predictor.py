import streamlit as st
import pandas as pd
import numpy as np
from millify import millify
from pathlib import Path
import plotly.graph_objects as go
import joblib
import gc

st.set_page_config(
    page_title="Customer Predictor",
    page_icon="🔍",
    layout="wide"
)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login from the main page.")
    st.stop()

BASE_DIR = Path(__file__).parent.parent

@st.cache_resource
def load_models():
    cat_model = joblib.load(BASE_DIR/"models"/"catboost_model.pkl")
    lgb_model = joblib.load(BASE_DIR/"models"/"lightgbm_model.pkl")
    xgb_model = joblib.load(BASE_DIR/"models"/"xgboost_model.pkl")
    feature_cols = joblib.load(BASE_DIR/"models"/"feature_columns.pkl")
    ensemble_info = joblib.load(BASE_DIR/"models"/"ensemble_info.pkl")
    encoders = joblib.load(BASE_DIR/"models"/"encoders.pkl")
    model_metrics = joblib.load(BASE_DIR/"models"/"model_metrics.pkl")
    return cat_model, lgb_model, xgb_model, feature_cols, ensemble_info, encoders, model_metrics

@st.cache_data
def load_test():
    return pd.read_parquet(BASE_DIR/"cleaned_data"/"test_final.parquet")

try:
    cat_model, lgb_model, xgb_model, feature_cols, ensemble_info, encoders, model_metrics = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.exception(e)
    st.stop()

try:
    test = load_test()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

st.header("Customer Default Predictor")

customer_id = st.selectbox(
    "Select Customer ID",
    sorted(test["SK_ID_CURR"].unique())
)

customer_data = test[test["SK_ID_CURR"] == customer_id].copy()

st.subheader("Customer Profile")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Income", millify(customer_data["AMT_INCOME_TOTAL"].iloc[0]))
with c2:
    st.metric("Credit Amount", millify(customer_data["AMT_CREDIT"].iloc[0]))
with c3:
    st.metric("Annuity", millify(customer_data["AMT_ANNUITY"].iloc[0]))
with c4:
    st.metric("Loan To Income", f"{customer_data['LOAN_TO_INCOME'].iloc[0]:.2f}")

st.divider()
st.subheader("Model Settings")

selected_threshold = st.slider(
    "Decision Threshold",
    min_value=0.10, max_value=0.90,
    value=float(ensemble_info["threshold"]),
    step=0.05
)
st.caption(f"Current Threshold: {selected_threshold:.2f}")

if st.button("Predict Default Risk", use_container_width=True):

    gc.collect()
    X = customer_data.drop(columns=["SK_ID_CURR"], errors="ignore").copy()

    for col in X.select_dtypes(include='category').columns:
        X[col] = X[col].astype(str)

    for col, encoder in encoders.items():
        if col in X.columns:
            try:
                X[col] = encoder.transform(X[col].astype(str))
            except:
                pass

    missing_cols = [col for col in feature_cols if col not in X.columns]
    if missing_cols:
        st.error(f"Missing {len(missing_cols)} model features.")
        st.stop()

    X = X[feature_cols]

    try:
        cat_prob = float(np.array(cat_model.predict_proba(X)).flatten()[1])
        lgb_prob = float(np.array(lgb_model.predict_proba(X)).flatten()[1])
        xgb_prob = float(np.array(xgb_model.predict_proba(X)).flatten()[1])
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.exception(e)
        st.stop()

    probability = (
        cat_prob * ensemble_info["cat_weight"]
        + lgb_prob * ensemble_info["lgb_weight"]
        + xgb_prob * ensemble_info["xgb_weight"]
    )

    prediction = int(probability >= selected_threshold)

    if probability >= 0.60:
        risk_segment = "High Risk"
        risk_color = "#640D0D"
    elif probability >= 0.30:
        risk_segment = "Medium Risk"
        risk_color = "#3D4F4A"
    else:
        risk_segment = "Low Risk"
        risk_color = "#61867B"

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Default Probability", f"{probability*100:.2f}%")
    with r2:
        st.metric("Prediction", "Likely Default" if prediction == 1 else "Likely Non Default")
    with r3:
        st.metric("Risk Segment", risk_segment)

    st.divider()

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Default Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 30], "color": "#61867B"},
                {"range": [30, 60], "color": "#3D4F4A"},
                {"range": [60, 100], "color": "#640D0D"}
            ]
        }
    ))
    gauge.update_layout(paper_bgcolor="#82B1A3", font=dict(color="black"))
    st.plotly_chart(gauge, use_container_width=True)

    if probability >= 0.60:
        st.error("❌ High Risk — Reject or enhanced review. Reduce credit limit. Stricter monitoring.")
    elif probability >= 0.30:
        st.warning("⚠️ Medium Risk — Conditional approval. Moderate limit. Monitor repayment.")
    else:
        st.success("✅ Low Risk — Approve. Eligible for standard products. Consider premium offers.")

    st.subheader("Key Customer Information")

    def safe_metric(value, decimals=2, use_millify=False):
        if pd.isna(value):
            return "N/A"
        if use_millify:
            return millify(value)
        return f"{value:.{decimals}f}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("External Score", safe_metric(customer_data["EXT_SOURCE_MEAN"].iloc[0]))
        val = customer_data["ACTIVE_CREDIT_COUNT"].iloc[0]
        st.metric("Active Credits", "N/A" if pd.isna(val) else str(int(val)))
    with c2:
        st.metric("Loan To Income", safe_metric(customer_data["LOAN_TO_INCOME"].iloc[0]))
        st.metric("Total Debt", safe_metric(customer_data["TOTAL_DEBT"].iloc[0], use_millify=True))
    with c3:
        st.metric("Limit Usage", safe_metric(customer_data["AVG_LIMIT_USAGE_RATIO"].iloc[0]))
        st.metric("Payment Delay", safe_metric(customer_data["AVG_PAYMENT_DELAY"].iloc[0]))

st.divider()
st.header("Model Performance")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("ROC-AUC", round(model_metrics["roc_auc"], 4))
with m2:
    st.metric("Catboost Weight", model_metrics["cat_weight"])
with m3:
    st.metric("LightGBM", model_metrics["lgb_weight"])
with m4:
    st.metric("XGBoost Weight", model_metrics['xgb_weight'])
with m5:
    st.metric("Threshold", model_metrics['threshold'])