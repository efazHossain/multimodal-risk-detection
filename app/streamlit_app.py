from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_PATH = BASE_DIR / "data" / "sample" / "sample_customers.csv"

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import sys

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.generate_support_notes import make_note
from src.validate_input import validate_scoring_input

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    preprocess_struct = joblib.load(DATA_DIR / "preprocess_struct.joblib")
    text_model = joblib.load(DATA_DIR / "text_model.joblib")
    model = joblib.load(MODELS_DIR / "model_logreg.joblib")

    feature_names = None
    fn_path = DATA_DIR / "feature_names.json"
    if fn_path.exists():
        with open(fn_path, "r") as f:
            feature_names = json.load(f)

    X_all = np.load(DATA_DIR / "X.npy")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_all), size=min(600, len(X_all)), replace=False)
    X_bg = X_all[idx]

    explainer = shap.Explainer(model, X_bg, feature_names=feature_names)
    return preprocess_struct, text_model, model, explainer, feature_names

preprocess_struct, text_model, model, explainer, feature_names = load_artifacts()


@st.cache_data
def load_sample_template() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)

# ---------- UI ----------
st.title("Multimodal Churn Risk Detector (Explainable)")

st.write("Upload a CSV (single row or multiple rows) OR use the form to predict churn risk with SHAP explanations.")

tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])

def preprocess_rows(df_in: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    # Ensure TotalCharges numeric like notebooks
    df_in = df_in.copy()
    if "TotalCharges" in df_in.columns:
        df_in["TotalCharges"] = pd.to_numeric(df_in["TotalCharges"], errors="coerce")

    # Create support notes if not provided
    if "support_notes" not in df_in.columns:
        df_in["support_notes"] = df_in.apply(lambda r: make_note(r.to_dict()), axis=1)

    # Structured features = drop ID/target/text if present
    drop_cols = [c for c in ["customerID", "Churn", "support_notes"] if c in df_in.columns]
    X_struct = df_in.drop(columns=drop_cols)

    X_struct_processed = preprocess_struct.transform(X_struct)

    text_embeddings = text_model.encode(
        df_in["support_notes"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    X_combined = np.hstack([X_struct_processed, text_embeddings])
    return X_combined, df_in

def predict_and_explain(X_combined: np.ndarray, row_index: int = 0):
    proba = model.predict_proba(X_combined)[:, 1]
    shap_values = explainer(X_combined[row_index:row_index+1])
    return proba, shap_values


def add_revenue_at_risk(df_out: pd.DataFrame) -> pd.DataFrame:
    df_out = df_out.copy()
    if "MonthlyCharges" in df_out.columns:
        monthly = pd.to_numeric(df_out["MonthlyCharges"], errors="coerce").fillna(0.0)
        df_out["expected_revenue_at_risk"] = monthly * df_out["churn_probability"]
    return df_out


def render_validation_messages(errors: list[str], warnings: list[str]) -> None:
    if errors:
        st.error("Input validation failed.")
        for message in errors:
            st.write(f"- {message}")

    if warnings:
        st.warning("Input validation warnings")
        for message in warnings:
            st.write(f"- {message}")


def render_revenue_dashboard(df_out: pd.DataFrame) -> None:
    if "expected_revenue_at_risk" not in df_out.columns:
        return

    total_risk = float(df_out["expected_revenue_at_risk"].sum())
    avg_risk = float(df_out["expected_revenue_at_risk"].mean())
    high_risk_count = int((df_out["churn_probability"] >= 0.5).sum())

    st.subheader("Revenue At Risk")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Expected Monthly Revenue At Risk", f"${total_risk:,.2f}")
    metric_cols[1].metric("Average Revenue At Risk / Customer", f"${avg_risk:,.2f}")
    metric_cols[2].metric("Customers Above 0.50 Churn Risk", f"{high_risk_count}")

    top_cols = [col for col in ["customerID", "Contract", "PaymentMethod", "MonthlyCharges"] if col in df_out.columns]
    top_cols += ["churn_probability", "expected_revenue_at_risk"]
    top_risk = df_out.sort_values("expected_revenue_at_risk", ascending=False).head(10)
    st.write("Top customers by expected revenue at risk:")
    st.dataframe(top_risk[top_cols])

    if "Contract" in df_out.columns:
        contract_summary = (
            df_out.groupby("Contract", dropna=False)["expected_revenue_at_risk"]
            .sum()
            .sort_values(ascending=False)
        )
        st.write("Revenue at risk by contract type:")
        st.bar_chart(contract_summary)


def build_manual_row(values: dict) -> pd.DataFrame:
    template = load_sample_template().iloc[[0]].copy()

    defaults = {
        "customerID": "MANUAL-ENTRY",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "Churn": "No",
    }

    for col, value in defaults.items():
        if col in template.columns:
            template.at[template.index[0], col] = value

    for col, value in values.items():
        if col in template.columns:
            template.at[template.index[0], col] = value

    return template

with tab1:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        st.write("Preview:", df_up.head())
        validation = validate_scoring_input(df_up, preprocess_struct)
        render_validation_messages(validation.errors, validation.warnings)

        if not validation.errors:
            Xc, prepared_df = preprocess_rows(validation.cleaned_df)
            proba = model.predict_proba(Xc)[:, 1]
            df_out = prepared_df.copy()
            df_out["churn_probability"] = proba
            df_out = add_revenue_at_risk(df_out)

            st.subheader("Predictions")
            st.dataframe(df_out)
            render_revenue_dashboard(df_out)

            pick = st.number_input("Row to explain (0-indexed)", min_value=0, max_value=len(df_up)-1, value=0)
            _, shap_values = predict_and_explain(Xc, row_index=int(pick))

            st.subheader("SHAP Explanation (Selected Row)")

            # Waterfall
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            st.pyplot(fig)
            plt.close(fig)

            # Top contributors
            vals = shap_values.values[0]
            names = shap_values.feature_names if shap_values.feature_names is not None else [f"Feature {i}" for i in range(len(vals))]
            topk = 12
            idx = np.argsort(np.abs(vals))[-topk:][::-1]
            contrib = pd.DataFrame({"feature": np.array(names)[idx], "shap_value": vals[idx]})
            st.write("Top contributing features:")
            st.dataframe(contrib)

with tab2:
    st.subheader("Manual Entry")

    tenure = st.slider("tenure (months)", 0, 72, 12)
    monthly = st.slider("MonthlyCharges", 0, 150, 70)
    total = st.number_input("TotalCharges", value=float(monthly * max(tenure, 1)))

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    techsupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    payment = st.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.info("Manual entry fills the remaining Telco fields from a sample template row before scoring.")

    if st.button("Predict (Manual)"):
        row = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": techsupport,
            "PaymentMethod": payment
        }
        df_one = build_manual_row(row)
        try:
            validation = validate_scoring_input(df_one, preprocess_struct)
            render_validation_messages(validation.errors, validation.warnings)
            if not validation.errors:
                Xc, prepared_df = preprocess_rows(validation.cleaned_df)
                proba, shap_values = predict_and_explain(Xc, row_index=0)
                df_out = prepared_df.copy()
                df_out["churn_probability"] = proba
                df_out = add_revenue_at_risk(df_out)

                st.metric("Churn probability", f"{proba[0]:.3f}")
                render_revenue_dashboard(df_out)

                fig = plt.figure()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Manual entry failed due to schema mismatch: {e}")
            st.write("Tip: Use Upload CSV with a row from the original dataset schema.")
