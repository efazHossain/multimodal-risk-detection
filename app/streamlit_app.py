from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# ---------- MUST MATCH YOUR NOTE GENERATION ----------
def make_note(row: dict) -> str:
    notes = []

    tenure = float(row.get("tenure", 0) or 0)
    if tenure <= 6:
        notes.append("New customer still evaluating the service.")
    elif tenure >= 48:
        notes.append("Long-term customer with stable history.")

    contract = row.get("Contract", "")
    if contract == "Month-to-month":
        notes.append("Month-to-month plan; higher sensitivity to dissatisfaction.")
    elif contract == "Two year":
        notes.append("Two-year contract; lower churn tendency.")

    mc = float(row.get("MonthlyCharges", 0) or 0)
    if mc >= 90:
        notes.append("Customer mentions high monthly cost concerns.")
    elif mc <= 30:
        notes.append("Low monthly cost; fewer billing complaints.")

    pm = str(row.get("PaymentMethod", "")).lower()
    if "electronic check" in pm:
        notes.append("Payment via electronic check; occasional billing friction noted.")

    if row.get("InternetService", "") == "Fiber optic":
        notes.append("Fiber service; intermittent performance complaints reported.")

    if row.get("TechSupport", "") == "No":
        notes.append("No tech support; support resolution concerns mentioned.")

    if not notes:
        notes.append("No major issues reported recently.")
    return " ".join(notes)

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

# ---------- UI ----------
st.title("Multimodal Churn Risk Detector (Explainable)")

st.write("Upload a CSV (single row or multiple rows) OR use the form to predict churn risk with SHAP explanations.")

tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])

def preprocess_rows(df_in: pd.DataFrame) -> np.ndarray:
    # Ensure TotalCharges numeric like notebooks
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
    return X_combined

def predict_and_explain(X_combined: np.ndarray, row_index: int = 0):
    proba = model.predict_proba(X_combined)[:, 1]
    shap_values = explainer(X_combined[row_index:row_index+1])
    return proba, shap_values

with tab1:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        st.write("Preview:", df_up.head())

        Xc = preprocess_rows(df_up)
        proba = model.predict_proba(Xc)[:, 1]
        df_out = df_up.copy()
        df_out["churn_probability"] = proba

        st.subheader("Predictions")
        st.dataframe(df_out)

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
    st.subheader("Manual Entry (Minimal fields)")

    tenure = st.slider("tenure (months)", 0, 72, 12)
    monthly = st.slider("MonthlyCharges", 0, 150, 70)
    total = st.number_input("TotalCharges", value=float(monthly * max(tenure, 1)))

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    techsupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    payment = st.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # Fill remaining columns with safe defaults (must match training columns)
    # Easiest: require a template CSV row from the dataset (recommended in README).
    st.info("Manual entry uses a minimal set. For full compatibility, upload a CSV row from the Telco dataset schema.")

    if st.button("Predict (Manual)"):
        # Minimal row (may fail if your preprocess expects full schema)
        row = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": techsupport,
            "PaymentMethod": payment,
        }
        df_one = pd.DataFrame([row])
        try:
            Xc = preprocess_rows(df_one)
            proba, shap_values = predict_and_explain(Xc, row_index=0)

            st.metric("Churn probability", f"{proba[0]:.3f}")

            fig = plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Manual entry failed due to schema mismatch: {e}")
            st.write("Tip: Use Upload CSV with a row from the original dataset schema.")
