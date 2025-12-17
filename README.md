# Multimodal Churn Risk Detection

An end-to-end, explainable machine learning system that predicts customer churn by combining structured customer data with unstructured text features, deployed as an interactive Streamlit application with SHAP-based explanations.

---

## 🚀 Project Overview

Customer churn is a major business risk across subscription-based industries.  
This project builds a **multimodal churn prediction system** that:

- Uses **structured customer, service, and billing data**
- Incorporates **unstructured support-note style text** via NLP embeddings
- Produces **interpretable predictions** using SHAP
- Deploys predictions and explanations through a **Streamlit web app**

The focus is not only predictive performance, but **model transparency and usability** in real-world decision-making.

---

## 📊 Dataset

- **IBM Telco Customer Churn Dataset** (Kaggle)
- Includes customer demographics, service subscriptions, billing information, and churn labels
- Unstructured text is **synthetically generated** from existing attributes to simulate customer support notes

> Synthetic text is used to demonstrate multimodal modeling and explainability in a realistic business setting.

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (EDA)
- Target imbalance analysis
- Numeric feature distributions and outliers
- Churn rates by contract type, service category, and payment method
- Identification of key churn drivers (e.g., tenure, contract length)

### 2. Feature Engineering
- Numeric features: median imputation
- Categorical features: one-hot encoding
- Text features: Sentence-BERT (`all-MiniLM-L6-v2`) embeddings
- Concatenation of structured and unstructured feature spaces

### 3. Modeling
Two models were trained and compared:
- **Logistic Regression** (baseline, interpretable)
- **XGBoost** (nonlinear ensemble)

Evaluation metrics:
- ROC-AUC
- Precision–Recall AUC

**Best model:** Logistic Regression  
Chosen for strong performance in high-dimensional feature space and superior interpretability.

### 4. Explainability
- **Global explainability:** SHAP beeswarm plots (notebook)
- **Local explainability:** SHAP waterfall plots (Streamlit app)
- Feature names mapped back to human-readable variables

### 5. Deployment
- Interactive **Streamlit app**
- Upload CSVs or select individual rows
- View churn probabilities and SHAP explanations in real time

---

## 📈 Results

| Model | ROC-AUC | PR-AUC |
|-----|--------|--------|
| Logistic Regression | **0.846** | **0.650** |
| XGBoost | 0.832 | 0.625 |

**Key insight:**  
The engineered feature space is largely linearly separable, allowing a simpler linear model to outperform a more complex ensemble while remaining explainable.

---

## 🔍 Example Explanation

The Streamlit app provides **customer-level explanations**, showing how factors such as:

- Customer tenure
- Monthly charges
- Contract type (month-to-month vs long-term)
- Internet service type

combine to increase or decrease churn risk.

This enables actionable insights rather than black-box predictions.

---

## 🖥️ Streamlit Application

**Features:**
- Upload CSV with one or more customers
- Predict churn probability
- Select a row for detailed SHAP waterfall explanation
- View top contributing features

Run locally:
```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
