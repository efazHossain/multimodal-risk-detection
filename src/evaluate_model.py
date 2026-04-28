"""
Evaluate trained churn models and generate reporting artifacts.

Outputs:
- reports/model_evaluation.csv
- reports/model_evaluation.json
- reports/figures/confusion_matrix_logreg.png
- reports/figures/roc_curve_logreg.png
- reports/figures/pr_curve_logreg.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    x_test_path = PROCESSED_DIR / "X_test.npy"
    y_test_path = PROCESSED_DIR / "y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "Missing X_test.npy or y_test.npy.\n"
            "Run this first: python src/train_model.py"
        )

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    return X_test, y_test


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Calculate lift among the top k fraction of highest-risk customers.

    Example:
    k=0.10 means top 10% of customers ranked by churn probability.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be between 0 and 1.")

    n_top = max(1, int(len(y_true) * k))
    ranked_indices = np.argsort(y_prob)[::-1]
    top_indices = ranked_indices[:n_top]

    top_churn_rate = y_true[top_indices].mean()
    overall_churn_rate = y_true.mean()

    if overall_churn_rate == 0:
        return 0.0

    return float(top_churn_rate / overall_churn_rate)


def evaluate_model(model_name: str, model_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "lift_at_top_10_percent": lift_at_k(y_test, y_prob, k=0.10),
    }

    return {key: float(value) if key != "model" else value for key, value in metrics.items()}


def save_logreg_plots(X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Save standard evaluation plots for the Logistic Regression model.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "model_logreg.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Run this first: python src/train_model.py"
        )

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_logreg.png", dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve - Logistic Regression")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_logreg.png", dpi=300)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve - Logistic Regression")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve_logreg.png", dpi=300)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data()

    model_paths = {
        "logistic_regression": MODELS_DIR / "model_logreg.joblib",
        "xgboost": MODELS_DIR / "model_xgb.joblib",
    }

    results = []

    for model_name, model_path in model_paths.items():
        if model_path.exists():
            print(f"Evaluating {model_name}...")
            results.append(evaluate_model(model_name, model_path, X_test, y_test))
        else:
            print(f"Skipping missing model: {model_path}")

    if not results:
        raise FileNotFoundError(
            "No trained models found.\n"
            "Run this first: python src/train_model.py"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_evaluation.csv", index=False)

    with open(REPORTS_DIR / "model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_logreg_plots(X_test, y_test)

    print("Evaluation complete.")
    print(results_df)
    print(f"Saved: {REPORTS_DIR / 'model_evaluation.csv'}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()