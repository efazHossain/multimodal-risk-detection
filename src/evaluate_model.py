"""
Evaluate trained churn models and generate reporting artifacts.

Outputs:
- reports/model_evaluation.csv
- reports/model_evaluation.json
- reports/figures/confusion_matrix_logreg.png
- reports/figures/roc_curve_logreg.png
- reports/figures/pr_curve_logreg.png
- reports/figures/lift_curve_logreg.png
- reports/figures/gain_curve_logreg.png
- reports/logreg_lift_gain_table.csv
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


def build_lift_gain_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Build decile-level cumulative gain and lift metrics.
    """
    ranked = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob",
        ascending=False,
    ).reset_index(drop=True)
    ranked["bucket"] = pd.qcut(
        ranked.index + 1,
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1

    total_positives = max(1, int(ranked["y_true"].sum()))
    baseline_rate = ranked["y_true"].mean()

    rows = []
    cumulative_positives = 0
    for bucket, group in ranked.groupby("bucket", sort=True):
        bucket_size = len(group)
        positives = int(group["y_true"].sum())
        cumulative_positives += positives
        population_fraction = group.index.max() + 1
        population_fraction /= len(ranked)
        gain = cumulative_positives / total_positives
        response_rate = group["y_true"].mean() if bucket_size else 0.0
        lift = (response_rate / baseline_rate) if baseline_rate > 0 else 0.0

        rows.append(
            {
                "decile": int(bucket),
                "customers_in_bin": bucket_size,
                "positives_in_bin": positives,
                "response_rate": float(response_rate),
                "cumulative_positives": cumulative_positives,
                "cumulative_gain": float(gain),
                "lift": float(lift),
                "population_fraction": float(population_fraction),
            }
        )

    return pd.DataFrame(rows)


def save_lift_gain_plots(y_test: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Save lift and cumulative gain charts for the Logistic Regression model.
    """
    lift_gain_df = build_lift_gain_table(y_test, y_prob, n_bins=10)
    lift_gain_df.to_csv(REPORTS_DIR / "logreg_lift_gain_table.csv", index=False)

    gain_plot_df = pd.concat(
        [
            pd.DataFrame({"population_fraction": [0.0], "cumulative_gain": [0.0]}),
            lift_gain_df[["population_fraction", "cumulative_gain"]],
        ],
        ignore_index=True,
    )
    plt.figure()
    plt.plot(gain_plot_df["population_fraction"], gain_plot_df["cumulative_gain"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Cumulative Gains Curve - Logistic Regression")
    plt.xlabel("Population Fraction Contacted")
    plt.ylabel("Fraction of Churners Captured")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gain_curve_logreg.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(lift_gain_df["decile"], lift_gain_df["lift"], marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.title("Lift Curve - Logistic Regression")
    plt.xlabel("Decile (Highest Risk to Lowest Risk)")
    plt.ylabel("Lift vs Average Churn Rate")
    plt.xticks(lift_gain_df["decile"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lift_curve_logreg.png", dpi=300)
    plt.close()


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

    save_lift_gain_plots(y_test, y_prob)


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
