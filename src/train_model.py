"""
Train churn prediction models using processed multimodal features.

Outputs:
- models/model_logreg.joblib
- models/model_xgb.joblib
- data/processed/X_train.npy
- data/processed/X_test.npy
- data/processed/y_train.npy
- data/processed/y_test.npy
- reports/model_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from make_features import DEFAULT_INPUT, build_features


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42


def load_or_create_features(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X/y if they exist. Otherwise, build them from the raw dataset.
    """
    x_path = PROCESSED_DIR / "X.npy"
    y_path = PROCESSED_DIR / "y.npy"

    if x_path.exists() and y_path.exists():
        print("Loading existing processed features...")
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        print("Processed features not found. Building features first...")
        X, y, _ = build_features(input_path)

    return X, y


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate binary classification predictions.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }


def train_models(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train Logistic Regression and XGBoost models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    logreg = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    print("Training Logistic Regression...")
    logreg.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb.fit(X_train, y_train)

    joblib.dump(logreg, MODELS_DIR / "model_logreg.joblib")
    joblib.dump(xgb, MODELS_DIR / "model_xgb.joblib")

    logreg_prob = logreg.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    metrics = {
        "logistic_regression": evaluate_predictions(y_test, logreg_prob),
        "xgboost": evaluate_predictions(y_test, xgb_prob),
    }

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Saved: {MODELS_DIR / 'model_logreg.joblib'}")
    print(f"Saved: {MODELS_DIR / 'model_xgb.joblib'}")
    print(f"Saved metrics: {REPORTS_DIR / 'model_metrics.json'}")
    print(json.dumps(metrics, indent=2))

    return metrics


def main(input_path: Path) -> None:
    X, y = load_or_create_features(input_path)
    train_models(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    main(args.input)