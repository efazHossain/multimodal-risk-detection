"""
Build structured + text features for multimodal churn prediction.

Outputs:
- data/processed/X.npy
- data/processed/y.npy
- data/processed/preprocess_struct.joblib
- data/processed/text_model.joblib
- data/processed/feature_names.json
- data/processed/churn_with_notes.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_support_notes import add_support_notes


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Handles sklearn version differences:
    newer versions use sparse_output, older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the IBM Telco churn dataset.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        df = df[df["Churn"].notna()].copy()

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Separate structured features, target, and support notes.
    """
    if "Churn" not in df.columns:
        raise ValueError("Expected target column `Churn` was not found.")

    if "support_notes" not in df.columns:
        df = add_support_notes(df)

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        bad_values = df.loc[y.isna(), "Churn"].unique()
        raise ValueError(f"Unexpected Churn values found: {bad_values}")

    drop_cols = [col for col in ["customerID", "Churn", "support_notes"] if col in df.columns]
    X_struct = df.drop(columns=drop_cols)
    support_notes = df["support_notes"].fillna("No support note available.")

    return X_struct, y.astype(int).to_numpy(), support_notes


def build_preprocessor(X_struct: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for structured customer data.
    """
    numeric_cols = X_struct.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X_struct.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_structured_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return human-readable feature names from the fitted ColumnTransformer.
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"structured_feature_{i}" for i in range(preprocessor.transformers_.shape[0])]


def build_features(input_path: Path = DEFAULT_INPUT) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build combined structured + text feature matrix.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Place the Telco CSV in data/raw/ or pass --input path/to/file.csv"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = clean_telco_data(df)
    df = add_support_notes(df)

    notes_output = PROCESSED_DIR / "churn_with_notes.csv"
    df.to_csv(notes_output, index=False)

    X_struct, y, support_notes = split_features_target(df)

    preprocessor = build_preprocessor(X_struct)
    X_struct_processed = preprocessor.fit_transform(X_struct)

    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    X_text = text_model.encode(
        support_notes.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    X = np.hstack([X_struct_processed, X_text])

    structured_names = preprocessor.get_feature_names_out().tolist()
    text_names = [f"text_embedding_{i:03d}" for i in range(X_text.shape[1])]
    feature_names = structured_names + text_names

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)

    joblib.dump(preprocessor, PROCESSED_DIR / "preprocess_struct.joblib")
    joblib.dump(text_model, PROCESSED_DIR / "text_model.joblib")

    with open(PROCESSED_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("Feature build complete.")
    print(f"Saved: {PROCESSED_DIR / 'X.npy'}")
    print(f"Saved: {PROCESSED_DIR / 'y.npy'}")
    print(f"Saved: {PROCESSED_DIR / 'preprocess_struct.joblib'}")
    print(f"Saved: {PROCESSED_DIR / 'text_model.joblib'}")
    print(f"Saved: {PROCESSED_DIR / 'feature_names.json'}")
    print(f"Feature matrix shape: {X.shape}")

    return X, y, feature_names


def main(input_path: Path) -> None:
    build_features(input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    main(args.input)