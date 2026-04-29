"""
Validation helpers for batch churn scoring inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer


@dataclass
class ValidationResult:
    cleaned_df: pd.DataFrame
    errors: list[str]
    warnings: list[str]


def _get_transformer_columns(preprocessor: ColumnTransformer, transformer_name: str) -> list[str]:
    for name, _, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []


def validate_scoring_input(df: pd.DataFrame, preprocessor: ColumnTransformer) -> ValidationResult:
    """
    Validate an uploaded or manually created scoring dataframe against the
    fitted structured preprocessor.
    """
    cleaned_df = df.copy()
    errors: list[str] = []
    warnings: list[str] = []

    numeric_cols = _get_transformer_columns(preprocessor, "num")
    categorical_cols = _get_transformer_columns(preprocessor, "cat")
    required_cols = numeric_cols + categorical_cols

    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(cleaned_df=cleaned_df, errors=errors, warnings=warnings)

    unexpected_cols = [
        col for col in cleaned_df.columns if col not in required_cols + ["customerID", "Churn", "support_notes"]
    ]
    if unexpected_cols:
        warnings.append(f"Extra columns will be ignored by the model: {unexpected_cols}")

    if "customerID" in cleaned_df.columns:
        duplicate_ids = int(cleaned_df["customerID"].astype(str).duplicated().sum())
        if duplicate_ids > 0:
            warnings.append(f"Found {duplicate_ids} duplicate customerID values.")

    for col in numeric_cols:
        original = cleaned_df[col]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int(converted.isna().sum() - original.isna().sum())
        if invalid_count > 0:
            warnings.append(
                f"Column `{col}` contains {invalid_count} non-numeric values; they will be treated as missing."
            )
        cleaned_df[col] = converted

    null_columns = [col for col in required_cols if cleaned_df[col].isna().all()]
    if null_columns:
        errors.append(f"Required columns are entirely empty: {null_columns}")

    high_null_cols = [
        col for col in required_cols if 0 < cleaned_df[col].isna().mean() >= 0.3
    ]
    if high_null_cols:
        warnings.append(f"Columns with 30% or more missing values: {high_null_cols}")

    if "tenure" in cleaned_df.columns and (cleaned_df["tenure"] < 0).fillna(False).any():
        warnings.append("Some `tenure` values are negative.")

    if "MonthlyCharges" in cleaned_df.columns and (cleaned_df["MonthlyCharges"] < 0).fillna(False).any():
        warnings.append("Some `MonthlyCharges` values are negative.")

    if "TotalCharges" in cleaned_df.columns and (cleaned_df["TotalCharges"] < 0).fillna(False).any():
        warnings.append("Some `TotalCharges` values are negative.")

    if {"MonthlyCharges", "TotalCharges"}.issubset(cleaned_df.columns):
        too_low_total = (
            cleaned_df["TotalCharges"] < cleaned_df["MonthlyCharges"]
        ).fillna(False)
        if too_low_total.any():
            warnings.append(
                f"{int(too_low_total.sum())} rows have `TotalCharges` lower than `MonthlyCharges`."
            )

    cat_pipeline = None
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            cat_pipeline = transformer
            cat_cols = list(cols)
            break
    else:
        cat_cols = []

    if cat_pipeline is not None:
        onehot = cat_pipeline.named_steps.get("onehot")
        if onehot is not None and hasattr(onehot, "categories_"):
            for col, seen_values in zip(cat_cols, onehot.categories_):
                incoming = (
                    cleaned_df[col].dropna().astype(str).str.strip().unique().tolist()
                )
                unknown = sorted(set(incoming) - set(map(str, seen_values)))
                if unknown:
                    warnings.append(
                        f"Column `{col}` contains unseen categories that will be ignored: {unknown}"
                    )

    return ValidationResult(cleaned_df=cleaned_df, errors=errors, warnings=warnings)
