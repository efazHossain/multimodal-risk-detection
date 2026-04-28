"""
Generate synthetic support-note style text for the IBM Telco churn dataset.

This creates a lightweight unstructured text column called `support_notes`
from existing structured customer attributes. These notes are used later
to create Sentence-BERT embeddings for multimodal churn prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "churn_with_notes.csv"


def make_note(row: dict) -> str:
    """
    Create a synthetic support note for one customer row.

    This logic should stay aligned with the Streamlit app's make_note function
    so training-time and app-time text features are consistent.
    """
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

    monthly_charges = float(row.get("MonthlyCharges", 0) or 0)
    if monthly_charges >= 90:
        notes.append("Customer mentions high monthly cost concerns.")
    elif monthly_charges <= 30:
        notes.append("Low monthly cost; fewer billing complaints.")

    payment_method = str(row.get("PaymentMethod", "")).lower()
    if "electronic check" in payment_method:
        notes.append("Payment via electronic check; occasional billing friction noted.")

    if row.get("InternetService", "") == "Fiber optic":
        notes.append("Fiber service; intermittent performance complaints reported.")

    if row.get("TechSupport", "") == "No":
        notes.append("No tech support; support resolution concerns mentioned.")

    if not notes:
        notes.append("No major issues reported recently.")

    return " ".join(notes)


def add_support_notes(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Add a support_notes column to a dataframe.

    Parameters
    ----------
    df:
        Input customer dataframe.
    overwrite:
        If True, replace an existing support_notes column.
    """
    df = df.copy()

    if "support_notes" in df.columns and not overwrite:
        return df

    df["support_notes"] = df.apply(lambda row: make_note(row.to_dict()), axis=1)
    return df


def main(input_path: Path, output_path: Path, overwrite: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Place the Telco CSV in data/raw/ or pass --input path/to/file.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = add_support_notes(df, overwrite=overwrite)
    df.to_csv(output_path, index=False)

    print(f"Saved dataset with support notes to: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite support_notes if it already exists.",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.overwrite)