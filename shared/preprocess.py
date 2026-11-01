"""Dataset validation and reproducible train/test splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "pima_diabetes.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the source CSV without changing valid measurements."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    frame = frame.loc[:, required].copy()
    for column in required:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column {column!r} must contain numeric values.") from error

    frame = frame.replace([np.inf, -np.inf], np.nan)
    target_values = set(frame[TARGET].dropna().unique())
    if frame[TARGET].isna().any() or not target_values.issubset({0, 1}):
        raise ValueError("Outcome must contain only non-missing binary labels 0 and 1.")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Outcome must contain both classes 0 and 1.")
    if frame[FEATURES].isna().all(axis=None):
        raise ValueError("All feature values are missing.")

    return frame


def split_dataset(
    frame: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one stratified split; learned transforms belong inside model pipelines."""
    dataset = load_raw_dataset() if frame is None else frame.copy()
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = dataset.loc[:, FEATURES]
    y = dataset[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return dataset facts suitable for reports and dashboard summaries."""
    return {
        "rows": int(len(frame)),
        "features": len(FEATURES),
        "feature_names": FEATURES,
        "target": TARGET,
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts().sort_index().items()
        },
        "zero_counts_in_measurement_fields": {
            column: int(frame[column].eq(0).sum())
            for column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        },
    }


def save_split(frame: pd.DataFrame | None = None) -> None:
    """Write raw, unscaled split files for inspection and reproducibility."""
    dataset = load_raw_dataset() if frame is None else frame.copy()
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = X_train.copy()
    train_frame[TARGET] = y_train
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    train_frame.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_frame.to_csv(PROCESSED_DIR / "test.csv", index=False)

    metadata = dataset_summary(dataset)
    metadata.update(
        {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "preprocessing": (
                "Raw feature values are saved. Imputation and scaling are fitted "
                "inside each training fold and final model pipeline."
            ),
        }
    )
    with (PROCESSED_DIR / "split_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    frame = load_raw_dataset()
    save_split(frame)
    summary = dataset_summary(frame)
    print(f"Validated {summary['rows']} rows and {summary['features']} features.")
    print(f"Class counts: {summary['class_counts']}")
    print(f"Split files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
