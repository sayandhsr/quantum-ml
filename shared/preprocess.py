
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================
# PROJECT PATHS
# =====================================================

ROOT_DIR = Path(
    r"C:\Users\Dell\OneDrive\Documents\tinkerspace"
)

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
    "Age"
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# =====================================================
# LOAD DATASET
# =====================================================

def load_raw():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = FEATURES + [TARGET]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[required].copy()

    return df


# =====================================================
# CLEAN INVALID ZERO VALUES
# =====================================================

def clean_zeros(df):

    df = df.copy()

    for column in ZERO_AS_MISSING:
        df[column] = df[column].replace(0, np.nan)

    return df


# =====================================================
# SPLIT AND SCALE
# =====================================================

def split_and_scale(df):

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    missing_counts = X_train.isna().sum()

    # Calculate medians using training data only.
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Fit scaler only on training data.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
        "medians": medians,
        "missing_counts": missing_counts,
        "feature_names": FEATURES
    }


# =====================================================
# SAVE PROCESSED DATA
# =====================================================

def save_processed(data):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df = data["X_train"].copy()
    train_df[TARGET] = data["y_train"].values

    test_df = data["X_test"].copy()
    test_df[TARGET] = data["y_test"].values

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    with open(
        PROCESSED_DIR / "splits.pkl",
        "wb"
    ) as file:

        pickle.dump(
            {
                "scaler": data["scaler"],
                "medians": data["medians"],
                "features": data["feature_names"]
            },
            file
        )

    print("\nProcessed dataset saved successfully.")

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Testing samples:",
        len(test_df)
    )


# =====================================================
# LOAD PROCESSED DATA
# =====================================================

def load_processed():

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    metadata_path = PROCESSED_DIR / "splits.pkl"

    if not all([
        train_path.exists(),
        test_path.exists(),
        metadata_path.exists()
    ]):

        raise FileNotFoundError(
            "Processed files not found. Run preprocessing first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\nLoading Pima Diabetes Dataset...")

    df = load_raw()

    print("\nOriginal dataset shape:", df.shape)

    df = clean_zeros(df)

    print("\nMissing values after zero replacement:")

    print(df[FEATURES].isna().sum())

    data = split_and_scale(df)

    save_processed(data)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
