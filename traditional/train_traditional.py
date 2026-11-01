
import sys
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# =====================================================
# PROJECT ROOT AND IMPORT PATH
# =====================================================

ROOT = Path(
    r"C:\Users\Dell\OneDrive\Documents\tinkerspace"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.preprocess import load_processed
from shared.metrics import (
    calculate_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics
)

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

MODEL_DIR = ROOT / "traditional" / "models"
RESULT_DIR = ROOT / "traditional" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# TRAINING
# =====================================================

def main():

    print("\nLoading shared processed dataset...")

    X_train, X_test, y_train, y_test, metadata = (
        load_processed()
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
    }

    results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # =================================================
    # TRAIN EACH MODEL
    # =================================================

    for name, model in models.items():

        print("\n" + "=" * 50)
        print("Training:", name)
        print("=" * 50)

        # Cross-validation uses training data only.
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        print(
            "5-Fold CV Accuracy:",
            round(cv_scores.mean(), 4)
        )

        # Fit final model on the complete training split.
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate test metrics.
        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        metrics["cv_accuracy_mean"] = float(
            cv_scores.mean()
        )

        metrics["cv_accuracy_std"] = float(
            cv_scores.std()
        )

        results[name] = metrics

        safe_name = name.lower().replace(" ", "_")

        # Save trained model.
        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib"
        )

        # Save metrics.
        save_metrics(
            metrics,
            RESULT_DIR / f"{safe_name}_metrics.json"
        )

        # Save confusion matrix.
        plot_confusion_matrix(
            y_test,
            predictions,
            f"{name} Confusion Matrix",
            RESULT_DIR / f"{safe_name}_confusion.png"
        )

        # Save ROC curve.
        plot_roc_curve(
            y_test,
            probabilities,
            f"{name} ROC Curve",
            RESULT_DIR / f"{safe_name}_roc.png"
        )

        print_metrics(name, metrics)

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    # =================================================
    # SAVE COMPARISON RESULTS
    # =================================================

    comparison = pd.DataFrame(results).T

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv"
    )

    with open(
        RESULT_DIR / "all_metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # =================================================
    # COMPARISON CHART
    # =================================================

    metrics_to_plot = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    comparison[metrics_to_plot].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Traditional ML Model Comparison")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "model_comparison.png",
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 50)
    print("TRADITIONAL ML TRAINING COMPLETED")
    print("=" * 50)

    print("\nModels saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()
