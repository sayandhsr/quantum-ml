
from __future__ import annotations
import time
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics,
    select_f1_threshold,
)
from shared.preprocess import (
    FEATURES,
    RANDOM_STATE,
    ROOT_DIR,
    split_dataset,
    load_raw_dataset,
)


MODEL_DIR = ROOT_DIR / "traditional" / "models"
RESULT_DIR = ROOT_DIR / "traditional" / "results"
N_SPLITS = 5


def build_searches() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Create modest, training-only searches with fold-local preprocessing."""
    def pipeline(classifier: Any, scale: bool = True) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    return {
        "logistic_regression": (
            pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale=False,
            ),
            {
                "classifier__max_depth": [None, 6],
                "classifier__min_samples_leaf": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "extra_trees": (
            pipeline(
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                scale=False,
            ),
            {
                "classifier__max_depth": [None, 6],
                "classifier__min_samples_leaf": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE), scale=False),
            {
                "classifier__n_estimators": [80, 140],
                "classifier__learning_rate": [0.03, 0.08],
                "classifier__max_depth": [1, 2],
            },
        ),
        "gaussian_nb": (
            pipeline(GaussianNB()),
            {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
    }


def _cv_summary(search: GridSearchCV, metric: str) -> tuple[float, float]:
    row = search.best_index_
    results = search.cv_results_
    scores = np.asarray(
        [results[f"split{fold}_test_{metric}"][row] for fold in range(N_SPLITS)],
        dtype=float,
    )
    return float(scores.mean()), float(scores.std())


def train_and_evaluate() -> dict[str, dict[str, Any]]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset()
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "f1": "f1",
        "recall": "recall",
    }
    results: dict[str, dict[str, Any]] = {}
    searches = build_searches()

    print(f"Training on {len(X_train)} rows; held-out test has {len(X_test)} rows.")
    for model_name, (estimator, parameter_grid) in searches.items():
        started = time.perf_counter()
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Select the operating threshold from out-of-fold training predictions only.
        oof_probabilities = cross_val_predict(
            best_model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = select_f1_threshold(y_train, oof_probabilities)
        test_probabilities = best_model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)
        test_metrics = calculate_metrics(y_test, test_predictions, test_probabilities)
        elapsed_seconds = time.perf_counter() - started

        cv_auc_mean, cv_auc_std = _cv_summary(search, "roc_auc")
        cv_f1_mean, cv_f1_std = _cv_summary(search, "f1")
        cv_recall_mean, cv_recall_std = _cv_summary(search, "recall")
        cv_accuracy_mean, cv_accuracy_std = _cv_summary(search, "accuracy")
        cv_pr_auc_mean, cv_pr_auc_std = _cv_summary(search, "average_precision")

        model_results: dict[str, Any] = {
            **test_metrics,
            "threshold": threshold,
            "cv_roc_auc_mean": cv_auc_mean,
            "cv_roc_auc_std": cv_auc_std,
            "cv_pr_auc_mean": cv_pr_auc_mean,
            "cv_pr_auc_std": cv_pr_auc_std,
            "cv_accuracy_mean": cv_accuracy_mean,
            "cv_accuracy_std": cv_accuracy_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "cv_recall_mean": cv_recall_mean,
            "cv_recall_std": cv_recall_std,
            "best_parameters": search.best_params_,
            "training_seconds": elapsed_seconds,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "features": FEATURES,
        }
        results[model_name] = model_results

        joblib.dump(
            {
                "model": best_model,
                "threshold": threshold,
                "features": FEATURES,
                "model_name": model_name,
            },
            MODEL_DIR / f"{model_name}.joblib",
        )
        save_metrics(model_results, RESULT_DIR / f"{model_name}_metrics.json")
        plot_confusion_matrix(
            y_test,
            test_predictions,
            f"{model_name.replace('_', ' ').title()} — held-out test",
            RESULT_DIR / f"{model_name}_confusion.png",
        )
        plot_roc_curve(
            y_test,
            test_probabilities,
            f"{model_name.replace('_', ' ').title()} — ROC",
            RESULT_DIR / f"{model_name}_roc.png",
        )
        plot_precision_recall_curve(
            y_test,
            test_probabilities,
            f"{model_name.replace('_', ' ').title()} — precision-recall",
            RESULT_DIR / f"{model_name}_precision_recall.png",
        )
        print(
            f"{model_name}: test ROC-AUC={test_metrics['roc_auc']:.3f}, "
            f"PR-AUC={test_metrics['pr_auc']:.3f}, recall={test_metrics['recall']:.3f}, "
            f"F1={test_metrics['f1_score']:.3f}, threshold={threshold:.3f}"
        )

    comparison = pd.DataFrame(results).T
    comparison.to_csv(RESULT_DIR / "model_comparison.csv", index_label="model")
    save_metrics(results, RESULT_DIR / "all_metrics.json")
    comparison[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    ].plot(kind="bar", figsize=(12, 6))
    plt.title("Classical models — held-out test metrics")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_comparison.png", dpi=150)
    plt.close()
    return results


def main() -> None:
    train_and_evaluate()
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Evaluation results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
