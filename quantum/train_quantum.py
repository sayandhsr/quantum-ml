"""Train a small local variational quantum classifier and controlled baseline."""

from __future__ import annotations

import pickle
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantum.model import build_circuit, quantum_probability
from shared.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from shared.preprocess import (
    FEATURES,
    RANDOM_STATE,
    ROOT_DIR,
    load_raw_dataset,
    split_dataset,
)


MODEL_DIR = ROOT_DIR / "quantum" / "models"
RESULT_DIR = ROOT_DIR / "quantum" / "results"
N_QUBITS = 4
MAX_TRAINING_SAMPLES = 80
MAX_ITERATIONS = 50
FEATURE_REPS = 1
ANSATZ_REPS = 1


def _choose_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> list[str]:
    scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    chosen = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [feature_names[index] for index in chosen[:N_QUBITS]]


def _fit_controlled_classical(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "gaussian_nb": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", GaussianNB()),
                ]
            ),
            {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
    }
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy"}
    best: GridSearchCV | None = None
    best_name = ""
    for name, (pipeline, parameters) in candidates.items():
        search = GridSearchCV(
            pipeline,
            parameters,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        if best is None or search.best_score_ > best.best_score_:
            best, best_name = search, name

    if best is None:
        raise RuntimeError("No controlled classical baseline was fitted.")
    probabilities = best.best_estimator_.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, predictions, probabilities)
    metrics.update(
        {
            "model": best_name,
            "threshold": 0.5,
            "cv_roc_auc_mean": float(best.best_score_),
            "cv_roc_auc_std": float(best.cv_results_["std_test_roc_auc"][best.best_index_]),
            "best_parameters": best.best_params_,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
        }
    )
    return metrics


def train_quantum() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_raw_dataset()
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    subset_size = min(MAX_TRAINING_SAMPLES, len(X_train))
    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=subset_size,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    subset_imputed = pd.DataFrame(
        imputer.fit_transform(X_subset),
        columns=FEATURES,
        index=X_subset.index,
    )
    X_subset_scaled = scaler.fit_transform(subset_imputed)
    test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=FEATURES,
        index=X_test.index,
    )
    X_test_scaled = scaler.transform(test_imputed)
    y_subset_array = y_subset.to_numpy(dtype=int)
    selected_features = _choose_features(
        X_subset_scaled,
        y_subset_array,
        FEATURES,
    )
    selected_indices = [FEATURES.index(feature) for feature in selected_features]

    X_quantum = X_subset_scaled[:, selected_indices]
    X_test_quantum = X_test_scaled[:, selected_indices]
    X_quantum = np.clip(X_quantum, -2.0, 2.0) * (np.pi / 2.0)
    X_test_quantum = np.clip(X_test_quantum, -2.0, 2.0) * (np.pi / 2.0)
    y_test_array = y_test.to_numpy(dtype=int)

    circuit, feature_parameters, ansatz_parameters = build_circuit(
        N_QUBITS,
        FEATURE_REPS,
        ANSATZ_REPS,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    initial_parameters = rng.uniform(-0.2, 0.2, size=len(ansatz_parameters))
    loss_history: list[float] = []
    objective_evaluations = 0
    optimization_started = time.perf_counter()

    def objective(weights: np.ndarray) -> float:
        nonlocal objective_evaluations
        probabilities = np.asarray(
            [
                quantum_probability(
                    circuit,
                    feature_parameters,
                    ansatz_parameters,
                    row,
                    weights,
                )
                for row in X_quantum
            ],
            dtype=float,
        )
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_subset_array * np.log(probabilities)
            + (1 - y_subset_array) * np.log(1 - probabilities)
        )
        objective_evaluations += 1
        loss_history.append(float(loss))
        return float(loss)

    optimization = minimize(
        objective,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5, "tol": 1e-4},
    )
    optimization_seconds = time.perf_counter() - optimization_started
    best_parameters = np.asarray(optimization.x, dtype=float)

    test_probabilities = np.asarray(
        [
            quantum_probability(
                circuit,
                feature_parameters,
                ansatz_parameters,
                row,
                best_parameters,
            )
            for row in X_test_quantum
        ],
        dtype=float,
    )
    threshold = 0.5
    test_predictions = (test_probabilities >= threshold).astype(int)
    metrics = calculate_metrics(y_test_array, test_predictions, test_probabilities)
    metrics.update(
        {
            "threshold": threshold,
            "training_rows": int(len(X_subset)),
            "test_rows": int(len(X_test)),
            "selected_features": selected_features,
            "n_qubits": N_QUBITS,
            "feature_reps": FEATURE_REPS,
            "ansatz_reps": ANSATZ_REPS,
            "trainable_parameters": len(ansatz_parameters),
            "optimizer": "COBYLA",
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "objective_evaluations": int(objective_evaluations),
            "final_training_loss": float(optimization.fun),
            "optimization_seconds": float(optimization_seconds),
            "simulation": "local exact statevector",
        }
    )
    model_data = {
        "parameters": best_parameters,
        "features": FEATURES,
        "selected_features": selected_features,
        "n_qubits": N_QUBITS,
        "feature_reps": FEATURE_REPS,
        "ansatz_reps": ANSATZ_REPS,
        "imputer": imputer,
        "scaler": scaler,
        "threshold": threshold,
        "optimizer": "COBYLA",
        "training_rows": int(len(X_subset)),
        "training_sample_indices": [int(index) for index in X_subset.index],
        "metrics": metrics,
    }
    with (MODEL_DIR / "quantum_model.pkl").open("wb") as file:
        pickle.dump(model_data, file)
    save_metrics(metrics, RESULT_DIR / "quantum_metrics.json")

    plot_confusion_matrix(
        y_test_array,
        test_predictions,
        "Quantum model — held-out test",
        RESULT_DIR / "quantum_confusion.png",
    )
    plot_roc_curve(
        y_test_array,
        test_probabilities,
        "Quantum model — ROC",
        RESULT_DIR / "quantum_roc.png",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(loss_history)
    axis.set(
        title="Quantum optimization loss",
        xlabel="Objective evaluation",
        ylabel="Training binary cross-entropy",
    )
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "quantum_loss.png", dpi=150)
    plt.close(figure)
    circuit.draw(output="mpl", fold=-1)
    plt.savefig(RESULT_DIR / "quantum_circuit.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    controlled = _fit_controlled_classical(
        X_subset.loc[:, selected_features],
        y_subset,
        X_test.loc[:, selected_features],
        y_test,
    )
    controlled.update(
        {
            "selected_features": selected_features,
            "comparison_design": (
                "Same stratified training subset, held-out test split, and selected "
                "four input features as the quantum classifier."
            ),
        }
    )
    save_metrics(controlled, RESULT_DIR / "controlled_classical_metrics.json")
    print(
        f"Quantum test ROC-AUC={metrics['roc_auc']:.3f}, "
        f"PR-AUC={metrics['pr_auc']:.3f}, recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1_score']:.3f}"
    )
    print(
        f"COBYLA status={optimization.status}: {optimization.message}; "
        f"evaluations={objective_evaluations}; runtime={optimization_seconds:.1f}s"
    )
    return {"quantum": metrics, "controlled_classical": controlled}


def main() -> None:
    train_quantum()
    print(f"Quantum model saved in: {MODEL_DIR}")
    print(f"Quantum evaluation results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
