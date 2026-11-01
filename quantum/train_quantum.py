
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator

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
# CONFIGURATION
# =====================================================

MODEL_DIR = ROOT / "quantum" / "models"
RESULT_DIR = ROOT / "quantum" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4
MAX_TRAINING_SAMPLES = 80
MAX_ITERATIONS = 20

SIMULATOR = AerSimulator(
    method="statevector"
)


# =====================================================
# FEATURE SELECTION
# =====================================================

def select_features(X_train, y_train, X_test):

    scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=42
    )

    selected_indices = np.argsort(scores)[-N_QUBITS:]

    selected_features = [
        X_train.columns[index]
        for index in selected_indices
    ]

    return (
        X_train[selected_features],
        X_test[selected_features],
        selected_features
    )


# =====================================================
# BUILD QUANTUM CIRCUIT
# =====================================================

def build_circuit():

    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=2,
        entanglement="linear"
    )

    ansatz = RealAmplitudes(
        num_qubits=N_QUBITS,
        reps=2,
        entanglement="linear"
    )

    circuit = QuantumCircuit(N_QUBITS)

    circuit.compose(
        feature_map,
        inplace=True
    )

    circuit.compose(
        ansatz,
        inplace=True
    )

    # Keep the parameter groups explicitly.
    feature_parameters = list(feature_map.parameters)
    ansatz_parameters = list(ansatz.parameters)

    return (
        circuit,
        feature_parameters,
        ansatz_parameters
    )


# =====================================================
# QUANTUM PROBABILITY
# =====================================================

def quantum_probability(
    circuit,
    feature_parameters,
    ansatz_parameters,
    feature_values,
    weights
):

    parameter_values = {}

    # Bind input data to feature-map parameters.
    for parameter, value in zip(
        feature_parameters,
        feature_values
    ):

        parameter_values[parameter] = float(value)

    # Bind trainable weights to ansatz parameters.
    for parameter, value in zip(
        ansatz_parameters,
        weights
    ):

        parameter_values[parameter] = float(value)

    bound_circuit = circuit.assign_parameters(
        parameter_values
    )

    bound_circuit.save_statevector()

    compiled = transpile(
        bound_circuit,
        SIMULATOR
    )

    result = SIMULATOR.run(
        compiled
    ).result()

    statevector = np.asarray(
        result.get_statevector(compiled),
        dtype=complex
    )

    # Probability of measuring qubit 0 as |1>.
    probability = sum(
        abs(amplitude) ** 2
        for index, amplitude in enumerate(statevector)
        if index & 1
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================

def main():

    print("\nLoading shared processed dataset...")

    X_train, X_test, y_train, y_test, metadata = (
        load_processed()
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -------------------------------------------------
    # Select four features using training data only.
    # -------------------------------------------------

    X_train_selected, X_test_selected, selected_features = (
        select_features(
            X_train,
            y_train,
            X_test
        )
    )

    print("\nSelected quantum features:")

    for feature in selected_features:
        print("-", feature)

    # -------------------------------------------------
    # Select a reproducible training subset.
    # -------------------------------------------------

    rng = np.random.default_rng(42)

    if len(X_train_selected) > MAX_TRAINING_SAMPLES:

        selected_rows = rng.choice(
            len(X_train_selected),
            size=MAX_TRAINING_SAMPLES,
            replace=False
        )

        X_quantum = X_train_selected.iloc[
            selected_rows
        ].to_numpy()

        y_quantum = y_train.iloc[
            selected_rows
        ].to_numpy()

    else:

        X_quantum = X_train_selected.to_numpy()
        y_quantum = y_train.to_numpy()

    # -------------------------------------------------
    # Convert features to bounded circuit angles.
    # -------------------------------------------------

    X_quantum = np.clip(
        X_quantum,
        -2,
        2
    ) * (np.pi / 2)

    X_test_quantum = np.clip(
        X_test_selected.to_numpy(),
        -2,
        2
    ) * (np.pi / 2)

    # -------------------------------------------------
    # Build circuit.
    # -------------------------------------------------

    (
        circuit,
        feature_parameters,
        ansatz_parameters
    ) = build_circuit()

    print("\nNumber of qubits:", N_QUBITS)
    print("Trainable parameters:", len(ansatz_parameters))

    initial_parameters = rng.uniform(
        -np.pi,
        np.pi,
        len(ansatz_parameters)
    )

    history = []

    # =================================================
    # OBJECTIVE FUNCTION
    # =================================================

    def objective(weights):

        probabilities = []

        for features in X_quantum:

            probability = quantum_probability(
                circuit,
                feature_parameters,
                ansatz_parameters,
                features,
                weights
            )

            probabilities.append(probability)

        probabilities = np.clip(
            probabilities,
            1e-7,
            1 - 1e-7
        )

        # Binary cross-entropy.
        loss = -np.mean(
            y_quantum * np.log(probabilities)
            +
            (1 - y_quantum) * np.log(1 - probabilities)
        )

        history.append(float(loss))

        return float(loss)

    # =================================================
    # OPTIMISATION
    # =================================================

    print("\nStarting COBYLA optimisation...")

    optimisation = minimize(
        objective,
        initial_parameters,
        method="COBYLA",
        options={
            "maxiter": MAX_ITERATIONS,
            "rhobeg": 0.5
        }
    )

    best_parameters = optimisation.x

    print("\nOptimisation completed.")
    print("Final loss:", optimisation.fun)
    print("Optimiser success:", optimisation.success)
    print("Optimiser message:", optimisation.message)

    # =================================================
    # TEST EVALUATION
    # =================================================

    print("\nEvaluating test dataset...")

    y_prob = []

    for features in X_test_quantum:

        probability = quantum_probability(
            circuit,
            feature_parameters,
            ansatz_parameters,
            features,
            best_parameters
        )

        y_prob.append(probability)

    y_prob = np.asarray(y_prob)

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print_metrics(
        "Quantum Machine Learning",
        metrics
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # =================================================
    # SAVE MODEL
    # =================================================

    model_data = {
        "parameters": best_parameters,
        "features": selected_features,
        "n_qubits": N_QUBITS,
        "feature_parameters": feature_parameters,
        "ansatz_parameters": ansatz_parameters
    }

    with open(
        MODEL_DIR / "quantum_model.pkl",
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )

    save_metrics(
        metrics,
        RESULT_DIR / "quantum_metrics.json"
    )

    # =================================================
    # CONFUSION MATRIX
    # =================================================

    plot_confusion_matrix(
        y_test,
        y_pred,
        "Quantum Model Confusion Matrix",
        RESULT_DIR / "quantum_confusion.png"
    )

    # =================================================
    # ROC CURVE
    # =================================================

    plot_roc_curve(
        y_test,
        y_prob,
        "Quantum Model ROC Curve",
        RESULT_DIR / "quantum_roc.png"
    )

    # =================================================
    # LOSS CURVE
    # =================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        history,
        marker="o"
    )

    plt.title("Quantum Optimisation Loss")
    plt.xlabel("Objective Evaluation")
    plt.ylabel("Binary Cross-Entropy Loss")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "quantum_loss.png",
        dpi=150
    )

    plt.close()

    # =================================================
    # QUANTUM CIRCUIT DIAGRAM
    # =================================================

    circuit.draw(
        output="mpl",
        fold=-1
    )

    plt.savefig(
        RESULT_DIR / "quantum_circuit.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 50)
    print("QUANTUM TRAINING COMPLETED")
    print("=" * 50)

    print("\nModel saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()
