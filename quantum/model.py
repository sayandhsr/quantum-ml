"""Circuit construction and inference helpers shared by quantum training and UI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def build_circuit(
    n_qubits: int = 4,
    feature_reps: int = 1,
    ansatz_reps: int = 1,
) -> tuple[QuantumCircuit, tuple[Any, ...], tuple[Any, ...]]:
    """Build a data-encoding circuit and a trainable hardware-light ansatz.

    RY rotations encode each selected feature. RZZ interactions encode pairwise
    feature products, while each variational layer applies local RY/RZ rotations
    and a nearest-neighbour CZ entangling pattern.
    """
    if n_qubits < 2 or feature_reps < 1 or ansatz_reps < 1:
        raise ValueError("The circuit requires >=2 qubits and positive repetition counts.")

    features = ParameterVector("x", n_qubits)
    weights = ParameterVector("theta", 2 * n_qubits * ansatz_reps)
    circuit = QuantumCircuit(n_qubits)

    for _ in range(feature_reps):
        for qubit in range(n_qubits):
            circuit.ry(features[qubit], qubit)
        for qubit in range(n_qubits - 1):
            circuit.rzz(features[qubit] * features[qubit + 1], qubit, qubit + 1)

    parameter_index = 0
    for _ in range(ansatz_reps):
        for qubit in range(n_qubits):
            circuit.ry(weights[parameter_index], qubit)
            parameter_index += 1
            circuit.rz(weights[parameter_index], qubit)
            parameter_index += 1
        for qubit in range(n_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    return circuit, tuple(features), tuple(weights)


def quantum_probability(
    circuit: QuantumCircuit,
    feature_parameters: Sequence[Any],
    ansatz_parameters: Sequence[Any],
    feature_values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Return the exact simulated probability that qubit zero is measured as 1."""
    if len(feature_parameters) != len(feature_values):
        raise ValueError("Feature parameter and input lengths do not match.")
    if len(ansatz_parameters) != len(weights):
        raise ValueError("Ansatz parameter and weight lengths do not match.")
    if not np.isfinite(np.asarray(feature_values, dtype=float)).all():
        raise ValueError("Quantum input features must be finite.")
    if not np.isfinite(np.asarray(weights, dtype=float)).all():
        raise ValueError("Quantum parameters must be finite.")

    bindings = {
        parameter: float(value)
        for parameter, value in zip(feature_parameters, feature_values)
    }
    bindings.update(
        {
            parameter: float(value)
            for parameter, value in zip(ansatz_parameters, weights)
        }
    )
    state = Statevector.from_instruction(circuit.assign_parameters(bindings))
    probability = float(state.probabilities()[1::2].sum())
    return float(np.clip(probability, 0.0, 1.0))


def _input_frame(
    features: pd.DataFrame | Mapping[str, float] | Sequence[float],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        frame = features.loc[:, list(feature_names)].copy()
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([features]).loc[:, list(feature_names)]
    else:
        values = list(features)
        if len(values) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} input feature values.")
        frame = pd.DataFrame([values], columns=list(feature_names))
    return frame


def predict_quantum_probability(
    model_data: Mapping[str, Any],
    features: pd.DataFrame | Mapping[str, float] | Sequence[float],
) -> float:
    """Apply the saved training transforms and evaluate one quantum prediction."""
    feature_names = list(model_data["features"])
    selected_features = list(model_data["selected_features"])
    input_frame = _input_frame(features, feature_names)
    imputed = pd.DataFrame(
        model_data["imputer"].transform(input_frame),
        columns=feature_names,
        index=input_frame.index,
    )
    scaled = model_data["scaler"].transform(imputed)
    scaled_frame = pd.DataFrame(scaled, columns=feature_names)
    selected_values = scaled_frame.loc[:, selected_features].to_numpy(dtype=float)[0]
    angles = np.clip(selected_values, -2.0, 2.0) * (np.pi / 2.0)

    circuit, feature_parameters, ansatz_parameters = build_circuit(
        n_qubits=int(model_data["n_qubits"]),
        feature_reps=int(model_data["feature_reps"]),
        ansatz_reps=int(model_data["ansatz_reps"]),
    )
    return quantum_probability(
        circuit,
        feature_parameters,
        ansatz_parameters,
        angles,
        model_data["parameters"],
    )
