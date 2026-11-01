"""High-performance Qiskit Data Re-Uploading Quantum Neural Network (DR-QNN)."""

from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_drqnn_circuit(n_qubits: int = 4, n_layers: int = 2) -> tuple[QuantumCircuit, list[Any], ParameterVector]:
    qc = QuantumCircuit(n_qubits, name="DR_QNN")
    x_layers = [ParameterVector(f"x_l{l}", n_qubits) for l in range(n_layers)]
    n_weights = n_layers * n_qubits * 2
    theta_params = ParameterVector("theta", n_weights)
    t_idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x_layers[l][q], q)
        for q in range(n_qubits):
            qc.rz(theta_params[t_idx], q)
            t_idx += 1
            qc.ry(theta_params[t_idx], q)
            t_idx += 1
        for q in range(n_qubits):
            qc.cz(q, (q + 1) % n_qubits)
    flat_inputs = [p for vec in x_layers for p in vec]
    return qc, flat_inputs, theta_params


class FastVectorizedQiskitEngine:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    @staticmethod
    def _ry(angles: np.ndarray) -> np.ndarray:
        half = angles / 2.0
        cos = np.cos(half)
        sin = np.sin(half)
        if np.ndim(angles) == 0:
            return np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)
        B = len(angles)
        U = np.zeros((B, 2, 2), dtype=np.complex128)
        U[:, 0, 0] = cos
        U[:, 0, 1] = -sin
        U[:, 1, 0] = sin
        U[:, 1, 1] = cos
        return U

    @staticmethod
    def _rz(angles: np.ndarray) -> np.ndarray:
        half = angles / 2.0
        if np.ndim(angles) == 0:
            return np.array([[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128)
        B = len(angles)
        U = np.zeros((B, 2, 2), dtype=np.complex128)
        U[:, 0, 0] = np.exp(-1j * half)
        U[:, 1, 1] = np.exp(1j * half)
        return U

    def _apply_gate(self, state: np.ndarray, U: np.ndarray, qubit: int) -> np.ndarray:
        s_swapped = np.swapaxes(state, 1, qubit + 1)
        if U.ndim == 2:
            res = np.einsum("ij,bj...->bi...", U, s_swapped)
        else:
            res = np.einsum("bij,bj...->bi...", U, s_swapped)
        return np.swapaxes(res, 1, qubit + 1)

    def _apply_cz(self, state: np.ndarray, c: int, t: int) -> np.ndarray:
        s = state.copy()
        sl = [slice(None)] * (self.n_qubits + 1)
        sl[c + 1] = 1
        sl[t + 1] = 1
        s[tuple(sl)] *= -1.0
        return s

    def forward(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        B = len(X)
        state = np.zeros((B,) + (2,) * self.n_qubits, dtype=np.complex128)
        state[:, 0, 0] = 1.0
        w_grid = weights.reshape(self.n_layers, self.n_qubits, 2)
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                state = self._apply_gate(state, self._ry(X[:, q]), q)
            for q in range(self.n_qubits):
                state = self._apply_gate(state, self._rz(w_grid[l, q, 0]), q)
                state = self._apply_gate(state, self._ry(w_grid[l, q, 1]), q)
            for q in range(self.n_qubits):
                state = self._apply_cz(state, q, (q + 1) % self.n_qubits)
        probs = np.abs(state) ** 2
        exp_z = []
        for q in range(2):
            axes = tuple(i for i in range(1, self.n_qubits + 1) if i != q + 1)
            p_q = np.sum(probs, axis=axes)
            exp_z.append(p_q[:, 0] - p_q[:, 1])
        return np.stack(exp_z, axis=1)

    def backward_parameter_shift(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        B = len(X)
        n_weights = len(weights)
        grad = np.zeros((B, 2, n_weights))
        shift = np.pi / 2.0
        for p in range(n_weights):
            w_plus = weights.copy()
            w_plus[p] += shift
            w_minus = weights.copy()
            w_minus[p] -= shift
            z_plus = self.forward(X, w_plus)
            z_minus = self.forward(X, w_minus)
            grad[:, :, p] = 0.5 * (z_plus - z_minus)
        return grad


class QiskitDataReuploadingClassifier:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, learning_rate: float = 0.08, epochs: int = 30, batch_size: int = 25, random_state: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.engine = FastVectorizedQiskitEngine(n_qubits, n_layers)
        self.circuit, self.input_params, self.weight_params = build_drqnn_circuit(n_qubits, n_layers)
        rng = np.random.RandomState(random_state)
        self.num_weights = len(self.weight_params)
        self.weights = rng.uniform(-np.pi, np.pi, self.num_weights)
        self.gamma = np.array([2.5, 2.5])
        self.beta = np.zeros(2)
        self.m_w, self.v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.m_g, self.v_g = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.m_b, self.v_b = np.zeros_like(self.beta), np.zeros_like(self.beta)
        self.step_count = 0

    def get_circuit_ascii(self) -> str:
        try:
            return str(self.circuit.draw(output="text"))
        except Exception:
            return f"Qiskit Circuit: {self.n_qubits} qubits, {self.n_layers} re-uploading layers, {self.num_weights} weights."

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.engine.forward(X, self.weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        z = self.forward(X_arr)
        logits = z * self.gamma + self.beta
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        N = len(X)
        indices = np.random.permutation(N)
        n_batches = int(np.ceil(N / self.batch_size))
        total_loss = 0.0
        grad_norms = []
        for b in range(n_batches):
            idx = indices[b * self.batch_size : min((b + 1) * self.batch_size, N)]
            X_b, y_b = X[idx], y[idx]
            B = len(X_b)
            y_oh = np.zeros((B, 2))
            y_oh[np.arange(B), y_b] = 1.0
            z = self.engine.forward(X_b, self.weights)
            logits = z * self.gamma + self.beta
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
            total_loss += loss
            dlogits = (probs - y_oh) / B
            dz = dlogits * self.gamma
            dgamma = np.sum(dlogits * z, axis=0)
            dbeta = np.sum(dlogits, axis=0)
            weight_grad = self.engine.backward_parameter_shift(X_b, self.weights)
            dweights = np.sum(dz[:, :, None] * weight_grad, axis=(0, 1))
            gnorm = float(np.linalg.norm(dweights))
            grad_norms.append(gnorm)
            self.step_count += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for param, grad, m, v in [
                (self.weights, dweights, self.m_w, self.v_w),
                (self.gamma, dgamma, self.m_g, self.v_g),
                (self.beta, dbeta, self.m_b, self.v_b),
            ]:
                np.clip(grad, -5.0, 5.0, out=grad)
                m[:] = b1 * m + (1.0 - b1) * grad
                v[:] = b2 * v + (1.0 - b2) * (grad ** 2)
                m_hat = m / (1.0 - b1 ** self.step_count)
                v_hat = v / (1.0 - b2 ** self.step_count)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        avg_loss = total_loss / n_batches
        train_acc = float(np.mean(self.predict(X) == y))
        mean_grad = float(np.mean(grad_norms))
        return avg_loss, train_acc, mean_grad

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, verbose: bool = False) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"loss": [], "train_acc": [], "test_acc": [], "grad_norm": [], "time": []}
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            t_ep = time.time()
            loss, train_acc, gnorm = self.train_epoch(X_train, y_train)
            dur = time.time() - t_ep
            test_acc = float(np.mean(self.predict(X_test) == y_test)) if X_test is not None and y_test is not None else 0.0
            history["loss"].append(loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["grad_norm"].append(gnorm)
            history["time"].append(dur)
        return history
