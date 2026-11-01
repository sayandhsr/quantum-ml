"""Quick environment and dataset smoke check."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "scipy",
    "qiskit",
    "qiskit_aer",
    "streamlit",
    "plotly",
    "joblib",
]


def main() -> int:
    print("=" * 52)
    print("PROJECT ENVIRONMENT CHECK")
    print("=" * 52)
    print(f"Python: {sys.version.split()[0]}")

    failed: list[str] = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "available")
            print(f"[OK] {package}: {version}")
        except (ImportError, OSError) as error:
            print(f"[MISSING] {package}: {error}")
            failed.append(package)

    dataset = ROOT / "data" / "pima_diabetes.csv"
    if dataset.is_file():
        print(f"[OK] Dataset: {dataset}")
    else:
        print(f"[MISSING] Dataset: {dataset}")
        failed.append("dataset")

    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        result = AerSimulator().run(circuit, shots=100, seed_simulator=42).result()
        if not result.get_counts():
            raise RuntimeError("Simulator returned no measurement counts.")
        print(f"[OK] Qiskit Aer simulator: {result.get_counts()}")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"[FAILED] Qiskit Aer simulator: {error}")
        failed.append("qiskit_aer simulation")

    print("=" * 52)
    if failed:
        print("Checks failed:", ", ".join(failed))
        return 1
    print("All environment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
