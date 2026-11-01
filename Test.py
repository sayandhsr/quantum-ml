
from pathlib import Path
import importlib
import sys

ROOT = Path(
    r"C:\Users\Dell\OneDrive\Documents\tinkerspace"
)

sys.path.insert(0, str(ROOT))

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
    "joblib"
]


def main():

    print("=" * 50)
    print("PROJECT ENVIRONMENT CHECK")
    print("=" * 50)

    failed = []

    for package in PACKAGES:

        try:

            importlib.import_module(package)

            print(f"[OK] {package}")

        except ImportError:

            print(f"[MISSING] {package}")

            failed.append(package)

    dataset = ROOT / "data" / "pima_diabetes.csv"

    print("\nDataset:")

    if dataset.exists():

        print("[OK]", dataset)

    else:

        print("[MISSING]", dataset)

    print("\nQiskit Simulator:")

    try:

        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(2)

        circuit.h(0)
        circuit.cx(0, 1)

        circuit.measure_all()

        simulator = AerSimulator()

        result = simulator.run(
            circuit,
            shots=100
        ).result()

        print("[OK] AerSimulator")

        print(
            result.get_counts()
        )

    except Exception as error:

        print("[FAILED]", error)

    print("\n" + "=" * 50)

    if failed:

        print("Missing packages:", failed)

    else:

        print("Environment checks completed.")

    print("=" * 50)


if __name__ == "__main__":
    main()
