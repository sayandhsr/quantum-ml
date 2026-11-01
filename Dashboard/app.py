
def predict_quantum(features):

    model_path = QUANTUM_MODELS / "quantum_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Quantum model not found. Train it first."
        )

    with open(model_path, "rb") as file:
        model_data = pickle.load(file)

    parameters = model_data["parameters"]
    n_qubits = model_data["n_qubits"]

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2,
        entanglement="linear"
    )

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,
        entanglement="linear"
    )

    circuit = QuantumCircuit(n_qubits)

    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    feature_parameters = list(feature_map.parameters)
    ansatz_parameters = list(ansatz.parameters)

    parameter_values = {}

    for parameter, value in zip(
        feature_parameters,
        features
    ):
        parameter_values[parameter] = float(value)

    for parameter, value in zip(
        ansatz_parameters,
        parameters
    ):
        parameter_values[parameter] = float(value)

    bound = circuit.assign_parameters(parameter_values)

    simulator = AerSimulator(method="statevector")

    bound.save_statevector()

    compiled = transpile(bound, simulator)

    result = simulator.run(compiled).result()

    statevector = np.asarray(
        result.get_statevector(compiled),
        dtype=complex
    )

    probability = sum(
        abs(amplitude) ** 2
        for index, amplitude in enumerate(statevector)
        if index & 1
    )

    return float(np.clip(probability, 0.0, 1.0))
