from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from quantum.model import build_circuit, predict_quantum_probability
from shared.metrics import calculate_metrics, select_f1_threshold
from shared.preprocess import (
    FEATURES,
    TARGET,
    dataset_summary,
    load_raw_dataset,
    split_dataset,
)
from traditional.train_traditional import build_searches


class DatasetTests(unittest.TestCase):
    def test_dataset_schema_and_observed_values(self) -> None:
        frame = load_raw_dataset()
        self.assertEqual(frame.columns.tolist(), FEATURES + [TARGET])
        self.assertEqual(set(frame[TARGET].unique()), {0, 1})
        self.assertEqual(frame.isna().sum().sum(), 0)
        summary = dataset_summary(frame)
        self.assertEqual(summary["rows"], 500)
        self.assertEqual(summary["duplicate_rows"], 0)
        self.assertTrue(
            all(value == 0 for value in summary["zero_counts_in_measurement_fields"].values())
        )

    def test_split_is_deterministic_stratified_and_preserves_raw_features(self) -> None:
        frame = load_raw_dataset()
        first = split_dataset(frame)
        second = split_dataset(frame)
        for index, (first_part, second_part) in enumerate(zip(first, second)):
            if index < 2:
                pd.testing.assert_frame_equal(first_part, second_part)
            else:
                pd.testing.assert_series_equal(first_part, second_part)
        X_train, X_test, y_train, y_test = first
        self.assertEqual((len(X_train), len(X_test)), (400, 100))
        self.assertEqual(X_train.columns.tolist(), FEATURES)
        self.assertEqual(y_train.value_counts().to_dict(), {0: 265, 1: 135})
        self.assertEqual(y_test.value_counts().to_dict(), {0: 66, 1: 34})
        self.assertTrue(np.isfinite(X_train.to_numpy()).all())


class ModelTests(unittest.TestCase):
    def test_classical_pipeline_saves_and_reloads_probability_model(self) -> None:
        X_train, _, y_train, _ = split_dataset()
        pipeline, _ = build_searches()["logistic_regression"]
        pipeline.fit(X_train, y_train)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"model": pipeline, "threshold": 0.5, "features": FEATURES}, path)
            loaded = joblib.load(path)
            probabilities = loaded["model"].predict_proba(X_train.iloc[:3])[:, 1]
        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_metrics_and_training_only_threshold(self) -> None:
        truth = np.array([0, 0, 1, 1, 1, 0])
        probabilities = np.array([0.1, 0.3, 0.55, 0.8, 0.7, 0.2])
        threshold = select_f1_threshold(truth, probabilities)
        metrics = calculate_metrics(truth, probabilities >= threshold, probabilities)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)
        self.assertIn("pr_auc", metrics)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (2, 2))

    def test_quantum_parameter_binding_and_saved_preprocessing(self) -> None:
        circuit, feature_parameters, ansatz_parameters = build_circuit()
        self.assertEqual(len(feature_parameters), 4)
        self.assertEqual(len(ansatz_parameters), 8)
        X_train, _, y_train, _ = split_dataset()
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=FEATURES,
            index=X_train.index,
        )
        scaler.fit(imputed)
        model_data = {
            "features": FEATURES,
            "selected_features": FEATURES[:4],
            "imputer": imputer,
            "scaler": scaler,
            "n_qubits": 4,
            "feature_reps": 1,
            "ansatz_reps": 1,
            "parameters": np.zeros(len(ansatz_parameters)),
            "threshold": 0.5,
        }
        del circuit, y_train
        probability = predict_quantum_probability(model_data, X_train.iloc[0])
        self.assertTrue(0 <= probability <= 1)


if __name__ == "__main__":
    unittest.main()
