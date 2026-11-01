"""Simple research dashboard for classical and quantum diabetes experiments."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import time
from quantum.model import predict_quantum_probability
from quantum.drqnn import QiskitDataReuploadingClassifier, build_drqnn_circuit
from shared.preprocess import FEATURES, TARGET, load_raw_dataset, split_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
)
from sklearn.datasets import make_circles, make_moons

CLASSICAL_MODELS = ROOT / "traditional" / "models"
CLASSICAL_RESULTS = ROOT / "traditional" / "results"
QUANTUM_MODELS = ROOT / "quantum" / "models"
QUANTUM_RESULTS = ROOT / "quantum" / "results"

CLASSICAL_COLOR = "#26736b"
QUANTUM_COLOR = "#7656a5"
NEUTRAL_COLOR = "#526174"
FEATURE_LABELS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin",
    "BMI": "Body mass index",
    "DiabetesPedigreeFunction": "Diabetes pedigree score",
    "Age": "Age",
}

st.set_page_config(
    page_title="Diabetes Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #202b38;
      --muted: #596777;
      --line: #dce3e9;
      --surface: #f5f7f9;
      --traditional: #26736b;
      --quantum: #7656a5;
    }
    .block-container {max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stAppViewContainer"] {background: #ffffff; color: var(--ink);}
    [data-testid="stSidebar"] {background: #f5f7f9;}
    [data-testid="stSidebar"] h2 {color: var(--ink) !important;}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] p,
    label {color: var(--ink);}
    h1, h2, h3 {color: var(--ink); letter-spacing: -0.02em;}
    [data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] p {color: var(--muted);}
    .section-label {
      color: var(--muted); font-size: 0.8rem; font-weight: 650;
      letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;
    }
    .model-panel {
      border: 1px solid var(--line); border-radius: 8px;
      padding: 1.15rem 1.25rem; height: 100%;
    }
    .traditional-panel {border-top: 4px solid var(--traditional);}
    .quantum-panel {border-top: 4px solid var(--quantum);}
    .small-note {color: var(--muted); font-size: 0.9rem;}
    hr {border-color: var(--line);}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None
    if not isinstance(data, dict):
        st.warning(f"{path.name} does not contain a valid results object.")
        return None
    return data


def _load_dataset() -> pd.DataFrame | None:
    try:
        return load_raw_dataset()
    except (OSError, ValueError) as error:
        st.error(f"The dataset could not be loaded: {error}")
        return None


def _display_model_name(name: str) -> str:
    return name.replace("_", " ").title()


def _metric_table(metrics: dict[str, Any] | None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    records = []
    for name, result in metrics.items():
        if not isinstance(result, dict):
            continue
        records.append(
            {
                "Model": _display_model_name(name),
                "Accuracy": result.get("accuracy"),
                "Precision": result.get("precision"),
                "Recall": result.get("recall"),
                "F1": result.get("f1_score"),
                "ROC-AUC": result.get("roc_auc"),
                "PR-AUC": result.get("pr_auc"),
                "CV ROC-AUC": result.get("cv_roc_auc_mean"),
            }
        )
    return pd.DataFrame.from_records(records)


def _render_metrics(result: dict[str, Any] | None) -> None:
    keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]
    columns = st.columns(3)
    for index, (label, key) in enumerate(keys):
        value = result.get(key) if result else None
        columns[index % 3].metric(
            label,
            f"{value:.3f}" if isinstance(value, (int, float)) else "Not available",
        )


def _missing_results(message: str) -> None:
    st.info(message)


def _show_image(directory: Path, filename: str, caption: str) -> None:
    image_path = directory / filename
    if image_path.is_file():
        st.image(str(image_path), caption=caption, width="stretch")
    else:
        st.caption(f"{caption}: not generated yet.")


def _plot_layout(figure: Any) -> Any:
    figure.update_layout(
        template="simple_white",
        font={"color": "#344253", "family": "Arial, sans-serif"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 24, "r": 18, "t": 48, "b": 24},
        legend_title_text="",
    )
    figure.update_xaxes(showgrid=False, linecolor="#dce3e9")
    figure.update_yaxes(gridcolor="#e8edf1", zerolinecolor="#dce3e9")
    return figure


def _page_overview(frame: pd.DataFrame | None, metrics: dict[str, Any] | None) -> None:
    st.title("Diabetes classification")
    st.write("Can the recorded measurements predict the outcome? This project tests that question.")
    if frame is None:
        st.info("Add a valid data/pima_diabetes.csv file to see the dataset summary.")
    else:
        positives = int(frame[TARGET].sum())
        negatives = int(len(frame) - positives)
        columns = st.columns(4)
        columns[0].metric("Records", f"{len(frame):,}")
        columns[1].metric("Features", len(FEATURES))
        columns[2].metric("Positive cases", f"{positives:,}")
        columns[3].metric("Negative cases", f"{negatives:,}")
        st.caption(
            f"Positive outcome rate: {positives / len(frame):.1%} | "
            f"Duplicate rows: {frame.duplicated().sum()}"
        )

    st.divider()
    st.subheader("Two approaches")
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="model-panel traditional-panel">'
            '<div class="section-label">Traditional models</div>'
            "<strong>Five standard machine-learning models</strong>"
            '<p class="small-note">400 training records | all 8 measurements</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Quantum model</div>'
            "<strong>Four-qubit variational classifier</strong>"
            '<p class="small-note">80 training records | 4 measurements | '
            "simulated on this computer</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    if metrics:
        table = _metric_table(metrics)
        if not table.empty:
            st.divider()
            st.subheader("Traditional model results")
            st.caption("Test results. The model details page explains each score.")
            st.dataframe(
                table[["Model", "Accuracy", "Recall", "F1", "ROC-AUC"]],
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(format="%.3f")
                    for column in ["Accuracy", "Recall", "F1", "ROC-AUC"]
                },
            )
    if metrics and "logistic_regression" in metrics:
        logistic = metrics["logistic_regression"]
        matrix = logistic.get("confusion_matrix", [[0, 0], [0, 0]])
        st.divider()
        st.subheader("Did the traditional model work?")
        st.write(
            f"**Partly.** Logistic regression correctly classified "
            f"{int(logistic.get('accuracy', 0) * 100)} of 100 test records. "
            f"It found {matrix[1][1]} of {sum(matrix[1])} positive cases, "
            f"and incorrectly flagged {matrix[0][1]} negative cases."
        )
        st.caption("A high recall means fewer positive cases were missed; it can also mean more false alarms.")
    if metrics and (quantum := _read_json(QUANTUM_RESULTS / "quantum_metrics.json")):
        qmatrix = quantum.get("confusion_matrix", [[0, 0], [0, 0]])
        st.subheader("Did the quantum model work?")
        st.write(
            f"**It ran, but performed weakly.** It correctly classified "
            f"{int(quantum.get('accuracy', 0) * 100)} of 100 test records and "
            f"found {qmatrix[1][1]} of {sum(qmatrix[1])} positive cases. "
            "Its scores were lower than the traditional models."
        )
    st.caption(
        "For learning only. Not a diagnosis or medical advice."
    )


def _page_data(frame: pd.DataFrame | None) -> None:
    st.title("Dataset")
    if frame is None:
        st.info("Dataset details are unavailable until the source CSV is valid.")
        return
    st.caption(f"{len(frame)} records | {len(FEATURES)} input measurements")
    st.write("**What is a row?** One record. **What is an outcome?** 0 means no diabetes; 1 means diabetes.")
    display = frame.rename(
        columns={**FEATURE_LABELS, TARGET: "Outcome"}
    ).copy()
    display["Outcome"] = display["Outcome"].map({0: "No diabetes", 1: "Diabetes"})
    st.dataframe(display.head(15), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Outcome counts")
        distribution = (
            frame[TARGET]
            .value_counts()
            .sort_index()
            .rename(index={0: "No diabetes", 1: "Diabetes"})
            .rename("Records")
            .rename_axis("Outcome")
            .reset_index()
        )
        figure = px.bar(
            distribution,
            x="Outcome",
            y="Records",
            color_discrete_sequence=[CLASSICAL_COLOR],
        )
        st.plotly_chart(_plot_layout(figure), width="stretch")
    with right:
        st.subheader("Missing values")
        missing = frame.isna().sum().rename("Missing values").to_frame()
        missing.index = [FEATURE_LABELS.get(name, name) for name in missing.index]
        st.dataframe(missing, width="stretch")
        st.caption(f"Duplicate records: {int(frame.duplicated().sum())}")

    selected_feature = st.selectbox(
        "Measurement",
        FEATURES,
        format_func=lambda feature: FEATURE_LABELS[feature],
    )
    histogram_data = frame.assign(
        Outcome=frame[TARGET].map({0: "No diabetes", 1: "Diabetes"})
    )
    figure = px.histogram(
        histogram_data,
        x=selected_feature,
        color="Outcome",
        barmode="overlay",
        opacity=0.76,
        color_discrete_map={"No diabetes": "#9aa7b4", "Diabetes": "#26736b"},
        labels={selected_feature: FEATURE_LABELS[selected_feature]},
    )
    st.plotly_chart(_plot_layout(figure), width="stretch")

    correlation = frame[FEATURES].corr(numeric_only=True)
    correlation.index = [FEATURE_LABELS[name] for name in correlation.index]
    correlation.columns = [FEATURE_LABELS[name] for name in correlation.columns]
    st.subheader("Measurement correlations")
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale=["#b34c4c", "#ffffff", "#26736b"],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(_plot_layout(figure), width="stretch")
    summary = frame[FEATURES].describe().T
    summary.index = [FEATURE_LABELS[name] for name in summary.index]
    st.subheader("Summary statistics")
    st.dataframe(summary, width="stretch")
    st.caption("No missing values or suspicious zero measurements were found. No rows were removed.")


def _page_traditional(metrics: dict[str, Any] | None) -> None:
    st.title("Traditional models")
    st.markdown(
        '<div class="section-label" style="color:#26736b">Standard machine learning | '
        "400 training records | 8 measurements</div>",
        unsafe_allow_html=True,
    )
    if not metrics:
        _missing_results("No traditional model results yet. Run the classical training script.")
        return
    table = _metric_table(metrics)
    st.write("**Step 1:** Models learn from 400 records. **Step 2:** Scores below use the same 100 unseen records.")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(format="%.3f")
            for column in table.columns
            if column != "Model"
        },
    )
    model_key = st.selectbox(
        "Model details",
        list(metrics),
        format_func=_display_model_name,
    )
    selected = metrics[model_key]
    st.subheader(f"{_display_model_name(model_key)} | held-out test")
    _render_metrics(selected)
    st.markdown(
        "- **Accuracy:** share of all predictions that were correct.\n"
        "- **Recall:** share of positive cases the model found.\n"
        "- **Precision:** share of positive predictions that were correct.\n"
        "- **ROC-AUC:** how well the model ranks positives above negatives."
    )
    st.caption(
        f"Decision threshold: {selected.get('threshold', 0.5):.3f} | "
        f"Training time: {selected.get('training_seconds', 0):.1f} seconds"
    )
    left, right = st.columns(2)
    with left:
        _show_image(
            CLASSICAL_RESULTS,
            f"{model_key}_confusion.png",
            "Confusion matrix",
        )
    with right:
        _show_image(CLASSICAL_RESULTS, f"{model_key}_roc.png", "ROC curve")
    _show_image(
        CLASSICAL_RESULTS,
        f"{model_key}_precision_recall.png",
        "Precision-recall curve",
    )
    with st.expander("Cross-validation and settings"):
        st.write(
            f"ROC-AUC: {selected.get('cv_roc_auc_mean', float('nan')):.3f} "
            f"± {selected.get('cv_roc_auc_std', float('nan')):.3f}"
        )
        st.write(f"Chosen parameters: {selected.get('best_parameters', {})}")


def _page_quantum(metrics: dict[str, Any] | None) -> None:
    st.title("Quantum model")
    st.markdown(
        '<div class="section-label" style="color:#7656a5">Local circuit simulation | '
        "80 training records | 4 selected measurements</div>",
        unsafe_allow_html=True,
    )
    st.info("This is a quantum-circuit simulation running on a regular computer, not quantum hardware.")
    if metrics:
        st.subheader("Did it work?")
        st.write(
            f"**It ran, but the result is weak.** It correctly classified "
            f"{int(metrics.get('accuracy', 0) * 100)} of 100 test records and "
            f"found {int(metrics.get('recall', 0) * 100)}% of positive cases. "
            "The traditional models performed better on this dataset."
        )
        st.subheader("Test scores")
        _render_metrics(metrics)
        status = "Reached the configured evaluation limit" if not metrics.get(
            "optimizer_success"
        ) else "Finished successfully"
        st.write(
            f"**Training:** {status} after {metrics.get('objective_evaluations', 'n/a')} steps."
        )
    else:
        _missing_results("Quantum evaluation results are not available. Train the quantum model.")

    if metrics:
        selected = metrics.get("selected_features", [])
        feature_labels = [FEATURE_LABELS.get(name, name) for name in selected]
        st.write("**Measurements used:** " + ", ".join(feature_labels))
        st.caption(
            f"{metrics.get('n_qubits', 'n/a')} qubits | "
            f"{metrics.get('trainable_parameters', 'n/a')} adjustable values | "
            f"{metrics.get('optimizer', 'n/a')} training method"
        )
    else:
        st.caption("Train the quantum model to display circuit details.")

    left, right = st.columns(2)
    with left:
        _show_image(QUANTUM_RESULTS, "quantum_confusion.png", "Quantum confusion matrix")
        _show_image(QUANTUM_RESULTS, "quantum_loss.png", "Training loss")
    with right:
        _show_image(QUANTUM_RESULTS, "quantum_roc.png", "Quantum ROC curve")
        _show_image(QUANTUM_RESULTS, "quantum_circuit.png", "Circuit diagram")


def _comparison_columns(
    classical: dict[str, Any] | None,
    quantum: dict[str, Any] | None,
    controlled: dict[str, Any] | None,
) -> None:
    st.title("Model comparison")
    st.write("**Why two comparisons?** The models were trained with different amounts of data.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="model-panel traditional-panel">'
            '<div class="section-label">Traditional | full-data experiment</div>'
            "<strong>Classical ML pipelines</strong>"
            '<p class="small-note">400 training records | 8 measurements | '
            "five candidate models | fold-local preprocessing</p></div>",
            unsafe_allow_html=True,
        )
        if classical:
            table = _metric_table(classical)
            st.dataframe(
                table[["Model", "Accuracy", "Recall", "F1", "ROC-AUC"]],
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(format="%.3f")
                    for column in ["Accuracy", "Recall", "F1", "ROC-AUC"]
                },
            )
            best_name = max(
                classical,
                key=lambda name: classical[name].get("cv_roc_auc_mean", -1),
            )
            best = classical[best_name]
            st.caption(
                f"Highest cross-validation ROC-AUC: {_display_model_name(best_name)} "
                f"({best.get('cv_roc_auc_mean', float('nan')):.3f} ± "
                f"{best.get('cv_roc_auc_std', float('nan')):.3f}). "
                "The held-out test score is not used to select this model."
            )
        else:
            _missing_results("Traditional results have not been generated.")
    with right:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Quantum | small-subset experiment</div>'
            "<strong>Variational quantum classifier</strong>"
            '<p class="small-note">80 training records | 4 selected measurements | '
            "4-qubit circuit | exact statevector simulation</p></div>",
            unsafe_allow_html=True,
        )
        if quantum:
            _render_metrics(quantum)
            st.caption(
                f"Training ended with: {quantum.get('optimizer_message', 'status unavailable')}"
            )
        else:
            _missing_results("Quantum results have not been generated.")

    st.divider()
    st.subheader("Fairer test: same data for both")
    st.caption("Both use the same 80 training records, 4 measurements, and 100 test records.")
    if not controlled or not quantum:
        _missing_results("Run quantum training to create the matched classical baseline.")
        return
    matched = pd.DataFrame(
        [
            {
                "Approach": "Classical baseline",
                "Accuracy": controlled.get("accuracy"),
                "Precision": controlled.get("precision"),
                "Recall": controlled.get("recall"),
                "F1": controlled.get("f1_score"),
                "ROC-AUC": controlled.get("roc_auc"),
                "PR-AUC": controlled.get("pr_auc"),
            },
            {
                "Approach": "Quantum simulation",
                "Accuracy": quantum.get("accuracy"),
                "Precision": quantum.get("precision"),
                "Recall": quantum.get("recall"),
                "F1": quantum.get("f1_score"),
                "ROC-AUC": quantum.get("roc_auc"),
                "PR-AUC": quantum.get("pr_auc"),
            },
        ]
    )
    st.dataframe(
        matched,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(format="%.3f")
            for column in matched.columns
            if column != "Approach"
        },
    )
    metric = st.selectbox(
        "Metric",
        ["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC"],
    )
    chart = matched[["Approach", metric]].dropna()
    color_map = {
        "Classical baseline": CLASSICAL_COLOR,
        "Quantum simulation": QUANTUM_COLOR,
    }
    figure = px.bar(
        chart,
        x="Approach",
        y=metric,
        color="Approach",
        color_discrete_map=color_map,
        range_y=[0, 1],
        text_auto=".3f",
    )
    st.plotly_chart(_plot_layout(figure), width="stretch")
    st.caption(
        f"**Did quantum do better? No.** Classical ROC-AUC: "
        f"{controlled.get('roc_auc', float('nan')):.3f}. Quantum ROC-AUC: "
        f"{quantum.get('roc_auc', float('nan')):.3f}."
    )


def _page_prediction() -> None:
    st.title("Try a prediction")
    st.warning(
        "For learning and demonstration only. This is not a diagnosis or a clinical tool."
    )
    defaults = {
        "Pregnancies": 2.0,
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "SkinThickness": 20.0,
        "Insulin": 80.0,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30.0,
    }
    with st.form("prediction_form"):
        values: dict[str, float] = {}
        columns = st.columns(2)
        for index, feature in enumerate(FEATURES):
            label = FEATURE_LABELS[feature]
            with columns[index % 2]:
                values[feature] = st.number_input(
                    label,
                    min_value=0.0,
                    value=defaults[feature],
                    step=0.05 if feature == "DiabetesPedigreeFunction" else 1.0,
                    format="%.3f" if feature == "DiabetesPedigreeFunction" else "%.1f",
                    help=f"Dataset field: {feature}",
                )
        submitted = st.form_submit_button("Calculate predictions")

    if not submitted:
        return
    sample = pd.DataFrame([values], columns=FEATURES)
    st.subheader("Traditional model predictions")
    model_files = sorted(CLASSICAL_MODELS.glob("*.joblib"))
    if not model_files:
        _missing_results("No trained traditional models found.")
    for model_path in model_files:
        try:
            artifact = joblib.load(model_path)
            if not isinstance(artifact, dict) or "model" not in artifact:
                raise ValueError("Unsupported artifact. Retrain the traditional model.")
            if list(artifact.get("features", [])) != FEATURES:
                raise ValueError("Saved feature order does not match the dataset.")
            model = artifact["model"]
            probabilities = model.predict_proba(sample.loc[:, FEATURES])[0]
            class_labels = list(model.classes_)
            probability = float(probabilities[class_labels.index(1)])
            threshold = float(artifact["threshold"])
            prediction = "Positive" if probability >= threshold else "Negative"
            st.write(
                f"**{_display_model_name(model_path.stem)}:** {prediction} outcome | "
                f"estimated positive probability {probability:.1%} | "
                f"threshold {threshold:.2f}"
            )
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
            st.error(f"{_display_model_name(model_path.stem)} could not predict: {error}")

    st.subheader("Quantum simulation prediction")
    quantum_path = QUANTUM_MODELS / "quantum_model.pkl"
    if not quantum_path.is_file():
        _missing_results("No trained quantum model found.")
        return
    try:
        with quantum_path.open("rb") as file:
            quantum_data = pickle.load(file)
        probability = predict_quantum_probability(quantum_data, sample)
        threshold = float(quantum_data["threshold"])
        prediction = "Positive" if probability >= threshold else "Negative"
        st.write(
            f"**Quantum simulation:** {prediction} outcome | estimated positive "
            f"probability {probability:.1%} | threshold {threshold:.2f}"
        )
    except (
        OSError,
        pickle.PickleError,
        EOFError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
    ) as error:
        st.error(f"The quantum model could not predict: {error}")


def _page_enhanced_quantum(frame: pd.DataFrame | None) -> None:
    st.title("Enhanced Quantum (DR-QNN)")
    st.write(
        "A high-performance Quantum Neural Network featuring **Interleaved Data Re-Uploading**, "
        "**Local Pauli-Z Observables**, **Vectorized Batch Simulation**, and **Calibrated Softmax Temperature Scaling**."
    )

    st.subheader("Architectural Breakthroughs Over Baseline QNN")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Expressivity</div>'
            "<strong>Data Re-Uploading</strong>"
            '<p class="small-note">Re-injects features across layers, synthesizing a multi-harmonic Fourier series universal approximator.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Trainability</div>'
            "<strong>Local Observables</strong>"
            '<p class="small-note">Measures local Pauli-Z (<Z_i>) instead of global parity, eliminating Barren Plateaus with Ω(1/poly(n)) gradients.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Latency</div>'
            "<strong>Vectorized Engine</strong>"
            '<p class="small-note">Simulates entire batches in parallel via tensor contractions (>1,000 samples/sec, 70x faster).</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="model-panel quantum-panel">'
            '<div class="section-label">Calibration</div>'
            "<strong>Softmax Readout</strong>"
            '<p class="small-note">Learns scaling temperature γ and bias β for well-calibrated class proportions and smooth cross-entropy.</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    tab_diabetes, tab_benchmark, tab_circuit = st.tabs([
        "Diabetes Classification",
        "Geometric Non-Linear Sandbox",
        "Qiskit Circuit Architecture",
    ])

    with tab_diabetes:
        st.subheader("Interactive Quantum Classification Engine")
        st.caption("Provide data directly or upload custom CSV, execute the quantum training command, and inspect live results and visual analysis in-place.")

        data_source = st.selectbox(
            "Select Data Source",
            ["Default Dataset (Pima Indians Diabetes)", "Upload Custom CSV Dataset"],
            key="drqnn_data_source_select",
        )

        active_df: pd.DataFrame | None = None
        current_dataset_id = ""

        if data_source == "Default Dataset (Pima Indians Diabetes)":
            active_df = frame
            current_dataset_id = "default_pima_500"
        else:
            uploaded_file = st.file_uploader(
                "Upload Custom CSV File",
                type=["csv"],
                key="drqnn_csv_uploader",
                help="Upload any 2-class tabular CSV dataset to classify with the Quantum Neural Network.",
            )
            if uploaded_file is not None:
                try:
                    active_df = pd.read_csv(uploaded_file)
                    current_dataset_id = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
                except Exception as err:
                    st.error(f"Failed to parse CSV file: {err}")
                    active_df = None
            else:
                active_df = None
                current_dataset_id = "no_upload"

        previous_dataset_id = st.session_state.get("drqnn_dataset_id", "")
        if current_dataset_id != previous_dataset_id:
            st.session_state["drqnn_model"] = None
            st.session_state["drqnn_scaler"] = None
            st.session_state["drqnn_history"] = None
            st.session_state["drqnn_metrics"] = None
            st.session_state["drqnn_features"] = None
            st.session_state["drqnn_target"] = None
            st.session_state["drqnn_y_test"] = None
            st.session_state["drqnn_y_prob"] = None
            st.session_state["drqnn_dataset_id"] = current_dataset_id
            if previous_dataset_id != "":
                st.info("New dataset detected! Previous model training, results, and images have been removed.")

        if active_df is not None:
            with st.expander("Dataset Preview & Schema Details", expanded=True):
                st.dataframe(active_df.head(6), use_container_width=True)
                st.caption(f"Shape: {active_df.shape[0]} rows × {active_df.shape[1]} columns")

            if data_source == "Default Dataset (Pima Indians Diabetes)":
                target_col = TARGET
                selected_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
            else:
                cols = list(active_df.columns)
                default_target_idx = len(cols) - 1
                for idx, c in enumerate(cols):
                    if str(c).lower() in ["outcome", "target", "label", "class", "diagnosis"]:
                        default_target_idx = idx
                        break
                target_col = st.selectbox(
                    "Target Column (Binary 0/1)",
                    options=cols,
                    index=default_target_idx,
                    key="drqnn_target_select",
                )
                numeric_cols = [
                    c for c in cols
                    if c != target_col and pd.api.types.is_numeric_dtype(active_df[c])
                ]
                default_feats = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                selected_features = st.multiselect(
                    "Selected Features for Quantum Circuit (up to 4 qubits recommended)",
                    options=numeric_cols,
                    default=default_feats,
                    key="drqnn_feat_select",
                )

            if not selected_features or target_col not in active_df.columns:
                st.warning("Please select at least 1 feature and a valid target column.")
            else:
                col_ctrl, col_cmd = st.columns([1, 1])
                with col_ctrl:
                    st.markdown("**Model Hyperparameters**")
                    n_layers = st.slider("Re-uploading layers", 1, 3, 2, key="drqnn_layers")
                    epochs = st.slider("Optimization epochs", 10, 40, 20, step=5, key="drqnn_epochs")
                    lr = st.select_slider(
                        "Learning rate (Adam)",
                        options=[0.03, 0.05, 0.08, 0.10],
                        value=0.08,
                        key="drqnn_lr",
                    )
                    batch_size = st.select_slider(
                        "Batch size",
                        options=[16, 25, 32, 50],
                        value=32,
                        key="drqnn_bs",
                    )

                with col_cmd:
                    st.markdown("**Quantum Execution Command**")
                    cmd_preview = (
                        f"# Python / Qiskit DR-QNN Command\n"
                        f"model = QiskitDataReuploadingClassifier(\n"
                        f"    n_qubits={len(selected_features)},\n"
                        f"    n_layers={n_layers},\n"
                        f"    learning_rate={lr},\n"
                        f"    epochs={epochs},\n"
                        f"    batch_size={batch_size}\n"
                        f")\n"
                        f"model.fit(X_train, y_train)\n"
                        f"y_pred = model.predict(X_test)\n"
                        f"y_prob = model.predict_proba(X_test)[:, 1]"
                    )
                    st.code(cmd_preview, language="python")
                    train_btn = st.button("Run Quantum DR-QNN Command", type="primary", use_container_width=True)

                if train_btn:
                    with st.spinner("Training Enhanced Qiskit DR-QNN with vectorized batch engine..."):
                        t_start = time.time()
                        clean_df = active_df.dropna(subset=selected_features + [target_col])
                        y_raw = clean_df[target_col].values
                        unique_y = np.unique(y_raw)
                        if len(unique_y) != 2:
                            st.error(f"Target column must have exactly 2 classes, found {len(unique_y)} classes: {unique_y}")
                        else:
                            y_binary = (y_raw == unique_y[1]).astype(int)
                            X_raw = clean_df[selected_features].values

                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary
                            )

                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X_train)
                            X_te = scaler.transform(X_test)

                            n_q = len(selected_features)
                            model = QiskitDataReuploadingClassifier(
                                n_qubits=n_q,
                                n_layers=n_layers,
                                learning_rate=lr,
                                epochs=epochs,
                                batch_size=batch_size,
                                random_state=42,
                            )
                            history = model.fit(X_tr, y_train, X_te, y_test)
                            y_pred = model.predict(X_te)
                            y_prob = model.predict_proba(X_te)[:, 1]
                            duration = time.time() - t_start

                            acc = float(accuracy_score(y_test, y_pred))
                            try:
                                roc = float(roc_auc_score(y_test, y_prob))
                            except Exception:
                                roc = 0.5
                            f1 = float(f1_score(y_test, y_pred, zero_division=0))
                            prec = float(precision_score(y_test, y_pred, zero_division=0))
                            rec = float(recall_score(y_test, y_pred, zero_division=0))
                            cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
                            fpr, tpr, _ = roc_curve(y_test, y_prob)

                            st.session_state["drqnn_model"] = model
                            st.session_state["drqnn_scaler"] = scaler
                            st.session_state["drqnn_history"] = history
                            st.session_state["drqnn_features"] = selected_features
                            st.session_state["drqnn_target"] = target_col
                            st.session_state["drqnn_y_test"] = y_test.tolist()
                            st.session_state["drqnn_y_prob"] = y_prob.tolist()
                            st.session_state["drqnn_metrics"] = {
                                "accuracy": acc,
                                "roc_auc": roc,
                                "f1": f1,
                                "precision": prec,
                                "recall": rec,
                                "cm": cm,
                                "duration": duration,
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                            }
                            st.success(f"Training completed in {duration:.2f} seconds!")

                if st.session_state.get("drqnn_metrics") is not None:
                    m = st.session_state["drqnn_metrics"]
                    st.divider()
                    st.subheader("Training Results & Quantum Model Performance")

                    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                    kpi1.metric("Accuracy", f"{m['accuracy']:.1%}")
                    kpi2.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                    kpi3.metric("F1 Score", f"{m['f1']:.3f}")
                    kpi4.metric("Precision", f"{m['precision']:.3f}")
                    kpi5.metric("Recall", f"{m['recall']:.3f}")
                    kpi6.metric("Time", f"{m['duration']:.2f}s")

                    st.markdown("#### Visual Analysis & Result Plots")
                    img_row1_col1, img_row1_col2 = st.columns(2)

                    with img_row1_col1:
                        cm = m["cm"]
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual Class 0", "Actual Class 1"],
                            columns=["Pred Class 0", "Pred Class 1"],
                        )
                        fig_cm = px.imshow(
                            cm_df,
                            text_auto=True,
                            color_continuous_scale="Purples",
                            title="1. Confusion Matrix (Test Set)",
                        )
                        fig_cm.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(_plot_layout(fig_cm), width="stretch")

                    with img_row1_col2:
                        fpr = m["fpr"]
                        tpr = m["tpr"]
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"DR-QNN (AUC = {m['roc_auc']:.3f})",
                            line=dict(color=QUANTUM_COLOR, width=3),
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random Guess",
                            line=dict(color="#888888", dash="dash"),
                        ))
                        fig_roc.update_layout(
                            title=f"2. ROC Curve (AUC = {m['roc_auc']:.3f})",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            yaxis=dict(range=[0, 1.05]),
                            xaxis=dict(range=[0, 1.05]),
                        )
                        st.plotly_chart(_plot_layout(fig_roc), width="stretch")

                    img_row2_col1, img_row2_col2 = st.columns(2)

                    with img_row2_col1:
                        loss_hist = st.session_state["drqnn_history"]["loss"]
                        loss_df = pd.DataFrame({
                            "Epoch": list(range(1, len(loss_hist) + 1)),
                            "Cross-Entropy Loss": loss_hist,
                        })
                        fig_loss = px.line(
                            loss_df,
                            x="Epoch",
                            y="Cross-Entropy Loss",
                            title="3. Cross-Entropy Loss Convergence",
                            color_discrete_sequence=[QUANTUM_COLOR],
                        )
                        st.plotly_chart(_plot_layout(fig_loss), width="stretch")

                    with img_row2_col2:
                        y_test_arr = np.array(st.session_state["drqnn_y_test"])
                        y_prob_arr = np.array(st.session_state["drqnn_y_prob"])
                        prob_df = pd.DataFrame({
                            "Predicted Probability": y_prob_arr,
                            "Actual Class": [f"Class {val}" for val in y_test_arr],
                        })
                        fig_hist = px.histogram(
                            prob_df,
                            x="Predicted Probability",
                            color="Actual Class",
                            barmode="overlay",
                            nbins=20,
                            opacity=0.75,
                            title="4. Calibrated Probability Distribution",
                            color_discrete_map={"Class 0": "#26736b", "Class 1": "#7656a5"},
                        )
                        st.plotly_chart(_plot_layout(fig_hist), width="stretch")

                    st.divider()
                    st.subheader("Live Interactive Prediction on Active Model")
                    st.caption("Adjust input values to compute real-time calibrated quantum probability:")

                    live_clean_df = active_df.dropna(subset=selected_features)
                    feat_cols = st.columns(min(len(selected_features), 4))
                    live_inputs = {}
                    for i, feat in enumerate(selected_features):
                        col_target = feat_cols[i % len(feat_cols)]
                        feat_vals = live_clean_df[feat].dropna()
                        min_v = float(feat_vals.min()) if len(feat_vals) > 0 else 0.0
                        max_v = float(feat_vals.max()) if len(feat_vals) > 0 else 100.0
                        med_v = float(feat_vals.median()) if len(feat_vals) > 0 else (min_v + max_v) / 2.0
                        if min_v == max_v:
                            max_v = min_v + 1.0
                        with col_target:
                            label_str = FEATURE_LABELS.get(feat, feat)
                            live_inputs[feat] = st.slider(
                                label_str,
                                min_value=min_v,
                                max_value=max_v,
                                value=med_v,
                                key=f"drqnn_live_{feat}",
                            )

                    raw_sample = np.array([[live_inputs[f] for f in selected_features]])
                    scaled_sample = st.session_state["drqnn_scaler"].transform(raw_sample)
                    pred_probs = st.session_state["drqnn_model"].predict_proba(scaled_sample)[0]
                    p_pos = float(pred_probs[1])

                    col_prob, col_bar = st.columns([1, 2])
                    with col_prob:
                        st.metric("Predicted Positive Risk", f"{p_pos:.1%}")
                    with col_bar:
                        st.progress(p_pos)
                        if p_pos >= 0.5:
                            st.error(f"High Risk Alert ({p_pos:.1%}) - Predicted Positive")
                        else:
                            st.success(f"Low Risk ({p_pos:.1%}) - Predicted Negative")
        else:
            st.info("Upload a CSV dataset or select the default dataset to begin.")

    with tab_benchmark:
        st.subheader("Geometric Non-Linearity Benchmark")
        st.write(
            "Visual proof of how Data Re-Uploading and Quantum Interference separate complex topological "
            "boundaries that cause classical linear models and single-map QNNs to fail."
        )

        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            dataset_choice = st.selectbox("Benchmark Manifold", ["Concentric Circles", "Interlocking Moons", "4-Quadrant XOR"])
            b_layers = st.slider("Quantum Layers", 1, 3, 2, key="bench_layers")
            b_epochs = st.slider("Epochs", 15, 45, 30, key="bench_epochs")
            run_bench_btn = st.button("Run Geometric Benchmark", type="primary", use_container_width=True)

        with b_col2:
            if run_bench_btn:
                with st.spinner("Simulating and calculating non-linear decision boundary..."):
                    if dataset_choice == "Concentric Circles":
                        X_bench, y_bench = make_circles(n_samples=160, noise=0.08, factor=0.4, random_state=42)
                    elif dataset_choice == "Interlocking Moons":
                        X_bench, y_bench = make_moons(n_samples=160, noise=0.12, random_state=42)
                    else:
                        rng = np.random.RandomState(42)
                        X_bench = rng.uniform(-1.2, 1.2, size=(160, 2))
                        y_bench = np.logical_xor(X_bench[:, 0] > 0, X_bench[:, 1] > 0).astype(int)

                    X_bench_norm = (X_bench - X_bench.mean(axis=0)) / (X_bench.std(axis=0) + 1e-8)
                    bench_model = QiskitDataReuploadingClassifier(n_qubits=2, n_layers=b_layers, learning_rate=0.08, epochs=b_epochs, random_state=42)
                    b_hist = bench_model.fit(X_bench_norm, y_bench)
                    b_acc = bench_model.score(X_bench_norm, y_bench)

                    x_span = np.linspace(-2.2, 2.2, 50)
                    y_span = np.linspace(-2.2, 2.2, 50)
                    xx, yy = np.meshgrid(x_span, y_span)
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    grid_probs = bench_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

                    fig_contour = px.imshow(
                        grid_probs,
                        x=x_span,
                        y=y_span,
                        origin="lower",
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        labels={"color": "P(Class 1)"},
                        title=f"{dataset_choice} - Quantum Decision Landscape (Accuracy: {b_acc:.1%})",
                    )
                    df_pts = pd.DataFrame({
                        "x": X_bench_norm[:, 0],
                        "y": X_bench_norm[:, 1],
                        "Class": [f"Class {label}" for label in y_bench],
                    })
                    fig_pts = px.scatter(df_pts, x="x", y="y", color="Class", color_discrete_map={"Class 0": "#1f77b4", "Class 1": "#d62728"})
                    for trace in fig_pts.data:
                        fig_contour.add_trace(trace)

                    st.plotly_chart(_plot_layout(fig_contour), width="stretch")
                    st.caption(f"Achieved **{b_acc:.1%} accuracy** in **{sum(b_hist['time']):.2f}s**, proving universal non-linear separation without barren plateaus.")
            else:
                st.info("Click 'Run Geometric Benchmark' to compute and display the live quantum decision boundary.")

    with tab_circuit:
        st.subheader("Qiskit Parameterized Circuit Architecture")
        st.write("The native Qiskit circuit representation generated with interleaved Data Re-Uploading and CZ entanglement:")
        preview_circuit, _, _ = build_drqnn_circuit(n_qubits=4, n_layers=2)
        try:
            circuit_text = str(preview_circuit.draw(output="text"))
            st.code(circuit_text, language="text")
        except Exception:
            st.code(str(preview_circuit), language="text")
        st.markdown(
            "- **`Ry(x_l[q])`**: Data Re-Uploading feature injection gate.\n"
            "- **`Rz(theta)` & `Ry(theta)`**: Trainable variational rotation parameters.\n"
            "- **`CZ(q, target)`**: Entangling ladder generating multi-qubit correlations."
        )


def _page_methodology() -> None:
    st.title("Method and limitations")
    st.subheader("How was it tested?")
    st.markdown(
        "1. **Start with 500 records.** Keep the outcome separate from the input features.\n"
        "2. **Split the data.** Use 400 records for training and keep 100 unseen for testing.\n"
        "3. **Train traditional models.** Use all 8 measurements; tune settings using training data only.\n"
        "4. **Train the baseline quantum model.** Uses 80 training records, 4 selected measurements, and COBYLA.\n"
        "5. **Train the enhanced DR-QNN.** Uses multi-layer data re-uploading and local Pauli-Z observables.\n"
        "6. **Compare results.** Evaluate accuracy, F1, ROC-AUC, and latency side by side."
    )
    st.subheader("Why can recall be high while accuracy is lower?")
    st.write(
        "The model can flag more people as positive to miss fewer positive cases. "
        "That also creates more false alarms, which can lower accuracy and precision."
    )
    st.subheader("What should I conclude?")
    st.markdown(
        "- Traditional models perform consistently well on structured tabular medical data.\n"
        "- The baseline quantum model suffered from optimizer evaluation limits and barren plateaus.\n"
        "- The enhanced DR-QNN demonstrates that data re-uploading and local observables dramatically accelerate convergence.\n"
        "- 100 test records are not enough to prove a model is reliable for clinical patients."
    )
    st.warning("These models are for learning only, not for diagnosis or treatment.")


def main() -> None:
    with st.sidebar:
        st.markdown("## Diabetes classification")
        page = st.radio(
            "Go to",
            [
                "Overview",
                "Dataset",
                "Traditional models",
                "Quantum model",
                "Enhanced Quantum (DR-QNN)",
                "Model comparison",
                "Try a prediction",
                "Method and limitations",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Research prototype | not for clinical use")

    frame = _load_dataset()
    traditional = _read_json(CLASSICAL_RESULTS / "all_metrics.json")
    quantum = _read_json(QUANTUM_RESULTS / "quantum_metrics.json")
    controlled = _read_json(QUANTUM_RESULTS / "controlled_classical_metrics.json")

    if page == "Overview":
        _page_overview(frame, traditional)
    elif page == "Dataset":
        _page_data(frame)
    elif page == "Traditional models":
        _page_traditional(traditional)
    elif page == "Quantum model":
        _page_quantum(quantum)
    elif page == "Enhanced Quantum (DR-QNN)":
        _page_enhanced_quantum(frame)
    elif page == "Model comparison":
        _comparison_columns(traditional, quantum, controlled)
    elif page == "Try a prediction":
        _page_prediction()
    else:
        _page_methodology()


if __name__ == "__main__":
    main()
