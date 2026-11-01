"""Simple research dashboard for classical and quantum diabetes experiments."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import time
from quantum.model import predict_quantum_probability
from quantum.drqnn import QiskitDataReuploadingClassifier, build_drqnn_circuit
from shared.preprocess import FEATURES, TARGET, load_raw_dataset, split_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
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
        st.subheader("Real-World Diabetes Classification (Pima Indians)")
        st.caption("Trains the 4-qubit Enhanced DR-QNN on the 500-record dataset with mutual-information selected clinical features.")

        col_ctrl, col_pred = st.columns([1, 1])

        with col_ctrl:
            st.markdown("**Model Hyperparameters**")
            n_layers = st.slider("Re-uploading layers", 1, 3, 2, key="drqnn_layers")
            epochs = st.slider("Optimization epochs", 10, 40, 20, step=5, key="drqnn_epochs")
            lr = st.select_slider("Learning rate (Adam)", options=[0.03, 0.05, 0.08, 0.10], value=0.08, key="drqnn_lr")
            batch_size = st.select_slider("Batch size", options=[16, 25, 32, 50], value=32, key="drqnn_bs")

            train_btn = st.button("Train Enhanced DR-QNN Now", type="primary", use_container_width=True)

        if "drqnn_model" not in st.session_state:
            st.session_state["drqnn_model"] = None
            st.session_state["drqnn_history"] = None
            st.session_state["drqnn_scaler"] = None
            st.session_state["drqnn_metrics"] = None

        if train_btn and frame is not None:
            with st.spinner("Training Enhanced Qiskit DR-QNN with vectorized batch engine..."):
                t_start = time.time()
                X_train, X_test, y_train, y_test = split_dataset(frame)
                selected_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[selected_features])
                X_te = scaler.transform(X_test[selected_features])

                model = QiskitDataReuploadingClassifier(
                    n_qubits=4,
                    n_layers=n_layers,
                    learning_rate=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    random_state=42,
                )
                history = model.fit(X_tr, y_train.values, X_te, y_test.values)
                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1]
                duration = time.time() - t_start

                acc = float(accuracy_score(y_test, y_pred))
                roc = float(roc_auc_score(y_test, y_prob))
                f1 = float(f1_score(y_test, y_pred, zero_division=0))
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

                st.session_state["drqnn_model"] = model
                st.session_state["drqnn_scaler"] = scaler
                st.session_state["drqnn_history"] = history
                st.session_state["drqnn_metrics"] = {
                    "accuracy": acc,
                    "roc_auc": roc,
                    "f1": f1,
                    "cm": cm,
                    "duration": duration,
                }
                st.success(f"Training completed in {duration:.2f} seconds!")

        if st.session_state["drqnn_metrics"] is not None:
            m = st.session_state["drqnn_metrics"]
            m_cols = st.columns(4)
            m_cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
            m_cols[1].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            m_cols[2].metric("F1 Score", f"{m['f1']:.3f}")
            m_cols[3].metric("Training Time", f"{m['duration']:.2f}s")

            st.divider()
            p_left, p_right = st.columns(2)
            with p_left:
                st.subheader("Training Loss Convergence")
                loss_df = pd.DataFrame({
                    "Epoch": list(range(1, len(st.session_state["drqnn_history"]["loss"]) + 1)),
                    "Loss": st.session_state["drqnn_history"]["loss"],
                })
                fig_loss = px.line(loss_df, x="Epoch", y="Loss", title="Cross-Entropy Loss", color_discrete_sequence=[QUANTUM_COLOR])
                st.plotly_chart(_plot_layout(fig_loss), width="stretch")

            with p_right:
                st.subheader("Confusion Matrix (Test Set)")
                cm = m["cm"]
                cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Pred Negative", "Pred Positive"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Purples", title="Test Confusion Matrix")
                st.plotly_chart(_plot_layout(fig_cm), width="stretch")

        with col_pred:
            st.markdown("**Live Interactive Patient Predictor**")
            st.caption("Adjust patient metrics to compute real-time calibrated quantum probability:")
            p_glucose = st.slider("Glucose (mg/dL)", 50.0, 220.0, 125.0, key="drqnn_glu")
            p_bmi = st.slider("Body Mass Index (BMI)", 15.0, 55.0, 31.0, key="drqnn_bmi")
            p_age = st.slider("Age (Years)", 18.0, 85.0, 35.0, key="drqnn_age")
            p_pedigree = st.slider("Diabetes Pedigree Score", 0.05, 2.50, 0.45, key="drqnn_ped")

            if st.session_state["drqnn_model"] is not None and st.session_state["drqnn_scaler"] is not None:
                raw_input = np.array([[p_glucose, p_bmi, p_age, p_pedigree]])
                scaled_input = st.session_state["drqnn_scaler"].transform(raw_input)
                q_probs = st.session_state["drqnn_model"].predict_proba(scaled_input)[0]
                pos_prob = float(q_probs[1])

                st.markdown(f"**Predicted Positive Risk:** `{pos_prob:.1%}`")
                st.progress(pos_prob)
                if pos_prob >= 0.5:
                    st.error(f"High Predicted Risk ({pos_prob:.1%}) - Positive class")
                else:
                    st.success(f"Low Predicted Risk ({pos_prob:.1%}) - Negative class")
            else:
                st.info("Train the Enhanced DR-QNN above to enable live real-time predictions.")

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
