"""Shared binary-classification metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
) -> dict[str, Any]:
    """Calculate thresholded metrics and ranking metrics for the positive class."""
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    if not (len(truth) == len(prediction) == len(probability)):
        raise ValueError("Labels, predictions, and probabilities must have equal lengths.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite values between 0 and 1.")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Metrics require both binary target classes 0 and 1.")

    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1_score": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
    }


def select_f1_threshold(y_true: Any, y_prob: Any) -> float:
    """Choose an F1 threshold using training-only out-of-fold probabilities."""
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    if len(truth) != len(probability) or len(truth) == 0:
        raise ValueError("Labels and probabilities must be non-empty and equally sized.")
    if not np.isfinite(probability).all():
        raise ValueError("Threshold selection requires finite probabilities.")
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = np.flatnonzero(f1_values == f1_values.max())
    return float(thresholds[best[-1]])


def save_metrics(metrics: dict[str, Any], output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    title: str,
    output_path: Path | str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], labels=["No diabetes", "Diabetes"])
    axis.set_yticks([0, 1], labels=["No diabetes", "Diabetes"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_roc_curve(
    y_true: Any,
    y_prob: Any,
    title: str,
    output_path: Path | str,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.legend()
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_precision_recall_curve(
    y_true: Any,
    y_prob: Any,
    title: str,
    output_path: Path | str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label="Model")
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.legend()
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 45}\n{name}\n{'=' * 45}")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value:.4f}")
    print("Confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
