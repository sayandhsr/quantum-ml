
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def calculate_metrics(y_true, y_pred, y_prob):

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist()
    }

    return metrics


def save_metrics(metrics, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    output_path
):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150)

    plt.close(fig)


def plot_roc_curve(
    y_true,
    y_prob,
    title,
    output_path
):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label="ROC Curve"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(title)

    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150)

    plt.close(fig)


def print_metrics(name, metrics):

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    for key, value in metrics.items():

        if key != "confusion_matrix":
            print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")

    print(
        np.array(metrics["confusion_matrix"])
    )
