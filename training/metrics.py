"""Classification metrics and confusion-matrix reporting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict:
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(len(class_names))),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=list(range(len(class_names))),
        ).tolist(),
    }


def write_metrics(metrics: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    destination: str | Path,
    title: str,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
