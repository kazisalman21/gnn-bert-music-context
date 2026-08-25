"""Small plotting helpers used by the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def _score_axis_upper(values: list[float], minimum: float = 0.5) -> float:
    """Use 0.50 for low-score plots and the full 1.00 range otherwise."""
    observed_max = max(values, default=0.0)
    return minimum if observed_max <= minimum else 1.0


def _finish(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_f1_history(history: dict, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = history["epoch"]
    plotted_values = []
    if history.get("val_macro_f1"):
        ax.plot(epochs, history["val_macro_f1"], marker="o", markersize=3, label="Validation Macro-F1")
        plotted_values.extend(history["val_macro_f1"])
    if history.get("val_micro_f1"):
        ax.plot(epochs, history["val_micro_f1"], marker="s", markersize=3, label="Validation Micro-F1")
        plotted_values.extend(history["val_micro_f1"])
    ax.set(xlabel="Epoch", ylabel="F1 score", title=title)
    ax.set_ylim(0, _score_axis_upper(plotted_values))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.25)
    ax.legend()
    _finish(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], path: str | Path,
                          title: str) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted label", ylabel="True label", title=title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black", fontsize=8)
    _finish(fig, path)


def plot_model_comparison(names: list[str], values: list[float], path: str | Path,
                          title: str, ylabel: str = "Macro-F1") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][:len(names)])
    ax.set(ylabel=ylabel, title=title)
    ax.set_ylim(0, _score_axis_upper(values))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    _finish(fig, path)


def plot_per_tag_metrics(tags: list[str], f1_values: list[float], auc_values: list[float],
                         path: str | Path) -> None:
    positions = np.arange(len(tags))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(positions - width / 2, f1_values, width, label="F1")
    ax.bar(positions + width / 2, auc_values, width, label="AUC-PR")
    ax.set_xticks(positions, tags, rotation=55, ha="right")
    ax.set(ylabel="Score", title="Task 3 per-tag performance")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _finish(fig, path)


def plot_tsne(points: np.ndarray, labels: np.ndarray, class_names: list[str],
              path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=16, alpha=0.7, label=class_name)
    ax.set(title=title, xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.15)
    _finish(fig, path)


def plot_attention_heatmap(tokens: list[str], weights: np.ndarray, path: str | Path,
                           title: str) -> None:
    weights = np.asarray(weights).reshape(1, -1)
    fig_width = max(7, len(tokens) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 2.2))
    image = ax.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    ax.set_xticks(range(len(tokens)), tokens, rotation=55, ha="right")
    ax.set_yticks([0], ["attention"])
    ax.set_title(title)
    _finish(fig, path)
