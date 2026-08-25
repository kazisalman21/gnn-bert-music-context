"""Draw the implemented Task 3 architecture as a report-ready figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   linewidth=1.3, color="#444444"))


def main() -> None:
    figure, axis = plt.subplots(figsize=(13, 5.2))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5.2)
    axis.axis("off")

    blue = "#d9e8f5"
    orange = "#fde0c5"
    green = "#dcefd7"
    purple = "#e8dcf2"

    box(axis, 0.3, 3.35, 1.6, 0.75, "Audio\n22.05 kHz", blue)
    box(axis, 2.35, 3.35, 1.8, 0.75, "5 s segments\n(2 s for cases)", blue)
    box(axis, 4.6, 3.35, 1.9, 0.75, "77-D node features\n+ graph edges", blue)
    box(axis, 7.0, 3.35, 1.7, 0.75, "GraphSAGE\nmean pooling", green)
    box(axis, 9.2, 3.35, 1.45, 0.75, "Graph vector\n$g$", green)

    box(axis, 0.3, 1.05, 1.6, 0.75, "Context text\nor caption", orange)
    box(axis, 2.35, 1.05, 1.8, 0.75, "DistilBERT\ntokenizer", orange)
    box(axis, 4.6, 1.05, 1.9, 0.75, "DistilBERT\ntoken states $H$", orange)
    box(axis, 7.25, 1.05, 2.1, 0.75, "$Q=gW_Q$\n$K=HW_K$, $V=HW_V$", purple)
    box(axis, 10.0, 1.9, 1.65, 0.9, "Cross-attention\n$[g;AV]$", purple)
    box(axis, 11.95, 1.9, 0.85, 0.9, "19-tag\noutput", green)

    for start, end in [
        ((1.9, 3.72), (2.35, 3.72)), ((4.15, 3.72), (4.6, 3.72)),
        ((6.5, 3.72), (7.0, 3.72)), ((8.7, 3.72), (9.2, 3.72)),
        ((1.9, 1.42), (2.35, 1.42)), ((4.15, 1.42), (4.6, 1.42)),
        ((6.5, 1.42), (7.25, 1.42)), ((9.35, 1.42), (10.0, 2.08)),
        ((10.65, 3.35), (10.65, 2.8)), ((11.65, 2.35), (11.95, 2.35)),
    ]:
        arrow(axis, start, end)

    axis.text(6.5, 4.75, "Implemented GNN--BERT cross-attention pipeline", ha="center",
              fontsize=15, fontweight="bold")
    axis.text(6.5, 0.25,
              "Ablations bypass one branch (BERT-only/GNN-only) or replace cross-attention with concatenation.",
              ha="center", fontsize=9, color="#444444")
    output_path = ROOT / "plots/task3_architecture.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
