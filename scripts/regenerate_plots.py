"""Regenerate plots from saved experiment JSON files without retraining models."""

from __future__ import annotations

import json
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import plot_f1_history, plot_model_comparison
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def read_result(filename: str) -> dict:
    return json.loads((RESULTS / filename).read_text(encoding="utf-8"))


def main() -> None:
    task1 = read_result("task1_results.json")
    plot_f1_history(task1["history"], PLOTS / "task1_f1_curves.png",
                    "Task 1 validation F1 by epoch")

    task2 = read_result("task2_results.json")
    plot_f1_history(task2["gnn"]["history"], PLOTS / "task2_gnn_learning_curve.png",
                    "Task 2 GraphSAGE validation F1")
    plot_f1_history(task2["cnn"]["history"], PLOTS / "task2_cnn_learning_curve.png",
                    "Task 2 CNN validation F1")
    plot_model_comparison(
        ["GraphSAGE", "CNN"],
        [task2["gnn"]["test_metrics"]["macro_f1"], task2["cnn"]["test_metrics"]["macro_f1"]],
        PLOTS / "task2_model_comparison.png",
        "Task 2 model comparison",
    )

    for mode in ("gnn_only", "bert_only", "concat", "cross_attention"):
        result_path = RESULTS / f"task3_{mode}_results.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            plot_f1_history(
                result["history"],
                PLOTS / f"task3_{mode}_learning_curve.png",
                f"Task 3 {mode.replace('_', ' ')} validation F1",
            )


if __name__ == "__main__":
    main()
