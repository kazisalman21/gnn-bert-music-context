"""Validate final evidence and faculty deliverables without creating substitute results."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/final_validation.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks = []

    def check(name: str, condition: bool, detail) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    required = [
        "results/task1_results.json", "results/task2_results.json",
        "results/task3_results.json", "results/task4_results.json",
        "results/metrics.json", "results/faculty_evaluation_table.csv",
        "checkpoints/task1_best.pt", "checkpoints/task2_gnn_best.pt",
        "checkpoints/task2_cnn_best.pt", "checkpoints/task3_bert_only_best.pt",
        "checkpoints/task3_gnn_only_best.pt", "checkpoints/task3_concat_best.pt",
        "checkpoints/task3_cross_attention_best.pt", "notebooks/eda.ipynb",
        "checkpoints/task4_contrastive_best.pt",
        "notebooks/demo_context.ipynb", "report/final_report.pdf",
        "retrieval_examples/task4_top3_retrievals.json",
    ]
    missing = [relative for relative in required if not (ROOT / relative).exists()]
    check("required core artifacts", not missing, {"missing": missing})

    for notebook in (ROOT / "notebooks/eda.ipynb", ROOT / "notebooks/demo_context.ipynb"):
        try:
            payload = read_json(notebook)
            valid = payload.get("nbformat") == 4 and bool(payload.get("cells"))
            detail = {"cells": len(payload.get("cells", [])), "nbformat": payload.get("nbformat")}
        except Exception as exc:
            valid, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"notebook JSON: {notebook.name}", valid, detail)

    musiccaps = {
        split: read_json(ROOT / f"data/splits/musiccaps_{split}.json")
        for split in ("train", "val", "test")
    }
    musiccaps_overlap = {
        "train_val": len(set(musiccaps["train"]) & set(musiccaps["val"])),
        "train_test": len(set(musiccaps["train"]) & set(musiccaps["test"])),
        "val_test": len(set(musiccaps["val"]) & set(musiccaps["test"])),
    }
    check("MusicCaps split ID overlap", not any(musiccaps_overlap.values()), musiccaps_overlap)

    gtzan = read_json(ROOT / "data/splits/gtzan_splits.json")
    check(
        "GTZAN exact-audio cross-split duplicates",
        gtzan["audit"]["cross_split_exact_audio_duplicate_groups"] == 0,
        gtzan["audit"],
    )
    mtat = read_json(ROOT / "data/splits/mtat_splits.json")
    check(
        "MTAT clip/exact-song overlap",
        not any(mtat["audit"]["clip_overlap"].values())
        and not any(mtat["audit"]["exact_song_tuple_overlap"].values()),
        mtat["audit"],
    )

    graph_checks = []
    for graph_dir in (
        ROOT / "data/processed/gtzan_graphs",
        ROOT / "data/processed/mtat_graphs",
        ROOT / "data/processed/musiccaps_graphs",
    ):
        for path in sorted(graph_dir.glob("*.pt"))[:20]:
            item = torch.load(path, map_location="cpu", weights_only=False)
            graph_checks.append({
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "nodes": int(item["x"].shape[0]),
                "features": int(item["x"].shape[1]),
                "edges": int(item["edge_index"].shape[1]),
                "finite": bool(torch.isfinite(item["x"]).all()),
            })
    graphs_valid = len(graph_checks) >= 60 and all(
        item["nodes"] > 0 and item["features"] == 77 and item["edges"] > 0 and item["finite"]
        for item in graph_checks
    )
    check("60 sampled real graphs", graphs_valid, graph_checks)

    if (ROOT / "results/task1_results.json").exists():
        task1 = read_json(ROOT / "results/task1_results.json")
        metrics = task1["test_metrics"]
        check(
            "Task 1 held-out metrics",
            metrics["num_samples"] == len(musiccaps["test"])
            and metrics["num_tags"] == task1["vocabulary_size"]
            and all(0 <= metrics[key] <= 1 for key in ("macro_f1", "micro_f1", "mean_auc_pr")),
            {key: metrics[key] for key in ("num_samples", "num_tags", "macro_f1", "micro_f1", "mean_auc_pr")},
        )

    if (ROOT / "results/task2_results.json").exists():
        task2 = read_json(ROOT / "results/task2_results.json")
        expected = {split: len(gtzan[split]) for split in ("train", "val", "test")}
        valid = task2["split_counts"] == expected
        for mode in ("gnn", "cnn"):
            metrics = task2[mode]["test_metrics"]
            valid = valid and metrics["num_samples"] == expected["test"]
            valid = valid and all(0 <= metrics[key] <= 1 for key in ("accuracy", "macro_f1"))
        check("Task 2 clean-split metrics", valid, {"expected_split": expected})

    if (ROOT / "results/task3_results.json").exists():
        task3 = read_json(ROOT / "results/task3_results.json")
        modes = {"bert_only", "gnn_only", "concat", "cross_attention"}
        valid = set(task3["results"]) == modes
        for mode in modes:
            metrics = task3["results"][mode]["test_metrics"]
            valid = valid and metrics["num_samples"] == task3["split_counts"]["test"]
            valid = valid and metrics["num_tags"] == 19
            valid = valid and all(0 <= metrics[key] <= 1 for key in ("macro_f1", "micro_f1", "mean_auc_pr"))
        check("Task 3 four-way held-out metrics", valid, {"modes": sorted(task3["results"])})

    if (ROOT / "results/task4_results.json").exists():
        task4 = read_json(ROOT / "results/task4_results.json")
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        expected_ranges = config["task4"]["protocol_ranges"]
        protocol = task4["provenance"]
        valid = all(task4["split_counts"][name] >= 2 for name in ("train", "val", "test"))
        valid = valid and protocol["experiment_version"] == "task4-musiccaps-real-audio-v2"
        valid = valid and protocol["requested_ranges"] == expected_ranges
        valid = valid and "untrained_initialization_test_metrics" not in task4
        valid = valid and "untrained_initialization_validation_metrics" in task4
        for direction in ("caption_to_audio", "audio_to_caption"):
            valid = valid and all(
                0 <= task4[direction][f"R@{k}"] <= 1 for k in (1, 5, 10)
            )
            valid = valid and all(
                abs(
                    task4["random_retrieval_baseline"][direction][f"R@{k}"]
                    - min(k, task4["split_counts"]["test"])
                    / task4["split_counts"]["test"]
                ) < 1e-9
                for k in (1, 5, 10)
            )
        examples = read_json(ROOT / "retrieval_examples/task4_top3_retrievals.json")
        test_range = expected_ranges["test"]
        requested_test_ids = set(
            musiccaps["test"][
                int(test_range["start"]):
                int(test_range["start"]) + int(test_range["limit"])
            ]
        )
        valid = valid and len(examples) == 10
        valid = valid and all(len(item["top_3"]) == 3 for item in examples)
        valid = valid and all(
            item["paired_ytid"] in requested_test_ids
            and all(match["ytid"] in requested_test_ids for match in item["top_3"])
            for item in examples
        )
        check(
            "Task 4 held-out retrieval and examples",
            valid,
            {
                "split_counts": task4["split_counts"],
                "caption_to_audio": task4["caption_to_audio"],
                "audio_to_caption": task4["audio_to_caption"],
                "qualitative_examples": len(examples),
                "experiment_version": protocol["experiment_version"],
                "requested_ranges": protocol["requested_ranges"],
            },
        )

    required_plots = [
        "task1_f1_curves.png", "task2_model_comparison.png", "task2_gnn_confusion_matrix.png",
        "task2_cnn_confusion_matrix.png", "task3_ablation_macro_f1.png",
        "task3_ablation_auc_pr.png", "task3_per_tag_metrics.png", "task3_tsne_genre.png",
        "task3_tsne_mood.png", "task3_musiccaps_case_studies.png", "task3_architecture.png",
        "task4_learning_curve.png", "task4_retrieval_recall.png",
    ]
    plot_details = []
    plots_valid = True
    for filename in required_plots:
        path = ROOT / "plots" / filename
        if not path.exists():
            plots_valid = False
            plot_details.append({"file": filename, "missing": True})
            continue
        with Image.open(path) as image:
            width, height = image.size
        plots_valid = plots_valid and width >= 800 and height >= 500
        plot_details.append({"file": filename, "width": width, "height": height})
    check("faculty plots", plots_valid, plot_details)

    report_path = ROOT / "report/final_report.pdf"
    if report_path.exists():
        document = pymupdf.open(report_path)
        pages = len(document)
        document.close()
        check("report page count", 6 <= pages <= 10, {"pages": pages, "required": "6-10"})

    passed = all(item["passed"] for item in checks)
    payload = {"passed": passed, "checks": checks}
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
