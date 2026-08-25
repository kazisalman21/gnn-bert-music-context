"""Generate three held-out Task 3 graph/text alignment case studies.

The examples are selected deterministically from the saved test predictions:
one high, one median, and one low example-level F1 case.  This avoids presenting
three hand-picked successes.  Every value and attention weight comes from the
trained cross-attention checkpoint and a real MTAT test clip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_task3 import MTATContextDataset, collate_batch, model_inputs
from src.dataset import CONTEXT_TAGS, TARGET_TAGS
from src.fusion_model import FusionModel
from src.train import set_seed


def example_f1(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    true_positive = (labels * predictions).sum(axis=1)
    denominator = labels.sum(axis=1) + predictions.sum(axis=1)
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float),
                     where=denominator > 0)


def choose_cases(archive, threshold: float) -> list[tuple[str, int, float]]:
    labels = archive["labels"].astype(np.int32)
    probabilities = archive["probabilities"]
    predictions = (probabilities >= threshold).astype(np.int32)
    scores = example_f1(labels, predictions)

    # Qualitative examples are required to have at least one annotated target.
    eligible = np.flatnonzero(labels.sum(axis=1) > 0)
    if len(eligible) < 3:
        raise RuntimeError("Fewer than three target-positive held-out examples are available")
    ordered = eligible[np.argsort(scores[eligible], kind="stable")]
    selections = [
        ("low", int(ordered[0])),
        ("median", int(ordered[len(ordered) // 2])),
        ("high", int(ordered[-1])),
    ]
    return [(name, index, float(scores[index])) for name, index in selections]


def unique_edges(graph: dict) -> list[tuple[int, int, float, str]]:
    edge_index = graph["edge_index"].cpu().numpy()
    edge_attr = graph.get("edge_attr")
    weights = (
        edge_attr.squeeze(-1).cpu().numpy()
        if edge_attr is not None
        else np.ones(edge_index.shape[1], dtype=np.float32)
    )
    found: dict[tuple[int, int], tuple[float, str]] = {}
    for position in range(edge_index.shape[1]):
        source, target = map(int, edge_index[:, position])
        if source == target:
            continue
        pair = tuple(sorted((source, target)))
        kind = "temporal" if abs(source - target) == 1 else "similarity"
        value = float(weights[position])
        if pair not in found or value > found[pair][0]:
            found[pair] = (value, kind)
    return [(source, target, value, kind) for (source, target), (value, kind) in found.items()]


def plot_case(case_number: int, selection_name: str, sample_f1: float, clip_id: int,
              graph: dict, text: str, tokens: list[str], attention: np.ndarray,
              true_tags: list[str], predictions: list[tuple[str, float]], output_path: Path) -> None:
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.1], hspace=0.55)
    graph_ax = fig.add_subplot(grid[0])

    node_count = int(graph["x"].shape[0])
    x_positions = np.arange(node_count, dtype=float)
    y_positions = np.zeros(node_count, dtype=float)
    edges = unique_edges(graph)
    similarity_edges = sorted(
        [edge for edge in edges if edge[3] == "similarity"], key=lambda item: item[2], reverse=True
    )
    strongest_similarity = {(edge[0], edge[1]) for edge in similarity_edges[:3]}

    temporal_label_used = False
    similarity_label_used = False
    for source, target, weight, kind in edges:
        distance = target - source
        height = 0.2 + 0.16 * abs(distance)
        curve_x = np.linspace(source, target, 60)
        phase = np.linspace(0, np.pi, 60)
        curve_y = height * np.sin(phase)
        highlighted = (source, target) in strongest_similarity
        if kind == "temporal":
            color, style, width, alpha = "#4c78a8", "-", 1.8, 0.80
            label = "temporal connection" if not temporal_label_used else None
            temporal_label_used = True
        else:
            color, style = "#d62728", "--"
            width, alpha = (3.0, 0.95) if highlighted else (1.2, 0.35)
            label = "top acoustic-similarity connection" if highlighted and not similarity_label_used else None
            similarity_label_used = similarity_label_used or highlighted
        graph_ax.plot(curve_x, curve_y, color=color, linestyle=style, linewidth=width,
                      alpha=alpha, label=label)

    graph_ax.scatter(x_positions, y_positions, s=520, color="#54a24b", edgecolor="black", zorder=3)
    for node in range(node_count):
        graph_ax.text(node, 0, f"S{node + 1}", ha="center", va="center", color="white",
                      fontsize=9, fontweight="bold", zorder=4)
    graph_ax.set_xlim(-0.6, max(node_count - 0.4, 0.6))
    graph_ax.set_ylim(-0.22, max(1.15, 0.35 + 0.18 * node_count))
    graph_ax.set_xticks(x_positions, [f"{5 * node}-{5 * (node + 1)} s" for node in range(node_count)])
    graph_ax.set_yticks([])
    graph_ax.set_title(f"Case {case_number}: MTAT clip {clip_id} ({selection_name} held-out F1={sample_f1:.2f})")
    if temporal_label_used or similarity_label_used:
        graph_ax.legend(loc="upper right", fontsize=8)
    for spine in ("left", "right", "top"):
        graph_ax.spines[spine].set_visible(False)

    attention_ax = fig.add_subplot(grid[1])
    heatmap = attention.reshape(1, -1)
    image = attention_ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0)
    attention_ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=9)
    attention_ax.set_yticks([0], ["mean attention"])
    attention_ax.set_title(f"Context: {text}", fontsize=10)
    fig.colorbar(image, ax=attention_ax, fraction=0.02, pad=0.02)

    prediction_text = ", ".join(f"{tag} ({score:.2f})" for tag, score in predictions)
    fig.text(0.02, 0.02, "Ground truth: " + (", ".join(true_tags) or "none"), fontsize=9)
    fig.text(0.02, 0.002, "Predicted: " + (prediction_text or "none above threshold"), fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    checkpoint_path = ROOT / "checkpoints/task3_cross_attention_best.pt"
    result_path = ROOT / "results/task3_cross_attention_results.json"
    archive_path = ROOT / "results/task3_test_embeddings.npz"
    for path in (checkpoint_path, result_path, archive_path):
        if not path.exists():
            raise FileNotFoundError(f"Required real Task 3 artifact is missing: {path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    threshold = float(result["threshold_tuned_on_validation"])
    archive = np.load(archive_path)
    selected = choose_cases(archive, threshold)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    annotations = pd.read_csv(ROOT / "data/raw/magnatagatune/annotations_final.csv", sep="\t")
    graph_dir = ROOT / "data/processed/mtat_graphs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(
        gnn_input_dim=int(checkpoint["feature_dim"]),
        num_labels=len(TARGET_TAGS),
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]),
        freeze_bert=True,
        fusion_mode="cross_attention",
        projection_dim=int(config["fusion"]["projection_dim"]),
        num_attention_heads=int(config["fusion"]["num_attention_heads"]),
        dropout=float(config["fusion"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = []
    for case_number, (selection_name, archive_index, saved_f1) in enumerate(selected, start=1):
        clip_id = int(archive["clip_ids"][archive_index])
        dataset = MTATContextDataset(
            annotations, [clip_id], graph_dir, tokenizer, int(config["bert"]["max_length"])
        )
        if len(dataset) != 1:
            raise RuntimeError(f"Could not reconstruct held-out clip {clip_id}")
        item = dataset[0]
        batch = collate_batch([item])
        with torch.no_grad():
            logits, extras = model(**model_inputs(batch, device))
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            attention = extras["attention_weights"][0, :, 0, :].mean(dim=0).cpu().numpy()

        valid_length = int(batch["attention_mask"][0].sum().item())
        token_ids = batch["input_ids"][0, :valid_length].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        attention = attention[:valid_length]
        labels = item["labels"].numpy().astype(np.int32)
        true_tags = [TARGET_TAGS[i] for i in np.flatnonzero(labels)]
        predicted_indices = np.flatnonzero(probabilities >= threshold)
        if len(predicted_indices) == 0:
            predicted_indices = np.argsort(probabilities)[-3:][::-1]
        predicted = sorted(
            [(TARGET_TAGS[i], float(probabilities[i])) for i in predicted_indices],
            key=lambda pair: pair[1], reverse=True,
        )
        graph = torch.load(graph_dir / f"{clip_id}.pt", map_location="cpu", weights_only=False)
        output_path = ROOT / f"plots/task3_case_study_{case_number}_attention.png"
        plot_case(
            case_number, selection_name, saved_f1, clip_id, graph, item["text"], tokens,
            attention, true_tags, predicted, output_path,
        )
        records.append({
            "case": case_number,
            "selection": selection_name,
            "clip_id": clip_id,
            "example_f1": saved_f1,
            "context_text": item["text"],
            "active_context_tags": [
                tag for tag in CONTEXT_TAGS if int(dataset.rows[0][tag]) == 1],
            "ground_truth_target_tags": true_tags,
            "predictions_shown": [{"tag": tag, "probability": score} for tag, score in predicted],
            "threshold": threshold,
            "figure": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        })

    output = {
        "dataset": "MagnaTagATune held-out shards d-f",
        "selection_protocol": "lowest, median, and highest example-level F1 among target-positive test clips",
        "attention": "mean cross-attention over four heads; evaluation mode",
        "cases": records,
    }
    (ROOT / "results/task3_case_studies.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
