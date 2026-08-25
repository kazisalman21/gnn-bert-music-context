"""Generate three real MusicCaps graph/caption alignment panels.

The Task 3 cross-attention checkpoint is trained on MTAT. These panels are a
qualitative transfer check on real held-out MusicCaps audio and expert captions;
they are not reported as MusicCaps classification or retrieval metrics.
"""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_task3_case_studies import unique_edges
from src.dataset import TARGET_TAGS
from src.fusion_model import FusionModel
from src.train import set_seed


def draw_graph(axis, graph: dict) -> None:
    node_count = int(graph["x"].shape[0])
    x_positions = np.arange(node_count, dtype=float)
    edges = unique_edges(graph)
    similarity_edges = sorted(
        [edge for edge in edges if edge[3] == "similarity"], key=lambda edge: edge[2], reverse=True
    )
    strongest = {(edge[0], edge[1]) for edge in similarity_edges[:3]}
    labels_used = set()

    for source, target, weight, kind in edges:
        curve_x = np.linspace(source, target, 60)
        curve_y = (0.18 + 0.14 * abs(target - source)) * np.sin(np.linspace(0, np.pi, 60))
        if kind == "temporal":
            color, style, width, alpha, label = "#4c78a8", "-", 1.7, 0.8, "temporal"
        else:
            highlighted = (source, target) in strongest
            color, style = "#d62728", "--"
            width, alpha, label = (2.8, 0.95, "strong similarity") if highlighted else (1.0, 0.3, None)
        if label in labels_used:
            label = None
        elif label:
            labels_used.add(label)
        axis.plot(curve_x, curve_y, color=color, linestyle=style, linewidth=width,
                  alpha=alpha, label=label)

    axis.scatter(x_positions, np.zeros(node_count), s=420, color="#54a24b", edgecolor="black", zorder=3)
    for node in range(node_count):
        axis.text(node, 0, f"S{node + 1}", ha="center", va="center", color="white",
                  fontweight="bold", fontsize=9, zorder=4)
    axis.set_xlim(-0.5, max(node_count - 0.5, 0.5))
    axis.set_ylim(-0.2, max(0.9, 0.25 + node_count * 0.13))
    axis.set_xticks(x_positions, [f"{2 * i}-{2 * (i + 1)}s" for i in range(node_count)])
    axis.set_yticks([])
    axis.set_xlabel("Downloaded MusicCaps interval")
    axis.legend(loc="upper right", fontsize=7)
    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)


def parse_aspects(value) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return [str(item).strip().lower() for item in parsed]


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    checkpoint_path = ROOT / "checkpoints/task3_cross_attention_best.pt"
    result_path = ROOT / "results/task3_cross_attention_results.json"
    if not checkpoint_path.exists() or not result_path.exists():
        raise FileNotFoundError("Complete the real Task 3 cross-attention run first")

    graph_dir = ROOT / "data/processed/musiccaps_graphs"
    test_ids = json.loads((ROOT / "data/splits/musiccaps_test.json").read_text(encoding="utf-8"))
    selected_ids = [ytid for ytid in test_ids if (graph_dir / f"{ytid}.pt").exists()][:3]
    if len(selected_ids) < 3:
        raise RuntimeError("At least three real held-out MusicCaps graphs are required")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_result = json.loads(result_path.read_text(encoding="utf-8"))
    threshold = float(run_result["threshold_tuned_on_validation"])
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv").set_index("ytid", drop=False)
    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(
        gnn_input_dim=int(checkpoint["feature_dim"]), num_labels=len(TARGET_TAGS),
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]), freeze_bert=True,
        fusion_mode="cross_attention", projection_dim=int(config["fusion"]["projection_dim"]),
        num_attention_heads=int(config["fusion"]["num_attention_heads"]),
        dropout=float(config["fusion"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    figure, axes = plt.subplots(3, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [1.0, 1.35]})
    records = []
    for row_number, ytid in enumerate(selected_ids):
        row = metadata.loc[ytid]
        caption = str(row["caption"])
        saved_graph = torch.load(graph_dir / f"{ytid}.pt", map_location="cpu", weights_only=False)
        graph = Batch.from_data_list([
            Data(x=saved_graph["x"], edge_index=saved_graph["edge_index"])
        ])
        encoded = tokenizer(
            caption, max_length=int(config["bert"]["max_length"]), truncation=True,
            padding=False, return_tensors="pt",
        )
        with torch.no_grad():
            logits, extras = model(
                graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device),
                encoded["input_ids"].to(device), encoded["attention_mask"].to(device),
            )
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        attention = extras["attention_weights"][0, :, 0, :].mean(dim=0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())

        content_indices = [i for i, token in enumerate(tokens) if token not in {"[CLS]", "[SEP]", "[PAD]"}]
        ranked = sorted(content_indices, key=lambda index: attention[index], reverse=True)[:18]
        displayed = sorted(ranked)
        displayed_tokens = [f"{tokens[index]}" for index in displayed]
        displayed_weights = attention[displayed]

        graph_axis = axes[row_number, 0]
        draw_graph(graph_axis, saved_graph)
        graph_axis.set_title(f"Case {row_number + 1}: {ytid} segment graph", fontsize=11)

        attention_axis = axes[row_number, 1]
        image = attention_axis.imshow(displayed_weights.reshape(1, -1), cmap="YlOrRd", aspect="auto", vmin=0)
        attention_axis.set_xticks(range(len(displayed_tokens)), displayed_tokens, rotation=45, ha="right", fontsize=8)
        attention_axis.set_yticks([0], ["attention"])
        attention_axis.set_title(textwrap.fill(caption[:230], width=88), fontsize=8.5)
        figure.colorbar(image, ax=attention_axis, fraction=0.022, pad=0.02)

        predicted_indices = np.flatnonzero(probabilities >= threshold)
        if len(predicted_indices) == 0:
            predicted_indices = np.argsort(probabilities)[-3:][::-1]
        predictions = sorted(
            [(TARGET_TAGS[index], float(probabilities[index])) for index in predicted_indices],
            key=lambda pair: pair[1], reverse=True,
        )
        exact_targets = [tag for tag in parse_aspects(row["aspect_list"]) if tag in TARGET_TAGS]
        prediction_line = ", ".join(f"{tag}={score:.2f}" for tag, score in predictions[:6])
        attention_axis.set_xlabel(
            f"Exact target aspects: {', '.join(exact_targets) or 'none'} | Model: {prediction_line}",
            fontsize=8,
        )
        records.append({
            "case": row_number + 1,
            "ytid": ytid,
            "split": "test",
            "caption": caption,
            "exact_task3_target_aspects": exact_targets,
            "threshold": threshold,
            "predictions": [{"tag": tag, "probability": score} for tag, score in predictions],
            "displayed_tokens": [
                {"position": int(index), "token": tokens[index], "mean_attention": float(attention[index])}
                for index in displayed
            ],
        })

    figure.suptitle(
        "Task 3 qualitative transfer: real held-out MusicCaps graphs and expert captions",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = ROOT / "plots/task3_musiccaps_case_studies.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    output = {
        "dataset": "MusicCaps held-out AudioSet-eval partition",
        "model_training_dataset": "MagnaTagATune",
        "interpretation": "qualitative transfer only; not a MusicCaps performance estimate",
        "selection": "first three available held-out clips in the frozen test split",
        "attention": "mean over four cross-attention heads; 18 highest-attention content tokens shown in sequence order",
        "figure": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "cases": records,
    }
    (ROOT / "results/task3_musiccaps_case_studies.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
