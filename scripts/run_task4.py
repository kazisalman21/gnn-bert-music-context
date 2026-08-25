"""Task 4: contrastive audio-graph and MusicCaps caption retrieval."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contrastive import ContrastiveDualEncoder, InfoNCELoss, compute_retrieval_metrics
from src.dataset import TARGET_TAGS
from src.evaluate import compute_multilabel_metrics, tune_threshold
from src.fusion_model import FusionModel
from src.provenance import file_sha256, stable_hash
from src.train import set_seed


EXPERIMENT_VERSION = "task4-musiccaps-real-audio-v2"


def parse_aspects(value) -> list[str]:
    """Parse the published aspect list without evaluating arbitrary code."""
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = []
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


class Task4Dataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, split_ids: list[str], split_name: str,
                 graph_dir: Path, tokenizer, max_length: int):
        indexed = metadata.set_index("ytid", drop=False)
        self.rows = []
        self.graphs = []
        for ytid in split_ids:
            graph_path = graph_dir / f"{ytid}.pt"
            if ytid not in indexed.index or not graph_path.exists():
                continue
            item = torch.load(graph_path, map_location="cpu", weights_only=False)
            saved_split = item.get("metadata", {}).get("split")
            if saved_split != split_name:
                raise RuntimeError(f"Graph {ytid} is marked {saved_split}, expected {split_name}")
            self.rows.append(indexed.loc[ytid])
            self.graphs.append((item["x"], item["edge_index"]))

        self.captions = [str(row["caption"]) for row in self.rows]
        encoded = tokenizer(
            self.captions, max_length=max_length, truncation=True, padding=False
        )
        self.input_ids = [torch.tensor(values, dtype=torch.long) for values in encoded["input_ids"]]
        self.attention_masks = [
            torch.tensor(values, dtype=torch.long) for values in encoded["attention_mask"]
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        aspects = parse_aspects(row["aspect_list"])
        graph_x, edge_index = self.graphs[index]
        return {
            "ytid": str(row["ytid"]),
            "caption": self.captions[index],
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "aspects": aspects,
            "tag_labels": torch.tensor(
                [float(tag in aspects) for tag in TARGET_TAGS], dtype=torch.float32
            ),
            "graph_x": graph_x,
            "edge_index": edge_index,
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
        }


def collate_batch(items: list[dict]) -> dict:
    graphs = Batch.from_data_list([
        Data(x=item["graph_x"], edge_index=item["edge_index"]) for item in items
    ])
    max_length = max(item["input_ids"].numel() for item in items)
    input_ids = [
        nn.functional.pad(item["input_ids"], (0, max_length - item["input_ids"].numel()))
        for item in items
    ]
    attention_masks = [
        nn.functional.pad(
            item["attention_mask"], (0, max_length - item["attention_mask"].numel())
        )
        for item in items
    ]
    return {
        "graph_x": graphs.x,
        "edge_index": graphs.edge_index,
        "graph_batch": graphs.batch,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "tag_labels": torch.stack([item["tag_labels"] for item in items]),
        "ytids": [item["ytid"] for item in items],
        "captions": [item["caption"] for item in items],
        "start_s": [item["start_s"] for item in items],
        "end_s": [item["end_s"] for item in items],
        "aspects": [item["aspects"] for item in items],
    }


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {
        "graph_x": batch["graph_x"].to(device),
        "graph_edge_index": batch["edge_index"].to(device),
        "graph_batch": batch["graph_batch"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


@torch.no_grad()
def evaluate_retrieval(model, loader, criterion, device: torch.device) -> dict:
    model.eval()
    losses = []
    graph_embeddings = []
    text_embeddings = []
    labels = []
    details = {key: [] for key in ("ytids", "captions", "start_s", "end_s", "aspects")}
    for batch in loader:
        graph_emb, text_emb = model(**model_inputs(batch, device))
        losses.append(criterion(graph_emb, text_emb).item())
        graph_embeddings.append(graph_emb.cpu())
        text_embeddings.append(text_emb.cpu())
        labels.append(batch["tag_labels"].numpy())
        for key in details:
            details[key].extend(batch[key])
    graph_embeddings = torch.cat(graph_embeddings)
    text_embeddings = torch.cat(text_embeddings)
    return {
        "loss": float(np.mean(losses)),
        "graph_embeddings": graph_embeddings,
        "text_embeddings": text_embeddings,
        "metrics": compute_retrieval_metrics(graph_embeddings, text_embeddings),
        "tag_labels": np.concatenate(labels),
        **details,
    }


def validation_score(metrics: dict) -> float:
    return float(
        (metrics["caption_to_audio"]["R@5"] + metrics["audio_to_caption"]["R@5"]) / 2
    )


def random_retrieval_baseline(sample_count: int) -> dict:
    values = {f"R@{k}": min(k, sample_count) / sample_count for k in (1, 5, 10)}
    return {"caption_to_audio": values.copy(), "audio_to_caption": values.copy()}


def write_retrieval_examples(test: dict, output_dir: Path) -> list[dict]:
    """Save the ten faculty-requested caption-to-top-3 examples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    similarities = test["text_embeddings"] @ test["graph_embeddings"].t()
    top_k = min(3, similarities.shape[1])
    indices = similarities.topk(top_k, dim=1).indices
    examples = []
    for query_index in range(min(10, len(test["ytids"]))):
        matches = []
        for rank, candidate_index in enumerate(indices[query_index].tolist(), start=1):
            matches.append({
                "rank": rank,
                "ytid": test["ytids"][candidate_index],
                "start_s": test["start_s"][candidate_index],
                "end_s": test["end_s"][candidate_index],
                "similarity": float(similarities[query_index, candidate_index]),
                "is_paired_clip": candidate_index == query_index,
            })
        examples.append({
            "query_index": query_index,
            "query_caption": test["captions"][query_index],
            "paired_ytid": test["ytids"][query_index],
            "top_3": matches,
        })

    (output_dir / "task4_top3_retrievals.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    lines = ["# Task 4: ten held-out caption-to-audio retrievals", ""]
    for number, example in enumerate(examples, start=1):
        lines.extend([
            f"## Query {number}", "", example["query_caption"], "",
            "| Rank | Clip | Similarity | Paired |",
            "|---:|---|---:|:---:|",
        ])
        for match in example["top_3"]:
            paired = "yes" if match["is_paired_clip"] else "no"
            lines.append(
                f"| {match['rank']} | {match['ytid']} "
                f"({match['start_s']:.0f}-{match['end_s']:.0f}s) | "
                f"{match['similarity']:.4f} | {paired} |"
            )
        lines.append("")
    (output_dir / "task4_top3_retrievals.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return examples


@torch.no_grad()
def encode_tag_prompts(model, tokenizer, max_length: int,
                       device: torch.device) -> torch.Tensor:
    prompts = [f"This music is described as {tag}." for tag in TARGET_TAGS]
    encoded = tokenizer(
        prompts, max_length=max_length, padding=True, truncation=True,
        return_tensors="pt",
    )
    return model.encode_text(
        encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    ).cpu()


@torch.no_grad()
def task3_transfer_metrics(loader, config: dict, device: torch.device) -> dict:
    """Apply the completed Task 3 model without MusicCaps fine-tuning."""
    checkpoint = torch.load(
        ROOT / "checkpoints/task3_cross_attention_best.pt",
        map_location="cpu", weights_only=True,
    )
    model = FusionModel(
        gnn_input_dim=int(checkpoint["feature_dim"]), num_labels=len(TARGET_TAGS),
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]), freeze_bert=True,
        fusion_mode="cross_attention",
        projection_dim=int(config["fusion"]["projection_dim"]),
        num_attention_heads=int(config["fusion"]["num_attention_heads"]),
        dropout=float(config["fusion"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities = []
    labels = []
    for batch in loader:
        logits, _ = model(**model_inputs(batch, device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["tag_labels"].numpy())
    return compute_multilabel_metrics(
        np.concatenate(labels), np.concatenate(probabilities),
        threshold=float(checkpoint["threshold"]), tag_names=TARGET_TAGS,
    )


def plot_history(history: dict, validation_count: int,
                 output_path: Path) -> None:
    fig, (loss_axis, recall_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="validation")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("InfoNCE loss")
    loss_axis.set_ylim(0, max(history["train_loss"] + history["val_loss"]) * 1.12)
    loss_axis.legend()

    recall_axis.plot(
        history["epoch"], history["val_mean_r5"],
        marker="o", label="trained model", color="#2ca02c",
    )
    chance = min(5, validation_count) / validation_count
    recall_axis.axhline(
        chance, color="#777777", linestyle="--", label="random ranking"
    )
    recall_axis.set_xlabel("Epoch")
    recall_axis.set_ylabel("Mean bidirectional R@5")
    recall_top = max(0.20, max(history["val_mean_r5"]) * 1.20, chance * 1.20)
    recall_axis.set_ylim(0, min(1.0, recall_top))
    recall_axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_retrieval(metrics: dict, baseline: dict, output_path: Path) -> None:
    labels = [
        "C-A R@1", "C-A R@5", "C-A R@10",
        "A-C R@1", "A-C R@5", "A-C R@10",
    ]
    trained = [
        metrics[direction][f"R@{k}"]
        for direction in ("caption_to_audio", "audio_to_caption")
        for k in (1, 5, 10)
    ]
    chance = [
        baseline[direction][f"R@{k}"]
        for direction in ("caption_to_audio", "audio_to_caption")
        for k in (1, 5, 10)
    ]
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(8, 4))
    chance_bars = axis.bar(
        positions - 0.2, chance, width=0.4, label="random ranking"
    )
    trained_bars = axis.bar(
        positions + 0.2, trained, width=0.4, label="trained dual encoder"
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, min(1.0, max(0.15, max(chance + trained) * 1.25)))
    axis.bar_label(chance_bars, fmt="%.3f", fontsize=7, padding=2)
    axis.bar_label(trained_bars, fmt="%.3f", fontsize=7, padding=2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    schedule = config["task4"]["training"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(schedule["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(schedule["batch_size"]))
    args = parser.parse_args()

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = ROOT / "data/processed/musiccaps_graphs"
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv")
    metadata["ytid"] = metadata["ytid"].astype(str)
    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])

    split_paths = {
        name: ROOT / f"data/splits/musiccaps_{name}.json"
        for name in ("train", "val", "test")
    }
    full_split_ids = {
        name: [
            str(value)
            for value in json.loads(path.read_text(encoding="utf-8"))
        ]
        for name, path in split_paths.items()
    }
    requested_ranges = config["task4"]["protocol_ranges"]
    split_ids = {}
    for name, ids in full_split_ids.items():
        start = int(requested_ranges[name]["start"])
        limit = int(requested_ranges[name]["limit"])
        split_ids[name] = ids[start:start + limit]
        if len(split_ids[name]) != limit:
            raise RuntimeError(
                f"Task 4 {name} range requests {limit} IDs from offset {start}, "
                f"but the frozen split supplies only {len(split_ids[name])}"
            )
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    if any(set(split_ids[left]) & set(split_ids[right]) for left, right in pairs):
        raise RuntimeError("MusicCaps IDs overlap across frozen splits")

    datasets = {
        name: Task4Dataset(
            metadata, split_ids[name], name, graph_dir, tokenizer,
            int(config["bert"]["max_length"]),
        )
        for name in ("train", "val", "test")
    }
    counts = {name: len(dataset) for name, dataset in datasets.items()}
    if min(counts.values()) < 2:
        raise RuntimeError(
            f"Task 4 needs at least two real graph-caption pairs per split: {counts}"
        )
    print("Task 4 usable real-audio split counts:", counts, flush=True)

    generator = torch.Generator().manual_seed(int(config["seed"]))
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_batch,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    feature_dim = int(datasets["train"][0]["graph_x"].shape[1])
    model = ContrastiveDualEncoder(
        gnn_input_dim=feature_dim,
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]),
        freeze_bert=bool(config["task4"]["freeze_bert"]),
        projection_dim=int(config["task4"]["projection_dim"]),
        dropout=float(config["gnn"]["dropout"]),
    ).to(device)
    task3_gnn_path = ROOT / "checkpoints/task3_gnn_only_best.pt"
    if bool(config["task4"]["initialize_gnn_from_task3"]):
        task3_state = torch.load(
            task3_gnn_path, map_location="cpu", weights_only=True
        )["model_state_dict"]
        encoder_state = {
            key.removeprefix("gnn_encoder."): value
            for key, value in task3_state.items()
            if key.startswith("gnn_encoder.")
        }
        model.gnn_encoder.load_state_dict(encoder_state, strict=True)
        print("Initialized Task 4 GraphSAGE from the Task 3 audio encoder.")
    criterion = InfoNCELoss(float(config["task4"]["temperature"]))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(schedule["learning_rate"]),
        weight_decay=float(schedule["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    results_dir = ROOT / config["paths"]["results"]
    plots_dir = ROOT / config["paths"]["plots"]
    examples_dir = ROOT / config["paths"]["retrieval_examples"]
    checkpoint_path = (
        ROOT / config["paths"]["checkpoints"] / "task4_contrastive_best.pt"
    )
    for directory in (
        results_dir, plots_dir, examples_dir, checkpoint_path.parent
    ):
        directory.mkdir(parents=True, exist_ok=True)

    initial_validation = evaluate_retrieval(
        model, loaders["val"], criterion, device
    )
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_mean_r5": [],
    }
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if bool(config["task4"]["freeze_bert"]):
            # Frozen transformer dropout should not add noise between pairs.
            model.bert.bert.eval()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            graph_emb, text_emb = model(**model_inputs(batch, device))
            loss = criterion(graph_emb, text_emb)
            if not torch.isfinite(loss):
                raise RuntimeError("Task 4 produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip"])
            )
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        validation = evaluate_retrieval(
            model, loaders["val"], criterion, device
        )
        score = validation_score(validation["metrics"])
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(validation["loss"])
        history["val_mean_r5"].append(score)
        print(
            f"Task 4 epoch {epoch:02d}: "
            f"loss={history['train_loss'][-1]:.4f}, "
            f"val loss={validation['loss']:.4f}, mean R@5={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(schedule["early_stopping_patience"]):
                break

    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )["model_state_dict"]
    )
    validation = evaluate_retrieval(
        model, loaders["val"], criterion, device
    )
    test = evaluate_retrieval(model, loaders["test"], criterion, device)
    random_baseline = random_retrieval_baseline(counts["test"])
    examples = write_retrieval_examples(test, examples_dir)

    prompt_embeddings = encode_tag_prompts(
        model, tokenizer, int(config["bert"]["max_length"]), device
    )
    validation_scores = (
        (validation["text_embeddings"] @ prompt_embeddings.t()).numpy() + 1
    ) / 2
    test_scores = (
        (test["text_embeddings"] @ prompt_embeddings.t()).numpy() + 1
    ) / 2
    zero_shot_threshold = float(
        tune_threshold(validation["tag_labels"], validation_scores)
    )
    zero_shot_metrics = compute_multilabel_metrics(
        test["tag_labels"], test_scores,
        threshold=zero_shot_threshold, tag_names=TARGET_TAGS,
    )

    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    supervised_transfer = task3_transfer_metrics(
        loaders["test"], config, device
    )

    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": int(config["seed"]),
        "split_sha256": {
            name: file_sha256(path) for name, path in split_paths.items()
        },
        "usable_ids_sha256": {
            name: stable_hash({
                "ids": [str(row["ytid"]) for row in dataset.rows]
            })
            for name, dataset in datasets.items()
        },
        "split_counts": counts,
        "requested_ranges": requested_ranges,
        "feature_dim": feature_dim,
        "temperature": float(config["task4"]["temperature"]),
        "projection_dim": int(config["task4"]["projection_dim"]),
        "freeze_bert": bool(config["task4"]["freeze_bert"]),
        "initialize_gnn_from_task3": bool(
            config["task4"]["initialize_gnn_from_task3"]
        ),
        "task3_gnn_checkpoint_sha256": (
            file_sha256(task3_gnn_path)
            if bool(config["task4"]["initialize_gnn_from_task3"])
            else None
        ),
        "training": {
            **schedule,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
        },
    }
    result = {
        "task": "Task 4: Cross-modal MusicCaps alignment",
        "dataset": (
            "MusicCaps public metadata with recovered real YouTube intervals"
        ),
        "split_counts": counts,
        "sample_reduction_note": (
            "Only source videos successfully recovered at their published "
            "MusicCaps intervals are used. No missing clip is replaced or "
            "synthesized."
        ),
        "caption_to_audio": test["metrics"]["caption_to_audio"],
        "audio_to_caption": test["metrics"]["audio_to_caption"],
        "test_loss": test["loss"],
        "validation_metrics_at_best_checkpoint": validation["metrics"],
        "random_retrieval_baseline": random_baseline,
        "untrained_initialization_validation_metrics": (
            initial_validation["metrics"]
        ),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mean_r5": best_score,
        "training_time_seconds": time.time() - started,
        "parameter_counts": {
            "total": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "qualitative_examples_file": (
            "retrieval_examples/task4_top3_retrievals.json"
        ),
        "qualitative_example_count": len(examples),
        "zero_shot_tag_transfer": {
            "labels": TARGET_TAGS,
            "ground_truth": "exact matches in MusicCaps aspect_list",
            "task4_caption_to_tag_prompt": zero_shot_metrics,
            "task4_threshold_tuned_on_validation": zero_shot_threshold,
            "task3_cross_attention_direct_transfer": supervised_transfer,
        },
        "device": str(device),
        "provenance": {
            "experiment_signature": stable_hash(protocol),
            **protocol,
        },
    }
    (results_dir / "task4_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        results_dir / "task4_test_embeddings.npz",
        graph_embeddings=test["graph_embeddings"].numpy(),
        text_embeddings=test["text_embeddings"].numpy(),
        ytids=np.asarray(test["ytids"]),
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "projection_dim": int(config["task4"]["projection_dim"]),
        "temperature": float(config["task4"]["temperature"]),
        "freeze_bert": bool(config["task4"]["freeze_bert"]),
        "provenance": result["provenance"],
    }, checkpoint_path)
    plot_history(
        history, counts["val"], plots_dir / "task4_learning_curve.png"
    )
    plot_retrieval(
        test["metrics"], random_baseline,
        plots_dir / "task4_retrieval_recall.png",
    )
    print(json.dumps({
        "split_counts": counts,
        "caption_to_audio": result["caption_to_audio"],
        "audio_to_caption": result["audio_to_caption"],
        "best_epoch": best_epoch,
    }, indent=2))


if __name__ == "__main__":
    main()
