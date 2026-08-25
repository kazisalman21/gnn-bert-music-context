"""Task 3: GNN-BERT fusion with a four-model ablation on MTAT."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CONTEXT_TAGS, TARGET_GENRE_TAGS, TARGET_MOOD_TAGS, TARGET_TAGS
from src.evaluate import compute_multilabel_metrics, tune_threshold
from src.fusion_model import FusionModel
from src.plotting import (
    plot_f1_history,
    plot_model_comparison,
    plot_per_tag_metrics,
    plot_tsne,
)
from src.train import set_seed
from src.provenance import file_sha256, stable_hash


EXPERIMENT_VERSION = "task3-official-shards-rich77-v2"


class MTATContextDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, clip_ids: list[int], graph_dir: Path,
                 tokenizer, max_length: int):
        indexed = annotations.set_index("clip_id", drop=False)
        self.rows = []
        for clip_id in clip_ids:
            graph_path = graph_dir / f"{int(clip_id)}.pt"
            if int(clip_id) in indexed.index and graph_path.exists():
                self.rows.append(indexed.loc[int(clip_id)])
        self.graph_dir = graph_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.contexts = [self.make_context(row) for row in self.rows]
        # Context sentences never change between epochs. Tokenizing once avoids
        # repeating hundreds of thousands of identical CPU operations during
        # the long GNN-only run.
        encoded = self.tokenizer(
            self.contexts,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        self.input_ids = [torch.tensor(values, dtype=torch.long) for values in encoded["input_ids"]]
        self.attention_masks = [
            torch.tensor(values, dtype=torch.long) for values in encoded["attention_mask"]
        ]
        self.graphs = []
        for row in self.rows:
            clip_id = int(row["clip_id"])
            item = torch.load(
                self.graph_dir / f"{clip_id}.pt", map_location="cpu", weights_only=False
            )
            self.graphs.append((item["x"], item["edge_index"]))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def make_context(row: pd.Series) -> str:
        active = [tag for tag in CONTEXT_TAGS if int(row[tag]) == 1]
        if not active:
            return "This music features no described instruments or vocals."
        if len(active) == 1:
            return f"This music features {active[0]}."
        if len(active) == 2:
            return f"This music features {active[0]} and {active[1]}."
        return f"This music features {', '.join(active[:-1])}, and {active[-1]}."

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        clip_id = int(row["clip_id"])
        graph_x, graph_edge_index = self.graphs[index]
        text = self.contexts[index]
        return {
            "clip_id": clip_id,
            "text": text,
            "graph_x": graph_x,
            "graph_edge_index": graph_edge_index,
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": torch.tensor([float(row[tag]) for tag in TARGET_TAGS], dtype=torch.float32),
        }


def collate_batch(items: list[dict]) -> dict:
    graph_batch = Batch.from_data_list([
        Data(x=item["graph_x"], edge_index=item["graph_edge_index"]) for item in items
    ])
    max_length = max(item["input_ids"].numel() for item in items)
    input_ids = []
    attention_masks = []
    for item in items:
        padding = max_length - item["input_ids"].numel()
        input_ids.append(nn.functional.pad(item["input_ids"], (0, padding), value=0))
        attention_masks.append(nn.functional.pad(item["attention_mask"], (0, padding), value=0))
    return {
        "graph_x": graph_batch.x,
        "graph_edge_index": graph_batch.edge_index,
        "graph_batch": graph_batch.batch,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack([item["labels"] for item in items]),
        "clip_ids": [item["clip_id"] for item in items],
        "texts": [item["text"] for item in items],
    }


def model_inputs(batch: dict, device) -> dict:
    return {
        "graph_x": batch["graph_x"].to(device),
        "graph_edge_index": batch["graph_edge_index"].to(device),
        "graph_batch": batch["graph_batch"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_embeddings: bool = False) -> dict:
    model.eval()
    losses = []
    probabilities = []
    labels = []
    clip_ids = []
    embeddings = []
    for batch in loader:
        logits, extras = model(**model_inputs(batch, device))
        target = batch["labels"].to(device)
        losses.append(criterion(logits, target).item())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())
        clip_ids.extend(batch["clip_ids"])
        if collect_embeddings:
            embeddings.append(extras["z"].cpu().numpy())
    return {
        "loss": float(np.mean(losses)),
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "clip_ids": np.asarray(clip_ids, dtype=np.int64),
        "embeddings": np.concatenate(embeddings) if embeddings else None,
    }


def train_epochs(model, train_loader, val_loader, criterion, device, checkpoint_path: Path,
                  history: dict, phase: str, epochs: int, learning_rate: float,
                  weight_decay: float, patience: int, best_score: float) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stale = 0

    for _ in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(**model_inputs(batch, device))
            loss = criterion(logits, batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Task 3 {phase} produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        validation = evaluate(model, val_loader, criterion, device)
        metrics = compute_multilabel_metrics(
            validation["labels"], validation["probabilities"], threshold=0.5
        )
        epoch = len(history["epoch"]) + 1
        history["epoch"].append(epoch)
        history["phase"].append(phase)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(validation["loss"])
        history["val_macro_f1"].append(metrics["macro_f1"])
        history["val_micro_f1"].append(metrics["micro_f1"])
        print(
            f"Task 3 {history['mode']} epoch {epoch:02d} ({phase}): "
            f"loss={history['train_loss'][-1]:.4f}, "
            f"val macro-F1={metrics['macro_f1']:.4f}, "
            f"val micro-F1={metrics['micro_f1']:.4f}",
            flush=True,
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            stale = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return best_score


def set_bert_trainability(model: FusionModel, mode: str) -> None:
    if not hasattr(model, "bert_encoder"):
        return
    for parameter in model.bert_encoder.bert.parameters():
        parameter.requires_grad = mode == "all"
    if mode == "last_layer":
        for parameter in model.bert_encoder.bert.transformer.layer[-1].parameters():
            parameter.requires_grad = True


def train_mode(mode: str, loaders: dict, feature_dim: int, criterion, config: dict,
               device, args, checkpoint_dir: Path, embedding_path: Path,
               provenance: dict) -> dict:
    schedule = config["task3"]["training"]
    model = FusionModel(
        gnn_input_dim=feature_dim,
        num_labels=len(TARGET_TAGS),
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]),
        freeze_bert=mode != "gnn_only",
        fusion_mode=mode,
        projection_dim=int(config["fusion"]["projection_dim"]),
        num_attention_heads=int(config["fusion"]["num_attention_heads"]),
        dropout=float(config["fusion"]["dropout"]),
    ).to(device)

    checkpoint_path = checkpoint_dir / f"task3_{mode}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = {
        "mode": mode, "epoch": [], "phase": [], "train_loss": [], "val_loss": [],
        "val_macro_f1": [], "val_micro_f1": [],
    }
    started = time.time()

    if mode == "gnn_only":
        best_score = train_epochs(
            model, loaders["train"], loaders["val"], criterion, device,
            checkpoint_path, history, "full", args.gnn_epochs,
            float(schedule["gnn_learning_rate"]),
            float(schedule["weight_decay"]), 8, -1.0,
        )
    else:
        set_bert_trainability(model, "frozen")
        best_score = train_epochs(
            model, loaders["train"], loaders["val"], criterion, device,
            checkpoint_path, history, "bert_frozen", args.frozen_epochs,
            float(schedule["frozen_learning_rate"]),
            float(schedule["weight_decay"]), 3, -1.0,
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
        )
        set_bert_trainability(model, "last_layer")
        best_score = train_epochs(
            model, loaders["train"], loaders["val"], criterion, device,
            checkpoint_path, history, "partial_fine_tune", args.finetune_epochs,
            float(schedule["finetune_learning_rate"]),
            float(schedule["weight_decay"]), 2, best_score,
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    )
    validation = evaluate(model, loaders["val"], criterion, device)
    threshold = float(tune_threshold(
        validation["labels"], validation["probabilities"], metric="macro_f1"
    ))
    test = evaluate(
        model, loaders["test"], criterion, device,
        collect_embeddings=mode == "cross_attention",
    )
    metrics = compute_multilabel_metrics(
        test["labels"], test["probabilities"], threshold=threshold, tag_names=TARGET_TAGS
    )
    result = {
        "mode": mode,
        "best_validation_macro_f1_at_0.5": best_score,
        "threshold_tuned_on_validation": threshold,
        "test_loss": test["loss"],
        "test_metrics": metrics,
        "history": history,
        "training_time_seconds": time.time() - started,
        "parameter_counts": model.count_parameters(),
        "provenance": provenance,
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "mode": mode,
        "feature_dim": feature_dim,
        "target_tags": TARGET_TAGS,
        "threshold": threshold,
        "split_protocol": "official MTAT shards 0-b/c/d-f",
        "seed": int(config["seed"]),
        "provenance": provenance,
    }, checkpoint_path)

    if mode == "cross_attention":
        np.savez_compressed(
            embedding_path,
            embeddings=test["embeddings"],
            labels=test["labels"],
            probabilities=test["probabilities"],
            clip_ids=test["clip_ids"],
        )
    return result


def make_tsne(config: dict) -> dict:
    archive = np.load(ROOT / config["paths"]["results"] / "task3_test_embeddings.npz")
    embeddings = archive["embeddings"]
    labels = archive["labels"]
    plots_dir = ROOT / config["paths"]["plots"]
    seed = int(config["seed"])
    metadata = {}

    for name, indices, class_names in [
        ("genre", list(range(len(TARGET_GENRE_TAGS))), TARGET_GENRE_TAGS),
        ("mood", list(range(len(TARGET_GENRE_TAGS), len(TARGET_TAGS))), TARGET_MOOD_TAGS),
    ]:
        subset_labels = labels[:, indices]
        mask = subset_labels.sum(axis=1) == 1
        selected = np.flatnonzero(mask)
        if len(selected) > 3000:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(selected, size=3000, replace=False))
        if len(selected) < 10:
            metadata[name] = {"plotted": False, "reason": "fewer than 10 single-active examples"}
            continue
        class_index = subset_labels[selected].argmax(axis=1)
        perplexity = min(int(config["evaluation"]["tsne_perplexity"]), max(5, (len(selected) - 1) // 3))
        points = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=int(config["evaluation"]["tsne_n_iter"]),
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(embeddings[selected])
        plot_tsne(
            points, class_index, class_names,
            plots_dir / f"task3_tsne_{name}.png",
            f"Task 3 fused embeddings by ground-truth {name}",
        )
        metadata[name] = {
            "plotted": True,
            "samples": len(selected),
            "selection": f"exactly one active ground-truth {name} tag",
            "perplexity": perplexity,
            "iterations": int(config["evaluation"]["tsne_n_iter"]),
            "seed": seed,
        }
    return metadata


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    schedule = config["task3"]["training"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", default="gnn_only,bert_only,concat,cross_attention",
        help="Comma-separated subset of modes to train",
    )
    parser.add_argument("--gnn-epochs", type=int, default=int(schedule["gnn_epochs"]))
    parser.add_argument("--frozen-epochs", type=int, default=int(schedule["frozen_epochs"]))
    parser.add_argument("--finetune-epochs", type=int, default=int(schedule["finetune_epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(schedule["batch_size"]))
    args = parser.parse_args()

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = ROOT / "data/processed/mtat_graphs"
    if not any(graph_dir.glob("*.pt")):
        raise FileNotFoundError("Run scripts/preprocess_mtat.py before Task 3")

    annotations = pd.read_csv(ROOT / "data/raw/magnatagatune/annotations_final.csv", sep="\t")
    missing = [tag for tag in CONTEXT_TAGS + TARGET_TAGS if tag not in annotations.columns]
    if missing:
        raise ValueError(f"Required tags are missing: {missing}")
    if set(CONTEXT_TAGS) & set(TARGET_TAGS):
        raise ValueError("Context and target tag sets must remain lexically disjoint")

    split_payload = json.loads((ROOT / "data/splits/mtat_splits.json").read_text(encoding="utf-8"))
    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    datasets = {
        name: MTATContextDataset(
            annotations, split_payload[name], graph_dir, tokenizer,
            int(config["bert"]["max_length"]),
        )
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_batch,
        )
        for name, dataset in datasets.items()
    }
    print("Task 3 usable split counts:", {name: len(data) for name, data in datasets.items()}, flush=True)

    first_graph = torch.load(
        graph_dir / f"{int(datasets['train'].rows[0]['clip_id'])}.pt",
        map_location="cpu", weights_only=False,
    )
    feature_dim = int(first_graph["x"].shape[1])
    train_rows = pd.DataFrame(datasets["train"].rows)
    positive = train_rows[TARGET_TAGS].sum(axis=0).to_numpy(dtype=np.float32)
    negative = len(train_rows) - positive
    positive_weight = np.clip(negative / np.maximum(positive, 1), 1, 20)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))

    results_dir = ROOT / config["paths"]["results"]
    plots_dir = ROOT / config["paths"]["plots"]
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    requested_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    valid_modes = {"gnn_only", "bert_only", "concat", "cross_attention"}
    if not set(requested_modes) <= valid_modes:
        raise ValueError(f"Unknown Task 3 modes: {set(requested_modes) - valid_modes}")
    if len(requested_modes) != len(set(requested_modes)):
        raise ValueError("Task 3 modes must not be repeated")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_results_dir = results_dir / "task3_runs" / run_id
    run_checkpoint_dir = ROOT / config["paths"]["checkpoints"] / "task3_runs" / run_id
    run_results_dir.mkdir(parents=True, exist_ok=False)
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    first_metadata = first_graph.get("metadata", {})
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": int(config["seed"]),
        "split_file_sha256": file_sha256(ROOT / "data/splits/mtat_splits.json"),
        "split_counts": {name: len(data) for name, data in datasets.items()},
        "preprocessing_version": first_metadata.get("preprocessing_version"),
        "preprocessing_signature": first_metadata.get("preprocessing_signature"),
        "feature_dim": feature_dim,
        "target_tags": TARGET_TAGS,
        "bert": config["bert"],
        "gnn": config["gnn"],
        "fusion": config["fusion"],
        "training": {
            **schedule,
            "gnn_epochs": args.gnn_epochs,
            "frozen_epochs": args.frozen_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
        },
    }
    provenance = {
        "run_id": run_id,
        "experiment_signature": stable_hash(protocol),
        **protocol,
    }

    mode_results = {}
    for mode in requested_modes:
        mode_provenance = {**provenance, "mode": mode}
        result = train_mode(
            mode, loaders, feature_dim, criterion, config, device, args,
            run_checkpoint_dir, run_results_dir / "task3_test_embeddings.npz",
            mode_provenance,
        )
        mode_results[mode] = result
        (run_results_dir / f"task3_{mode}_results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        plot_f1_history(
            result["history"],
            plots_dir / (
                f"task3_{mode}_learning_curve.png"
                if set(requested_modes) == valid_modes
                else f"task3_{mode}_{run_id}_learning_curve.png"
            ),
            f"Task 3 {mode.replace('_', ' ')} validation F1",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(requested_modes) == valid_modes:
        for mode, result in mode_results.items():
            (results_dir / f"task3_{mode}_results.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            shutil.copy2(
                run_checkpoint_dir / f"task3_{mode}_best.pt",
                ROOT / config["paths"]["checkpoints"] / f"task3_{mode}_best.pt",
            )
        shutil.copy2(
            run_results_dir / "task3_test_embeddings.npz",
            results_dir / "task3_test_embeddings.npz",
        )
        order = ["bert_only", "gnn_only", "concat", "cross_attention"]
        macro_values = [mode_results[mode]["test_metrics"]["macro_f1"] for mode in order]
        auc_values = [mode_results[mode]["test_metrics"]["mean_auc_pr"] for mode in order]
        labels = ["BERT only", "GNN only", "Concat", "Cross-attention"]
        plot_model_comparison(labels, macro_values, plots_dir / "task3_ablation_macro_f1.png",
                              "Task 3 four-way ablation: Macro-F1")
        plot_model_comparison(labels, auc_values, plots_dir / "task3_ablation_auc_pr.png",
                              "Task 3 four-way ablation: mean AUC-PR", ylabel="Mean AUC-PR")

        cross_tags = mode_results["cross_attention"]["test_metrics"]["per_tag"]
        plot_per_tag_metrics(
            [item["tag"] for item in cross_tags],
            [item["f1"] for item in cross_tags],
            [item["auc_pr"] for item in cross_tags],
            plots_dir / "task3_per_tag_metrics.png",
        )
        tsne_metadata = make_tsne(config)
        consolidated = {
            "task": "Task 3: disjoint context-to-target GNN-BERT fusion on MTAT",
            "dataset": "MagnaTagATune",
            "context_tags": CONTEXT_TAGS,
            "target_genre_tags": TARGET_GENRE_TAGS,
            "target_mood_tags": TARGET_MOOD_TAGS,
            "split_counts": {name: len(data) for name, data in datasets.items()},
            "split_audit": split_payload["audit"],
            "leakage_statement": (
                "Context and target tags are lexically disjoint. Both originate from the same "
                "annotation corpus, so indirect statistical associations remain."
            ),
            "results": {mode: mode_results[mode] for mode in order},
            "tsne": tsne_metadata,
            "device": str(device),
            "seed": int(config["seed"]),
            "provenance": provenance,
        }
        (results_dir / "task3_results.json").write_text(
            json.dumps(consolidated, indent=2), encoding="utf-8"
        )
        print(json.dumps({mode: mode_results[mode]["test_metrics"] for mode in order}, indent=2))
    else:
        print(
            "Task 3 partial run saved separately at "
            f"{run_results_dir.relative_to(ROOT)}. Canonical Task 3 results were not changed."
        )


if __name__ == "__main__":
    main()
