"""Task 2: compare GraphSAGE and a mel-spectrogram CNN on GTZAN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_singlelabel_metrics
from src.gnn_model import GNNClassifier
from src.plotting import plot_confusion_matrix, plot_f1_history, plot_model_comparison
from src.train import set_seed


class GraphDataset(Dataset):
    def __init__(self, track_ids: list[str], graph_dir: Path):
        self.track_ids = track_ids
        self.graph_dir = graph_dir

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> Data:
        track_id = self.track_ids[index]
        item = torch.load(self.graph_dir / f"{track_id}.pt", map_location="cpu", weights_only=False)
        graph = Data(x=item["x"], edge_index=item["edge_index"], edge_attr=item.get("edge_attr"))
        graph.y = item["y"].long().reshape(1)
        graph.track_id = track_id
        return graph


class MelDataset(Dataset):
    def __init__(self, track_ids: list[str], mel_dir: Path):
        self.track_ids = track_ids
        self.mel_dir = mel_dir

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int):
        item = torch.load(
            self.mel_dir / f"{self.track_ids[index]}.pt",
            map_location="cpu",
            weights_only=False,
        )
        return item["log_mel"].float().unsqueeze(0), item["y"].long()


def mel_collate(batch):
    max_frames = max(features.shape[-1] for features, _ in batch)
    padded = []
    for features, _ in batch:
        if features.shape[-1] < max_frames:
            features = nn.functional.pad(features, (0, max_frames - features.shape[-1]))
        padded.append(features)
    return torch.stack(padded), torch.stack([label for _, label in batch])


class MelCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, features):
        return self.classifier(self.features(features))


def forward_batch(model, batch, mode: str, device):
    if mode == "gnn":
        batch = batch.to(device)
        return model(batch.x, batch.edge_index, batch.batch), batch.y
    features, labels = batch
    return model(features.to(device)), labels.to(device)


@torch.no_grad()
def evaluate(model, loader, mode: str, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    labels = []
    for batch in loader:
        logits, target = forward_batch(model, batch, mode, device)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(target.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
        predictions,
        labels,
    )


def train(model, train_loader, val_loader, mode: str, device, checkpoint_path: Path,
          epochs: int, learning_rate: float, weight_decay: float,
          patience: int) -> tuple[dict, float, int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(3, patience // 3)
    )
    criterion = nn.CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_accuracy": [], "val_macro_f1": []}
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        correct = 0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward_batch(model, batch, mode, device)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{mode} produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += labels.numel()

        val_accuracy, val_f1, _, _ = evaluate(model, val_loader, mode, device)
        scheduler.step(val_f1)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["train_accuracy"].append(correct / seen)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_f1)
        print(
            f"Task 2 {mode} epoch {epoch:03d}: loss={history['train_loss'][-1]:.4f}, "
            f"train accuracy={history['train_accuracy'][-1]:.3f}, "
            f"val accuracy={val_accuracy:.3f}, val macro-F1={val_f1:.3f}",
            flush=True,
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    )
    return history, best_score, best_epoch, time.time() - started


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    schedule = config["task2"]["training"]
    graph_config = config["task2"]["graph_model"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-epochs", type=int, default=int(schedule["gnn_epochs"]))
    parser.add_argument("--cnn-epochs", type=int, default=int(schedule["cnn_epochs"]))
    args = parser.parse_args()

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((ROOT / "data/splits/gtzan_splits.json").read_text(encoding="utf-8"))
    genres = splits["genres"]
    graph_dir = ROOT / "data/processed/gtzan_graphs"
    mel_dir = ROOT / "data/processed/gtzan_mels"
    for track_id in splits["train"] + splits["val"] + splits["test"]:
        if not (graph_dir / f"{track_id}.pt").exists() or not (mel_dir / f"{track_id}.pt").exists():
            raise FileNotFoundError("Run scripts/preprocess_gtzan.py before Task 2")

    graph_loaders = {
        name: GraphDataLoader(
            GraphDataset(splits[name], graph_dir),
            batch_size=int(schedule["gnn_batch_size"]),
            shuffle=name == "train",
            num_workers=int(config["training"]["num_workers"]),
        )
        for name in ("train", "val", "test")
    }
    mel_loaders = {
        name: DataLoader(
            MelDataset(splits[name], mel_dir),
            batch_size=int(schedule["cnn_batch_size"]),
            shuffle=name == "train", collate_fn=mel_collate,
            num_workers=int(config["training"]["num_workers"]),
        )
        for name in ("train", "val", "test")
    }

    sample = torch.load(graph_dir / f"{splits['train'][0]}.pt", map_location="cpu", weights_only=False)
    feature_dim = int(sample["x"].shape[1])
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    results_dir = ROOT / config["paths"]["results"]
    plots_dir = ROOT / config["paths"]["plots"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    gnn = GNNClassifier(
        input_dim=feature_dim,
        hidden_dim=int(graph_config["hidden_dim"]),
        num_layers=int(graph_config["num_layers"]),
        num_classes=len(genres),
        dropout=float(graph_config["dropout"]),
    ).to(device)
    gnn_path = checkpoint_dir / "task2_gnn_best.pt"
    gnn_history, gnn_best, gnn_epoch, gnn_time = train(
        gnn, graph_loaders["train"], graph_loaders["val"], "gnn", device,
        gnn_path, args.gnn_epochs, learning_rate=float(schedule["learning_rate"]),
        weight_decay=float(schedule["weight_decay"]),
        patience=int(schedule["gnn_patience"]),
    )
    gnn_accuracy, gnn_f1, gnn_predictions, gnn_labels = evaluate(
        gnn, graph_loaders["test"], "gnn", device
    )
    gnn_metrics = compute_singlelabel_metrics(gnn_labels, gnn_predictions, genres)
    torch.save({
        "model_state_dict": gnn.state_dict(),
        "feature_dim": feature_dim,
        "model_config": graph_config,
        "genres": genres,
        "split_protocol": splits["protocol"],
        "seed": int(config["seed"]),
    }, gnn_path)

    cnn = MelCNN(len(genres)).to(device)
    cnn_path = checkpoint_dir / "task2_cnn_best.pt"
    cnn_history, cnn_best, cnn_epoch, cnn_time = train(
        cnn, mel_loaders["train"], mel_loaders["val"], "cnn", device,
        cnn_path, args.cnn_epochs, learning_rate=float(schedule["learning_rate"]),
        weight_decay=float(schedule["weight_decay"]),
        patience=int(schedule["cnn_patience"]),
    )
    cnn_accuracy, cnn_f1, cnn_predictions, cnn_labels = evaluate(
        cnn, mel_loaders["test"], "cnn", device
    )
    cnn_metrics = compute_singlelabel_metrics(cnn_labels, cnn_predictions, genres)
    torch.save({
        "model_state_dict": cnn.state_dict(),
        "mel_bins": int(config["audio"]["mel_bins"]),
        "genres": genres,
        "split_protocol": splits["protocol"],
        "seed": int(config["seed"]),
    }, cnn_path)

    result = {
        "task": "Task 2: GTZAN audio-only GNN versus CNN",
        "dataset": "GTZAN",
        "split_counts": {name: len(splits[name]) for name in ("train", "val", "test")},
        "split_protocol": splits["protocol"],
        "gnn": {
            "test_metrics": gnn_metrics,
            "best_validation_macro_f1": gnn_best,
            "best_epoch": gnn_epoch,
            "training_time_seconds": gnn_time,
            "feature_dim": feature_dim,
            "history": gnn_history,
        },
        "cnn": {
            "test_metrics": cnn_metrics,
            "best_validation_macro_f1": cnn_best,
            "best_epoch": cnn_epoch,
            "training_time_seconds": cnn_time,
            "mel_bins": int(config["audio"]["mel_bins"]),
            "history": cnn_history,
        },
        "device": str(device),
        "seed": int(config["seed"]),
    }
    (results_dir / "task2_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_f1_history(gnn_history, plots_dir / "task2_gnn_learning_curve.png", "Task 2 GraphSAGE validation F1")
    plot_f1_history(cnn_history, plots_dir / "task2_cnn_learning_curve.png", "Task 2 CNN validation F1")
    plot_confusion_matrix(gnn_metrics["confusion_matrix"], genres,
                          plots_dir / "task2_gnn_confusion_matrix.png", "Task 2 GraphSAGE confusion matrix")
    plot_confusion_matrix(cnn_metrics["confusion_matrix"], genres,
                          plots_dir / "task2_cnn_confusion_matrix.png", "Task 2 CNN confusion matrix")
    plot_model_comparison(["GraphSAGE", "CNN"], [gnn_f1, cnn_f1],
                          plots_dir / "task2_model_comparison.png", "Task 2 model comparison")
    print(json.dumps({"gnn": gnn_metrics, "cnn": cnn_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
