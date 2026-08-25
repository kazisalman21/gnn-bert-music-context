"""Task 1: DistilBERT caption-to-tag baseline on MusicCaps."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bert_encoder import BertEncoder
from src.evaluate import compute_multilabel_metrics, tune_threshold
from src.plotting import plot_f1_history
from src.train import set_seed


def parse_tags(value) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(tag).strip().lower() for tag in parsed if str(tag).strip()]


class CaptionTagDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, ids: list[str], tag_to_index: dict[str, int],
                 tokenizer, max_length: int):
        indexed = metadata.set_index("ytid", drop=False)
        missing = [ytid for ytid in ids if ytid not in indexed.index]
        if missing:
            raise ValueError(f"Missing {len(missing)} MusicCaps IDs from metadata")
        self.rows = indexed.loc[ids].reset_index(drop=True)
        self.tag_to_index = tag_to_index
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        tokens = self.tokenizer(
            str(row["caption"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.zeros(len(self.tag_to_index), dtype=torch.float32)
        for tag in parse_tags(row["aspect_list"]):
            if tag in self.tag_to_index:
                labels[self.tag_to_index[tag]] = 1.0
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "ytid": str(row["ytid"]),
            "caption": str(row["caption"]),
        }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    losses = []
    probabilities = []
    labels = []
    ids = []
    captions = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        target = batch["labels"].to(device)
        losses.append(criterion(logits, target).item())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())
        ids.extend(batch["ytid"])
        captions.extend(batch["caption"])
    return (
        float(np.mean(losses)),
        np.concatenate(probabilities),
        np.concatenate(labels),
        ids,
        captions,
    )


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                phase: str, epochs: int, history: dict, checkpoint_path: Path,
                best_score: float, patience: int = 4) -> float:
    stale_epochs = 0
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Task 1 produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        val_loss, val_probs, val_labels, _, _ = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_multilabel_metrics(val_labels, val_probs, threshold=0.5)
        epoch_number = len(history["epoch"]) + 1
        history["epoch"].append(epoch_number)
        history["phase"].append(phase)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        print(
            f"Task 1 epoch {epoch_number:02d} ({phase}): "
            f"loss={history['train_loss'][-1]:.4f}, "
            f"val macro-F1={val_metrics['macro_f1']:.4f}, "
            f"val micro-F1={val_metrics['micro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return best_score


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    schedule = config["task1"]["training"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-epochs", type=int, default=int(schedule["frozen_epochs"]))
    parser.add_argument("--finetune-epochs", type=int, default=int(schedule["finetune_epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(schedule["batch_size"]))
    args = parser.parse_args()

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = ROOT / config["paths"]["data_splits"]
    splits = {
        name: json.loads((split_dir / f"musiccaps_{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test")
    }
    vocab_payload = json.loads((split_dir / "musiccaps_tag_vocab.json").read_text(encoding="utf-8"))
    tag_vocabulary = vocab_payload["tags"]
    tag_to_index = {tag: index for index, tag in enumerate(tag_vocabulary)}
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv", dtype={"ytid": str})

    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    datasets = {
        name: CaptionTagDataset(
            metadata, ids, tag_to_index, tokenizer, int(config["bert"]["max_length"])
        )
        for name, ids in splits.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=int(config["training"]["num_workers"]),
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size * 2, shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size * 2, shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        ),
    }

    model = BertEncoder(
        num_labels=len(tag_vocabulary),
        model_name=config["bert"]["model_name"],
        freeze_bert=True,
        hidden_dim=int(config["bert"]["hidden_dim"]),
    ).to(device)

    positive_counts = np.zeros(len(tag_vocabulary), dtype=np.float32)
    for row_index in range(len(datasets["train"])):
        positive_counts += datasets["train"][row_index]["labels"].numpy()
    negative_counts = len(datasets["train"]) - positive_counts
    positive_weight = np.clip(negative_counts / np.maximum(positive_counts, 1), 1, 20)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))

    checkpoints_dir = ROOT / config["paths"]["checkpoints"]
    results_dir = ROOT / config["paths"]["results"]
    plots_dir = ROOT / config["paths"]["plots"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "task1_best.pt"

    history = {
        "epoch": [], "phase": [], "train_loss": [], "val_loss": [],
        "val_macro_f1": [], "val_micro_f1": [],
    }
    started = time.time()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(schedule["frozen_learning_rate"]),
        weight_decay=float(schedule["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.frozen_epochs, 1))
    best_score = train_phase(
        model, loaders["train"], loaders["val"], criterion, optimizer, scheduler,
        device, "frozen", args.frozen_epochs, history, checkpoint_path, -1.0,
        patience=int(schedule["early_stopping_patience"]),
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"])
    for parameter in model.bert.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(schedule["finetune_learning_rate"]),
        weight_decay=float(schedule["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.finetune_epochs, 1))
    best_score = train_phase(
        model, loaders["train"], loaders["val"], criterion, optimizer, scheduler,
        device, "fine_tune", args.finetune_epochs, history, checkpoint_path, best_score,
        patience=int(schedule["early_stopping_patience"]),
    )

    best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(best_state)
    val_loss, val_probs, val_labels, _, _ = evaluate(model, loaders["val"], criterion, device)
    threshold = float(tune_threshold(val_labels, val_probs, metric="macro_f1"))
    test_loss, test_probs, test_labels, test_ids, test_captions = evaluate(
        model, loaders["test"], criterion, device
    )
    metrics = compute_multilabel_metrics(
        test_labels, test_probs, threshold=threshold, tag_names=tag_vocabulary
    )

    examples = []
    for index in range(min(5, len(test_ids))):
        top_indices = np.argsort(test_probs[index])[-5:][::-1]
        true_indices = np.flatnonzero(test_labels[index] > 0)
        examples.append({
            "ytid": test_ids[index],
            "caption": test_captions[index],
            "true_tags_in_vocabulary": [tag_vocabulary[i] for i in true_indices],
            "top_5_predictions": [
                {"tag": tag_vocabulary[i], "probability": float(test_probs[index, i])}
                for i in top_indices
            ],
        })

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": config["bert"]["model_name"],
        "tag_vocabulary": tag_vocabulary,
        "threshold": threshold,
        "split_protocol": "AudioSet eval test; seeded 90/10 non-eval train/validation",
        "seed": seed,
    }
    torch.save(final_checkpoint, checkpoint_path)

    result = {
        "task": "Task 1: MusicCaps caption to aspect-tag proxy",
        "dataset": "MusicCaps",
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "vocabulary_size": len(tag_vocabulary),
        "minimum_training_frequency": vocab_payload["min_freq"],
        "threshold_tuned_on_validation": threshold,
        "best_validation_macro_f1": best_score,
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "test_metrics": metrics,
        "history": history,
        "examples": examples,
        "training_time_seconds": time.time() - started,
        "device": str(device),
        "seed": seed,
    }
    (results_dir / "task1_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_f1_history(history, plots_dir / "task1_f1_curves.png", "Task 1 validation F1 by epoch")
    print(json.dumps({"test_metrics": metrics, "threshold": threshold}, indent=2), flush=True)


if __name__ == "__main__":
    main()
