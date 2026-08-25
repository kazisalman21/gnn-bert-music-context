"""Inference smoke tests using real held-out records and trained checkpoints."""

from __future__ import annotations

import json
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_task1 import CaptionTagDataset
from scripts.run_task2 import MelCNN
from scripts.run_task3 import MTATContextDataset, collate_batch, model_inputs
from scripts.run_task4 import (
    Task4Dataset,
    collate_batch as task4_collate_batch,
    model_inputs as task4_model_inputs,
)
from src.bert_encoder import BertEncoder
from src.contrastive import ContrastiveDualEncoder, compute_retrieval_metrics
from src.dataset import TARGET_TAGS
from src.fusion_model import FusionModel
from src.gnn_model import GNNClassifier


def task1_check(config: dict, device: torch.device) -> None:
    checkpoint_path = ROOT / "checkpoints/task1_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tags = checkpoint["tag_vocabulary"]
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint["model_name"])
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv")
    test_ids = json.loads((ROOT / "data/splits/musiccaps_test.json").read_text(encoding="utf-8"))
    dataset = CaptionTagDataset(
        metadata, [test_ids[0]], {tag: index for index, tag in enumerate(tags)},
        tokenizer, int(config["bert"]["max_length"]),
    )
    sample = dataset[0]
    model = BertEncoder(
        len(tags), checkpoint["model_name"], freeze_bert=True,
        hidden_dim=int(config["bert"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            sample["input_ids"].unsqueeze(0).to(device),
            sample["attention_mask"].unsqueeze(0).to(device),
        )
    assert logits.shape == (1, len(tags)) and torch.isfinite(logits).all()
    print(f"PASS Task 1: real MusicCaps {sample['ytid']} -> {tuple(logits.shape)}")


def task2_check(config: dict, device: torch.device) -> None:
    splits = json.loads((ROOT / "data/splits/gtzan_splits.json").read_text(encoding="utf-8"))
    track_id = splits["test"][0]
    graph_item = torch.load(
        ROOT / f"data/processed/gtzan_graphs/{track_id}.pt", map_location="cpu", weights_only=False
    )
    graph = Batch.from_data_list([
        Data(x=graph_item["x"], edge_index=graph_item["edge_index"])
    ]).to(device)
    gnn_checkpoint = torch.load(ROOT / "checkpoints/task2_gnn_best.pt", map_location="cpu", weights_only=True)
    graph_config = gnn_checkpoint.get("model_config", config["task2"]["graph_model"])
    gnn = GNNClassifier(
        input_dim=int(gnn_checkpoint["feature_dim"]),
        hidden_dim=int(graph_config["hidden_dim"]),
        num_layers=int(graph_config["num_layers"]),
        num_classes=len(gnn_checkpoint["genres"]),
        dropout=float(graph_config["dropout"]),
    ).to(device)
    gnn.load_state_dict(gnn_checkpoint["model_state_dict"])
    gnn.eval()
    with torch.no_grad():
        gnn_logits = gnn(graph.x, graph.edge_index, graph.batch)

    mel_item = torch.load(
        ROOT / f"data/processed/gtzan_mels/{track_id}.pt", map_location="cpu", weights_only=False
    )
    cnn_checkpoint = torch.load(ROOT / "checkpoints/task2_cnn_best.pt", map_location="cpu", weights_only=True)
    cnn = MelCNN(len(cnn_checkpoint["genres"])).to(device)
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn.eval()
    with torch.no_grad():
        cnn_logits = cnn(mel_item["log_mel"].float().unsqueeze(0).unsqueeze(0).to(device))

    assert gnn_logits.shape == cnn_logits.shape == (1, 10)
    assert torch.isfinite(gnn_logits).all() and torch.isfinite(cnn_logits).all()
    print(f"PASS Task 2: real GTZAN {track_id} -> GNN/CNN {tuple(gnn_logits.shape)}")


def task3_check(config: dict, device: torch.device) -> None:
    checkpoint_path = ROOT / "checkpoints/task3_cross_attention_best.pt"
    if not checkpoint_path.exists():
        tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
        annotations = pd.read_csv(ROOT / "data/raw/magnatagatune/annotations_final.csv", sep="\t")
        available = sorted(
            int(path.stem) for path in (ROOT / "data/processed/mtat_graphs").glob("*.pt")
        )[:2]
        if len(available) < 2:
            raise RuntimeError("Two real MTAT graphs are required for the Task 3 smoke check")
        dataset = MTATContextDataset(
            annotations, available, ROOT / "data/processed/mtat_graphs", tokenizer,
            int(config["bert"]["max_length"]),
        )
        batch = collate_batch([dataset[0], dataset[1]])
        for mode in ("gnn_only", "bert_only", "concat", "cross_attention"):
            model = FusionModel(
                gnn_input_dim=int(batch["graph_x"].shape[1]), num_labels=len(TARGET_TAGS),
                gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
                gnn_num_layers=int(config["gnn"]["num_layers"]),
                bert_model_name=config["bert"]["model_name"],
                bert_hidden_dim=int(config["bert"]["hidden_dim"]), freeze_bert=True,
                fusion_mode=mode, projection_dim=int(config["fusion"]["projection_dim"]),
                num_attention_heads=int(config["fusion"]["num_attention_heads"]),
                dropout=float(config["fusion"]["dropout"]),
            ).to(device)
            model.train()
            logits, _ = model(**model_inputs(batch, device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].to(device)
            )
            loss.backward()
            assert logits.shape == (2, len(TARGET_TAGS)) and torch.isfinite(loss)
            assert any(
                parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
            )
            print(f"PASS Task 3 {mode}: two real MTAT clips -> loss={loss.item():.4f}")
            del model, logits, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    splits = json.loads((ROOT / "data/splits/mtat_splits.json").read_text(encoding="utf-8"))
    annotations = pd.read_csv(ROOT / "data/raw/magnatagatune/annotations_final.csv", sep="\t")
    dataset = MTATContextDataset(
        annotations, splits["test"], ROOT / "data/processed/mtat_graphs", tokenizer,
        int(config["bert"]["max_length"]),
    )
    sample = dataset[0]
    batch = collate_batch([sample])
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
    with torch.no_grad():
        logits, extras = model(**model_inputs(batch, device))
    assert logits.shape == (1, len(TARGET_TAGS)) and torch.isfinite(logits).all()
    assert "attention_weights" in extras
    print(f"PASS Task 3: real MTAT {sample['clip_id']} -> {tuple(logits.shape)}")


def task4_check(config: dict, device: torch.device) -> None:
    checkpoint = torch.load(
        ROOT / "checkpoints/task4_contrastive_best.pt",
        map_location="cpu", weights_only=True,
    )
    tokenizer = DistilBertTokenizer.from_pretrained(config["bert"]["model_name"])
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv")
    full_test_ids = json.loads(
        (ROOT / "data/splits/musiccaps_test.json").read_text(encoding="utf-8")
    )
    test_range = config["task4"]["protocol_ranges"]["test"]
    start = int(test_range["start"])
    requested_ids = full_test_ids[start:start + int(test_range["limit"])]
    dataset = Task4Dataset(
        metadata, requested_ids, "test",
        ROOT / "data/processed/musiccaps_graphs",
        tokenizer, int(config["bert"]["max_length"]),
    )
    if len(dataset) < 2:
        raise RuntimeError("Two final-protocol MusicCaps graphs are required for Task 4")
    batch = task4_collate_batch([dataset[0], dataset[1]])
    model = ContrastiveDualEncoder(
        gnn_input_dim=int(checkpoint["feature_dim"]),
        gnn_hidden_dim=int(config["gnn"]["hidden_dim"]),
        gnn_num_layers=int(config["gnn"]["num_layers"]),
        bert_model_name=config["bert"]["model_name"],
        bert_hidden_dim=int(config["bert"]["hidden_dim"]),
        freeze_bert=bool(checkpoint["freeze_bert"]),
        projection_dim=int(checkpoint["projection_dim"]),
        dropout=float(config["gnn"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        graph_embeddings, text_embeddings = model(
            **task4_model_inputs(batch, device)
        )
    metrics = compute_retrieval_metrics(graph_embeddings, text_embeddings)
    assert graph_embeddings.shape == text_embeddings.shape == (
        2, int(checkpoint["projection_dim"])
    )
    assert torch.isfinite(graph_embeddings).all()
    assert torch.isfinite(text_embeddings).all()
    assert torch.allclose(
        graph_embeddings.norm(dim=1), torch.ones(2, device=device), atol=1e-5
    )
    assert all(
        0.0 <= value <= 1.0
        for direction in metrics.values()
        for value in direction.values()
    )
    print(
        f"PASS Task 4: two final-protocol MusicCaps clips -> "
        f"{tuple(graph_embeddings.shape)}"
    )


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    task1_check(config, device)
    task2_check(config, device)
    task3_check(config, device)
    task4_check(config, device)
    print("REAL-CHECKPOINT SMOKE TESTS: PASS (Tasks 1-4)")


if __name__ == "__main__":
    main()
