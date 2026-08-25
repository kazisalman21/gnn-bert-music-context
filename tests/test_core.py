"""Fast regression tests for metrics, provenance, and safe packaging."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_submission import SCRIPT_FILES, SOURCE_DIRECTORIES
from src.contrastive import InfoNCELoss, compute_retrieval_metrics
from src.provenance import cache_matches, stable_hash


def test_perfect_retrieval_has_full_recall() -> None:
    embeddings = torch.eye(4)
    metrics = compute_retrieval_metrics(embeddings, embeddings)
    for direction in ("caption_to_audio", "audio_to_caption"):
        assert metrics[direction] == {
            "R@1": 1.0,
            "R@5": 1.0,
            "R@10": 1.0,
        }


def test_infonce_is_finite_and_backpropagates() -> None:
    graph = torch.randn(4, 8, requires_grad=True)
    text = torch.randn(4, 8, requires_grad=True)
    graph_norm = torch.nn.functional.normalize(graph, dim=1)
    text_norm = torch.nn.functional.normalize(text, dim=1)
    loss = InfoNCELoss(temperature=0.07)(graph_norm, text_norm)
    loss.backward()

    assert torch.isfinite(loss)
    assert graph.grad is not None and torch.isfinite(graph.grad).all()
    assert text.grad is not None and torch.isfinite(text.grad).all()


def test_provenance_hash_is_order_independent() -> None:
    first = {"sample_rate": 22050, "features": ["mfcc", "chroma"]}
    reordered = {"features": ["mfcc", "chroma"], "sample_rate": 22050}
    changed = {"features": ["mfcc"], "sample_rate": 22050}

    assert stable_hash(first) == stable_hash(reordered)
    assert stable_hash(first) != stable_hash(changed)


def test_cache_rejects_version_or_setting_drift() -> None:
    settings = {"sample_rate": 22050, "window_seconds": 5.0}
    metadata = {
        "preprocessing_version": "v2",
        "preprocessing_signature": stable_hash(settings),
    }

    assert cache_matches(metadata, "v2", settings)
    assert not cache_matches(metadata, "v1", settings)
    assert not cache_matches(
        metadata, "v2", {"sample_rate": 16000, "window_seconds": 5.0}
    )


def test_submission_uses_an_explicit_script_allowlist() -> None:
    assert "scripts" not in SOURCE_DIRECTORIES
    assert "scripts/create_submission.py" in SCRIPT_FILES
    assert "scripts/run_task4.py" in SCRIPT_FILES
    assert all("legacy" not in path.lower() for path in SCRIPT_FILES)
