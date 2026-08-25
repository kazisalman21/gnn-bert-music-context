"""Preprocess MagnaTagATune audio into five-second segment graphs."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_features import load_and_resample, compute_rich_segment_features
from src.dataset import CONTEXT_TAGS, TARGET_TAGS
from src.graph_builder import build_segment_graph, save_graph, validate_graph
from src.provenance import cache_matches, stable_hash


CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
MTAT_DIR = ROOT / "data/raw/magnatagatune"
PROCESSED_DIR = ROOT / "data/processed/mtat_graphs"
SPLIT_PATH = ROOT / "data/splits/mtat_splits.json"
RESULTS_PATH = ROOT / "results/mtat_preprocessing.json"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

SR = int(CONFIG["audio"]["sample_rate"])
WINDOW_SEC = float(CONFIG["segmentation"]["mtat_window_sec"])
HOP_LENGTH = int(CONFIG["audio"]["mel_hop_length"])
N_FFT = int(CONFIG["audio"]["mel_n_fft"])
TAU = float(CONFIG["graph"]["tau"])
VERSION = "mtat-rich77-5s-v1"
SETTINGS = {
    "sample_rate": SR, "window_sec": WINDOW_SEC, "hop_length": HOP_LENGTH,
    "n_fft": N_FFT, "feature_dim": 77, "normalize": True, "tau": TAU,
    "temporal_edges": bool(CONFIG["graph"]["temporal_edges"]),
    "self_loops": bool(CONFIG["graph"]["self_loops"]),
    "directed": bool(CONFIG["graph"]["directed"]),
}
LEGACY_SETTINGS = {
    "sample_rate": 22050, "window_sec": 5.0, "hop_length": 512,
    "n_fft": 2048, "feature_dim": 77, "normalize": True, "tau": 0.5,
    "temporal_edges": True, "self_loops": True, "directed": False,
}


def context_sentence(row: pd.Series) -> str:
    active = [tag for tag in CONTEXT_TAGS if int(row[tag]) == 1]
    if not active:
        return "This music features no described instruments or vocals."
    if len(active) == 1:
        return f"This music features {active[0]}."
    if len(active) == 2:
        return f"This music features {active[0]} and {active[1]}."
    return f"This music features {', '.join(active[:-1])}, and {active[-1]}."


def cache_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        metadata = item.get("metadata", {})
        return (
            cache_matches(metadata, VERSION, SETTINGS, LEGACY_SETTINGS)
            and item["x"].shape[1] == 77
        )
    except Exception:
        return False


def main() -> None:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("Run scripts/prepare_splits.py before preprocessing MTAT")

    annotations = pd.read_csv(MTAT_DIR / "annotations_final.csv", sep="\t")
    missing_tags = [tag for tag in CONTEXT_TAGS + TARGET_TAGS if tag not in annotations.columns]
    if missing_tags:
        raise ValueError(f"Required MTAT tags are absent: {missing_tags}")

    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_by_clip = {
        int(clip_id): name
        for name in ("train", "val", "test")
        for clip_id in split_payload[name]
    }

    started = time.time()
    success = 0
    cached = 0
    failed = []
    node_counts = Counter()
    edge_counts = Counter()
    split_counts = Counter()

    for index, row in annotations.iterrows():
        clip_id = int(row["clip_id"])
        output_path = PROCESSED_DIR / f"{clip_id}.pt"

        if cache_is_current(output_path):
            item = torch.load(output_path, map_location="cpu", weights_only=False)
            cached += 1
            success += 1
            node_counts[int(item["x"].shape[0])] += 1
            edge_counts[int(item["edge_index"].shape[1])] += 1
            split_counts[split_by_clip[clip_id]] += 1
            continue

        audio_path = MTAT_DIR / str(row["mp3_path"])
        try:
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty audio: {audio_path}")

            audio = load_and_resample(str(audio_path), sr=SR)
            features = compute_rich_segment_features(
                audio,
                sr=SR,
                window_sec=WINDOW_SEC,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
                normalize=True,
            )
            graph = build_segment_graph(
                features,
                tau=TAU,
                temporal_edges=bool(CONFIG["graph"]["temporal_edges"]),
                self_loops=bool(CONFIG["graph"]["self_loops"]),
                directed=bool(CONFIG["graph"]["directed"]),
            )
            validation = validate_graph(graph)
            if not validation["valid"]:
                raise ValueError(f"invalid graph: {validation['issues']}")

            graph.y = torch.tensor([float(row[tag]) for tag in TARGET_TAGS], dtype=torch.float32)
            active_context = [tag for tag in CONTEXT_TAGS if int(row[tag]) == 1]
            active_targets = [tag for tag in TARGET_TAGS if int(row[tag]) == 1]
            save_graph(graph, str(output_path), metadata={
                "preprocessing_version": VERSION,
                "preprocessing_signature": stable_hash(SETTINGS),
                "clip_id": clip_id,
                "source_audio": str(row["mp3_path"]).replace("\\", "/"),
                "split": split_by_clip[clip_id],
                "sample_rate": SR,
                "window_sec": WINDOW_SEC,
                "feature_dim": 77,
                "feature_description": "MFCC/chroma/spectral/energy statistics",
                "tau": TAU,
                "num_segments": int(features.shape[0]),
                "context_tags": active_context,
                "context_text": context_sentence(row),
                "target_tags": active_targets,
            })
            success += 1
            node_counts[int(graph.x.shape[0])] += 1
            edge_counts[int(graph.edge_index.shape[1])] += 1
            split_counts[split_by_clip[clip_id]] += 1
        except Exception as exc:
            failed.append({"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"})

        if (index + 1) % 1000 == 0:
            elapsed = time.time() - started
            print(
                f"[{index + 1}/{len(annotations)}] success={success} "
                f"failed={len(failed)} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    report = {
        "preprocessing_version": VERSION,
        "preprocessing_signature": stable_hash(SETTINGS),
        "preprocessing_settings": SETTINGS,
        "requested": len(annotations),
        "successful": success,
        "cached": cached,
        "failed": len(failed),
        "failures": failed,
        "sample_rate": SR,
        "window_sec": WINDOW_SEC,
        "feature_dim": 77,
        "tau": TAU,
        "node_count_distribution": dict(sorted(node_counts.items())),
        "edge_count_distribution": dict(sorted(edge_counts.items())),
        "usable_by_split": dict(split_counts),
        "elapsed_seconds": time.time() - started,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
