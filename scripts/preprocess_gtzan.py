"""Preprocess GTZAN for the Task 2 GNN and CNN experiments."""

from __future__ import annotations

import json
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_features import (
    compute_rich_segment_features,
    extract_log_mel,
    load_and_resample,
    normalize_features,
)
from src.graph_builder import build_segment_graph, save_graph, validate_graph
from src.provenance import cache_matches, stable_hash


CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
AUDIO_DIR = ROOT / "data/raw/gtzan/Data/genres_original"
GRAPH_DIR = ROOT / "data/processed/gtzan_graphs"
MEL_DIR = ROOT / "data/processed/gtzan_mels"
SPLIT_PATH = ROOT / "data/splits/gtzan_splits.json"
RESULT_PATH = ROOT / "results/gtzan_preprocessing.json"
SPLIT_AUDIT_PATH = ROOT / "results/data_split_audit.json"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
MEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
SR = int(CONFIG["audio"]["sample_rate"])
WINDOW_SEC = float(CONFIG["segmentation"]["gtzan_window_sec"])
TAU = float(CONFIG["graph"]["tau"])
VERSION = "gtzan-rich77-mel128-5s-v1"
SETTINGS = {
    "sample_rate": SR,
    "window_sec": WINDOW_SEC,
    "hop_length": int(CONFIG["audio"]["mel_hop_length"]),
    "n_fft": int(CONFIG["audio"]["mel_n_fft"]),
    "mel_bins": int(CONFIG["audio"]["mel_bins"]),
    "feature_dim": 77,
    "normalize": True,
    "tau": TAU,
    "temporal_edges": bool(CONFIG["graph"]["temporal_edges"]),
    "self_loops": bool(CONFIG["graph"]["self_loops"]),
    "directed": bool(CONFIG["graph"]["directed"]),
}
LEGACY_SETTINGS = {
    "sample_rate": 22050, "window_sec": 5.0, "hop_length": 512,
    "n_fft": 2048, "mel_bins": 128, "feature_dim": 77,
    "normalize": True, "tau": 0.5, "temporal_edges": True,
    "self_loops": True, "directed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_stratified_split(successful: list[str], audio_paths: dict[str, Path], seed: int) -> tuple[dict, dict]:
    """Split each genre while keeping byte-identical audio in one partition."""
    file_hashes = {track_id: sha256_file(audio_paths[track_id]) for track_id in successful}
    partitions = {"train": [], "val": [], "test": []}

    for genre_index, genre in enumerate(GENRES):
        genre_ids = np.asarray([track_id for track_id in successful if track_id.startswith(f"{genre}.")])
        genre_groups = np.asarray([file_hashes[track_id] for track_id in genre_ids])
        first_split = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=seed + genre_index
        )
        train_indices, temporary_indices = next(first_split.split(genre_ids, groups=genre_groups))
        temporary_ids = genre_ids[temporary_indices]
        temporary_groups = genre_groups[temporary_indices]
        second_split = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 100 + genre_index
        )
        val_relative, test_relative = next(
            second_split.split(temporary_ids, groups=temporary_groups)
        )
        partitions["train"].extend(genre_ids[train_indices].tolist())
        partitions["val"].extend(temporary_ids[val_relative].tolist())
        partitions["test"].extend(temporary_ids[test_relative].tolist())

    rng = np.random.default_rng(seed)
    for name in partitions:
        rng.shuffle(partitions[name])

    hash_splits: dict[str, set[str]] = {}
    for name, track_ids in partitions.items():
        for track_id in track_ids:
            hash_splits.setdefault(file_hashes[track_id], set()).add(name)
    duplicate_groups = sum(
        count > 1 for count in Counter(file_hashes.values()).values()
    )
    cross_split_groups = sum(len(names) > 1 for names in hash_splits.values())
    audit = {
        "track_id_overlap": {
            "train_val": len(set(partitions["train"]) & set(partitions["val"])),
            "train_test": len(set(partitions["train"]) & set(partitions["test"])),
            "val_test": len(set(partitions["val"]) & set(partitions["test"])),
        },
        "exact_audio_duplicate_groups": int(duplicate_groups),
        "cross_split_exact_audio_duplicate_groups": int(cross_split_groups),
        "hash_algorithm": "SHA-256 over source WAV bytes",
    }
    if cross_split_groups:
        raise RuntimeError(f"GTZAN split leaked {cross_split_groups} exact-audio groups")
    return partitions, audit


def is_current(graph_path: Path, mel_path: Path) -> bool:
    if not graph_path.exists() or not mel_path.exists():
        return False
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        mel = torch.load(mel_path, map_location="cpu", weights_only=False)
        return (
            cache_matches(graph.get("metadata", {}), VERSION, SETTINGS, LEGACY_SETTINGS)
            and cache_matches(mel.get("metadata", {}), VERSION, SETTINGS, LEGACY_SETTINGS)
            and graph["x"].shape[1] == 77
            and mel["log_mel"].shape[0] == int(CONFIG["audio"]["mel_bins"])
        )
    except Exception:
        return False


def main() -> None:
    tracks = []
    for genre in GENRES:
        for audio_path in sorted((AUDIO_DIR / genre).glob("*.wav")):
            tracks.append((genre, audio_path.stem, audio_path))
    audio_paths = {track_id: audio_path for _, track_id, audio_path in tracks}
    if len(tracks) != 1000:
        print(f"Warning: expected 1000 GTZAN files, found {len(tracks)}", flush=True)

    started = time.time()
    successful = []
    failures = []
    node_counts = Counter()
    cached = 0

    for index, (genre, track_id, audio_path) in enumerate(tracks):
        graph_path = GRAPH_DIR / f"{track_id}.pt"
        mel_path = MEL_DIR / f"{track_id}.pt"
        label = GENRES.index(genre)
        try:
            if is_current(graph_path, mel_path):
                graph_item = torch.load(graph_path, map_location="cpu", weights_only=False)
                node_counts[int(graph_item["x"].shape[0])] += 1
                cached += 1
                successful.append(track_id)
                continue

            audio = load_and_resample(str(audio_path), sr=SR)
            log_mel = extract_log_mel(
                audio,
                sr=SR,
                n_mels=int(CONFIG["audio"]["mel_bins"]),
                n_fft=int(CONFIG["audio"]["mel_n_fft"]),
                hop_length=int(CONFIG["audio"]["mel_hop_length"]),
            )
            log_mel = normalize_features(log_mel).astype(np.float32)
            features = compute_rich_segment_features(
                audio,
                sr=SR,
                window_sec=WINDOW_SEC,
                hop_length=int(CONFIG["audio"]["mel_hop_length"]),
                n_fft=int(CONFIG["audio"]["mel_n_fft"]),
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
                raise ValueError(validation["issues"])

            graph.y = torch.tensor(label, dtype=torch.long)
            metadata = {
                "preprocessing_version": VERSION,
                "preprocessing_signature": stable_hash(SETTINGS),
                "track_id": track_id,
                "genre": genre,
                "sample_rate": SR,
                "window_sec": WINDOW_SEC,
                "feature_dim": 77,
                "tau": TAU,
                "duration_seconds": len(audio) / SR,
            }
            save_graph(graph, str(graph_path), metadata=metadata)
            torch.save({
                "log_mel": torch.from_numpy(log_mel).to(torch.float16),
                "y": torch.tensor(label, dtype=torch.long),
                "metadata": metadata,
            }, mel_path)
            successful.append(track_id)
            node_counts[int(graph.x.shape[0])] += 1
        except Exception as exc:
            failures.append({"track_id": track_id, "error": f"{type(exc).__name__}: {exc}"})

        if (index + 1) % 100 == 0:
            print(
                f"[{index + 1}/{len(tracks)}] successful={len(successful)} "
                f"failed={len(failures)}",
                flush=True,
            )

    split_ids, split_audit = grouped_stratified_split(
        successful, audio_paths, int(CONFIG["seed"])
    )
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    split_payload = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "genres": GENRES,
        "protocol": (
            "seeded per-genre group split (SHA-256 exact-audio groups), "
            "approximately 80/10/10"
        ),
        "seed": int(CONFIG["seed"]),
        "audit": split_audit,
    }
    SPLIT_PATH.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    report = {
        "preprocessing_version": VERSION,
        "preprocessing_signature": stable_hash(SETTINGS),
        "preprocessing_settings": SETTINGS,
        "requested": len(tracks),
        "successful": len(successful),
        "cached": cached,
        "failed": len(failures),
        "failures": failures,
        "graph_feature_dim": 77,
        "mel_bins": int(CONFIG["audio"]["mel_bins"]),
        "window_sec": WINDOW_SEC,
        "node_count_distribution": dict(sorted(node_counts.items())),
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_audit": split_audit,
        "elapsed_seconds": time.time() - started,
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if SPLIT_AUDIT_PATH.exists():
        combined_audit = json.loads(SPLIT_AUDIT_PATH.read_text(encoding="utf-8"))
    else:
        combined_audit = {}
    combined_audit["gtzan"] = {
        "protocol": split_payload["protocol"],
        "counts": report["split_counts"],
        **split_audit,
    }
    SPLIT_AUDIT_PATH.write_text(json.dumps(combined_audit, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
