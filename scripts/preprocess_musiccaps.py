"""Preprocess downloaded real MusicCaps intervals into segment graphs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_features import compute_rich_segment_features, load_and_resample
from src.graph_builder import build_segment_graph, save_graph, validate_graph
from src.provenance import cache_matches, stable_hash


CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
AUDIO_DIR = ROOT / "data/raw/musiccaps/audio"
GRAPH_DIR = ROOT / "data/processed/musiccaps_graphs"
METADATA_PATH = ROOT / "data/raw/musiccaps/musiccaps-public.csv"
STATUS_PATH = ROOT / "results/musiccaps_audio_download.json"
RESULT_PATH = ROOT / "results/musiccaps_preprocessing.json"
VERSION = "musiccaps-rich77-2s-v1"
SETTINGS = {
    "sample_rate": int(CONFIG["audio"]["sample_rate"]),
    "window_sec": float(CONFIG["segmentation"]["musiccaps_window_sec"]),
    "hop_length": int(CONFIG["audio"]["mel_hop_length"]),
    "n_fft": int(CONFIG["audio"]["mel_n_fft"]),
    "feature_dim": 77,
    "normalize": True,
    "tau": float(CONFIG["graph"]["tau"]),
    "temporal_edges": bool(CONFIG["graph"]["temporal_edges"]),
    "self_loops": bool(CONFIG["graph"]["self_loops"]),
    "directed": bool(CONFIG["graph"]["directed"]),
}
LEGACY_SETTINGS = {
    "sample_rate": 22050, "window_sec": 2.0, "hop_length": 512,
    "n_fft": 2048, "feature_dim": 77, "normalize": True, "tau": 0.5,
    "temporal_edges": True, "self_loops": True, "directed": False,
}
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        return (
            cache_matches(item.get("metadata", {}), VERSION, SETTINGS, LEGACY_SETTINGS)
            and item["x"].shape[1] == 77
        )
    except Exception:
        return False


def main() -> None:
    if not STATUS_PATH.exists():
        raise FileNotFoundError("Run scripts/download_musiccaps_audio.py first")
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    metadata = pd.read_csv(METADATA_PATH).set_index("ytid", drop=False)
    split_membership = {}
    for split in ("train", "val", "test"):
        for ytid in json.loads((ROOT / f"data/splits/musiccaps_{split}.json").read_text(encoding="utf-8")):
            split_membership[str(ytid)] = split

    usable_ids = [
        ytid for ytid, record in status["clips"].items()
        if record["status"] in {"downloaded", "cached"} and (AUDIO_DIR / f"{ytid}.wav").exists()
    ]
    started = time.time()
    successful = 0
    cached = 0
    failures = []
    for position, ytid in enumerate(usable_ids, start=1):
        output_path = GRAPH_DIR / f"{ytid}.pt"
        try:
            if is_current(output_path):
                cached += 1
                successful += 1
                continue
            audio = load_and_resample(str(AUDIO_DIR / f"{ytid}.wav"), sr=int(CONFIG["audio"]["sample_rate"]))
            features = compute_rich_segment_features(
                audio,
                sr=int(CONFIG["audio"]["sample_rate"]),
                window_sec=float(CONFIG["segmentation"]["musiccaps_window_sec"]),
                hop_length=int(CONFIG["audio"]["mel_hop_length"]),
                n_fft=int(CONFIG["audio"]["mel_n_fft"]),
                normalize=True,
            )
            graph = build_segment_graph(
                features,
                tau=float(CONFIG["graph"]["tau"]),
                temporal_edges=bool(CONFIG["graph"]["temporal_edges"]),
                self_loops=bool(CONFIG["graph"]["self_loops"]),
                directed=bool(CONFIG["graph"]["directed"]),
            )
            validation = validate_graph(graph)
            if not validation["valid"]:
                raise ValueError(validation["issues"])
            row = metadata.loc[ytid]
            save_graph(graph, str(output_path), metadata={
                "preprocessing_version": VERSION,
                "preprocessing_signature": stable_hash(SETTINGS),
                "ytid": ytid,
                "split": split_membership.get(ytid),
                "sample_rate": int(CONFIG["audio"]["sample_rate"]),
                "window_sec": float(CONFIG["segmentation"]["musiccaps_window_sec"]),
                "feature_dim": 77,
                "caption": str(row["caption"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
            })
            successful += 1
        except Exception as exc:
            failures.append({"ytid": ytid, "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{position}/{len(usable_ids)}] {ytid}", flush=True)

    report = {
        "preprocessing_version": VERSION,
        "preprocessing_signature": stable_hash(SETTINGS),
        "preprocessing_settings": SETTINGS,
        "requested_available_audio": len(usable_ids),
        "successful": successful,
        "cached": cached,
        "failed": len(failures),
        "failures": failures,
        "feature_dim": 77,
        "window_sec": float(CONFIG["segmentation"]["musiccaps_window_sec"]),
        "elapsed_seconds": time.time() - started,
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
