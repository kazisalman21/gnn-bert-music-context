"""Real-audio preprocessing gate for three GTZAN tracks.

The gate builds graphs in memory and uses a temporary directory for its save/load
check. It never writes into the canonical processed dataset.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_features import (
    compute_rich_segment_features,
    extract_chroma,
    extract_log_mel,
    load_and_resample,
    normalize_features,
)
from src.graph_builder import build_segment_graph, load_graph, save_graph, validate_graph


TRACKS = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("rock", "rock.00000.wav"),
]


def main() -> None:
    audio_root = ROOT / "data/raw/gtzan/Data/genres_original"
    with tempfile.TemporaryDirectory(prefix="cse425_graph_gate_") as temporary:
        temporary_dir = Path(temporary)
        for genre, filename in TRACKS:
            path = audio_root / genre / filename
            if not path.exists():
                raise FileNotFoundError(f"Required real GTZAN track is missing: {path}")
            audio = load_and_resample(str(path), sr=22050)
            log_mel = normalize_features(extract_log_mel(audio, sr=22050))
            chroma = extract_chroma(audio, sr=22050)
            features = compute_rich_segment_features(
                audio, sr=22050, window_sec=5.0, hop_length=512, n_fft=2048, normalize=True
            )
            graph = build_segment_graph(
                features, tau=0.5, temporal_edges=True, self_loops=True, directed=False
            )
            validation = validate_graph(graph)

            assert log_mel.shape[0] == 128
            assert chroma.shape[0] == 12
            assert features.shape == (6, 77)
            assert np.isfinite(log_mel).all() and np.isfinite(chroma).all()
            assert validation["valid"], validation["issues"]

            graph.y = torch.tensor(
                ["blues", "classical", "country", "disco", "hiphop",
                 "jazz", "metal", "pop", "reggae", "rock"].index(genre),
                dtype=torch.long,
            )
            saved_path = temporary_dir / f"{path.stem}.pt"
            save_graph(graph, str(saved_path), metadata={"source": str(path), "test_only": True})
            restored = load_graph(str(saved_path))
            assert restored.x.shape == graph.x.shape
            assert torch.equal(restored.edge_index, graph.edge_index)
            print(
                f"PASS {path.stem}: mel={tuple(log_mel.shape)}, "
                f"graph={graph.num_nodes} nodes/{graph.num_edges} edges, features=77"
            )
    print("REAL PREPROCESSING GATE: PASS")


if __name__ == "__main__":
    main()
