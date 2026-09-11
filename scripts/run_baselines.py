"""Generate simple majority and random baselines using frozen splits."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import TARGET_TAGS
from src.evaluate import compute_multilabel_metrics


def parse_tags(value) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError):
        return []
    return [str(tag).strip().lower() for tag in parsed] if isinstance(parsed, list) else []


def gtzan_baselines(rng) -> dict:
    splits = json.loads((ROOT / "data/splits/gtzan_splits.json").read_text(encoding="utf-8"))
    genres = splits["genres"]
    train_labels = np.array([genres.index(item.split(".")[0]) for item in splits["train"]])
    test_labels = np.array([genres.index(item.split(".")[0]) for item in splits["test"]])
    majority = Counter(train_labels).most_common(1)[0][0]
    class_probabilities = np.bincount(train_labels, minlength=len(genres)) / len(train_labels)
    predictions = {
        "majority": np.full_like(test_labels, majority),
        "uniform_random": rng.integers(0, len(genres), size=len(test_labels)),
        "stratified_random": rng.choice(len(genres), size=len(test_labels), p=class_probabilities),
    }
    return {
        name: {
            "accuracy": float(accuracy_score(test_labels, values)),
            "macro_f1": float(f1_score(test_labels, values, average="macro", zero_division=0)),
        }
        for name, values in predictions.items()
    }


def mtat_baselines(rng) -> dict:
    annotations = pd.read_csv(ROOT / "data/raw/magnatagatune/annotations_final.csv", sep="\t")
    splits = json.loads((ROOT / "data/splits/mtat_splits.json").read_text(encoding="utf-8"))
    indexed = annotations.set_index("clip_id")
    train_labels = indexed.loc[splits["train"], TARGET_TAGS].to_numpy(dtype=int)
    test_labels = indexed.loc[splits["test"], TARGET_TAGS].to_numpy(dtype=int)
    prevalence = train_labels.mean(axis=0)
    scores = {
        "all_zero": np.zeros_like(test_labels, dtype=float),
        "uniform_random": rng.random(test_labels.shape),
        "train_prevalence_random": rng.random(test_labels.shape) < prevalence,
    }
    results = {}
    for name, prediction_scores in scores.items():
        probabilities = prediction_scores.astype(float)
        results[name] = compute_multilabel_metrics(
            test_labels, probabilities, threshold=0.5, tag_names=TARGET_TAGS
        )
    return {
        "train_examples": len(train_labels),
        "test_examples": len(test_labels),
        "training_tag_prevalence": {
            tag: float(value) for tag, value in zip(TARGET_TAGS, prevalence)
        },
        "models": results,
    }


def musiccaps_baselines(rng) -> dict:
    metadata = pd.read_csv(ROOT / "data/raw/musiccaps/musiccaps-public.csv", dtype={"ytid": str})
    indexed = metadata.set_index("ytid")
    vocabulary = json.loads(
        (ROOT / "data/splits/musiccaps_tag_vocab.json").read_text(encoding="utf-8")
    )["tags"]
    tag_to_index = {tag: index for index, tag in enumerate(vocabulary)}
    split_ids = {
        name: json.loads((ROOT / f"data/splits/musiccaps_{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "test")
    }

    def matrix(ids):
        labels = np.zeros((len(ids), len(vocabulary)), dtype=int)
        for row_index, ytid in enumerate(ids):
            for tag in parse_tags(indexed.loc[ytid, "aspect_list"]):
                if tag in tag_to_index:
                    labels[row_index, tag_to_index[tag]] = 1
        return labels

    train_labels = matrix(split_ids["train"])
    test_labels = matrix(split_ids["test"])
    prevalence = train_labels.mean(axis=0)
    probabilities = rng.random(test_labels.shape) < prevalence
    return {
        "train_prevalence_random": compute_multilabel_metrics(
            test_labels, probabilities.astype(float), threshold=0.5, tag_names=vocabulary
        ),
        "all_zero": compute_multilabel_metrics(
            test_labels, np.zeros_like(test_labels, dtype=float), threshold=0.5, tag_names=vocabulary
        ),
    }


def gtzan_pca_mlp_baseline() -> dict:
    """B4 from faculty spec §8: PCA + MLP on hand-crafted audio features."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier

    splits = json.loads((ROOT / "data/splits/gtzan_splits.json").read_text(encoding="utf-8"))
    genres = splits["genres"]
    graph_dir = ROOT / "data/processed/gtzan_graphs"

    if not graph_dir.exists() or not any(graph_dir.glob("*.pt")):
        return {"skipped": True, "reason": "GTZAN graphs not found; run preprocess_gtzan.py first"}

    import torch

    def load_features(track_ids):
        features, labels = [], []
        for track_id in track_ids:
            path = graph_dir / f"{track_id}.pt"
            if not path.exists():
                continue
            item = torch.load(path, map_location="cpu", weights_only=False)
            x = item["x"].numpy()
            # Aggregate node features: [mean, std, min, max] → 4 × feat_dim
            # (mean alone is ~0 due to per-track normalization)
            vec = np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])
            features.append(vec)
            labels.append(int(item["y"]))
        return np.array(features), np.array(labels)

    train_features, train_labels = load_features(splits["train"])
    test_features, test_labels = load_features(splits["test"])

    if len(train_features) == 0 or len(test_features) == 0:
        return {"skipped": True, "reason": "No graph features could be loaded"}

    # Scale features before PCA (important: raw aggregates have different ranges)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # PCA dimensionality reduction
    n_components = min(50, train_scaled.shape[1], train_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    # Shallow MLP classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(train_pca, train_labels)
    predictions = mlp.predict(test_pca)

    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "pca_components": n_components,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "mlp_layers": [128, 64],
        "train_samples": len(train_features),
        "test_samples": len(test_features),
        "feature_dim_original": int(train_features.shape[1]),
    }


def graph_coherence_summary() -> dict:
    """Compute the optional graph coherence metric from faculty spec §6."""
    from src.evaluate import compute_graph_coherence
    import torch

    results = {}
    for name, graph_dir in [
        ("gtzan", ROOT / "data/processed/gtzan_graphs"),
        ("mtat", ROOT / "data/processed/mtat_graphs"),
    ]:
        if not graph_dir.exists() or not any(graph_dir.glob("*.pt")):
            results[name] = {"skipped": True}
            continue
        scores = []
        graph_files = sorted(graph_dir.glob("*.pt"))
        # Sample up to 200 graphs for speed
        sample = graph_files[:200] if len(graph_files) > 200 else graph_files
        for path in sample:
            item = torch.load(path, map_location="cpu", weights_only=False)
            score = compute_graph_coherence(
                item["x"].numpy(), item["edge_index"].numpy(), tau=0.5
            )
            scores.append(score)
        results[name] = {
            "mean_coherence": float(np.mean(scores)),
            "std_coherence": float(np.std(scores)),
            "min_coherence": float(np.min(scores)),
            "max_coherence": float(np.max(scores)),
            "graphs_sampled": len(scores),
            "total_graphs": len(graph_files),
            "tau": 0.5,
        }
    return results


def main() -> None:
    rng = np.random.default_rng(42)
    result = {
        "gtzan": gtzan_baselines(rng),
        "mtat_19_target_tags": mtat_baselines(rng),
        "musiccaps_task1": musiccaps_baselines(rng),
        "gtzan_pca_mlp": gtzan_pca_mlp_baseline(),
        "graph_coherence": graph_coherence_summary(),
        "note": "All prevalence estimates use training labels only; metrics use held-out test labels.",
    }
    output = ROOT / "results/baseline_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
