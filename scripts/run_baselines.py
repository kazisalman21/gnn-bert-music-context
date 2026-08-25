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


def main() -> None:
    rng = np.random.default_rng(42)
    result = {
        "gtzan": gtzan_baselines(rng),
        "mtat_19_target_tags": mtat_baselines(rng),
        "musiccaps_task1": musiccaps_baselines(rng),
        "note": "All prevalence estimates use training labels only; metrics use held-out test labels.",
    }
    output = ROOT / "results/baseline_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
