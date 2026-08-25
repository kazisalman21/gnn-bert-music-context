"""Create and audit every frozen train/validation/test split.

Protocols
---------
MusicCaps:
  * AudioSet evaluation rows are the held-out test set.
  * Remaining rows are split 90/10 into train/validation with seed 42.
  * The Task 1 label vocabulary is derived from training rows only.

MagnaTagATune:
  * Official archive shards 0-b are train, c is validation, d-f are test.
  * Clip, artist, and exact song-tuple overlap are audited and recorded.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SPLITS_DIR = ROOT / CONFIG["paths"]["data_splits"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results"]
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_aspects(value) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(tag).strip().lower() for tag in parsed if str(tag).strip()]


def prepare_musiccaps() -> dict:
    csv_path = ROOT / "data/raw/musiccaps/musiccaps-public.csv"
    df = pd.read_csv(csv_path)
    if df["ytid"].duplicated().any():
        raise ValueError("MusicCaps ytid values are not unique")

    test_df = df[df["is_audioset_eval"].astype(bool)].copy()
    trainval_df = df[~df["is_audioset_eval"].astype(bool)].copy()
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=float(CONFIG["task1"]["validation_fraction"]),
        random_state=int(CONFIG["seed"]),
        shuffle=True,
    )

    split_ids = {
        "train": train_df["ytid"].astype(str).tolist(),
        "val": val_df["ytid"].astype(str).tolist(),
        "test": test_df["ytid"].astype(str).tolist(),
    }
    for name, ids in split_ids.items():
        write_json(SPLITS_DIR / f"musiccaps_{name}.json", ids)

    train_tags = Counter()
    for value in train_df["aspect_list"]:
        train_tags.update(parse_aspects(value))
    minimum = int(CONFIG["task1"]["min_tag_frequency"])
    vocabulary = [tag for tag, count in train_tags.most_common() if count >= minimum]
    vocab_payload = {
        "tags": vocabulary,
        "counts": [int(train_tags[tag]) for tag in vocabulary],
        "min_freq": minimum,
        "source": "training split only",
        "train_examples": len(train_df),
    }
    write_json(SPLITS_DIR / "musiccaps_tag_vocab.json", vocab_payload)

    sets = {name: set(ids) for name, ids in split_ids.items()}
    audit = {
        "protocol": "AudioSet eval=test; non-eval deterministic 90/10 train/validation",
        "counts": {name: len(ids) for name, ids in split_ids.items()},
        "unique_ytid": int(df["ytid"].nunique()),
        "overlap": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
        "task1_vocabulary_size": len(vocabulary),
        "task1_min_training_frequency": minimum,
    }
    return audit


def prepare_mtat() -> dict:
    mtat_dir = ROOT / "data/raw/magnatagatune"
    annotations = pd.read_csv(mtat_dir / "annotations_final.csv", sep="\t")
    clip_info = pd.read_csv(mtat_dir / "clip_info_final.csv", sep="\t")
    if annotations["clip_id"].duplicated().any():
        raise ValueError("MTAT annotation clip_id values are not unique")

    shard = annotations["mp3_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    train_shards = set(CONFIG["task3"]["train_shards"])
    val_shards = set(CONFIG["task3"]["validation_shards"])
    test_shards = set(CONFIG["task3"]["test_shards"])
    known_shards = train_shards | val_shards | test_shards
    unknown = sorted(set(shard.unique()) - known_shards)
    if unknown:
        raise ValueError(f"Unexpected MTAT shards: {unknown}")

    split_ids = {
        "train": annotations.loc[shard.isin(train_shards), "clip_id"].astype(int).tolist(),
        "val": annotations.loc[shard.isin(val_shards), "clip_id"].astype(int).tolist(),
        "test": annotations.loc[shard.isin(test_shards), "clip_id"].astype(int).tolist(),
    }
    sets = {name: set(ids) for name, ids in split_ids.items()}
    union = set().union(*sets.values())
    if len(union) != len(annotations):
        raise ValueError("Official shard split does not cover every MTAT annotation")

    info_columns = ["clip_id", "artist", "title", "album"]
    info = clip_info[info_columns].drop_duplicates("clip_id")
    merged = annotations[["clip_id"]].merge(info, on="clip_id", how="left")
    artists = {
        name: set(merged.loc[merged["clip_id"].isin(ids), "artist"].dropna().astype(str))
        for name, ids in sets.items()
    }
    songs = {
        name: set(map(tuple, merged.loc[
            merged["clip_id"].isin(ids), ["artist", "title", "album"]
        ].fillna("").astype(str).values.tolist()))
        for name, ids in sets.items()
    }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    audit = {
        "protocol": "official MTAT shards: 0-b train, c validation, d-f test",
        "counts": {name: len(ids) for name, ids in split_ids.items()},
        "shards": {
            "train": sorted(train_shards),
            "val": sorted(val_shards),
            "test": sorted(test_shards),
        },
        "duplicate_clip_ids_within_split": {
            name: len(ids) - len(sets[name]) for name, ids in split_ids.items()
        },
        "clip_overlap": {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in pairs},
        "artist_overlap": {f"{a}_{b}": len(artists[a] & artists[b]) for a, b in pairs},
        "exact_song_tuple_overlap": {f"{a}_{b}": len(songs[a] & songs[b]) for a, b in pairs},
    }
    payload = {"train": split_ids["train"], "val": split_ids["val"], "test": split_ids["test"], "audit": audit}
    write_json(SPLITS_DIR / "mtat_splits.json", payload)
    return audit


def main() -> None:
    report = {
        "musiccaps": prepare_musiccaps(),
        "mtat": prepare_mtat(),
    }
    write_json(RESULTS_DIR / "data_split_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
