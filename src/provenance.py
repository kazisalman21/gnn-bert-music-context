"""Small helpers for reproducible preprocessing and experiment records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable_hash(payload: dict) -> str:
    """Return a deterministic SHA-256 hash for a JSON-compatible dictionary."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_matches(metadata: dict, version: str, settings: dict,
                  legacy_settings: dict | None = None) -> bool:
    """Check a cache fingerprint, with a bridge for audited older caches."""
    if metadata.get("preprocessing_version") != version:
        return False
    expected = stable_hash(settings)
    saved = metadata.get("preprocessing_signature")
    if saved is not None:
        return saved == expected
    return legacy_settings is not None and settings == legacy_settings
