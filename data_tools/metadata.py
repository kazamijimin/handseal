"""Shared metadata helpers for participant-aware HandSeal datasets."""

from __future__ import annotations

import json
from pathlib import Path


METADATA_FILENAME = "metadata.jsonl"
METADATA_FIELDS = (
    "participant_id",
    "session_id",
    "lighting",
    "background",
    "distance",
    "hand_position",
)
REQUIRED_METADATA_FIELDS = (
    "label",
    "split",
    *METADATA_FIELDS,
    "width",
    "height",
    "sha256",
)


def normalize_relative_path(value: str | Path) -> str:
    """Return a metadata path in the same slash-normalized form as manifests."""
    return Path(str(value).replace("\\", "/")).as_posix()


def load_metadata(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Load root metadata.jsonl and return rows indexed by relative image path."""
    metadata_path = root / METADATA_FILENAME
    if not metadata_path.exists():
        return {}, []

    rows: dict[str, dict] = {}
    errors: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or not value.get("path"):
                    raise ValueError("metadata row must be an object with a path")
                path = normalize_relative_path(value["path"])
                if path in rows:
                    raise ValueError(f"duplicate metadata path: {path}")
                rows[path] = value
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append({"line": line_number, "error": str(exc)})
    return rows, errors


def metadata_for_path(metadata: dict[str, dict], relative_path: str | Path) -> dict:
    return metadata.get(normalize_relative_path(relative_path), {})


def append_metadata(path: Path, record: dict) -> None:
    """Append one capture record without rewriting existing metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
