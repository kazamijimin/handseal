"""Create a separate JSONL manifest; never move or rewrite source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .audit_dataset import DEFAULT_CLASSES, IMAGE_EXTENSIONS, source_key


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, classes: list[str]) -> list[dict]:
    rows = []
    for split in ("train", "valid", "test"):
        for name in classes:
            directory = root / split / name
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(path) as image:
                    width, height = image.size
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "split": split,
                        "class_name": name,
                        "source_key": source_key(path),
                        "sha256": file_hash(path),
                        "width": width,
                        "height": height,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="handseal_dataset")
    parser.add_argument("--output", default="reports/dataset_manifest.jsonl")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    rows = build_manifest(root, args.classes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Manifest: {output.resolve()}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
