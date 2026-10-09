"""Audit a folder-based HandSeal dataset without modifying source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DEFAULT_CLASSES = ["dragon", "ram", "snake", "tiger", "no_seal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(path: Path) -> str:
    return SOURCE_SUFFIX.sub("", path.stem)


def audit_dataset(root: Path, class_names: list[str]) -> dict:
    counts = {
        split: {name: 0 for name in class_names}
        for split in ("train", "valid", "test")
    }
    records: list[dict] = []
    corrupt: list[dict] = []

    for split in counts:
        for name in class_names:
            class_dir = root / split / name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                counts[split][name] += 1
                record = {
                    "path": path.relative_to(root).as_posix(),
                    "split": split,
                    "class_name": name,
                    "source_key": source_key(path),
                    "sha256": sha256(path),
                }
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        record["width"], record["height"] = image.size
                        record["format"] = image.format
                except Exception as exc:  # Pillow raises several format-specific errors.
                    record["error"] = str(exc)
                    corrupt.append(record)
                records.append(record)

    hash_groups = defaultdict(list)
    source_groups = defaultdict(list)
    for record in records:
        hash_groups[record["sha256"]].append(record)
        source_groups[record["source_key"]].append(record)

    def cross_split_groups(groups):
        return [
            values
            for values in groups.values()
            if len({value["split"] for value in values}) > 1
        ]

    dimensions = Counter(
        f"{record.get('width', '?')}x{record.get('height', '?')}"
        for record in records
        if "error" not in record
    )
    return {
        "dataset_root": str(root.resolve()),
        "classes": class_names,
        "counts": counts,
        "total_images": len(records),
        "corrupt_images": corrupt,
        "dimensions": dict(dimensions),
        "exact_duplicate_groups": [
            values for values in hash_groups.values() if len(values) > 1
        ],
        "cross_split_exact_duplicate_groups": cross_split_groups(hash_groups),
        "cross_split_source_groups": cross_split_groups(source_groups),
        "records": records,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# HandSeal Dataset Audit",
        "",
        f"Dataset root: `{report['dataset_root']}`",
        f"Total supported images: **{report['total_images']}**",
        "",
        "## Counts",
        "",
        "| Split | " + " | ".join(report["classes"]) + " | Total |",
        "|---|" + "---|" * (len(report["classes"]) + 1),
    ]
    for split, values in report["counts"].items():
        total = sum(values.values())
        lines.append(
            f"| {split} | "
            + " | ".join(str(values[name]) for name in report["classes"])
            + f" | {total} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Corrupt images: {len(report['corrupt_images'])}",
            f"- Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
            f"- Cross-split exact duplicate groups: {len(report['cross_split_exact_duplicate_groups'])}",
            f"- Cross-split source-key groups: {len(report['cross_split_source_groups'])}",
            "",
            "The source dataset was not modified by this audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="handseal_dataset")
    parser.add_argument("--output", default="reports/dataset_audit.json")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    report = audit_dataset(root, args.classes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Audit JSON: {output.resolve()}")
    print(f"Audit Markdown: {markdown_path.resolve()}")
    print(f"Images: {report['total_images']}")
    print(f"Corrupt images: {len(report['corrupt_images'])}")
    print(
        "Cross-split source groups: "
        f"{len(report['cross_split_source_groups'])}"
    )
    return 2 if report["corrupt_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
