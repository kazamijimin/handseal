"""Audit a folder-based HandSeal dataset without modifying source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .metadata import (
    METADATA_FIELDS,
    REQUIRED_METADATA_FIELDS,
    load_metadata,
    metadata_for_path,
)


DEFAULT_CLASSES = ["dragon", "ram", "snake", "tiger", "no_seal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Return a small average hash for near-duplicate detection."""
    with Image.open(path) as image:
        image = image.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
        pixels = list(image.getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def source_key(path: Path) -> str:
    return SOURCE_SUFFIX.sub("", path.stem)


def _cross_split_groups(groups):
    return [
        values
        for values in groups.values()
        if len({value["split"] for value in values}) > 1
    ]


def _near_duplicate_groups(records: list[dict], max_distance: int = 4):
    candidates = [record for record in records if "perceptual_hash" in record]
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            distance = (
                int(candidates[left]["perceptual_hash"], 16)
                ^ int(candidates[right]["perceptual_hash"], 16)
            ).bit_count()
            if distance <= max_distance:
                union(left, right)

    groups = defaultdict(list)
    for index, record in enumerate(candidates):
        groups[find(index)].append(record)
    return [values for values in groups.values() if len(values) > 1]


def strict_issues(report: dict) -> list[str]:
    """Return issues that must stop a training run."""
    issues = []
    empty = report["empty_classes"]
    for split, classes in empty.items():
        if classes:
            issues.append(f"empty classes in {split}: {', '.join(classes)}")
    if report["corrupt_images"]:
        issues.append(f"corrupt images: {len(report['corrupt_images'])}")
    if report["unsupported_files"]:
        issues.append(f"unsupported files: {len(report['unsupported_files'])}")
    if report["metadata_errors"]:
        issues.append(f"metadata errors: {len(report['metadata_errors'])}")
    if report["missing_metadata_paths"]:
        issues.append(f"images missing required metadata: {len(report['missing_metadata_paths'])}")
    if report["metadata_mismatches"]:
        issues.append(f"metadata mismatches: {len(report['metadata_mismatches'])}")
    if report["cross_split_exact_duplicate_groups"]:
        issues.append(
            "cross-split exact duplicate groups: "
            f"{len(report['cross_split_exact_duplicate_groups'])}"
        )
    if report["cross_split_near_duplicate_groups"]:
        issues.append(
            "cross-split near-duplicate groups: "
            f"{len(report['cross_split_near_duplicate_groups'])}"
        )
    if report["cross_split_source_groups"]:
        issues.append(
            f"cross-split source groups: {len(report['cross_split_source_groups'])}"
        )
    if report["cross_split_participant_groups"]:
        issues.append(
            "participant leakage across splits: "
            f"{len(report['cross_split_participant_groups'])}"
        )
    if report["cross_split_session_groups"]:
        issues.append(
            "session leakage across splits: "
            f"{len(report['cross_split_session_groups'])}"
        )
    return issues


def audit_dataset(root: Path, class_names: list[str]) -> dict:
    counts = {
        split: {name: 0 for name in class_names}
        for split in ("train", "valid", "test")
    }
    records: list[dict] = []
    corrupt: list[dict] = []
    unsupported_files: list[str] = []
    metadata, metadata_errors = load_metadata(root)

    for split in counts:
        for name in class_names:
            class_dir = root / split / name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(root).as_posix()
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    unsupported_files.append(relative_path)
                    continue
                counts[split][name] += 1
                metadata_row = metadata_for_path(metadata, relative_path)
                record = {
                    "path": relative_path,
                    "split": split,
                    "class_name": name,
                    "source_key": source_key(path),
                    "sha256": sha256(path),
                }
                for field in METADATA_FIELDS:
                    record[field] = metadata_row.get(field)
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        record["width"], record["height"] = image.size
                        record["format"] = image.format
                        record["perceptual_hash"] = f"{perceptual_hash(path):016x}"
                except Exception as exc:  # Pillow raises several format-specific errors.
                    record["error"] = str(exc)
                    corrupt.append(record)
                records.append(record)

    hash_groups = defaultdict(list)
    source_groups = defaultdict(list)
    participant_groups = defaultdict(list)
    session_groups = defaultdict(list)
    metadata_mismatches = []
    missing_metadata_paths = []
    for record in records:
        hash_groups[record["sha256"]].append(record)
        source_groups[record["source_key"]].append(record)
        participant = record.get("participant_id")
        session = record.get("session_id")
        if participant:
            participant_groups[participant].append(record)
        if session:
            session_groups[session].append(record)
        metadata_row = metadata_for_path(metadata, record["path"])
        missing = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if not metadata_row.get(field)
        ]
        if not metadata_row:
            missing = ["metadata_row"]
        if missing:
            missing_metadata_paths.append({"path": record["path"], "fields": missing})
        if metadata_row:
            if metadata_row.get("label", metadata_row.get("class_name")) != record["class_name"]:
                metadata_mismatches.append(
                    {"path": record["path"], "field": "label", "value": metadata_row.get("label")}
                )
            if metadata_row.get("split") and metadata_row["split"] != record["split"]:
                metadata_mismatches.append(
                    {"path": record["path"], "field": "split", "value": metadata_row["split"]}
                )
        elif record["path"] not in metadata:
            continue

    empty_classes = {
        split: [name for name, count in values.items() if count == 0]
        for split, values in counts.items()
    }
    split_totals = {split: sum(values.values()) for split, values in counts.items()}
    balance = {
        split: {
            "minimum": min(values.values()) if values else 0,
            "maximum": max(values.values()) if values else 0,
            "total": split_totals[split],
            "ratio": (
                min(values.values()) / max(values.values())
                if values and max(values.values())
                else 0.0
            ),
        }
        for split, values in counts.items()
    }

    dimensions = Counter(
        f"{record.get('width', '?')}x{record.get('height', '?')}"
        for record in records
        if "error" not in record
    )
    aspect_ratios = Counter(
        f"{record['width'] / record['height']:.3f}"
        for record in records
        if "error" not in record and record.get("height")
    )
    near_duplicates = _near_duplicate_groups(records)
    cross_split_near_duplicates = _cross_split_groups(
        {
            index: values
            for index, values in enumerate(near_duplicates)
        }
    )
    return {
        "dataset_root": str(root.resolve()),
        "classes": class_names,
        "counts": counts,
        "total_images": len(records),
        "empty_classes": empty_classes,
        "balance": balance,
        "corrupt_images": corrupt,
        "unsupported_files": unsupported_files,
        "metadata_errors": metadata_errors,
        "missing_metadata_paths": missing_metadata_paths,
        "metadata_mismatches": metadata_mismatches,
        "metadata_rows": len(metadata),
        "dimensions": dict(dimensions),
        "aspect_ratios": dict(aspect_ratios),
        "exact_duplicate_groups": [
            values for values in hash_groups.values() if len(values) > 1
        ],
        "near_duplicate_groups": near_duplicates,
        "cross_split_exact_duplicate_groups": _cross_split_groups(hash_groups),
        "cross_split_near_duplicate_groups": cross_split_near_duplicates,
        "cross_split_source_groups": _cross_split_groups(source_groups),
        "cross_split_participant_groups": _cross_split_groups(participant_groups),
        "cross_split_session_groups": _cross_split_groups(session_groups),
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
            "## Balance",
            "",
            *[
                f"- {split}: min={values['minimum']} max={values['maximum']} "
                f"ratio={values['ratio']:.3f}"
                for split, values in report["balance"].items()
            ],
            "",
            "## Integrity checks",
            "",
            f"- Corrupt images: {len(report['corrupt_images'])}",
        f"- Unsupported files: {len(report['unsupported_files'])}",
            f"- Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"- Near-duplicate groups: {len(report['near_duplicate_groups'])}",
            f"- Cross-split exact duplicate groups: {len(report['cross_split_exact_duplicate_groups'])}",
        f"- Cross-split near-duplicate groups: {len(report['cross_split_near_duplicate_groups'])}",
            f"- Cross-split source-key groups: {len(report['cross_split_source_groups'])}",
        f"- Cross-split participant groups: {len(report['cross_split_participant_groups'])}",
        f"- Cross-split session groups: {len(report['cross_split_session_groups'])}",
        f"- Images missing metadata: {len(report['missing_metadata_paths'])}",
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when the dataset is not safe for training.",
    )
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
    issues = strict_issues(report) if args.strict else (
        ["corrupt images"] if report["corrupt_images"] else []
    )
    if issues:
        print("Audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
