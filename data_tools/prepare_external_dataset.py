"""Build an explicitly exploratory baseline from downloaded public datasets.

Unknown capture metadata stays null. The participant-strict audit is unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image, ImageOps

from .audit_dataset import DEFAULT_CLASSES, IMAGE_EXTENSIONS, audit_dataset, markdown_report, strict_issues
from .metadata import METADATA_FIELDS


def baseline_issues(report: dict) -> list[str]:
    """Permit unknown capture fields, but never absent rows or integrity/leakage errors."""
    issues = [issue for issue in strict_issues(report)
              if not issue.startswith("images missing required metadata:")]
    for missing in report["missing_metadata_paths"]:
        unexpected = set(missing["fields"]) - set(METADATA_FIELDS)
        if unexpected:
            issues.append(f"Missing verifiable metadata for {missing['path']}: {sorted(unexpected)}")
    return issues


def source_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def group_related(records: list[dict]) -> list[list[dict]]:
    """Join source images, exact/near duplicates, and known subjects transitively."""
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    keys = {}
    for i, row in enumerate(records):
        for field in ("source_key", "sha256", "participant_id"):
            if row.get(field):
                key = field, row[field]
                if key in keys:
                    union(i, keys[key])
                keys[key] = i
        for j in range(i):
            if (row["perceptual_hash"] ^ records[j]["perceptual_hash"]).bit_count() <= 4:
                union(i, j)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    return list(groups.values())


def assign_splits(groups: list[list[dict]]) -> tuple[list[tuple[dict, str]], list[dict]]:
    """Balance positive group splits; exclude negatives conflicting with their native split."""
    splits = ("train", "valid", "test")
    ratios = dict(zip(splits, (0.7, 0.15, 0.15)))
    positive = Counter(row["label"] for group in groups for row in group if not row["native_split"])
    counts = {split: Counter() for split in splits}
    selected, excluded = [], []
    ordered = sorted(groups, key=lambda g: (-sum(not r["native_split"] for r in g), g[0]["source_key"]))
    for group in ordered:
        additions = Counter(row["label"] for row in group if not row["native_split"])
        if additions:
            def score(split):
                return sum(((counts[split][label] + additions[label]) ** 2
                            - counts[split][label] ** 2)
                           / max(1, positive[label] * ratios[split]) for label in additions)
            chosen = min(splits, key=score)
            counts[chosen].update(additions)
        else:
            chosen = min((row["native_split"] for row in group), key=splits.index)
        for row in group:
            if row["native_split"] and row["native_split"] != chosen:
                excluded.append(row)
            else:
                selected.append((row, chosen))
    return selected, excluded


def image_record(path: Path, label: str, source: str, annotation: dict | None = None) -> dict:
    with Image.open(path) as original:
        image = original.convert("RGB")
        if annotation:
            boxes = annotation["annotation"]["bboxes"]
            width, height = image.size
            left = min(b[0] for b in boxes) * width
            top = min(b[1] for b in boxes) * height
            right = max(b[0] + b[2] for b in boxes) * width
            bottom = max(b[1] + b[3] for b in boxes) * height
            side = min(max(right-left, bottom-top) * 1.5, width, height)
            x = max(0, min((left+right-side)/2, width-side))
            y = max(0, min((top+bottom-side)/2, height-side))
            image = image.crop((int(x), int(y), int(x+side), int(y+side)))
        image = ImageOps.fit(image, (224, 224), method=Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    # Audit the actual encoded output pixels, including JPEG rounding.
    with Image.open(io.BytesIO(data)) as encoded:
        pixels = list(encoded.convert("L").resize((8, 8), Image.Resampling.BILINEAR).getdata())
    average = sum(pixels)/len(pixels)
    phash = 0
    for value in pixels:
        phash = (phash << 1) | int(value >= average)
    return {"source_key": f"{source}:{label}:{path.stem}", "source_path": str(path.resolve()),
            "source_dataset": source, "label": label, "sha256": hashlib.sha256(data).hexdigest(),
            "perceptual_hash": phash, "participant_id": annotation["participant_id"] if annotation else None,
            "native_split": annotation["split"] if annotation else None, "image_bytes": data,
            "original_label": "no_gesture" if annotation else label}


def prepare(naruto_root: Path, hagrid_root: Path, output: Path, seed: int = 42) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {output}")
    records = []
    for label in DEFAULT_CLASSES[:-1]:
        for split in ("train", "test"):
            records.extend(image_record(path, label, "naruto_kaggle")
                           for path in source_images(naruto_root/split/label))
    provenance = [json.loads(line) for line in (hagrid_root/"provenance.jsonl").read_text().splitlines()]
    rng = random.Random(seed)
    for split, limit in (("train", 400), ("valid", 100), ("test", 100)):
        candidates = sorted((r for r in provenance if r["split"] == split), key=lambda r: r["path"])
        rng.shuffle(candidates)
        people = set()
        picked = 0
        for row in candidates:
            if row["participant_id"] in people:
                continue
            people.add(row["participant_id"])
            records.append(image_record(hagrid_root/row["path"], "no_seal", "hagrid_v2", row))
            picked += 1
            if picked == limit:
                break
    groups = group_related(records)
    selected, excluded = assign_splits(groups)
    rows, seen = [], {}
    for record, split in selected:
        digest = record["sha256"]
        if digest in seen:
            if seen[digest] != record["label"]:
                raise ValueError("Identical image has conflicting labels")
            continue
        seen[digest] = record["label"]
        filename = record["source_dataset"] + "_" + digest + ".jpg"
        path = output/split/record["label"]/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(record["image_bytes"])
        rows.append({"path": path.relative_to(output).as_posix(), "label": record["label"],
                     "split": split, "width": 224, "height": 224, "sha256": digest,
                     **{field: record.get(field) for field in METADATA_FIELDS},
                     "source_dataset": record["source_dataset"], "source_path": record["source_path"],
                     "source_key": record["source_key"], "original_label": record["original_label"],
                     "native_split": record["native_split"], "split_policy": "exploratory_source_groups"})
    (output/"metadata.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    audit = audit_dataset(output, DEFAULT_CLASSES)
    (output/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output/"audit.md").write_text(markdown_report(audit), encoding="utf-8")
    issues = baseline_issues(audit)
    summary = {"seed": seed, "counts": audit["counts"], "images": len(rows),
               "related_groups": len(groups), "excluded_native_split_conflicts": len(excluded),
               "deduplicated_images": len(selected)-len(rows), "baseline_issues": issues,
               "participant_strict_issues": strict_issues(audit),
               "limitations": ["Naruto participant/session IDs are unknown; person-disjoint generalization is not established.",
                               "Naruto dataset license is Unknown in source metadata.",
                               "Source appearance may confound sign recognition; webcam validation remains necessary."]}
    (output/"preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if issues:
        raise ValueError("Exploratory baseline audit failed: " + "; ".join(issues))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naruto-root", type=Path, default=Path("local_setup/datasets/naruto_kaggle/extracted/Pure Naruto Hand Sign Data"))
    parser.add_argument("--hagrid-root", type=Path, default=Path("local_setup/datasets/hagrid/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("handseal_dataset_external_v2"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.naruto_root, args.hagrid_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
