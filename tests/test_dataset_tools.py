from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_tools.audit_dataset import audit_dataset, strict_issues
from data_tools.metadata import METADATA_FILENAME
from data_tools.review_dataset import create_contact_sheet
from data_tools.split_manifest import build_manifest
from training.datasets import ImageRecord, sample_weights


CLASSES = ["dragon", "ram", "snake", "tiger", "no_seal"]


def make_dataset(root: Path) -> None:
    for split in ("train", "valid", "test"):
        for label in CLASSES:
            (root / split / label).mkdir(parents=True)


class DatasetToolTests(unittest.TestCase):
    def test_audit_reports_balance_metadata_and_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            image_path = root / "train" / "dragon" / "P01_S01_dragon.jpg"
            Image.new("RGB", (32, 32), color="red").save(image_path)
            (root / "valid" / "dragon" / "P01_S02_dragon.jpg").write_bytes(
                image_path.read_bytes()
            )
            metadata = [
                {
                    "path": "train/dragon/P01_S01_dragon.jpg",
                    "label": "dragon",
                    "participant_id": "P01",
                    "session_id": "S01",
                    "split": "train",
                    "lighting": "room",
                    "background": "plain",
                    "distance": "near",
                    "hand_position": "center",
                },
                {
                    "path": "valid/dragon/P01_S02_dragon.jpg",
                    "label": "dragon",
                    "participant_id": "P01",
                    "session_id": "S02",
                    "split": "valid",
                    "lighting": "room",
                    "background": "plain",
                    "distance": "near",
                    "hand_position": "center",
                },
            ]
            (root / METADATA_FILENAME).write_text(
                "\n".join(json.dumps(row) for row in metadata) + "\n",
                encoding="utf-8",
            )

            report = audit_dataset(root, CLASSES)

            self.assertEqual(report["empty_classes"]["train"], ["ram", "snake", "tiger", "no_seal"])
            self.assertEqual(len(report["cross_split_exact_duplicate_groups"]), 1)
            self.assertEqual(len(report["cross_split_participant_groups"]), 1)
            self.assertGreaterEqual(len(report["cross_split_near_duplicate_groups"]), 1)
            self.assertTrue(strict_issues(report))

    def test_manifest_includes_capture_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            image_path = root / "train" / "dragon" / "sample.jpg"
            Image.new("RGB", (32, 32), color="blue").save(image_path)
            (root / METADATA_FILENAME).write_text(
                json.dumps(
                    {
                        "path": "train/dragon/sample.jpg",
                        "participant_id": "P01",
                        "session_id": "S01",
                        "lighting": "day",
                        "background": "plain",
                        "distance": "mid",
                        "hand_position": "center",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = build_manifest(root, CLASSES)
            self.assertEqual(rows[0]["participant_id"], "P01")
            self.assertEqual(rows[0]["session_id"], "S01")

    def test_contact_sheet_is_generated_for_visual_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.jpg"
            Image.new("RGB", (32, 32), color="green").save(image)
            output = root / "review" / "sheet.jpg"
            create_contact_sheet([image], "train/dragon", output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_balanced_sampling_weights_inverse_class_frequency(self):
        records = [
            ImageRecord(Path("a.jpg"), 0, "dragon", "train"),
            ImageRecord(Path("b.jpg"), 0, "dragon", "train"),
            ImageRecord(Path("c.jpg"), 1, "ram", "train"),
        ]
        weights = sample_weights(records).tolist()
        self.assertEqual(weights[0], weights[1])
        self.assertGreater(weights[2], weights[0])


if __name__ == "__main__":
    unittest.main()
