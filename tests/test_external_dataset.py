import unittest
import tempfile
from pathlib import Path
import json
from PIL import Image
from data_tools.audit_dataset import audit_dataset

from data_tools.prepare_external_dataset import baseline_issues, group_related, assign_splits, source_images


def record(name, label="dragon", phash=0, participant=None, native=None):
    return {"source_key": name, "sha256": name, "label": label,
            "perceptual_hash": phash, "participant_id": participant,
            "native_split": native}


class ExternalDatasetTests(unittest.TestCase):
    def test_audit_uses_manifest_source_identity_and_checks_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for split, color in (("train", "red"), ("valid", "blue")):
                path = root/split/"dragon"/f"{split}.jpg"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (32, 32), color).save(path)
                rows.append({"path": path.relative_to(root).as_posix(), "label": "dragon",
                             "split": split, "source_key": "shared-original-capture",
                             "width": 100, "height": 100, "sha256": "wrong-digest"})
            (root/"metadata.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            report = audit_dataset(root, ["dragon"])
            self.assertEqual(len(report["cross_split_source_groups"]), 1)
            fields = {row["field"] for row in report["metadata_mismatches"]}
            self.assertEqual(fields, {"width", "height", "sha256"})
            self.assertTrue(baseline_issues(report))

    def test_source_images_include_png_and_jpeg_without_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a.jpg", "b.png", "c.jpeg", "README.txt"):
                (root/name).touch()
            self.assertEqual([p.name for p in source_images(root)], ["a.jpg", "b.png", "c.jpeg"])

    def test_near_duplicate_links_are_transitive(self):
        rows = [record("a", phash=0), record("b", phash=15), record("c", phash=255)]
        self.assertEqual(len(group_related(rows)), 1)

    def test_unknown_people_are_not_invented_or_grouped_as_one_person(self):
        rows = [record("a", phash=0), record("b", phash=2**64-1)]
        self.assertEqual(len(group_related(rows)), 2)
        rows[0]["participant_id"] = rows[1]["participant_id"] = "known-person"
        self.assertEqual(len(group_related(rows)), 1)

    def test_hagrid_subject_splits_are_preserved_by_excluding_conflicts(self):
        group = [record("a", "no_seal", native="train"),
                 record("b", "no_seal", native="test")]
        selected, excluded = assign_splits([group])
        self.assertEqual([(r["source_key"], split) for r, split in selected], [("a", "train")])
        self.assertEqual([r["source_key"] for r in excluded], ["b"])

    def test_only_missing_capture_metadata_is_allowed_for_baseline(self):
        report = {"empty_classes": {}, "corrupt_images": [], "unsupported_files": [],
                  "metadata_errors": [], "metadata_mismatches": [],
                  "missing_metadata_paths": [{"path": "a", "fields": ["participant_id", "session_id"]}]}
        for kind in ("exact_duplicate", "near_duplicate", "source", "participant", "session"):
            report[f"cross_split_{kind}_groups"] = []
        self.assertEqual(baseline_issues(report), [])
        report["missing_metadata_paths"][0]["fields"] = ["metadata_row"]
        self.assertTrue(baseline_issues(report))
        report["missing_metadata_paths"] = []
        report["cross_split_exact_duplicate_groups"] = [[{}, {}]]
        self.assertTrue(baseline_issues(report))


if __name__ == "__main__":
    unittest.main()
