from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from data_tools.audit_dataset import audit_dataset
from training.config import class_names, load_config
from training.datasets import HandSealTransform
from training.models import build_model


class SetupTests(unittest.TestCase):
    def test_default_config_has_five_classes(self):
        config = load_config("training/configs/default.yaml")
        self.assertEqual(
            class_names(config),
            ["dragon", "ram", "snake", "tiger", "no_seal"],
        )

    def test_small_cnn_output_shape(self):
        model = build_model("small_cnn", num_classes=5)
        output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_preprocessing_shape_and_dtype(self):
        transform = HandSealTransform(
            size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            training=False,
        )
        image = Image.new("RGB", (320, 240), color=(120, 80, 40))
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_audit_detects_corrupt_file_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                for name in ("dragon", "ram", "snake", "tiger", "no_seal"):
                    (root / split / name).mkdir(parents=True)
            image_path = root / "train" / "dragon" / "sample.jpg"
            Image.new("RGB", (32, 32), color="red").save(image_path)
            (root / "valid" / "dragon" / "broken.jpg").write_bytes(b"not an image")

            report = audit_dataset(
                root,
                ["dragon", "ram", "snake", "tiger", "no_seal"],
            )

            self.assertEqual(report["total_images"], 2)
            self.assertEqual(len(report["corrupt_images"]), 1)
            self.assertTrue(image_path.exists())


if __name__ == "__main__":
    unittest.main()
