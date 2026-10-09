from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from app.artifact_inference import TorchScriptSignPredictor
from app.battle.state import BattleState, Jutsu, JutsuCatalog, SequenceTracker
from app.camera import center_crop, center_square_bounds, open_camera, split_player_zones
from app.demo_input import DemoInput
from app.sign_labels import display_name, prediction_words


JUTSU_PATH = Path(__file__).resolve().parents[1] / "app" / "battle" / "jutsu.yaml"


def feed_sign(tracker: SequenceTracker, label: str, start: float) -> float:
    tracker.update(label, 0.95, start)
    end = start + tracker.hold_seconds
    tracker.update(label, 0.95, end)
    return end


class FiveLogits(torch.nn.Module):
    def forward(self, inputs):
        return torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]], device=inputs.device).repeat(
            inputs.shape[0], 1
        )


class BattleTests(unittest.TestCase):
    def test_player_zones_split_with_center_gap(self):
        frame = np.zeros((10, 100, 3), dtype=np.uint8)
        left, right = split_player_zones(frame)
        self.assertEqual(left.shape[0], 10)
        self.assertEqual(right.shape[0], 10)
        self.assertLess(left.shape[1] + right.shape[1], frame.shape[1])

    def test_center_crop_matches_collector_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x1, y1, x2, y2 = center_square_bounds(frame.shape, 0.60)
        crop = center_crop(frame, 0.60)
        self.assertEqual(crop.shape, (y2 - y1, x2 - x1, 3))
        self.assertEqual(crop.shape[0], crop.shape[1])

    def test_camera_dependency_error_is_clear(self):
        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "OpenCV is required"):
                open_camera(0)

    def test_prediction_words_are_human_readable(self):
        self.assertEqual(display_name("dragon"), "Dragon")
        headline, detail = prediction_words("dragon", 0.95)
        self.assertEqual(headline, "Detected hand sign: Dragon")
        self.assertIn("confidence 95.0%", detail)

        headline, _ = prediction_words("snake", 0.50)
        self.assertEqual(headline, "Uncertain hand sign")
        headline, _ = prediction_words("no_seal", 0.99)
        self.assertEqual(headline, "No hand sign detected")

    def test_confidence_rejection(self):
        tracker = SequenceTracker()
        tracker.update("dragon", 0.79, 0.0)
        event = tracker.update("dragon", 0.79, 2.0)
        self.assertIsNone(event.accepted_sign)
        self.assertEqual(event.sequence, ())

    def test_stable_hold_accepts_first_sign(self):
        tracker = SequenceTracker()
        feed_sign(tracker, "dragon", 0.0)
        self.assertEqual(tuple(tracker.sequence), ("dragon",))
        self.assertFalse(tracker.complete)

    def test_duplicate_requires_neutral_transition(self):
        tracker = SequenceTracker()
        feed_sign(tracker, "dragon", 0.0)
        tracker.update("dragon", 0.95, 1.0)
        tracker.update("ram", 0.95, 1.5)
        self.assertEqual(tuple(tracker.sequence), ("dragon",))

    def test_sequence_timeout_resets_incomplete_sequence(self):
        tracker = SequenceTracker()
        feed_sign(tracker, "dragon", 0.0)
        event = tracker.update("no_seal", 0.99, 6.0)
        self.assertEqual(event.sequence, ())

    def test_jutsu_lookup_and_unknown_pair(self):
        catalog = JutsuCatalog.from_yaml(JUTSU_PATH)
        self.assertEqual(catalog.lookup(("dragon", "ram")).name, "Flame Burst")
        self.assertEqual(catalog.lookup(("dragon", "tiger")).damage, 0)
        self.assertEqual(catalog.lookup(("dragon", "tiger")).name, "Unknown Jutsu")

    def test_both_completed_sequences_resolve_simultaneously(self):
        battle = BattleState(JutsuCatalog.from_yaml(JUTSU_PATH), round_result_seconds=1.0)
        battle.update(
            {"player1": ("no_seal", 0.99), "player2": ("no_seal", 0.99)},
            0.0,
        )
        battle.update(
            {"player1": ("dragon", 0.95), "player2": ("snake", 0.95)},
            0.1,
        )
        battle.update(
            {"player1": ("dragon", 0.95), "player2": ("snake", 0.95)},
            0.85,
        )
        battle.update(
            {"player1": ("no_seal", 0.99), "player2": ("no_seal", 0.99)},
            1.0,
        )
        battle.update(
            {"player1": ("no_seal", 0.99), "player2": ("no_seal", 0.99)},
            1.4,
        )
        battle.update(
            {"player1": ("ram", 0.95), "player2": ("tiger", 0.95)},
            1.5,
        )
        battle.update(
            {"player1": ("ram", 0.95), "player2": ("tiger", 0.95)},
            2.25,
        )

        self.assertEqual(battle.phase, "round_result")
        self.assertEqual(battle.players["player1"].hp, 80)
        self.assertEqual(battle.players["player2"].hp, 75)
        self.assertEqual(battle.last_resolution.player1_jutsu.name, "Flame Burst")
        self.assertEqual(battle.last_resolution.player2_jutsu.name, "Shadow Bind")

    def test_game_over_and_reset(self):
        catalog = JutsuCatalog(
            {("dragon", "ram"): Jutsu(("dragon", "ram"), "Test Strike", 100)},
            Jutsu((), "Unknown Jutsu", 0),
        )
        battle = BattleState(catalog)
        battle.players["player1"].tracker.sequence = ["dragon", "ram"]
        battle.players["player1"].tracker._complete = True
        battle.players["player2"].tracker.sequence = ["dragon", "ram"]
        battle.players["player2"].tracker._complete = True
        battle.resolve_round(0.0)
        self.assertEqual(battle.phase, "game_over")
        self.assertEqual(battle.winner, "draw")
        battle.reset_match()
        self.assertEqual(battle.players["player1"].hp, 100)
        self.assertEqual(battle.phase, "collecting")

    def test_demo_input_operates_without_model_artifact(self):
        demo = DemoInput()
        demo.handle_key(ord("1"))
        demo.handle_key(ord("8"))
        self.assertEqual(demo.predictions()["player1"], ("dragon", 0.99))
        self.assertEqual(demo.predictions()["player2"], ("ram", 0.99))
        self.assertEqual(demo.handle_key(ord("r")), "reset")
        self.assertEqual(demo.predictions()["player1"][0], "no_seal")

    def test_torchscript_metadata_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "demo.pt"
            metadata_path = root / "demo.metadata.json"
            scripted = torch.jit.trace(FiveLogits(), torch.zeros(1, 3, 224, 224))
            scripted.save(model_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_path": model_path.name,
                        "class_names": ["dragon", "ram", "snake", "tiger", "no_seal"],
                        "image_size": 224,
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    }
                ),
                encoding="utf-8",
            )
            predictor = TorchScriptSignPredictor.from_metadata(metadata_path)
            prediction = predictor.predict(np.zeros((40, 40, 3), dtype=np.uint8))
            self.assertEqual(prediction.label, "snake")
            self.assertGreater(prediction.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
