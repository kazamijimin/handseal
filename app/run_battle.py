"""Run the local two-player HandSeal webcam battle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .artifact_inference import Prediction, TorchScriptSignPredictor
from .battle.state import BattleState, JutsuCatalog
from .camera import open_camera, split_player_zones
from .demo_input import DemoInput
from .ui import draw_battle_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUTSU = PROJECT_ROOT / "app" / "battle" / "jutsu.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "model"), default="demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--metadata", help="Export metadata JSON for model mode.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jutsu-config", default=str(DEFAULT_JUTSU))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "model" and not args.metadata:
        raise SystemExit("--metadata is required when --mode model is selected.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for the battle webcam. Install requirements.txt first."
        ) from exc

    catalog = JutsuCatalog.from_yaml(args.jutsu_config)
    battle = BattleState(catalog)
    demo = DemoInput() if args.mode == "demo" else None
    predictor = (
        TorchScriptSignPredictor.from_metadata(args.metadata, device=args.device)
        if args.mode == "model"
        else None
    )

    try:
        camera = open_camera(args.camera)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("The camera returned an unreadable frame.")
            now = time.monotonic()
            left_frame, right_frame = split_player_zones(frame)

            if demo is not None:
                raw_predictions = demo.predictions()
                display_predictions = {
                    player_id: Prediction(label, confidence)
                    for player_id, (label, confidence) in raw_predictions.items()
                }
            else:
                display_predictions = {
                    "player1": predictor.predict(left_frame),
                    "player2": predictor.predict(right_frame),
                }
                raw_predictions = {
                    player_id: (prediction.label, prediction.confidence)
                    for player_id, prediction in display_predictions.items()
                }

            battle.update(raw_predictions, now)
            snapshots = {
                player_id: player.tracker.snapshot(now)
                for player_id, player in battle.players.items()
            }
            draw_battle_overlay(frame, battle, display_predictions, snapshots, args.mode)
            cv2.imshow("HandSeal Two-Player Jutsu Battle", frame)
            key = cv2.waitKey(1) & 0xFF
            if demo is not None:
                action = demo.handle_key(key)
                if action == "reset":
                    battle.reset_match()
                elif action == "quit":
                    break
            elif key in (ord("q"), 27):
                break
            elif key == ord("r"):
                battle.reset_match()
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
