"""Run a single-player webcam hand-sign check without battle logic."""

from __future__ import annotations

import argparse
import time

from .artifact_inference import TorchScriptSignPredictor
from .challenge import ChallengeSession
from .challenge_ui import draw_challenge_overlay
from .camera import model_input, model_input_bounds, open_camera
from .sign_labels import prediction_words
from .sign_labels import display_name
from .voice import OptionalVoice


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preview", "model", "challenge"),
        default="preview",
        help="Preview, recognize signs, or run the target-sign challenge.",
    )
    parser.add_argument("--metadata", help="Export metadata JSON for model mode.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-ratio", type=float, default=0.60)
    parser.add_argument(
        "--input-mode",
        choices=("center", "full"),
        default="center",
        help="Use the trained center crop or the complete camera frame.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"model", "challenge"} and not args.metadata:
        raise SystemExit("--metadata is required for model and challenge modes.")
    if not 0 < args.crop_ratio <= 1:
        raise SystemExit("--crop-ratio must be greater than 0 and at most 1.")
    if not 0 < args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for the webcam. Install requirements.txt first."
        ) from exc

    try:
        predictor = (
            TorchScriptSignPredictor.from_metadata(
                args.metadata,
                device=args.device,
                resize_mode=("letterbox" if args.input_mode == "full" else "fit"),
            )
            if args.mode in {"model", "challenge"}
            else None
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not load model metadata: {exc}") from exc

    challenge = ChallengeSession(seed=args.seed) if args.mode == "challenge" else None
    voice = OptionalVoice() if challenge else None
    announced_target: str | None = None
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

            crop = model_input(frame, args.input_mode, args.crop_ratio)
            display = cv2.flip(frame, 1)
            x1, y1, x2, y2 = model_input_bounds(
                frame.shape,
                args.input_mode,
                args.crop_ratio,
            )

            if predictor is None:
                headline = "PREVIEW ONLY - no model loaded"
                detail = "Train and export a model to see hand-sign words"
                prediction = None
            else:
                prediction = predictor.predict(crop)
                if challenge is not None:
                    result = challenge.update_prediction(
                        prediction.label,
                        prediction.confidence,
                        now,
                    )
                    if result and voice is not None and voice.enabled:
                        if result.correct:
                            voice.speak(f"Correct. {display_name(result.target_label)}")
                        else:
                            voice.speak("Time up")
                    snapshot = challenge.snapshot(now)
                    if snapshot.target_label != announced_target and snapshot.target_label:
                        announced_target = snapshot.target_label
                        if voice is not None and voice.enabled:
                            voice.speak(f"Perform {display_name(announced_target)}")
                else:
                    headline, detail = prediction_words(
                        prediction.label,
                        prediction.confidence,
                        args.confidence_threshold,
                    )

            if challenge is not None and prediction is not None:
                snapshot = challenge.snapshot(now)
                draw_challenge_overlay(
                    display,
                    snapshot,
                    prediction,
                    (x1, y1, x2, y2),
                    args.confidence_threshold,
                    voice.enabled if voice is not None else False,
                    voice.error if voice is not None else None,
                )
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    display,
                    headline,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0) if predictor is not None else (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    detail,
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "Q or Esc: quit",
                    (20, display.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            window_title = (
                "HandSeal Target Sign Challenge"
                if challenge is not None
                else "HandSeal Single Hand-Sign Check"
            )
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if challenge is not None:
                if key == ord("r"):
                    challenge.reset(now)
                    announced_target = None
                elif key == ord("v") and voice is not None:
                    voice.toggle()
                    if voice.enabled and challenge.current_target:
                        voice.speak(f"Perform {display_name(challenge.current_target)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
