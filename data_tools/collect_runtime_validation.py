"""Capture model-input crops for webcam-only validation, never for training."""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from .collect_webcam import safe_token
from .metadata import append_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Export metadata JSON for the trained model.")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("centered", "multi_person", "distance", "background", "transition"),
    )
    parser.add_argument(
        "--expected-label",
        choices=("dragon", "ram", "snake", "tiger", "no_seal", "transition"),
        default="transition",
    )
    parser.add_argument("--output-root", default="runtime_validation")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--crop-ratio", type=float, default=0.60)
    parser.add_argument("--input-mode", choices=("center", "full"), default="center")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        raise SystemExit("--seconds and --interval must be positive.")
    if not 0 < args.crop_ratio <= 1:
        raise SystemExit("--crop-ratio must be greater than 0 and at most 1.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install requirements.txt first.") from exc

    from app.artifact_inference import TorchScriptSignPredictor
    from app.camera import model_input, model_input_bounds, open_camera
    from app.sign_labels import prediction_words

    try:
        predictor = TorchScriptSignPredictor.from_metadata(
            args.metadata,
            device=args.device,
            resize_mode=("letterbox" if args.input_mode == "full" else "fit"),
        )
        camera = open_camera(args.camera)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not start runtime validation: {exc}") from exc

    participant = safe_token(args.participant)
    session = safe_token(args.session)
    output_root = Path(args.output_root)
    output_dir = output_root / args.scenario / session
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"
    prefix = f"{participant}_{session}_{args.scenario}"
    saved_count = len(list(output_dir.glob(f"{prefix}_*.jpg")))
    started_at = time.monotonic()
    next_capture = started_at

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("The camera returned an unreadable frame.")
            now = time.monotonic()
            input_frame = model_input(frame, args.input_mode, args.crop_ratio)
            prediction = predictor.predict(input_frame)
            display = cv2.flip(frame, 1)
            height, width = display.shape[:2]
            x1, y1, x2, y2 = model_input_bounds(
                frame.shape,
                args.input_mode,
                args.crop_ratio,
            )
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)
            headline, detail = prediction_words(prediction.label, prediction.confidence)
            cv2.putText(display, headline, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, detail, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
            cv2.putText(
                display,
                f"VALIDATION: {args.scenario} | expected={args.expected_label} | Q: quit",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("HandSeal Runtime Validation", display)

            if now >= next_capture:
                crop = input_frame
                filename = f"{prefix}_{saved_count + 1:04d}.jpg"
                image_path = output_dir / filename
                if not cv2.imwrite(str(image_path), crop):
                    raise SystemExit(f"Could not write {image_path}.")
                append_metadata(
                    metadata_path,
                    {
                        "path": image_path.relative_to(output_root).as_posix(),
                        "purpose": "runtime_validation",
                        "participant_id": args.participant,
                        "session_id": args.session,
                        "scenario": args.scenario,
                        "expected_label": args.expected_label,
                        "predicted_label": prediction.label,
                        "confidence": prediction.confidence,
                        "split": "runtime_validation",
                        "width": int(crop.shape[1]),
                        "height": int(crop.shape[0]),
                        "sha256": sha256(image_path),
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                saved_count += 1
                next_capture += args.interval

            if now - started_at >= args.seconds:
                break
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved_count} runtime-validation images to {output_dir}.")
    print("These files are not part of handseal_dataset and are not used for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
