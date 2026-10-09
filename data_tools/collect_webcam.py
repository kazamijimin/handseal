"""Collect cropped webcam samples into a separate dataset root.

The default output is ``handseal_collected`` so the existing source dataset is
not changed accidentally. Use one participant identifier per person and collect
all configured classes under the same split policy.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .metadata import METADATA_FILENAME, append_metadata


CLASSES = ["dragon", "ram", "snake", "tiger", "no_seal"]


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    if not token:
        raise ValueError("participant and session identifiers must not be empty")
    return token


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_center(frame, ratio: float):
    height, width = frame.shape[:2]
    side = max(1, int(min(width, height) * ratio))
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--label", choices=CLASSES, required=True)
    parser.add_argument("--output-root", default="handseal_collected")
    parser.add_argument("--lighting", default="unspecified")
    parser.add_argument("--background", default="unspecified")
    parser.add_argument("--distance", default="unspecified")
    parser.add_argument("--hand-position", default="unspecified")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--images-per-clip", type=int, default=18)
    parser.add_argument("--crop-ratio", type=float, default=0.60)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for webcam collection. Install requirements.txt first."
        ) from exc

    if args.images_per_clip < 1 or args.seconds <= 0:
        raise SystemExit("--seconds must be positive and --images-per-clip must be at least 1.")

    participant = safe_token(args.participant)
    session = safe_token(args.session)
    output_root = Path(args.output_root)
    output_dir = output_root / args.split / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / METADATA_FILENAME
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    saved_count = len(
        list(output_dir.glob(f"{participant}_{session}_{args.label}_*.jpg"))
    )
    recording = False
    started_at = 0.0
    next_capture = 0.0
    captured = 0
    interval = args.seconds / args.images_per_clip

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("The camera returned an unreadable frame.")
            display = cv2.flip(frame, 1)
            height, width = display.shape[:2]
            side = int(min(width, height) * args.crop_ratio)
            x1 = (width - side) // 2
            y1 = (height - side) // 2
            cv2.rectangle(display, (x1, y1), (x1 + side, y1 + side), (255, 255, 255), 2)
            status = "RECORDING" if recording else "READY"
            cv2.putText(
                display,
                f"{args.participant} | {args.split}/{args.label} | {status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "R: record clip   Q: quit",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("HandSeal Dataset Collector", display)

            now = time.monotonic()
            if recording and now >= next_capture:
                crop = crop_center(frame, args.crop_ratio)
                filename = f"{participant}_{session}_{args.label}_{saved_count + captured + 1:04d}.jpg"
                image_path = output_dir / filename
                if not cv2.imwrite(str(image_path), crop):
                    raise SystemExit(f"Could not write {image_path}.")
                relative_path = image_path.relative_to(output_root).as_posix()
                append_metadata(
                    metadata_path,
                    {
                        "path": relative_path,
                        "label": args.label,
                        "participant_id": args.participant,
                        "session_id": args.session,
                        "split": args.split,
                        "lighting": args.lighting,
                        "background": args.background,
                        "distance": args.distance,
                        "hand_position": args.hand_position,
                        "width": int(crop.shape[1]),
                        "height": int(crop.shape[0]),
                        "sha256": sha256(image_path),
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                captured += 1
                next_capture += interval

            if recording and now - started_at >= args.seconds:
                recording = False
                saved_count += captured
                print(f"Saved {captured} images to {output_dir}.")
                captured = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not recording:
                recording = True
                started_at = time.monotonic()
                next_capture = started_at
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
