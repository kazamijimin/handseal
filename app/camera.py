"""Camera access and deterministic left/right player-zone cropping."""

from __future__ import annotations


def split_player_zones(frame, divider_ratio: float = 0.02):
    """Return left and right views, leaving a small center divider gap."""
    if frame is None or len(frame.shape) < 2:
        raise ValueError("A valid image frame is required.")
    width = frame.shape[1]
    divider = max(2, int(width * divider_ratio))
    midpoint = width // 2
    left_end = midpoint - divider // 2
    right_start = midpoint + (divider - divider // 2)
    if left_end <= 0 or right_start >= width:
        raise ValueError("Frame is too narrow for two player zones.")
    return frame[:, :left_end], frame[:, right_start:]


def open_camera(index: int = 0):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for the battle webcam. Install requirements.txt first."
        ) from exc
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {index}.")
    return camera
