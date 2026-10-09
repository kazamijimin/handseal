"""OpenCV rendering for the single-player target-sign challenge."""

from __future__ import annotations

from .sign_labels import display_name, prediction_words


def draw_challenge_overlay(
    frame,
    snapshot,
    prediction,
    crop_bounds,
    confidence_threshold: float,
    voice_enabled: bool,
    voice_error: str | None = None,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the challenge UI.") from exc

    x1, y1, x2, y2 = crop_bounds
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    target_text = (
        f"Perform: {display_name(snapshot.target_label)}"
        if snapshot.target_label
        else "Challenge complete"
    )
    if snapshot.phase == "complete":
        headline = f"Final score: {snapshot.score}"
        detail = (
            f"Accuracy {snapshot.accuracy:.0%} | "
            f"Avg response {snapshot.average_response_time:.1f}s"
        )
        summary = f"Best streak {snapshot.best_streak}"
    else:
        headline, detail = prediction_words(
            prediction.label,
            prediction.confidence,
            confidence_threshold,
        )

    lines = [
        (target_text, (0, 255, 255), 0.85),
        (headline, (0, 255, 0), 0.70),
        (detail, (220, 220, 220), 0.50),
        (
            f"Round {snapshot.round_number}/{snapshot.total_rounds}  "
            f"Score {snapshot.score}  Streak {snapshot.streak}",
            (255, 255, 255),
            0.58,
        ),
        (
            f"Time {snapshot.elapsed_seconds:.1f}s / 10.0s  "
            f"Stable {snapshot.stable_progress:.0%}",
            (220, 220, 220),
            0.52,
        ),
    ]
    if snapshot.phase == "complete":
        lines.append((summary, (255, 255, 255), 0.58))
    if snapshot.feedback:
        lines.append((snapshot.feedback, (0, 220, 255), 0.70))
    if voice_error:
        lines.append((f"Voice unavailable: {voice_error[:55]}", (0, 150, 255), 0.45))
    lines.append(
        (
            f"Voice {'ON' if voice_enabled else 'OFF'} | R restart | V voice | Q/Esc quit",
            (230, 230, 230),
            0.48,
        )
    )

    y = 38
    for text, color, scale in lines:
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 31
    return frame
