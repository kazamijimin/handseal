"""OpenCV rendering for the two-player battle prototype."""

from __future__ import annotations


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for the battle UI. Install requirements.txt first."
        ) from exc
    return cv2


def _text(cv2, frame, text, position, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _health_bar(cv2, frame, x, y, width, hp, color):
    height = 18
    cv2.rectangle(frame, (x, y), (x + width, y + height), (45, 45, 45), -1)
    filled = int(width * max(0, min(100, hp)) / 100)
    cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (230, 230, 230), 1)


def draw_battle_overlay(frame, battle, predictions, snapshots, mode: str):
    cv2 = _cv2()
    height, width = frame.shape[:2]
    midpoint = width // 2
    divider = max(2, int(width * 0.02))
    cv2.line(
        frame,
        (midpoint, 0),
        (midpoint, height),
        (190, 190, 190),
        divider,
    )
    cv2.rectangle(frame, (5, 5), (midpoint - 5, height - 5), (80, 120, 220), 2)
    cv2.rectangle(frame, (midpoint + 5, 5), (width - 5, height - 5), (220, 120, 80), 2)

    player1 = battle.players["player1"]
    player2 = battle.players["player2"]
    prediction1 = predictions["player1"]
    prediction2 = predictions["player2"]
    snapshot1 = snapshots["player1"]
    snapshot2 = snapshots["player2"]

    _text(cv2, frame, "PLAYER 1", (20, 32), color=(255, 220, 120))
    _text(cv2, frame, "PLAYER 2", (midpoint + 20, 32), color=(120, 220, 255))
    _health_bar(cv2, frame, 20, 48, max(80, midpoint - 50), player1.hp, (80, 180, 255))
    _health_bar(
        cv2,
        frame,
        midpoint + 20,
        48,
        max(80, midpoint - 50),
        player2.hp,
        (255, 150, 80),
    )
    _text(
        cv2,
        frame,
        f"{prediction1.label} {prediction1.confidence:.0%}",
        (20, 95),
    )
    _text(
        cv2,
        frame,
        f"{prediction2.label} {prediction2.confidence:.0%}",
        (midpoint + 20, 95),
    )
    _text(cv2, frame, " → ".join(snapshot1.sequence) or "? → ?", (20, 125))
    _text(cv2, frame, " → ".join(snapshot2.sequence) or "? → ?", (midpoint + 20, 125))
    _text(
        cv2,
        frame,
        f"Round {battle.round_number} | {battle.phase} | {mode}",
        (20, height - 55),
        scale=0.55,
    )
    _text(
        cv2,
        frame,
        "Demo keys: P1 1-5 | P2 7-0/- | R reset | Q quit",
        (20, height - 28),
        scale=0.5,
        color=(220, 220, 220),
    )

    if battle.last_resolution is not None:
        resolution = battle.last_resolution
        _text(
            cv2,
            frame,
            f"P1: {resolution.player1_jutsu.name} (-{resolution.damage_to_player2})",
            (20, 160),
            color=(255, 220, 120),
        )
        _text(
            cv2,
            frame,
            f"P2: {resolution.player2_jutsu.name} (-{resolution.damage_to_player1})",
            (midpoint + 20, 160),
            color=(120, 220, 255),
        )
    if battle.winner:
        winner_text = "DRAW" if battle.winner == "draw" else f"{battle.winner.upper()} WINS"
        _text(
            cv2,
            frame,
            winner_text,
            (max(20, midpoint - 120), height // 2),
            scale=1.0,
            color=(0, 255, 255),
            thickness=3,
        )
    return frame
