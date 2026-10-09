"""Human-readable words shown for model sign predictions."""

from __future__ import annotations


DISPLAY_NAMES = {
    "dragon": "Dragon",
    "ram": "Ram",
    "snake": "Snake",
    "tiger": "Tiger",
    "no_seal": "No hand sign",
}


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def prediction_words(label: str, confidence: float, threshold: float = 0.80):
    """Return the headline and supporting text for the webcam display."""
    name = display_name(label)
    if label == "no_seal":
        headline = "No hand sign detected"
    elif confidence < threshold:
        headline = "Uncertain hand sign"
    else:
        headline = f"Detected hand sign: {name}"
    detail = f"Best guess: {name} | confidence {confidence:.1%}"
    return headline, detail
