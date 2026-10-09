"""Keyboard-controlled predictions for testing the battle without a model."""

from __future__ import annotations

from dataclasses import dataclass


SIGN_KEYS = {
    "player1": {
        ord("1"): "dragon",
        ord("2"): "ram",
        ord("3"): "snake",
        ord("4"): "tiger",
        ord("5"): "no_seal",
    },
    "player2": {
        ord("7"): "dragon",
        ord("8"): "ram",
        ord("9"): "snake",
        ord("0"): "tiger",
        ord("-"): "no_seal",
    },
}


@dataclass
class DemoInput:
    """Hold a selected sign until another key changes the simulated prediction."""

    confidence: float = 0.99

    def __post_init__(self) -> None:
        self._labels = {"player1": "no_seal", "player2": "no_seal"}

    def handle_key(self, key: int) -> str | None:
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("r"):
            self._labels = {"player1": "no_seal", "player2": "no_seal"}
            return "reset"
        for player_id, key_map in SIGN_KEYS.items():
            if key in key_map:
                self._labels[player_id] = key_map[key]
                return None
        return None

    def predictions(self) -> dict[str, tuple[str, float]]:
        return {
            player_id: (label, self.confidence)
            for player_id, label in self._labels.items()
        }

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)
