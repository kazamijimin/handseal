"""Temporal sign recognition and deterministic two-player battle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


SIGN_LABELS = ("dragon", "ram", "snake", "tiger", "no_seal")
ACTIVE_SIGNS = frozenset(SIGN_LABELS) - {"no_seal"}


@dataclass(frozen=True)
class Jutsu:
    signs: tuple[str, ...]
    name: str
    damage: int


class JutsuCatalog:
    """Configuration-backed lookup for ordered sign pairs."""

    def __init__(self, jutsu: dict[tuple[str, ...], Jutsu], default: Jutsu) -> None:
        self._jutsu = jutsu
        self.default = default

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JutsuCatalog":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        entries = config.get("jutsu", [])
        lookup: dict[tuple[str, ...], Jutsu] = {}
        for entry in entries:
            signs = tuple(entry.get("signs", []))
            if len(signs) != 2 or any(sign not in ACTIVE_SIGNS for sign in signs):
                raise ValueError(f"Invalid jutsu signs: {signs}")
            if signs in lookup:
                raise ValueError(f"Duplicate jutsu signs: {signs}")
            damage = int(entry.get("damage", 0))
            if damage < 0:
                raise ValueError(f"Jutsu damage cannot be negative: {signs}")
            lookup[signs] = Jutsu(signs, str(entry["name"]), damage)

        default_config = config.get("default", {})
        default = Jutsu(
            signs=(),
            name=str(default_config.get("name", "Unknown Jutsu")),
            damage=int(default_config.get("damage", 0)),
        )
        if default.damage < 0:
            raise ValueError("Default jutsu damage cannot be negative.")
        return cls(lookup, default)

    def lookup(self, signs: tuple[str, ...] | list[str]) -> Jutsu:
        return self._jutsu.get(tuple(signs), self.default)

    @property
    def entries(self) -> tuple[Jutsu, ...]:
        return tuple(self._jutsu.values())


@dataclass(frozen=True)
class SequenceEvent:
    accepted_sign: str | None
    sequence: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class SequenceSnapshot:
    sequence: tuple[str, ...]
    candidate: str | None
    candidate_progress: float
    waiting_for_neutral: bool
    complete: bool


class SequenceTracker:
    """Convert frame predictions into two stable, ordered sign locks."""

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        hold_seconds: float = 0.75,
        neutral_seconds: float = 0.35,
        sequence_timeout: float = 5.0,
        max_signs: int = 2,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.neutral_seconds = neutral_seconds
        self.sequence_timeout = sequence_timeout
        self.max_signs = max_signs
        self.reset()

    def reset(self) -> None:
        self.sequence: list[str] = []
        self._candidate: str | None = None
        self._candidate_since: float | None = None
        self._neutral_since: float | None = None
        self._last_accepted_at: float | None = None
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def snapshot(self, now: float | None = None) -> SequenceSnapshot:
        progress = 0.0
        if self._candidate_since is not None and now is not None:
            progress = min(1.0, max(0.0, (now - self._candidate_since) / self.hold_seconds))
        return SequenceSnapshot(
            sequence=tuple(self.sequence),
            candidate=self._candidate,
            candidate_progress=progress,
            waiting_for_neutral=self._neutral_since is None and bool(self.sequence),
            complete=self._complete,
        )

    def _reset_incomplete_if_timed_out(self, now: float) -> None:
        if (
            self.sequence
            and not self._complete
            and self._last_accepted_at is not None
            and now - self._last_accepted_at > self.sequence_timeout
        ):
            self.reset()

    def update(self, label: str, confidence: float, now: float) -> SequenceEvent:
        if label not in SIGN_LABELS:
            raise ValueError(f"Unknown sign label: {label}")
        self._reset_incomplete_if_timed_out(now)
        if self._complete:
            return SequenceEvent(None, tuple(self.sequence), True)

        is_neutral = label == "no_seal" or confidence < self.confidence_threshold
        if is_neutral:
            self._candidate = None
            self._candidate_since = None
            if self.sequence and self._neutral_since is None:
                self._neutral_since = now
            return SequenceEvent(None, tuple(self.sequence), False)

        if self.sequence and self._neutral_since is None:
            # A second sign must be separated from the first by a neutral phase.
            return SequenceEvent(None, tuple(self.sequence), False)
        if self.sequence and now - self._neutral_since < self.neutral_seconds:
            return SequenceEvent(None, tuple(self.sequence), False)

        if self._candidate != label:
            self._candidate = label
            self._candidate_since = now
            return SequenceEvent(None, tuple(self.sequence), False)

        assert self._candidate_since is not None
        if now - self._candidate_since < self.hold_seconds:
            return SequenceEvent(None, tuple(self.sequence), False)

        self.sequence.append(label)
        self._last_accepted_at = now
        self._candidate = None
        self._candidate_since = None
        self._neutral_since = None
        if len(self.sequence) >= self.max_signs:
            self._complete = True
        return SequenceEvent(label, tuple(self.sequence), self._complete)


@dataclass
class PlayerBattleState:
    player_id: str
    hp: int = 100
    tracker: SequenceTracker = field(default_factory=SequenceTracker)


@dataclass(frozen=True)
class RoundResolution:
    player1_jutsu: Jutsu
    player2_jutsu: Jutsu
    damage_to_player1: int
    damage_to_player2: int
    winner: str | None


class BattleState:
    """Resolve complete two-player sequences simultaneously."""

    def __init__(self, catalog: JutsuCatalog, round_result_seconds: float = 1.5) -> None:
        self.catalog = catalog
        self.round_result_seconds = round_result_seconds
        self.players = {
            "player1": PlayerBattleState("player1"),
            "player2": PlayerBattleState("player2"),
        }
        self.round_number = 1
        self.phase = "collecting"
        self.last_resolution: RoundResolution | None = None
        self.winner: str | None = None
        self._round_reset_at: float | None = None

    def reset_match(self) -> None:
        for player in self.players.values():
            player.hp = 100
            player.tracker.reset()
        self.round_number = 1
        self.phase = "collecting"
        self.last_resolution = None
        self.winner = None
        self._round_reset_at = None

    def _advance_round_if_ready(self, now: float) -> None:
        if self.phase == "round_result" and self._round_reset_at is not None:
            if now >= self._round_reset_at:
                for player in self.players.values():
                    player.tracker.reset()
                self.round_number += 1
                self.phase = "collecting"
                self.last_resolution = None
                self._round_reset_at = None

    def update(
        self,
        predictions: dict[str, tuple[str, float]],
        now: float,
    ) -> dict[str, SequenceEvent]:
        self._advance_round_if_ready(now)
        if self.phase != "collecting":
            return {
                player_id: SequenceEvent(
                    None,
                    tuple(player.tracker.sequence),
                    player.tracker.complete,
                )
                for player_id, player in self.players.items()
            }

        events = {
            player_id: self.players[player_id].tracker.update(label, confidence, now)
            for player_id, (label, confidence) in predictions.items()
        }
        if all(player.tracker.complete for player in self.players.values()):
            self.resolve_round(now)
        return events

    def resolve_round(self, now: float) -> RoundResolution:
        player1_jutsu = self.catalog.lookup(tuple(self.players["player1"].tracker.sequence))
        player2_jutsu = self.catalog.lookup(tuple(self.players["player2"].tracker.sequence))
        damage_to_player1 = player2_jutsu.damage
        damage_to_player2 = player1_jutsu.damage
        self.players["player1"].hp = max(0, self.players["player1"].hp - damage_to_player1)
        self.players["player2"].hp = max(0, self.players["player2"].hp - damage_to_player2)

        if self.players["player1"].hp == 0 and self.players["player2"].hp == 0:
            winner = "draw"
        elif self.players["player1"].hp == 0:
            winner = "player2"
        elif self.players["player2"].hp == 0:
            winner = "player1"
        else:
            winner = None
        resolution = RoundResolution(
            player1_jutsu,
            player2_jutsu,
            damage_to_player1,
            damage_to_player2,
            winner,
        )
        self.last_resolution = resolution
        self.winner = winner
        self.phase = "game_over" if winner else "round_result"
        self._round_reset_at = None if winner else now + self.round_result_seconds
        return resolution
