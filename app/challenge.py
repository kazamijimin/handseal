"""Single-player target-sign challenge state and scoring."""

from __future__ import annotations

import random
from dataclasses import dataclass


POSITIVE_SIGNS = ("dragon", "ram", "snake", "tiger")
NEUTRAL_SIGN = "no_seal"


@dataclass(frozen=True)
class StableSignEvent:
    accepted_label: str | None
    progress: float
    waiting_for_neutral: bool


class StableSignDetector:
    """Accept a sign only after confidence and hold-time requirements."""

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        hold_seconds: float = 0.75,
        neutral_seconds: float = 0.35,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.neutral_seconds = neutral_seconds
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._candidate_since: float | None = None
        self._neutral_since: float | None = None
        self.last_progress = 0.0

    def update(self, label: str, confidence: float, now: float) -> StableSignEvent:
        is_neutral = label == NEUTRAL_SIGN or confidence < self.confidence_threshold
        if is_neutral:
            self._candidate = None
            self._candidate_since = None
            self.last_progress = 0.0
            if self._neutral_since is None:
                self._neutral_since = now
            return StableSignEvent(None, 0.0, True)

        if self._neutral_since is None and self._candidate is None:
            return StableSignEvent(None, 0.0, True)
        if self._neutral_since is not None and now - self._neutral_since < self.neutral_seconds:
            return StableSignEvent(None, 0.0, True)

        if self._candidate != label:
            self._candidate = label
            self._candidate_since = now
            self.last_progress = 0.0
            return StableSignEvent(None, 0.0, False)

        assert self._candidate_since is not None
        elapsed = now - self._candidate_since
        progress = min(1.0, max(0.0, elapsed / self.hold_seconds))
        self.last_progress = progress
        if elapsed + 1e-9 < self.hold_seconds:
            return StableSignEvent(None, progress, False)

        accepted = self._candidate
        self._candidate = None
        self._candidate_since = None
        self._neutral_since = None
        self.last_progress = 0.0
        return StableSignEvent(accepted, 1.0, True)


@dataclass(frozen=True)
class ChallengeResult:
    target_label: str
    predicted_label: str | None
    correct: bool
    attempt_count: int
    response_time_seconds: float
    score_awarded: int


@dataclass
class ChallengeRound:
    target_label: str
    started_at: float
    attempt_count: int = 0
    last_predicted_label: str | None = None


@dataclass(frozen=True)
class ChallengeSnapshot:
    phase: str
    target_label: str | None
    round_number: int
    total_rounds: int
    score: int
    streak: int
    best_streak: int
    elapsed_seconds: float
    stable_progress: float
    current_label: str
    current_confidence: float
    feedback: str
    accuracy: float
    average_response_time: float
    results: tuple[ChallengeResult, ...]


class ChallengeSession:
    """Run a reproducible ten-round target-sign session."""

    def __init__(
        self,
        seed: int = 42,
        total_rounds: int = 10,
        round_timeout_seconds: float = 10.0,
        feedback_seconds: float = 1.0,
        confidence_threshold: float = 0.80,
        hold_seconds: float = 0.75,
        neutral_seconds: float = 0.35,
    ) -> None:
        if total_rounds < 1:
            raise ValueError("total_rounds must be at least 1.")
        self.seed = seed
        self.total_rounds = total_rounds
        self.round_timeout_seconds = round_timeout_seconds
        self.feedback_seconds = feedback_seconds
        self.detector = StableSignDetector(
            confidence_threshold=confidence_threshold,
            hold_seconds=hold_seconds,
            neutral_seconds=neutral_seconds,
        )
        self._rng = random.Random(seed)
        self.reset(0.0)

    def _next_target(self) -> str:
        choices = [label for label in POSITIVE_SIGNS if label != self._previous_target]
        target = self._rng.choice(choices)
        self._previous_target = target
        return target

    def reset(self, now: float) -> None:
        self._rng = random.Random(self.seed)
        self._previous_target: str | None = None
        self.results: list[ChallengeResult] = []
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.phase = "active"
        self.feedback = ""
        self._feedback_until: float | None = None
        self._round: ChallengeRound | None = ChallengeRound(self._next_target(), now)
        self.last_label = NEUTRAL_SIGN
        self.last_confidence = 0.0
        self.detector.reset()

    @property
    def current_round(self) -> ChallengeRound | None:
        return self._round

    @property
    def current_target(self) -> str | None:
        return self._round.target_label if self._round else None

    @property
    def completed_rounds(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.correct for result in self.results) / len(self.results)

    @property
    def average_response_time(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.response_time_seconds for result in self.results) / len(self.results)

    def _record_result(
        self,
        now: float,
        predicted_label: str | None,
        correct: bool,
        score: int,
    ) -> ChallengeResult:
        assert self._round is not None
        response_time = min(
            self.round_timeout_seconds,
            max(0.0, now - self._round.started_at),
        )
        result = ChallengeResult(
            target_label=self._round.target_label,
            predicted_label=predicted_label,
            correct=correct,
            attempt_count=self._round.attempt_count,
            response_time_seconds=response_time,
            score_awarded=score,
        )
        self.results.append(result)
        self.score += score
        if correct:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.feedback = f"Correct: {result.target_label.title()}"
        else:
            self.streak = 0
            self.feedback = "Time up"
        if len(self.results) >= self.total_rounds:
            self.phase = "complete"
            self._round = None
            self._feedback_until = None
        else:
            self.phase = "feedback"
            self._feedback_until = now + self.feedback_seconds
        self.detector.reset()
        return result

    def _advance_feedback(self, now: float) -> None:
        if self.phase != "feedback" or self._feedback_until is None or now < self._feedback_until:
            return
        self._round = ChallengeRound(self._next_target(), now)
        self.phase = "active"
        self.feedback = ""
        self._feedback_until = None
        self.detector.reset()

    def update_prediction(self, label: str, confidence: float, now: float) -> ChallengeResult | None:
        self._advance_feedback(now)
        self.last_label = label
        self.last_confidence = confidence
        if self.phase != "active" or self._round is None:
            return None

        if now - self._round.started_at >= self.round_timeout_seconds:
            return self._record_result(
                now,
                self._round.last_predicted_label,
                False,
                0,
            )

        event = self.detector.update(label, confidence, now)
        if event.accepted_label is None:
            return None
        self._round.attempt_count += 1
        self._round.last_predicted_label = event.accepted_label
        correct = event.accepted_label == self._round.target_label
        if not correct:
            self.feedback = "Try again"
            self.detector.reset()
            return None
        return self._record_result(now, event.accepted_label, True, 100)

    def tick(self, now: float) -> ChallengeResult | None:
        self._advance_feedback(now)
        if self.phase != "active" or self._round is None:
            return None
        if now - self._round.started_at >= self.round_timeout_seconds:
            return self._record_result(
                now,
                self._round.last_predicted_label,
                False,
                0,
            )
        return None

    def snapshot(self, now: float) -> ChallengeSnapshot:
        elapsed = 0.0
        round_number = min(len(self.results) + 1, self.total_rounds)
        if self._round is not None:
            elapsed = min(self.round_timeout_seconds, max(0.0, now - self._round.started_at))
        return ChallengeSnapshot(
            phase=self.phase,
            target_label=self.current_target,
            round_number=round_number,
            total_rounds=self.total_rounds,
            score=self.score,
            streak=self.streak,
            best_streak=self.best_streak,
            elapsed_seconds=elapsed,
            stable_progress=self.detector.last_progress,
            current_label=self.last_label,
            current_confidence=self.last_confidence,
            feedback=self.feedback,
            accuracy=self.accuracy,
            average_response_time=self.average_response_time,
            results=tuple(self.results),
        )
