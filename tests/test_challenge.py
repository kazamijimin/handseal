from __future__ import annotations

import unittest

from app.challenge import ChallengeSession, StableSignDetector
from app.voice import OptionalVoice


def prime_detector(detector: StableSignDetector, start: float = 0.0) -> None:
    detector.update("no_seal", 0.99, start)


def confirm(detector: StableSignDetector, label: str, start: float):
    detector.update(label, 0.95, start)
    return detector.update(label, 0.95, start + detector.hold_seconds)


def confirm_target(session: ChallengeSession, now: float):
    session.update_prediction("no_seal", 0.99, now)
    target = session.current_target
    session.update_prediction(target, 0.95, now + 0.40)
    return session.update_prediction(target, 0.95, now + 0.40 + 0.75)


class ChallengeTests(unittest.TestCase):
    def test_stable_detector_requires_hold_and_confidence(self):
        detector = StableSignDetector()
        prime_detector(detector)
        first = detector.update("dragon", 0.79, 0.5)
        self.assertIsNone(first.accepted_label)
        event = confirm(detector, "dragon", 1.0)
        self.assertEqual(event.accepted_label, "dragon")

    def test_neutral_transition_is_required_after_acceptance(self):
        detector = StableSignDetector()
        prime_detector(detector)
        confirm(detector, "dragon", 1.0)
        blocked = detector.update("ram", 0.95, 2.0)
        self.assertIsNone(blocked.accepted_label)
        detector.update("no_seal", 0.99, 2.1)
        detector.update("no_seal", 0.99, 2.5)
        accepted = confirm(detector, "ram", 2.6)
        self.assertEqual(accepted.accepted_label, "ram")

    def test_targets_are_positive_and_not_consecutive_duplicates(self):
        session = ChallengeSession(seed=7, total_rounds=10)
        targets = []
        for index in range(10):
            targets.append(session.current_target)
            result = confirm_target(session, index * 3.0)
            self.assertTrue(result.correct)
            session.tick(index * 3.0 + 2.5)
        self.assertTrue(all(target in {"dragon", "ram", "snake", "tiger"} for target in targets))
        self.assertTrue(all(left != right for left, right in zip(targets, targets[1:])))

    def test_wrong_sign_does_not_advance_round(self):
        session = ChallengeSession(seed=1, total_rounds=2)
        session.update_prediction("no_seal", 0.99, 0.0)
        target = session.current_target
        wrong = next(label for label in ("dragon", "ram", "snake", "tiger") if label != target)
        session.update_prediction(wrong, 0.95, 0.4)
        session.update_prediction(wrong, 0.95, 1.15)
        self.assertEqual(session.completed_rounds, 0)
        self.assertEqual(session.current_target, target)
        self.assertEqual(session.current_round.attempt_count, 1)

    def test_timeout_resets_streak_and_records_round(self):
        session = ChallengeSession(seed=1, total_rounds=2)
        session.update_prediction("no_seal", 0.99, 0.0)
        result = session.tick(10.0)
        self.assertIsNotNone(result)
        self.assertFalse(result.correct)
        self.assertEqual(result.score_awarded, 0)
        self.assertEqual(session.streak, 0)
        self.assertEqual(session.completed_rounds, 1)

    def test_ten_round_summary_and_restart_are_deterministic(self):
        session = ChallengeSession(seed=12, total_rounds=10)
        first_target = session.current_target
        for index in range(10):
            result = confirm_target(session, index * 3.0)
            self.assertTrue(result.correct)
            session.tick(index * 3.0 + 2.5)
        self.assertEqual(session.phase, "complete")
        self.assertEqual(session.score, 1000)
        self.assertEqual(session.accuracy, 1.0)
        self.assertEqual(session.best_streak, 10)
        session.reset(0.0)
        self.assertEqual(session.current_target, first_target)
        self.assertEqual(session.score, 0)
        self.assertEqual(session.completed_rounds, 0)

    def test_voice_is_disabled_without_loading_optional_dependency(self):
        voice = OptionalVoice()
        self.assertFalse(voice.enabled)
        self.assertFalse(voice.available)
        self.assertFalse(voice.speak("Dragon"))


if __name__ == "__main__":
    unittest.main()
