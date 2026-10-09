"""Optional text-to-speech support for challenge announcements."""

from __future__ import annotations


class OptionalVoice:
    """Lazy voice adapter that never blocks visual challenge operation."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = False
        self.available = False
        self.error: str | None = None
        self._engine = None
        if enabled:
            self.toggle()

    def toggle(self) -> bool:
        if self.enabled:
            self.enabled = False
            return False
        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self.available = True
                self.error = None
            except Exception as exc:  # Voice is optional and platform-dependent.
                self.available = False
                self.error = str(exc)
                self.enabled = False
                return False
        self.enabled = True
        return True

    def speak(self, text: str) -> bool:
        if not self.enabled or self._engine is None:
            return False
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as exc:  # Keep visual operation alive if speech fails.
            self.error = str(exc)
            self.enabled = False
            return False
