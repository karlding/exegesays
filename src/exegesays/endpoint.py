"""End an utterance when speech leaves the source language, without waiting for silence."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 16000
TAIL_SECONDS = 1.0
CHECK_SECONDS = 0.4
NON_MANDARIN_CHECKS = 2
MIN_PREFIX_SECONDS = 0.35

TAIL_SAMPLES = int(TAIL_SECONDS * SAMPLE_RATE)
CHECK_SAMPLES = int(CHECK_SECONDS * SAMPLE_RATE)
MIN_PREFIX_SAMPLES = int(MIN_PREFIX_SECONDS * SAMPLE_RATE)


@dataclass
class SpeechCursor:
    """Sample count for the current speech segment.

    The count moves forward with each audio window. Callers read the VAD's
    sample list only to seed this count and to resync when a tail check is due.
    """

    samples: int = 0
    live: bool = False

    def reset(self) -> None:
        self.samples = 0
        self.live = False

    def advance(self, speaking: bool, window_size: int, seed: int = 0) -> None:
        if not speaking:
            self.reset()
            return
        if not self.live:
            self.samples = seed
            self.live = True
            return
        self.samples += window_size

    def resync(self, actual: int) -> None:
        self.samples = actual
        self.live = True


@dataclass
class LanguageSwitch:
    """Track source-language speech, then a run of other speech."""

    saw_source: bool = False
    non_source_streak: int = 0
    last_check_samples: int = 0
    source_end_samples: int = 0

    def reset(self) -> None:
        self.saw_source = False
        self.non_source_streak = 0
        self.last_check_samples = 0
        self.source_end_samples = 0

    def ready_for_check(self, current_samples: int) -> bool:
        if current_samples < TAIL_SAMPLES:
            return False
        return current_samples - self.last_check_samples >= CHECK_SAMPLES

    def mark_checked(self, end_samples: int) -> None:
        self.last_check_samples = end_samples

    def note_tail(self, is_source: bool, end_samples: int) -> bool:
        """Return True when the source language has stopped and translation should start."""
        self.last_check_samples = end_samples
        if is_source:
            self.saw_source = True
            self.non_source_streak = 0
            self.source_end_samples = end_samples
            return False
        if not self.saw_source:
            return False
        self.non_source_streak += 1
        return self.non_source_streak >= NON_MANDARIN_CHECKS

    def prefix_samples(self) -> int:
        return max(self.source_end_samples, MIN_PREFIX_SAMPLES)
