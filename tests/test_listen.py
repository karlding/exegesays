"""Tail checks keep a sample count instead of copying the VAD buffer every window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from exegesays.endpoint import CHECK_SAMPLES, TAIL_SAMPLES, LanguageSwitch, SpeechCursor
from exegesays.listen import _observe_speech, _queue_tail_check

WINDOW = 512


class _Vad:
    def __init__(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.reads = 0
        self.speech = True

    def grow(self, count: int) -> None:
        self.samples = np.concatenate(
            [self.samples, np.zeros(count, dtype=np.float32)]
        )

    def is_speech_detected(self) -> bool:
        return self.speech

    @property
    def current_segment(self) -> SimpleNamespace:
        self.reads += 1
        return SimpleNamespace(samples=self.samples)


class _Recognizer:
    def __init__(self) -> None:
        self.tails: list[int] = []

    def submit_tail(self, samples: np.ndarray) -> None:
        self.tails.append(int(samples.size))


def test_tail_check_reads_samples_on_the_check_interval() -> None:
    vad = _Vad()
    switch = LanguageSwitch()
    cursor = SpeechCursor()
    recognizer = _Recognizer()
    windows = (TAIL_SAMPLES + CHECK_SAMPLES * 2) // WINDOW
    for _ in range(windows):
        vad.grow(WINDOW)
        _observe_speech(cursor, vad, WINDOW)
        _queue_tail_check(vad, switch, cursor, recognizer, pending=False)
    # The sample list is read once to seed the cursor, then once per 0.4s check.
    # 56 windows would previously have read the list 56 times.
    assert windows == 56
    assert vad.reads == 1 + len(recognizer.tails)
    assert recognizer.tails == [16384, 23040]
