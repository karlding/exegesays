import io
import threading
import time
from types import SimpleNamespace

import numpy as np

from exegesays.output import TranslationWriter
from exegesays.worker import RecognizeWorker, TranslationWorker, _stable_audio


class _FakeTranslator:
    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate
        self.started: list[str] = []
        self.histories: list[list[tuple[str, str]]] = []

    def translate(self, text, history=None, on_token=None, on_reset=None):
        del on_reset
        self.started.append(text)
        self.histories.append(list(history or []))
        if text == "one":
            self.gate.wait(timeout=2)
        if on_token is not None:
            on_token(text.upper())
        return SimpleNamespace(text=text.upper(), mean_logprob=0.0, used_context=False)


def test_submit_does_not_wait_for_translation() -> None:
    gate = threading.Event()
    fake = _FakeTranslator(gate)
    buf = io.StringIO()
    worker = TranslationWorker(
        loader=lambda: fake,
        writer_factory=lambda: TranslationWriter(buf),
    )
    worker.start(timeout=2)
    started = time.perf_counter()
    worker.submit("one", show_source=False, verbose=False)
    worker.submit("two", show_source=False, verbose=False)
    assert time.perf_counter() - started < 0.2
    deadline = time.monotonic() + 2
    while fake.started != ["one"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake.started == ["one"]
    gate.set()
    worker.close(timeout=2)
    assert fake.started == ["one", "two"]
    assert fake.histories[1] == [("one", "ONE")]
    assert buf.getvalue() == "ONE\nTWO\n"


class _FakeRecognizer:
    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate
        self.calls: list[int] = []

    def transcribe(self, samples, sample_rate: int) -> tuple[str, str]:
        del sample_rate
        self.calls.append(len(samples))
        if len(self.calls) == 1:
            self.gate.wait(timeout=2)
        return "你好", "<|zh|>"


def test_submit_does_not_wait_for_recognition() -> None:
    gate = threading.Event()
    asr = _FakeRecognizer(gate)
    translator = TranslationWorker(
        loader=lambda: _FakeTranslator(threading.Event()),
        writer_factory=lambda: TranslationWriter(io.StringIO()),
    )
    translator.start(timeout=2)
    recognizer = RecognizeWorker(
        loader=lambda: asr,
        keep="zh",
        translator=translator,
        show_source=False,
        verbose=False,
    )
    recognizer.start(timeout=2)
    audio = np.zeros(1600, dtype=np.float32)
    started = time.perf_counter()
    recognizer.submit_utterance(audio)
    recognizer.submit_utterance(audio)
    assert time.perf_counter() - started < 0.2
    deadline = time.monotonic() + 2
    while asr.calls != [1600] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert asr.calls == [1600]
    gate.set()
    recognizer.close(timeout=2)
    translator.close(timeout=2)
    assert asr.calls == [1600, 1600]


def test_stable_audio_shares_owned_buffers_and_copies_views() -> None:
    owned = np.zeros(8, dtype=np.float32)
    assert _stable_audio(owned) is owned
    view = owned[:4]
    stable = _stable_audio(view)
    assert stable is not view
    view[0] = 1.0
    assert stable[0] == 0.0
