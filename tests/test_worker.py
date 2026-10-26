import io
import threading
import time
from types import SimpleNamespace

import numpy as np

from exegesays.endpoint import TAIL_SAMPLES
from exegesays.output import TranslationWriter
from exegesays.worker import (
    RecognizeWorker,
    SpeakerWorker,
    TranslationWorker,
    _join_kept,
    _stable_audio,
)


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
    deadline = time.monotonic() + 2
    while asr.calls != [1600, 1600] and time.monotonic() < deadline:
        time.sleep(0.01)
    recognizer.close(timeout=2)
    translator.close(timeout=2)
    assert asr.calls == [1600, 1600]


class _FakeEmbedder:
    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = list(vectors)
        self.calls: list[int] = []

    def embed(self, samples: np.ndarray) -> np.ndarray | None:
        self.calls.append(len(samples))
        if not self.vectors:
            return np.array([1.0, 0.0], dtype=np.float32)
        return self.vectors.pop(0)


def test_phrases_are_held_until_commit() -> None:
    translator = _FakeTranslator(threading.Event())
    buf = io.StringIO()
    mt = TranslationWorker(
        loader=lambda: translator,
        writer_factory=lambda: TranslationWriter(buf),
    )
    mt.start(timeout=2)
    ready = threading.Event()
    ready.set()
    asr = _FakeRecognizer(ready)
    recognizer = RecognizeWorker(
        loader=lambda: asr,
        keep="zh",
        translator=mt,
        show_source=False,
        verbose=False,
    )
    recognizer.start(timeout=2)
    audio = np.zeros(1600, dtype=np.float32)
    recognizer.submit_utterance(audio)
    recognizer.submit_utterance(audio)
    deadline = time.monotonic() + 2
    while asr.calls != [1600, 1600] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert asr.calls == [1600, 1600]
    assert translator.started == []
    recognizer.commit()
    while not translator.started and time.monotonic() < deadline:
        time.sleep(0.01)
    recognizer.close(timeout=2)
    mt.close(timeout=2)
    assert translator.started == ["你好你好"]


def test_english_phrases_join_with_spaces() -> None:
    assert _join_kept(["hello", "there"], "en") == "hello there"
    assert _join_kept(["你好", "世界"], "zh") == "你好世界"


def test_speaker_worker_sets_anchor_then_detects_a_change() -> None:
    embedder = _FakeEmbedder(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    worker = SpeakerWorker(loader=lambda: embedder)
    worker.start(timeout=2)
    audio = np.zeros(TAIL_SAMPLES, dtype=np.float32)
    voiced = np.full(TAIL_SAMPLES, 0.1, dtype=np.float32)
    worker.submit(voiced)
    worker.submit(voiced)
    worker.submit(voiced)
    deadline = time.monotonic() + 2
    results: list[bool] = []
    while len(results) < 3 and time.monotonic() < deadline:
        item = worker.poll()
        if item is not None:
            results.append(item[0])
        else:
            time.sleep(0.01)
    worker.close(timeout=2)
    assert results == [True, True, False]
    assert embedder.calls == [TAIL_SAMPLES, TAIL_SAMPLES, TAIL_SAMPLES]


def test_speaker_worker_reset_drops_stale_results() -> None:
    embedder = _FakeEmbedder(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    worker = SpeakerWorker(loader=lambda: embedder)
    worker.start(timeout=2)
    voiced = np.full(TAIL_SAMPLES, 0.1, dtype=np.float32)
    worker.submit(voiced)
    deadline = time.monotonic() + 2
    first = None
    while first is None and time.monotonic() < deadline:
        first = worker.poll()
        if first is None:
            time.sleep(0.01)
    assert first is not None and first[0] is True
    worker.reset()
    worker.submit(voiced)
    second = None
    while second is None and time.monotonic() < deadline:
        second = worker.poll()
        if second is None:
            time.sleep(0.01)
    worker.close(timeout=2)
    assert second is not None and second[0] is True


def test_stable_audio_shares_owned_buffers_and_copies_views() -> None:
    owned = np.zeros(8, dtype=np.float32)
    assert _stable_audio(owned) is owned
    view = owned[:4]
    stable = _stable_audio(view)
    assert stable is not view
    view[0] = 1.0
    assert stable[0] == 0.0
