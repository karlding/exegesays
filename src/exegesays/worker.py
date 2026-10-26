"""Run recognition and translation off the audio thread."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from exegesays.context import HISTORY_LIMIT
from exegesays.endpoint import TAIL_SAMPLES
from exegesays.mandarin import extract_kept
from exegesays.output import TranslationWriter
from exegesays.speaker import blend_anchor, same_speaker, voiced

_STOP = object()
_SAMPLE_RATE = 16000


class TranslationWorker:
    def __init__(
        self,
        loader: Callable[[], Any],
        writer_factory: Callable[[], TranslationWriter] | None = None,
    ) -> None:
        self._loader = loader
        self._writer_factory = writer_factory or TranslationWriter
        self._queue: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._main,
            name="exegesays-mt",
            daemon=True,
        )

    def start(self, timeout: float | None = None) -> None:
        self._thread.start()
        if timeout is None:
            return
        if not self._ready.wait(timeout=timeout):
            raise SystemExit("Translator did not start.")
        self.raise_if_failed()

    def submit(self, kept: str, *, show_source: bool, verbose: bool) -> None:
        self.raise_if_failed()
        self._queue.put((kept, show_source, verbose))

    def close(self, timeout: float = 120) -> None:
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise SystemExit(f"Translator failed: {self._error}") from self._error

    def _main(self) -> None:
        try:
            translator = self._loader()
            history: list[tuple[str, str]] = []
            self._ready.set()
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                kept, show_source, verbose = item
                _emit_translation(
                    translator,
                    history,
                    kept,
                    show_source=show_source,
                    verbose=verbose,
                    writer=self._writer_factory(),
                )
        except Exception as exc:
            self._error = exc
            self._ready.set()


class SpeakerWorker:
    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader
        self._queue: queue.Queue[Any] = queue.Queue()
        self._results: queue.Queue[tuple[int, bool, np.ndarray]] = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._epoch = 0
        self._thread = threading.Thread(
            target=self._main,
            name="exegesays-spk",
            daemon=True,
        )

    def start(self, timeout: float | None = None) -> None:
        self._thread.start()
        if timeout is None:
            return
        if not self._ready.wait(timeout=timeout):
            raise SystemExit("Speaker embedder did not start.")
        self.raise_if_failed()

    def submit(self, samples: np.ndarray) -> None:
        self.raise_if_failed()
        self._queue.put(("embed", self._epoch, _stable_audio(samples)))

    def reset(self) -> None:
        self.raise_if_failed()
        self._epoch += 1
        self._queue.put(("reset", self._epoch, None))

    def poll(self) -> tuple[bool, np.ndarray] | None:
        self.raise_if_failed()
        while True:
            try:
                epoch, same, snapshot = self._results.get_nowait()
            except queue.Empty:
                return None
            if epoch == self._epoch:
                return same, snapshot

    def close(self, timeout: float = 120) -> None:
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise SystemExit(f"Speaker embedder failed: {self._error}") from self._error

    def _main(self) -> None:
        try:
            embedder = self._loader()
            anchor: np.ndarray | None = None
            epoch = 0
            self._ready.set()
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                kind, item_epoch, samples = item
                if kind == "reset":
                    epoch = item_epoch
                    anchor = None
                    continue
                if item_epoch != epoch:
                    continue
                tail = samples[-TAIL_SAMPLES:]
                if anchor is not None and not voiced(tail):
                    self._results.put((epoch, True, samples))
                    continue
                embedding = embedder.embed(tail)
                if embedding is None:
                    self._results.put((epoch, True, samples))
                    continue
                if anchor is None:
                    anchor = embedding
                    self._results.put((epoch, True, samples))
                    continue
                matched = same_speaker(anchor, embedding)
                if matched:
                    anchor = blend_anchor(anchor, embedding)
                self._results.put((epoch, matched, samples))
        except Exception as exc:
            self._error = exc
            self._ready.set()


class RecognizeWorker:
    def __init__(
        self,
        loader: Callable[[], Any],
        keep: str,
        translator: TranslationWorker,
        *,
        show_source: bool,
        verbose: bool,
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        self._loader = loader
        self._keep = keep
        self._translator = translator
        self._show_source = show_source
        self._verbose = verbose
        self._sample_rate = sample_rate
        self._queue: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._main,
            name="exegesays-asr",
            daemon=True,
        )

    def start(self, timeout: float | None = None) -> None:
        self._thread.start()
        if timeout is None:
            return
        if not self._ready.wait(timeout=timeout):
            raise SystemExit("Recognizer did not start.")
        self.raise_if_failed()

    def submit_utterance(self, samples: np.ndarray) -> None:
        self.raise_if_failed()
        self._queue.put(("utterance", _stable_audio(samples)))

    def commit(self) -> None:
        self.raise_if_failed()
        self._queue.put(("commit", None))

    def close(self, timeout: float = 120) -> None:
        self._queue.put(("commit", None))
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise SystemExit(f"Recognizer failed: {self._error}") from self._error

    def _main(self) -> None:
        try:
            recognizer = self._loader()
            pending: list[str] = []
            self._ready.set()
            while True:
                item = self._queue.get()
                if item is _STOP:
                    self._flush(pending)
                    break
                kind, samples = item
                if kind == "commit":
                    self._flush(pending)
                    continue
                self._transcribe_utterance(recognizer, samples, pending)
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def _transcribe_utterance(
        self, recognizer: Any, samples: np.ndarray, pending: list[str]
    ) -> None:
        kept = self._kept(recognizer, samples, verbose=True)
        if kept:
            pending.append(kept)

    def _flush(self, pending: list[str]) -> None:
        if not pending:
            return
        kept = _join_kept(pending, self._keep)
        pending.clear()
        self._translator.submit(
            kept, show_source=self._show_source, verbose=self._verbose
        )

    def _kept(self, recognizer: Any, samples: np.ndarray, *, verbose: bool) -> str:
        if samples.size == 0:
            return ""
        if verbose and self._verbose:
            duration = samples.size / self._sample_rate
            peak = float(np.max(np.abs(samples)))
            _status(f"Speech {duration:.1f}s, peak {peak:.3f}")
        text, lang = recognizer.transcribe(samples, self._sample_rate)
        if verbose and self._verbose:
            _status(f"Recognized [{lang or '?'}]: {text or '(empty)'}")
        kept = extract_kept(text, self._keep, lang)
        if verbose and self._verbose:
            if kept:
                label = "Mandarin" if self._keep == "zh" else "English"
                _status(f"Heard {label}: {kept}")
            else:
                label = "Mandarin" if self._keep == "zh" else "English"
                _status(f"Ignored (no {label} kept)")
        return kept


def _join_kept(parts: list[str], keep: str) -> str:
    cleaned = [part.strip() for part in parts if part.strip()]
    if keep == "en":
        return " ".join(cleaned)
    return "".join(cleaned)


def _stable_audio(samples: np.ndarray) -> np.ndarray:
    """Share an owned contiguous buffer. Copy only views the caller may reuse."""
    audio = np.ascontiguousarray(samples, dtype=np.float32)
    if audio.base is None:
        return audio
    return np.array(audio, dtype=np.float32, copy=True)


def _emit_translation(
    translator: Any,
    history: list[tuple[str, str]],
    kept: str,
    *,
    show_source: bool,
    verbose: bool,
    writer: TranslationWriter,
) -> None:
    if verbose:
        _status("Translating...")
    if show_source:
        print(kept, flush=True)
    started = time.perf_counter()
    result = translator.translate(
        kept, history, on_token=writer.feed, on_reset=writer.reset
    )
    elapsed = time.perf_counter() - started
    translated = result.text
    if not translated:
        writer.reset()
        if verbose:
            _status(f"Translation returned nothing after {elapsed:.1f}s")
        return
    history.append((kept, translated))
    del history[:-HISTORY_LIMIT]
    writer.finish(translated)
    if verbose:
        extra = (
            f", used context, logprob={result.mean_logprob:.2f}"
            if result.used_context
            else f", logprob={result.mean_logprob:.2f}"
        )
        _status(f"Translated in {elapsed:.1f}s{extra}")


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
