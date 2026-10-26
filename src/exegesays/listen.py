"""Microphone capture, voice activity detection, and live translation."""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
import sherpa_onnx
import sounddevice as sd

from exegesays.endpoint import TURN_HOLD_SECONDS, SpeakerChange, SpeechCursor
from exegesays.paths import sense_voice_dir, speaker_embedding_path, translator_dir, vad_path
from exegesays.recognize import Recognizer
from exegesays.speaker import SpeakerEmbedder
from exegesays.translate import Translator
from exegesays.worker import RecognizeWorker, SpeakerWorker, TranslationWorker

SAMPLE_RATE = 16000


def list_input_devices() -> None:
    devices = sd.query_devices()
    default_input = _default_input_index()
    found = False
    for index, device in enumerate(devices):
        if device["max_input_channels"] < 1:
            continue
        found = True
        marker = ">" if index == default_input else " "
        print(f"{marker} {index}: {device['name']}")
    if not found:
        print("No microphone devices found")


def run(
    device: int | None,
    silence: float,
    show_source: bool,
    verbose: bool = False,
    target: str = "en",
) -> None:
    device_index = _default_input_index() if device is None else device
    device_info = sd.query_devices(device_index)
    if device_info["max_input_channels"] < 1:
        raise SystemExit(f"Device {device_index} is not an input: {device_info['name']}")
    keep = "zh" if target == "en" else "en"
    print(f"Loading models ({device_info['name']})...", flush=True)
    translator = TranslationWorker(
        loader=lambda: Translator(translator_dir(), target=target),
    )
    recognizer = RecognizeWorker(
        loader=lambda: Recognizer(
            model=sense_voice_dir() / "model.int8.onnx",
            tokens=sense_voice_dir() / "tokens.txt",
            language="en" if keep == "en" else "auto",
        ),
        keep=keep,
        translator=translator,
        show_source=show_source,
        verbose=verbose,
        sample_rate=SAMPLE_RATE,
    )
    speaker = SpeakerWorker(
        loader=lambda: SpeakerEmbedder(speaker_embedding_path()),
    )
    translator.start()
    recognizer.start()
    speaker.start()
    try:
        _run_session(
            device_index=device_index,
            silence=silence,
            verbose=verbose,
            keep=keep,
            recognizer=recognizer,
            speaker=speaker,
        )
    finally:
        recognizer.close()
        speaker.close()
        translator.close()


def _run_session(
    *,
    device_index: int,
    silence: float,
    verbose: bool,
    keep: str,
    recognizer: RecognizeWorker,
    speaker: SpeakerWorker,
) -> None:
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = str(vad_path())
    vad_config.silero_vad.threshold = 0.3
    vad_config.silero_vad.min_silence_duration = silence
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.silero_vad.max_speech_duration = 15
    vad_config.sample_rate = SAMPLE_RATE
    window_size = int(vad_config.silero_vad.window_size)
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

    ctx = mp.get_context("spawn")
    audio_queue: mp.Queue[tuple[str, Any]] = ctx.Queue()
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    proc = ctx.Process(
        target=_capture_process_main,
        args=(device_index, SAMPLE_RATE, window_size, audio_queue, stop_event, ready_event),
        daemon=True,
        name="exegesays-mic",
    )
    leftover = np.zeros((0,), dtype=np.float32)
    switch = SpeakerChange()
    cursor = SpeechCursor()
    tail_pending = False
    heard_speech = False
    commit_at = 0.0
    peak = 0.0
    last_level = time.monotonic()
    flush_event = threading.Event()

    print(
        f"Listening for {_source_label(keep)} → {_target_label(keep)}. "
        "Enter translates now. Ctrl+C to stop.",
        flush=True,
    )
    proc.start()
    stdin_thread = threading.Thread(
        target=_watch_enter,
        args=(flush_event,),
        daemon=True,
        name="exegesays-enter",
    )
    stdin_thread.start()
    try:
        if not ready_event.wait(timeout=15):
            raise SystemExit("Microphone capture did not start.")
        while True:
            recognizer.raise_if_failed()
            speaker.raise_if_failed()
            flushed = _apply_tail_result(
                vad, switch, cursor, speaker, recognizer, verbose
            )
            if flushed is not None:
                tail_pending = False
                commit_at = 0.0
                if flushed:
                    heard_speech = True
            if flush_event.is_set():
                flush_event.clear()
                leftover = _dump_buffer(
                    vad,
                    leftover,
                    window_size,
                    switch,
                    cursor,
                    speaker,
                    recognizer,
                    verbose,
                )
                tail_pending = False
                commit_at = 0.0
                heard_speech = True
            try:
                kind, payload = audio_queue.get(timeout=0.05)
            except queue.Empty:
                if not proc.is_alive():
                    raise SystemExit("Microphone capture stopped unexpectedly.")
                if _queue_tail_check(
                    vad, switch, cursor, speaker, tail_pending
                ):
                    tail_pending = True
                commit_at = _tick_hold_commit(vad, recognizer, commit_at)
                if (
                    verbose
                    and not heard_speech
                    and time.monotonic() - last_level >= 5
                ):
                    _status(
                        f"Still listening. Mic peak {peak:.3f} "
                        "(speech is usually above 0.02)"
                    )
                    peak = 0.0
                    last_level = time.monotonic()
                continue
            if kind == "error":
                raise SystemExit(str(payload))
            if kind == "overflow":
                if verbose:
                    _status("Audio input overflowed")
                continue
            chunk = np.ascontiguousarray(payload, dtype=np.float32)
            peak = max(peak, float(np.max(np.abs(chunk))) if chunk.size else 0.0)
            buf = chunk if leftover.size == 0 else np.concatenate([leftover, chunk])
            offset = 0
            while buf.size - offset >= window_size:
                vad.accept_waveform(buf[offset : offset + window_size])
                offset += window_size
                if _drain_completed_speech(
                    vad, switch, cursor, speaker, recognizer
                ):
                    heard_speech = True
                    tail_pending = False
                    commit_at = time.monotonic() + TURN_HOLD_SECONDS
                _observe_speech(cursor, vad, window_size)
                flushed = _apply_tail_result(
                    vad, switch, cursor, speaker, recognizer, verbose
                )
                if flushed is not None:
                    tail_pending = False
                    commit_at = 0.0
                    if flushed:
                        heard_speech = True
                if _queue_tail_check(
                    vad, switch, cursor, speaker, tail_pending
                ):
                    tail_pending = True
                commit_at = _tick_hold_commit(vad, recognizer, commit_at)
            leftover = buf[offset:]
    finally:
        stop_event.set()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.terminate()


def _capture_process_main(
    device_index: int,
    sample_rate: int,
    window_size: int,
    audio_queue: mp.Queue[tuple[str, Any]],
    stop_event: mp.Event,
    ready_event: mp.Event,
) -> None:
    """Read the microphone in a separate process so translation cannot stall it."""
    try:
        import numpy as np
        import sounddevice as sd

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            del frames, time_info
            if stop_event.is_set():
                raise sd.CallbackStop
            if status and status.input_overflow:
                audio_queue.put(("overflow", None))
            audio_queue.put(
                ("audio", np.array(indata[:, 0], dtype=np.float32, copy=True))
            )

        stream = sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=sample_rate,
            blocksize=window_size,
            callback=callback,
        )
        stream.start()
        ready_event.set()
        while not stop_event.wait(timeout=0.1):
            if stream.closed:
                break
        stream.stop()
        stream.close()
    except Exception as exc:
        audio_queue.put(("error", _capture_error_message(exc)))
        ready_event.set()


def _capture_error_message(exc: Exception) -> str:
    import sounddevice as sd

    if isinstance(exc, sd.PortAudioError):
        return (
            "Could not open the microphone. On macOS, allow microphone access "
            "for this terminal or Cursor in System Settings > Privacy & Security "
            f"> Microphone.\n{exc}"
        )
    return f"Microphone capture failed: {exc}"


def _watch_enter(flush_event: threading.Event) -> None:
    if not sys.stdin.isatty():
        return
    while True:
        try:
            line = sys.stdin.readline()
        except (OSError, EOFError):
            return
        if line == "":
            return
        flush_event.set()


def _drain_completed_speech(
    vad: sherpa_onnx.VoiceActivityDetector,
    switch: SpeakerChange,
    cursor: SpeechCursor,
    speaker: SpeakerWorker,
    recognizer: RecognizeWorker,
) -> bool:
    handled = False
    while not vad.empty():
        segment = np.array(vad.front.samples, dtype=np.float32)
        vad.pop()
        switch.reset()
        cursor.reset()
        speaker.reset()
        if segment.size:
            recognizer.submit_utterance(segment)
            handled = True
    return handled


def _dump_buffer(
    vad: sherpa_onnx.VoiceActivityDetector,
    leftover: np.ndarray,
    window_size: int,
    switch: SpeakerChange,
    cursor: SpeechCursor,
    speaker: SpeakerWorker,
    recognizer: RecognizeWorker,
    verbose: bool,
) -> np.ndarray:
    if leftover.size:
        pad = (-leftover.size) % window_size
        if pad:
            leftover = np.concatenate([leftover, np.zeros(pad, dtype=np.float32)])
        for index in range(0, leftover.size, window_size):
            vad.accept_waveform(leftover[index : index + window_size])
        leftover = np.zeros((0,), dtype=np.float32)
    handled = _drain_completed_speech(vad, switch, cursor, speaker, recognizer)
    try:
        vad.flush()
    except Exception:
        pass
    handled = _drain_completed_speech(vad, switch, cursor, speaker, recognizer) or handled
    if not handled:
        samples = _current_speech(vad)
        if samples.size:
            if verbose:
                _status("Enter: translating current buffer")
            recognizer.submit_utterance(samples)
            handled = True
    elif verbose:
        _status("Enter: flushed speech")
    recognizer.commit()
    vad.reset()
    switch.reset()
    cursor.reset()
    speaker.reset()
    if verbose and not handled:
        _status("Enter: no speech in the buffer")
    return leftover


def _tick_hold_commit(
    vad: sherpa_onnx.VoiceActivityDetector,
    recognizer: RecognizeWorker,
    commit_at: float,
) -> float:
    if vad.is_speech_detected():
        return 0.0
    if commit_at and time.monotonic() >= commit_at:
        recognizer.commit()
        return 0.0
    return commit_at


def _observe_speech(
    cursor: SpeechCursor,
    vad: sherpa_onnx.VoiceActivityDetector,
    window_size: int,
) -> None:
    speaking = bool(vad.is_speech_detected())
    if speaking and not cursor.live:
        cursor.advance(True, window_size, _current_speech(vad).size)
        return
    cursor.advance(speaking, window_size)


def _queue_tail_check(
    vad: sherpa_onnx.VoiceActivityDetector,
    switch: SpeakerChange,
    cursor: SpeechCursor,
    speaker: SpeakerWorker,
    pending: bool,
) -> bool:
    if pending or not vad.is_speech_detected():
        return False
    if not switch.ready_for_check(cursor.samples):
        return False
    samples = _current_speech(vad)
    cursor.resync(samples.size)
    if not switch.ready_for_check(samples.size):
        return False
    speaker.submit(samples)
    switch.mark_checked(samples.size)
    return True


def _apply_tail_result(
    vad: sherpa_onnx.VoiceActivityDetector,
    switch: SpeakerChange,
    cursor: SpeechCursor,
    speaker: SpeakerWorker,
    recognizer: RecognizeWorker,
    verbose: bool,
) -> bool | None:
    item = speaker.poll()
    if item is None:
        return None
    same_voice, snapshot = item
    if not switch.note_tail(same_voice, snapshot.size):
        return False
    prefix = snapshot[: switch.prefix_samples()]
    if verbose:
        _status("Speaker changed; translating now")
    if prefix.size:
        recognizer.submit_utterance(prefix)
    recognizer.commit()
    vad.reset()
    switch.reset()
    cursor.reset()
    speaker.reset()
    return True


def _current_speech(vad: sherpa_onnx.VoiceActivityDetector) -> np.ndarray:
    try:
        segment = vad.current_segment
    except Exception:
        return np.zeros(0, dtype=np.float32)
    samples = getattr(segment, "samples", None)
    if samples is None or len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.array(samples, dtype=np.float32)


def _source_label(keep: str) -> str:
    return "Mandarin" if keep == "zh" else "English"


def _target_label(keep: str) -> str:
    return "English" if keep == "zh" else "Chinese"


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _default_input_index() -> int:
    default = sd.default.device[0]
    if default is None or default < 0:
        raise SystemExit("No default microphone is available.")
    return int(default)
