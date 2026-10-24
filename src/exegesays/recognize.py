"""SenseVoice speech recognition through sherpa-onnx."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import sherpa_onnx


class Recognizer:
    def __init__(self, model: Path, tokens: Path, num_threads: int = 2) -> None:
        kwargs = {
            "model": str(model),
            "tokens": str(tokens),
            "num_threads": num_threads,
            "use_itn": True,
            "debug": False,
        }
        parameters = inspect.signature(
            sherpa_onnx.OfflineRecognizer.from_sense_voice
        ).parameters
        if "language" in parameters:
            kwargs["language"] = "auto"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> tuple[str, str]:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        text = getattr(result, "text", "") or ""
        lang = getattr(result, "lang", "") or getattr(result, "language", "") or ""
        return str(text), str(lang)
