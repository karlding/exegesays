from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exegesays.recognize import Recognizer


def test_recognizer_passes_language_to_sense_voice(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _OfflineRecognizer:
        @staticmethod
        def from_sense_voice(*, language: str = "auto", **kwargs):
            captured["language"] = language
            captured["kwargs"] = kwargs
            return SimpleNamespace()

    monkeypatch.setattr(
        "exegesays.recognize.sherpa_onnx.OfflineRecognizer",
        _OfflineRecognizer,
    )
    Recognizer(
        model=Path("model.int8.onnx"),
        tokens=Path("tokens.txt"),
        language="en",
    )
    assert captured["language"] == "en"


def test_transcribe_reads_text_and_lang(monkeypatch) -> None:
    class _Stream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="hello", lang="en")

        def accept_waveform(self, sample_rate: int, audio) -> None:
            del sample_rate, audio

    class _Impl:
        def create_stream(self) -> _Stream:
            return _Stream()

        def decode_stream(self, stream: _Stream) -> None:
            del stream

    class _OfflineRecognizer:
        @staticmethod
        def from_sense_voice(*, language: str = "auto", **kwargs):
            del language, kwargs
            return _Impl()

    monkeypatch.setattr(
        "exegesays.recognize.sherpa_onnx.OfflineRecognizer",
        _OfflineRecognizer,
    )
    recognizer = Recognizer(Path("model.int8.onnx"), Path("tokens.txt"), language="en")
    text, lang = recognizer.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert text == "hello"
    assert lang == "en"
