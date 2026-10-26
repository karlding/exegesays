"""Speaker embeddings for detecting a turn without waiting for silence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

SIMILARITY_THRESHOLD = 0.4
MIN_EMBED_SAMPLES = 8000
MIN_SPEECH_PEAK = 0.02
ANCHOR_EMA = 0.15


class SpeakerEmbedder:
    def __init__(self, model: Path, num_threads: int = 1) -> None:
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        )
        if not config.validate():
            raise SystemExit(f"Invalid speaker embedding config: {config}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if audio.size < MIN_EMBED_SAMPLES:
            return None
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        vector = self._extractor.compute(stream)
        return np.asarray(vector, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def same_speaker(anchor: np.ndarray, other: np.ndarray) -> bool:
    return cosine_similarity(anchor, other) >= SIMILARITY_THRESHOLD


def voiced(samples: np.ndarray) -> bool:
    if samples.size == 0:
        return False
    return float(np.max(np.abs(samples))) >= MIN_SPEECH_PEAK


def blend_anchor(anchor: np.ndarray, other: np.ndarray) -> np.ndarray:
    mixed = (1.0 - ANCHOR_EMA) * anchor + ANCHOR_EMA * other
    norm = float(np.linalg.norm(mixed))
    if norm == 0.0:
        return anchor
    return (mixed / norm).astype(np.float32)
