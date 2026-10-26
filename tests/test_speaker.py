import numpy as np
import pytest

from exegesays.speaker import cosine_similarity, same_speaker, voiced


def test_identical_vectors_are_the_same_speaker() -> None:
    vector = np.array([0.6, 0.8], dtype=np.float32)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert same_speaker(vector, vector) is True


def test_orthogonal_vectors_are_a_different_speaker() -> None:
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(left, right) == 0.0
    assert same_speaker(left, right) is False


def test_zero_vector_similarity_is_zero() -> None:
    zero = np.zeros(2, dtype=np.float32)
    other = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(zero, other) == 0.0


def test_close_vectors_still_count_as_the_same_speaker() -> None:
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.5, 0.5], dtype=np.float32)
    assert cosine_similarity(left, right) == pytest.approx(0.7071, abs=0.01)
    assert same_speaker(left, right) is True


def test_quiet_audio_is_not_voiced() -> None:
    assert voiced(np.zeros(1600, dtype=np.float32)) is False
    assert voiced(np.full(1600, 0.1, dtype=np.float32)) is True
