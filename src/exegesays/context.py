"""When a standalone translation should be retried with earlier utterances."""

from __future__ import annotations

import re

HISTORY_LIMIT = 3
LOW_LOGPROB = -1.5
_EN_CONTEXT = re.compile(
    r"\b(this|that|these|those|he|she|they|them|his|her|their|it|its|who)\b",
    re.IGNORECASE,
)
_CONTEXT_MARKERS = (
    "他们",
    "她们",
    "它们",
    "这个",
    "那个",
    "这些",
    "那些",
    "这里",
    "那里",
    "这样",
    "那样",
    "这么",
    "那么",
    "前者",
    "后者",
    "他",
    "她",
    "它",
    "其",
    "这",
    "那",
    "谁",
)


def looks_context_dependent(source: str, keep: str = "zh") -> bool:
    """True when the line likely refers to something said earlier."""
    text = source.strip()
    if not text:
        return False
    if keep == "en":
        return _EN_CONTEXT.search(text) is not None
    return any(marker in text for marker in _CONTEXT_MARKERS)


def should_retry_with_context(
    source: str,
    mean_logprob: float,
    has_history: bool,
    keep: str = "zh",
) -> bool:
    if not has_history:
        return False
    if looks_context_dependent(source, keep):
        return True
    return mean_logprob < LOW_LOGPROB
