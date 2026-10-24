"""Keep Mandarin spans from a mixed transcript."""

from __future__ import annotations

import re

_SPECIAL = re.compile(r"<\|[^|]*\|>")
_EXTRA_PUNCT = frozenset("…—～·")


def extract_kept(text: str, keep: str, lang: str = "") -> str:
    if keep == "en":
        return extract_english(text, lang)
    return extract_mandarin(text, lang)


def extract_mandarin(text: str, lang: str = "") -> str:
    """Return the Chinese to translate, or an empty string to ignore the utterance.

    SenseVoice often tags Mandarin as Cantonese, so language tags are not used to
    drop speech. Japanese and Korean are dropped when the transcript contains
    kana or Hangul. English words inside a Chinese utterance are removed. A
    result with no Han characters is ignored.
    """
    del lang
    raw = _SPECIAL.sub("", text or "")
    if _has_kana(raw) or _has_hangul(raw):
        return ""
    pieces = _keep_chinese_pieces(raw)
    if not any(_has_han(piece) for piece in pieces):
        return ""
    return "".join(pieces).strip()


def extract_english(text: str, lang: str = "") -> str:
    """Return the English to translate, dropping Chinese and other scripts."""
    del lang
    raw = _SPECIAL.sub("", text or "")
    if _has_kana(raw) or _has_hangul(raw):
        return ""
    pieces = _keep_english_pieces(raw)
    if not any(ch.isascii() and ch.isalpha() for piece in pieces for ch in piece):
        return ""
    return " ".join("".join(pieces).split())


def _has_han(text: str) -> bool:
    return any(_is_han(ch) for ch in text)


def _has_kana(text: str) -> bool:
    return any(0x3040 <= ord(ch) <= 0x30FF for ch in text)


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(ch) <= 0xD7AF for ch in text)


def _is_han(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_cjk_punct(ch: str) -> bool:
    code = ord(ch)
    return 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF or ch in _EXTRA_PUNCT


def _classify(ch: str) -> str:
    if _is_han(ch):
        return "han"
    if ch.isdigit():
        return "digit"
    if _is_cjk_punct(ch):
        return "punct"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isspace():
        return "space"
    return "other"


def _runs(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    kind = _classify(text[0])
    buf = [text[0]]
    runs: list[tuple[str, str]] = []
    for ch in text[1:]:
        next_kind = _classify(ch)
        if next_kind == kind:
            buf.append(ch)
        else:
            runs.append((kind, "".join(buf)))
            kind = next_kind
            buf = [ch]
    runs.append((kind, "".join(buf)))
    return _merge_decimals(runs)


def _merge_decimals(runs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    index = 0
    while index < len(runs):
        kind, value = runs[index]
        if (
            kind == "digit"
            and index + 2 < len(runs)
            and runs[index + 1] == ("other", ".")
            and runs[index + 2][0] == "digit"
        ):
            merged.append(("digit", value + "." + runs[index + 2][1]))
            index += 3
            continue
        merged.append((kind, value))
        index += 1
    return merged


def _neighbor_is(runs: list[tuple[str, str]], index: int, step: int, wanted: str) -> bool:
    cursor = index + step
    while 0 <= cursor < len(runs):
        kind, _value = runs[cursor]
        if kind == "space":
            cursor += step
            continue
        return kind == wanted
    return False


def _keep_chinese_pieces(text: str) -> list[str]:
    runs = _runs(text)
    kept: list[str] = []
    for index, (kind, value) in enumerate(runs):
        if kind == "han":
            kept.append(value)
        elif kind in {"punct", "digit"} and (
            _neighbor_is(runs, index, -1, "han") or _neighbor_is(runs, index, 1, "han")
        ):
            kept.append(value)
    return kept


def _keep_english_pieces(text: str) -> list[str]:
    runs = _runs(text)
    kept: list[str] = []
    for index, (kind, value) in enumerate(runs):
        if kind == "latin":
            kept.append(value)
        elif kind == "space" and (
            _neighbor_is(runs, index, -1, "latin")
            or _neighbor_is(runs, index, 1, "latin")
        ):
            kept.append(" ")
        elif kind in {"punct", "digit"} and (
            _neighbor_is(runs, index, -1, "latin")
            or _neighbor_is(runs, index, 1, "latin")
        ):
            kept.append(value)
        elif (
            kind == "other"
            and value in {".", "'", "’"}
            and (
                _neighbor_is(runs, index, -1, "latin")
                or _neighbor_is(runs, index, 1, "latin")
            )
        ):
            kept.append(value)
    return kept
