"""Write translation tokens to a stream as they arrive."""

from __future__ import annotations

import sys
from typing import TextIO


class TranslationWriter:
    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._parts: list[str] = []

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self._parts.append(piece)
        self._out.write(piece)
        self._out.flush()

    def reset(self) -> None:
        if not self._parts:
            return
        if self._out.isatty():
            self._out.write("\r\033[2K")
        else:
            self._out.write("\n")
        self._out.flush()
        self._parts.clear()

    def finish(self, text: str) -> None:
        displayed = "".join(self._parts)
        if displayed != text:
            self.reset()
            if text:
                self._out.write(text)
        self._out.write("\n")
        self._out.flush()
        self._parts.clear()
