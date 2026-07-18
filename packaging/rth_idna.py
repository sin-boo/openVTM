"""PyInstaller runtime hook: register idna codec (often missing from base_library.zip)."""

from __future__ import annotations

import codecs


def _encode(s: str, errors: str = "strict") -> tuple[bytes, int]:
    return s.encode("ascii", errors), len(s)


def _decode(b: bytes, errors: str = "strict") -> tuple[str, int]:
    return b.decode("ascii", errors), len(b)


def _search(name: str):
    if name == "idna":
        return codecs.CodecInfo(name="idna", encode=_encode, decode=_decode)
    return None


codecs.register(_search)
