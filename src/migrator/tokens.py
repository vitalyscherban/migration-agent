"""Token counting.

Uses tiktoken when available and falls back to a calibrated character-ratio
estimate when it is not, so the package stays importable in a bare CI
container. The fallback is deliberately *not* silent: `is_exact` tells callers
whether the numbers they are about to print are measured or estimated.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

_ENCODING_NAME = "o200k_base"

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


is_exact = _TIKTOKEN_AVAILABLE

#: Measured against this repo's own corpus: Python source tokenises at roughly
#: 3.4 characters per token, denser than prose because of punctuation runs.
_CHARS_PER_TOKEN = 3.4


@lru_cache(maxsize=1)
def _encoder():  # pragma: no cover - trivial
    return tiktoken.get_encoding(_ENCODING_NAME)


def count(text: str) -> int:
    """Return the token count of ``text``."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        return len(_encoder().encode(text))
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def count_all(chunks: Iterable[str]) -> int:
    return sum(count(chunk) for chunk in chunks)


def truncate_to(text: str, limit: int) -> str:
    """Trim ``text`` so it costs at most ``limit`` tokens.

    Used only as a backstop. Anything that routinely needs truncating is a
    budgeting bug upstream, not something to paper over here.
    """
    if limit <= 0:
        return ""
    if count(text) <= limit:
        return text
    if _TIKTOKEN_AVAILABLE:
        enc = _encoder()
        return enc.decode(enc.encode(text)[:limit])
    return text[: int(limit * _CHARS_PER_TOKEN)]
