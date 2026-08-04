"""Lazily-constructed, cached Gemini client.

The SDK client is thread-safe and reusable, so building one per request would
add needless connection setup to every message. It is created on first use
rather than at import so the app still boots with no API key configured.
"""
from __future__ import annotations

import threading
from typing import Any

from app.core.config import settings

_client: Any | None = None
_lock = threading.Lock()


def get_client() -> Any:
    """Return the shared ``genai.Client``. Raises if no key is configured."""
    global _client

    if _client is None:
        with _lock:
            if _client is None:  # re-check inside the lock
                if not settings.GEMINI_API_KEY:
                    raise RuntimeError("GEMINI_API_KEY is not configured")
                from google import genai

                _client = genai.Client(api_key=settings.GEMINI_API_KEY)

    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests and after a config change."""
    global _client
    with _lock:
        _client = None
