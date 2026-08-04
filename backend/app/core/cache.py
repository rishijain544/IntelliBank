"""Key-value backend used for rate limiting, refresh-token revocation and caching.

Uses Redis when ``REDIS_URL`` is configured and reachable; otherwise transparently
falls back to a thread-safe in-process store so the app always boots.
"""
from __future__ import annotations

import threading
import time
from typing import Protocol

from app.core.config import settings


class KVBackend(Protocol):
    name: str

    def incr_with_ttl(self, key: str, ttl_seconds: int) -> int: ...
    def ttl(self, key: str) -> int: ...
    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...
    def get(self, key: str) -> str | None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class InMemoryBackend:
    """Fallback backend. Adequate for single-process dev/demo deployments."""

    name = "in-memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.RLock()

    def _purge(self, key: str) -> None:
        item = self._data.get(key)
        if item and item[1] is not None and item[1] <= time.time():
            self._data.pop(key, None)

    def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        with self._lock:
            self._purge(key)
            cur, exp = self._data.get(key, ("0", None))
            new = int(cur) + 1
            if exp is None:
                exp = time.time() + ttl_seconds
            self._data[key] = (str(new), exp)
            return new

    def ttl(self, key: str) -> int:
        with self._lock:
            self._purge(key)
            item = self._data.get(key)
            if not item or item[1] is None:
                return -1
            return max(0, int(item[1] - time.time()))

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ttl_seconds if ttl_seconds else None)

    def get(self, key: str) -> str | None:
        with self._lock:
            self._purge(key)
            item = self._data.get(key)
            return item[0] if item else None

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisBackend:
    name = "redis"

    def __init__(self, url: str) -> None:
        import redis  # imported lazily so redis stays an optional runtime dep

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.5)
        self._r.ping()

    def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        pipe = self._r.pipeline()
        pipe.incr(key, 1)
        pipe.ttl(key)
        count, current_ttl = pipe.execute()
        if current_ttl is None or current_ttl < 0:
            self._r.expire(key, ttl_seconds)
        return int(count)

    def ttl(self, key: str) -> int:
        return int(self._r.ttl(key))

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._r.set(key, value, ex=ttl_seconds)

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._r.exists(key))


_backend: KVBackend | None = None


def get_kv() -> KVBackend:
    global _backend
    if _backend is None:
        if settings.REDIS_URL:
            try:
                _backend = RedisBackend(settings.REDIS_URL)
            except Exception:  # noqa: BLE001 - degrade gracefully, never block boot
                _backend = InMemoryBackend()
        else:
            _backend = InMemoryBackend()
    return _backend


def reset_kv() -> None:
    """Test helper."""
    global _backend
    _backend = None
