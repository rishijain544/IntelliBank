"""Fixed-window rate limiting dependency, backed by Redis or in-memory KV."""
from __future__ import annotations

import hashlib

from fastapi import HTTPException, Request, status

from app.core.cache import get_kv


def parse_rule(rule: str) -> tuple[int, int]:
    """Parse a ``"times/seconds"`` rule such as ``"8/300"``."""
    times, _, seconds = rule.partition("/")
    return int(times), int(seconds)


def client_fingerprint(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    ua = request.headers.get("user-agent", "")[:120]
    return hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:32]


class RateLimiter:
    """Usage: ``Depends(RateLimiter(settings.RATE_LIMIT_LOGIN, scope="login"))``."""

    def __init__(self, rule: str, scope: str = "default", by_user: bool = False) -> None:
        self.times, self.seconds = parse_rule(rule)
        self.scope = scope
        self.by_user = by_user

    async def __call__(self, request: Request) -> None:
        kv = get_kv()
        identity = client_fingerprint(request)
        if self.by_user:
            uid = getattr(request.state, "user_id", None)
            if uid:
                identity = f"u{uid}"
        key = f"rl:{self.scope}:{identity}"
        count = kv.incr_with_ttl(key, self.seconds)
        if count > self.times:
            retry = max(kv.ttl(key), 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for '{self.scope}'. Retry in {retry}s.",
                headers={"Retry-After": str(retry), "X-RateLimit-Limit": str(self.times)},
            )
