"""Password hashing, JWT issuance/verification, 2FA (TOTP) and misc token helpers."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt
import pyotp

from app.core.cache import get_kv
from app.core.config import settings

TokenType = Literal["access", "refresh"]

# bcrypt has a hard 72-byte input limit; pre-hashing keeps long passphrases safe
# and avoids silent truncation.
_PREHASH_PEPPER = b"smartbank-prehash-v1"


def _prehash(password: str) -> bytes:
    digest = hmac.new(_PREHASH_PEPPER, password.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest)  # 44 bytes, well under the limit


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode())
    except (ValueError, TypeError):
        return False


def password_strength_issues(password: str) -> list[str]:
    issues: list[str] = []
    if len(password) < 10:
        issues.append("must be at least 10 characters")
    if not any(c.islower() for c in password):
        issues.append("must contain a lowercase letter")
    if not any(c.isupper() for c in password):
        issues.append("must contain an uppercase letter")
    if not any(c.isdigit() for c in password):
        issues.append("must contain a digit")
    if not any(not c.isalnum() for c in password):
        issues.append("must contain a symbol")
    return issues


# --------------------------------------------------------------------------- JWT


def _now() -> datetime:
    return datetime.now(UTC)


def create_token(
    subject: int | str,
    token_type: TokenType,
    *,
    role: str = "customer",
    extra: dict[str, Any] | None = None,
) -> tuple[str, str, datetime]:
    """Return ``(encoded_jwt, jti, expires_at)``."""
    jti = uuid.uuid4().hex
    ttl = (
        timedelta(minutes=settings.ACCESS_TOKEN_TTL_MINUTES)
        if token_type == "access"
        else timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS)
    )
    issued = _now()
    expires = issued + ttl
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "type": token_type,
        "jti": jti,
        "iat": int(issued.timestamp()),
        "exp": int(expires.timestamp()),
        "iss": settings.APP_NAME,
    }
    if extra:
        payload.update(extra)
    encoded = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded, jti, expires


def decode_token(
    token: str,
    expected_type: TokenType | None = None,
    *,
    check_revocation: bool = True,
) -> dict[str, Any]:
    """Decode and validate a JWT. Raises ``jwt.PyJWTError`` subclasses on failure.

    ``check_revocation=False`` is used by the refresh endpoint: for refresh tokens
    the persisted ``RefreshToken.revoked`` flag is authoritative, and the caller
    needs to distinguish "revoked" from "invalid" in order to detect token reuse
    and revoke the whole family. Rejecting it here would make that unreachable.
    """
    payload = jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM],
        issuer=settings.APP_NAME,
        options={"require": ["exp", "sub", "jti"]},
    )
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError(f"expected {expected_type} token")
    if check_revocation and is_revoked(payload["jti"]):
        raise jwt.InvalidTokenError("token has been revoked")
    return payload


# ------------------------------------------------------------------ revocation


def revoke_jti(jti: str, ttl_seconds: int) -> None:
    get_kv().set(f"revoked:{jti}", "1", ttl_seconds=max(ttl_seconds, 1))


def is_revoked(jti: str) -> bool:
    return get_kv().exists(f"revoked:{jti}")


# ------------------------------------------------------------------------- 2FA


def new_totp_secret() -> str:
    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, account_email: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=account_email, issuer_name=settings.APP_NAME)


def verify_totp(secret: str, code: str) -> bool:
    if not secret or not code:
        return False
    return pyotp.TOTP(secret).verify(code.strip().replace(" ", ""), valid_window=1)


def current_totp(secret: str) -> str:
    """Only used by the seeder/demo helpers."""
    return pyotp.TOTP(secret).now()


# ----------------------------------------------------------------- misc tokens


def random_reference(prefix: str, length: int = 10) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "".join(secrets.choice(alphabet) for _ in range(length))


def mask_card(number: str) -> str:
    return f"**** **** **** {number[-4:]}" if len(number) >= 4 else "****"


def device_fingerprint(user_agent: str, ip: str) -> str:
    return hashlib.sha256(f"{user_agent}|{ip}".encode()).hexdigest()[:32]
