"""Shared FastAPI dependencies: authentication, RBAC, pagination and auditing."""
from __future__ import annotations

from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import decode_token
from app.models.enums import UserRole, UserStatus
from app.models.system import AuditLog
from app.models.user import User

# auto_error=False so a missing header yields our own 401 shape rather than
# FastAPI's, keeping error payloads consistent for the frontend.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve the caller from a Bearer access token.

    Status is re-checked on every request rather than trusted from the token, so
    an admin freezing an account takes effect immediately instead of when the
    access token happens to expire.
    """
    if credentials is None or not credentials.credentials:
        raise CREDENTIALS_ERROR

    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise CREDENTIALS_ERROR from exc

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CREDENTIALS_ERROR from exc

    user = db.get(User, user_id)
    if user is None:
        raise CREDENTIALS_ERROR

    if user.status == UserStatus.FROZEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is frozen. Contact support for assistance.",
        )
    if user.status == UserStatus.SUSPENDED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This account has been suspended."
        )

    # Expose the id for by-user rate limiting further down the stack.
    request.state.user_id = user.id
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_onboarding_user(user: CurrentUser) -> User:
    """Allow any non-frozen account, KYC-verified or not.

    Used for actions that carry no external compliance exposure: opening a basic
    account, adding simulated funds, moving money between accounts on this
    platform, and running a dry-run credit quote.

    This exists because gating those behind KYC indirectly disables the three ML
    features — a user with no account and no funds generates no transactions for
    the fraud model to score and no history for insights to analyse.

    ``get_current_user`` has already rejected frozen and suspended accounts, so
    reaching here means the caller is authenticated and in good standing.
    """
    return user


OnboardingUser = Annotated[User, Depends(get_onboarding_user)]


def get_active_user(user: CurrentUser) -> User:
    """Require a KYC-verified account.

    Reserved for actions a real bank gates on verified identity: external
    (interbank) transfers, card issuance, and binding loan applications.
    """
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Complete KYC verification to use this feature.",
        )
    return user


ActiveUser = Annotated[User, Depends(get_active_user)]


def get_admin_user(user: CurrentUser) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required"
        )
    return user


AdminUser = Annotated[User, Depends(get_admin_user)]


class Pagination:
    """Standard page/page_size query parameters with a hard upper bound."""

    def __init__(
        self,
        page: Annotated[int, Query(ge=1, le=10_000, description="1-indexed page number")] = 1,
        page_size: Annotated[int, Query(ge=1, le=100, description="Rows per page")] = 20,
    ) -> None:
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    def envelope(self, items: list, total: int) -> dict[str, Any]:
        total_pages = (total + self.page_size - 1) // self.page_size if total else 0
        return {
            "items": items,
            "total": total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": total_pages,
            "has_next": self.page < total_pages,
            "has_prev": self.page > 1,
        }


PageParams = Annotated[Pagination, Depends(Pagination)]


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def write_audit(
    db: Session,
    *,
    action: str,
    actor: User | None = None,
    request: Request | None = None,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    target_user_id: int | None = None,
    summary: str | None = None,
    before_state: dict[str, Any] | None = None,
    after_state: dict[str, Any] | None = None,
    success: bool = True,
) -> AuditLog:
    """Append an audit record.

    The row is added to the caller's session but not committed here, so the audit
    entry lands in the same transaction as the action it describes — an audit log
    that can commit independently of its action is worse than none.
    """
    entry = AuditLog(
        actor_id=actor.id if actor else None,
        actor_email=actor.email if actor else None,
        actor_role=actor.role if actor else None,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        target_user_id=target_user_id,
        summary=summary,
        before_state=before_state,
        after_state=after_state,
        ip_address=client_ip(request) if request else None,
        user_agent=request.headers.get("user-agent", "")[:400] if request else None,
        success=success,
    )
    db.add(entry)
    return entry
