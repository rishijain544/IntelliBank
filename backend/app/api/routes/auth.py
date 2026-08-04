"""Authentication, registration, KYC and session management.

Security choices worth noting:

* Login failures return one generic message regardless of cause, so the endpoint
  cannot be used to enumerate registered email addresses.
* Refresh tokens are persisted and rotated on every use. Presenting an already-
  rotated token revokes the whole family, which is the standard detection for a
  stolen refresh token being replayed.
* Government ID numbers are hashed before storage; only a masked form is kept for
  display.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import ActiveUser, CurrentUser, write_audit
from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import RateLimiter
from app.core.security import (
    create_token,
    decode_token,
    device_fingerprint,
    hash_password,
    new_totp_secret,
    revoke_jti,
    totp_provisioning_uri,
    verify_password,
    verify_totp,
)
from app.models.enums import AccountType, KycStatus, UserRole, UserStatus
from app.models.user import RefreshToken, User, UserDevice
from app.schemas import (
    KycSubmitRequest,
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    TwoFactorSetupResponse,
    TwoFactorVerifyRequest,
    UserResponse,
)
from app.services import banking, notifications as notif

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect email or password",
    headers={"WWW-Authenticate": "Bearer"},
)


def _hash_id(value: str) -> str:
    """Hash a government ID with an app-scoped salt.

    A bare SHA-256 of a 12-digit Aadhaar is brute-forceable in seconds, so the
    secret is mixed in. Real KYC would use a dedicated KMS-held key.
    """
    return hashlib.sha256(f"{settings.JWT_SECRET}:{value}".encode()).hexdigest()


def _issue_tokens(
    db: Session, user: User, request: Request, *, device: str | None = None
) -> TokenResponse:
    access, _, _ = create_token(user.id, "access", role=user.role)
    refresh, refresh_jti, refresh_exp = create_token(user.id, "refresh", role=user.role)

    db.add(
        RefreshToken(
            user_id=user.id,
            jti=refresh_jti,
            expires_at=refresh_exp,
            device_fingerprint=device,
            ip_address=request.client.host if request.client else None,
        )
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.ACCESS_TOKEN_TTL_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


def _track_device(db: Session, user: User, request: Request) -> tuple[str, bool]:
    """Upsert the calling device. Returns ``(fingerprint, is_new)``."""
    ua = request.headers.get("user-agent", "")
    ip = request.client.host if request.client else "unknown"
    fp = device_fingerprint(ua, ip)

    existing = db.execute(
        select(UserDevice).where(UserDevice.user_id == user.id, UserDevice.fingerprint == fp)
    ).scalar_one_or_none()

    now = datetime.now(UTC)
    if existing:
        existing.login_count += 1
        existing.last_seen_at = now
        return fp, False

    db.add(
        UserDevice(
            user_id=user.id,
            fingerprint=fp,
            user_agent=ua[:400],
            ip_address=ip,
            login_count=1,
            last_seen_at=now,
            # The first device seen is trusted implicitly; later ones are not.
            trusted=not db.execute(
                select(UserDevice.id).where(UserDevice.user_id == user.id).limit(1)
            ).first(),
        )
    )
    return fp, True


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RateLimiter(settings.RATE_LIMIT_REGISTER, scope="register"))],
    summary="Create an account (step 1 of onboarding)",
)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    existing = db.execute(select(User.id).where(User.email == payload.email)).first()
    if existing:
        # Registration cannot be silent about duplicates, but the message stays
        # neutral about whether the account is usable.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    if payload.phone:
        if db.execute(select(User.id).where(User.phone == payload.phone)).first():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this phone number already exists",
            )

    user = User(
        email=str(payload.email).lower(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name.strip(),
        phone=payload.phone,
        date_of_birth=payload.date_of_birth,
        role=UserRole.CUSTOMER,
        status=UserStatus.PENDING,
        kyc_status=KycStatus.NOT_STARTED,
        password_changed_at=datetime.now(UTC),
    )
    db.add(user)
    db.flush()

    notif.notify(
        db,
        user,
        notif_type="account_update",
        title="Welcome to IntelliBank",
        body="Complete your KYC verification to activate banking features.",
        action_url="/app/settings",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="user.register",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary=f"Account created for {user.email}",
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.post(
    "/kyc",
    response_model=UserResponse,
    summary="Submit KYC details (step 2 of onboarding)",
)
def submit_kyc(
    payload: KycSubmitRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    """Simulated KYC.

    Auto-verifies so the demo is usable end to end; a real deployment would queue
    this for manual or vendor review. Admins can still re-verify or reject.
    """
    if user.kyc_status == KycStatus.VERIFIED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="KYC is already verified"
        )

    pan_hash = _hash_id(payload.pan)
    clash = db.execute(
        select(User.id).where(User.pan_hash == pan_hash, User.id != user.id)
    ).first()
    if clash:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="These identity details are already registered to another account",
        )

    now = datetime.now(UTC)
    user.pan_hash = pan_hash
    user.pan_masked = f"{payload.pan[:3]}XXXX{payload.pan[-1]}"
    user.aadhaar_hash = _hash_id(payload.aadhaar)
    user.aadhaar_masked = f"XXXX-XXXX-{payload.aadhaar[-4:]}"
    user.id_document_type = payload.document_type
    user.id_document_name = payload.document_name
    user.address_line1 = payload.address_line1
    user.address_line2 = payload.address_line2
    user.city = payload.city
    user.state = payload.state
    user.postal_code = payload.postal_code
    user.annual_income = payload.annual_income
    user.employment_status = payload.employment_status
    user.employment_years = payload.employment_years
    user.dependents = payload.dependents
    user.housing_status = payload.housing_status
    user.kyc_status = KycStatus.VERIFIED
    user.kyc_submitted_at = now
    user.kyc_verified_at = now
    user.status = UserStatus.ACTIVE

    # Activating the customer opens their first account so the dashboard is not empty.
    has_account = db.execute(
        select(User.id).join(User.accounts).where(User.id == user.id).limit(1)
    ).first()
    if not has_account:
        banking.create_account(
            db, user, account_type=AccountType.SAVINGS, nickname="Primary Savings"
        )

    notif.notify(
        db,
        user,
        notif_type="account_update",
        title="KYC verified",
        body="Your identity has been verified and your savings account is now active.",
        action_url="/app/accounts",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="user.kyc_submit",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary="KYC submitted and auto-verified (simulated)",
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(RateLimiter(settings.RATE_LIMIT_LOGIN, scope="login"))],
    summary="Sign in and receive an access/refresh token pair",
)
def login(
    payload: LoginRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    user = db.execute(
        select(User).where(User.email == str(payload.email).lower())
    ).scalar_one_or_none()

    # Uniform failure response: no distinction between unknown email and bad password.
    if user is None:
        raise INVALID_CREDENTIALS

    now = datetime.now(UTC)
    if user.locked_until and user.locked_until > now:
        remaining = int((user.locked_until - now).total_seconds() / 60) + 1
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account temporarily locked after repeated failed attempts. Try again in {remaining} minute(s).",
        )

    if not verify_password(payload.password, user.hashed_password):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
            user.failed_login_attempts = 0
            notif.notify(
                db,
                user,
                notif_type="security",
                title="Account temporarily locked",
                body=f"We locked your account for {LOCKOUT_MINUTES} minutes after repeated failed sign-ins.",
                respect_preferences=False,
            )
        write_audit(
            db,
            action="user.login_failed",
            actor=user,
            request=request,
            entity_type="user",
            entity_id=user.id,
            summary="Incorrect password",
            success=False,
        )
        db.commit()
        raise INVALID_CREDENTIALS

    if user.status == UserStatus.FROZEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is frozen. Contact support for assistance.",
        )
    if user.status == UserStatus.SUSPENDED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This account has been suspended."
        )

    if user.two_factor_enabled:
        if not payload.totp_code:
            # 428 distinguishes "credentials fine, second factor required" from a
            # plain auth failure, so the client can render the code prompt.
            raise HTTPException(
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                detail="Two-factor authentication code required",
            )
        if not verify_totp(user.two_factor_secret or "", payload.totp_code):
            write_audit(
                db,
                action="user.login_2fa_failed",
                actor=user,
                request=request,
                entity_type="user",
                entity_id=user.id,
                success=False,
            )
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid two-factor authentication code",
            )

    fp, is_new_device = _track_device(db, user, request)
    if is_new_device:
        notif.notify_new_device(
            db, user, city=user.city, ip=request.client.host if request.client else None
        )

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now

    tokens = _issue_tokens(db, user, request, device=fp)
    write_audit(
        db,
        action="user.login",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary="New device" if is_new_device else "Known device",
    )
    db.commit()
    return tokens


@router.post("/refresh", response_model=TokenResponse, summary="Rotate a refresh token")
def refresh_tokens(
    payload: RefreshRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    try:
        # Revocation is checked against the persisted row below, not the KV list,
        # so that a replayed (already-rotated) token is distinguishable from a
        # forged one and can trigger the session-theft response.
        claims = decode_token(
            payload.refresh_token, expected_type="refresh", check_revocation=False
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        ) from exc

    stored = db.execute(
        select(RefreshToken).where(RefreshToken.jti == claims["jti"])
    ).scalar_one_or_none()

    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is not recognised"
        )

    if stored.revoked:
        # Replay of an already-rotated token: treat the family as compromised and
        # sign every session out rather than just rejecting this one request.
        for token in db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == stored.user_id, RefreshToken.revoked.is_(False)
            )
        ).scalars():
            token.revoked = True
            revoke_jti(token.jti, settings.REFRESH_TOKEN_TTL_DAYS * 86400)
        user = db.get(User, stored.user_id)
        if user:
            notif.notify(
                db,
                user,
                notif_type="security",
                title="All sessions signed out",
                body="We detected reuse of an expired session token and signed out every device as a precaution.",
                respect_preferences=False,
            )
        write_audit(
            db,
            action="auth.refresh_reuse_detected",
            actor=user,
            request=request,
            entity_type="refresh_token",
            entity_id=stored.jti,
            summary="Revoked all sessions after refresh-token reuse",
            success=False,
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token reuse detected. Please sign in again.",
        )

    user = db.get(User, stored.user_id)
    if user is None or user.status in (UserStatus.FROZEN, UserStatus.SUSPENDED):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This account cannot be refreshed"
        )

    stored.revoked = True
    revoke_jti(stored.jti, settings.REFRESH_TOKEN_TTL_DAYS * 86400)
    tokens = _issue_tokens(db, user, request, device=stored.device_fingerprint)
    db.commit()
    return tokens


@router.post("/logout", response_model=MessageResponse, summary="Revoke the current session")
def logout(
    payload: RefreshRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    try:
        claims = decode_token(payload.refresh_token, expected_type="refresh")
        stored = db.execute(
            select(RefreshToken).where(
                RefreshToken.jti == claims["jti"], RefreshToken.user_id == user.id
            )
        ).scalar_one_or_none()
        if stored and not stored.revoked:
            stored.revoked = True
            revoke_jti(stored.jti, settings.REFRESH_TOKEN_TTL_DAYS * 86400)
    except jwt.PyJWTError:
        # An unparseable token still results in a signed-out client, so this is
        # not surfaced as an error.
        pass

    write_audit(db, action="user.logout", actor=user, request=request, entity_type="user", entity_id=user.id)
    db.commit()
    return MessageResponse(message="Signed out successfully")


@router.post("/logout-all", response_model=MessageResponse, summary="Revoke every session")
def logout_all(
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    count = 0
    for token in db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id, RefreshToken.revoked.is_(False)
        )
    ).scalars():
        token.revoked = True
        revoke_jti(token.jti, settings.REFRESH_TOKEN_TTL_DAYS * 86400)
        count += 1

    write_audit(
        db,
        action="user.logout_all",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary=f"Revoked {count} session(s)",
    )
    db.commit()
    return MessageResponse(message=f"Signed out of {count} session(s)")


@router.get("/me", response_model=UserResponse, summary="Current user profile")
def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)


@router.post("/change-password", response_model=MessageResponse)
def change_password(
    payload: PasswordChangeRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    if not verify_password(payload.current_password, user.hashed_password):
        write_audit(
            db,
            action="user.password_change_failed",
            actor=user,
            request=request,
            entity_type="user",
            entity_id=user.id,
            success=False,
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )

    user.hashed_password = hash_password(payload.new_password)
    user.password_changed_at = datetime.now(UTC)

    # A password change invalidates every existing session.
    revoked = 0
    for token in db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id, RefreshToken.revoked.is_(False)
        )
    ).scalars():
        token.revoked = True
        revoke_jti(token.jti, settings.REFRESH_TOKEN_TTL_DAYS * 86400)
        revoked += 1

    notif.notify(
        db,
        user,
        notif_type="security",
        title="Password changed",
        body="Your password was changed and all other sessions were signed out.",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="user.password_change",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary=f"Password changed; {revoked} session(s) revoked",
    )
    db.commit()
    return MessageResponse(
        message="Password updated successfully",
        detail="Please sign in again on your other devices.",
    )


@router.post("/2fa/setup", response_model=TwoFactorSetupResponse)
def setup_two_factor(
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> TwoFactorSetupResponse:
    """Generate a TOTP secret. 2FA is not active until a code is confirmed."""
    if user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Two-factor authentication is already enabled",
        )

    secret = new_totp_secret()
    user.two_factor_secret = secret
    write_audit(
        db,
        action="user.2fa_setup_started",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
    )
    db.commit()
    return TwoFactorSetupResponse(
        secret=secret, provisioning_uri=totp_provisioning_uri(secret, user.email)
    )


@router.post("/2fa/enable", response_model=MessageResponse)
def enable_two_factor(
    payload: TwoFactorVerifyRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    if not user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Start 2FA setup first"
        )
    if not verify_totp(user.two_factor_secret, payload.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="That code is not valid"
        )

    user.two_factor_enabled = True
    notif.notify(
        db,
        user,
        notif_type="security",
        title="Two-factor authentication enabled",
        body="Your account now requires an authenticator code at sign-in.",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="user.2fa_enabled",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
    )
    db.commit()
    return MessageResponse(message="Two-factor authentication enabled")


@router.post("/2fa/disable", response_model=MessageResponse)
def disable_two_factor(
    payload: TwoFactorVerifyRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    """Disabling 2FA still requires a valid code, so a hijacked access token
    alone cannot strip the second factor."""
    if not user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Two-factor authentication is not enabled",
        )
    if not verify_totp(user.two_factor_secret or "", payload.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="That code is not valid"
        )

    user.two_factor_enabled = False
    user.two_factor_secret = None
    notif.notify(
        db,
        user,
        notif_type="security",
        title="Two-factor authentication disabled",
        body="Two-factor authentication was turned off for your account.",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="user.2fa_disabled",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
    )
    db.commit()
    return MessageResponse(message="Two-factor authentication disabled")
