"""Profile and settings management for the signed-in customer."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, write_audit
from app.core.database import get_db
from app.models.user import UserDevice
from app.schemas import (
    MessageResponse,
    NotificationPrefsRequest,
    ProfileUpdateRequest,
    UserResponse,
)

router = APIRouter(prefix="/profile", tags=["profile"])


@router.patch("", response_model=UserResponse, summary="Update profile details")
def update_profile(
    payload: ProfileUpdateRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in updates.items():
        setattr(user, field, value)

    write_audit(
        db,
        action="profile.update",
        actor=user,
        request=request,
        entity_type="user",
        entity_id=user.id,
        summary=f"Updated: {', '.join(updates)}" if updates else "No changes",
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.patch("/notifications", response_model=UserResponse, summary="Notification preferences")
def update_notification_prefs(
    payload: NotificationPrefsRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    for field, value in payload.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.get("/devices", summary="Known sign-in devices")
def list_devices(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> list[dict]:
    rows = (
        db.execute(
            select(UserDevice)
            .where(UserDevice.user_id == user.id)
            .order_by(UserDevice.last_seen_at.desc().nullslast())
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": d.id,
            "user_agent": d.user_agent,
            "ip_address": d.ip_address,
            "trusted": d.trusted,
            "login_count": d.login_count,
            "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
            "created_at": d.created_at.isoformat(),
        }
        for d in rows
    ]


@router.delete("/devices/{device_id}", response_model=MessageResponse)
def forget_device(
    device_id: int,
    request: Request,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    device = db.get(UserDevice, device_id)
    if device is None or device.user_id != user.id:
        return MessageResponse(message="Device not found")
    db.delete(device)
    write_audit(
        db,
        action="profile.forget_device",
        actor=user,
        request=request,
        entity_type="user_device",
        entity_id=device_id,
    )
    db.commit()
    return MessageResponse(message="Device removed from your trusted list")
