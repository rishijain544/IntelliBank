"""In-app notification helper.

Email delivery is *simulated*: rows are marked ``email_queued`` rather than sent,
because a demo project should not ship SMTP credentials. Swapping in a real
provider means implementing one function, not rewriting the call sites.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.enums import AlertSeverity, NotificationType
from app.models.system import Notification
from app.models.user import User


def notify(
    db: Session,
    user: User,
    *,
    notif_type: NotificationType | str,
    title: str,
    body: str,
    severity: AlertSeverity | str = AlertSeverity.LOW,
    action_url: str | None = None,
    meta: dict[str, Any] | None = None,
    respect_preferences: bool = True,
) -> Notification | None:
    """Create a notification, honouring the user's preferences.

    Security notifications ignore preferences: a customer must not be able to
    opt out of being told their password changed or a new device signed in.
    """
    ntype = str(notif_type)
    security_critical = ntype in {
        NotificationType.FRAUD_ALERT,
        NotificationType.SECURITY,
        NotificationType.NEW_DEVICE_LOGIN,
    }

    if respect_preferences and not security_critical:
        if ntype == NotificationType.LARGE_TRANSACTION and not user.notify_large_txn:
            return None
        if ntype == NotificationType.GENERAL and not user.notify_marketing:
            return None

    if ntype == NotificationType.NEW_DEVICE_LOGIN and not user.notify_login:
        return None

    entry = Notification(
        user_id=user.id,
        notif_type=ntype,
        severity=str(severity),
        title=title,
        body=body,
        action_url=action_url,
        meta=meta,
        email_queued=bool(user.notify_email),
    )
    db.add(entry)
    return entry


def notify_large_transaction(
    db: Session, user: User, *, amount: Decimal, reference: str, description: str
) -> Notification | None:
    if amount < Decimal(str(settings.LARGE_TXN_NOTIFY_THRESHOLD)):
        return None
    return notify(
        db,
        user,
        notif_type=NotificationType.LARGE_TRANSACTION,
        severity=AlertSeverity.MEDIUM,
        title=f"Large transaction: {amount:,.2f}",
        body=f"{description} (ref {reference}). If you did not authorise this, report it immediately.",
        action_url="/app/transactions",
        meta={"amount": str(amount), "reference": reference},
    )


def notify_new_device(db: Session, user: User, *, city: str | None, ip: str | None) -> Notification | None:
    where = city or ip or "an unrecognised location"
    return notify(
        db,
        user,
        notif_type=NotificationType.NEW_DEVICE_LOGIN,
        severity=AlertSeverity.MEDIUM,
        title="New device sign-in",
        body=f"Your account was accessed from a new device near {where}. Secure your account if this was not you.",
        action_url="/app/settings",
        meta={"city": city, "ip": ip},
    )


def notify_fraud_alert(
    db: Session, user: User, *, alert_ref: str, amount: Decimal, blocked: bool
) -> Notification:
    verb = "blocked" if blocked else "flagged for review"
    return notify(
        db,
        user,
        notif_type=NotificationType.FRAUD_ALERT,
        severity=AlertSeverity.CRITICAL if blocked else AlertSeverity.HIGH,
        title=f"Suspicious transaction {verb}",
        body=(
            f"A transaction of {amount:,.2f} was {verb} by our fraud detection system "
            f"(alert {alert_ref}). Confirm whether this was you."
        ),
        action_url="/app/fraud-center",
        meta={"alert_ref": alert_ref, "amount": str(amount), "blocked": blocked},
    )


def unread_count(db: Session, user_id: int) -> int:
    return int(
        db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == user_id, Notification.is_read.is_(False)
            )
        ).scalar_one()
        or 0
    )
