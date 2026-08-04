"""Fraud & Security Center, spending Insights, and notifications.

The customer-facing surface of the two risk models: fraud alerts are actionable
(confirm/dispute), anomaly alerts are informational nudges.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import ActiveUser, CurrentUser, PageParams, write_audit
from app.core.database import get_db
from app.models.banking import Account, Transaction
from app.models.enums import (
    AccountStatus,
    AlertStatus,
    LoanStatus,
    NotificationType,
    TransactionStatus,
    TransactionType,
)
from app.models.lending import CreditScore, Loan
from app.models.mixins import quantize
from app.models.risk import AnomalyAlert, FraudAlert
from app.models.system import Notification
from app.schemas import (
    AnomalyAlertResponse,
    CategoryBreakdown,
    DailySpend,
    DashboardResponse,
    FraudAlertResponse,
    FraudRespondRequest,
    InsightsResponse,
    MessageResponse,
    MonthlyTrend,
    NotificationResponse,
    Page,
    TransactionResponse,
)
from app.services import banking, notifications as notif

router = APIRouter(tags=["risk & insights"])

DEBIT_TYPES = [
    TransactionType.WITHDRAWAL.value,
    TransactionType.TRANSFER_OUT.value,
    TransactionType.CARD_PAYMENT.value,
    TransactionType.LOAN_REPAYMENT.value,
    TransactionType.FEE.value,
]
CREDIT_TYPES = [
    TransactionType.DEPOSIT.value,
    TransactionType.TRANSFER_IN.value,
    TransactionType.INTEREST.value,
    TransactionType.LOAN_DISBURSEMENT.value,
]


def _spend_window(user_id: int, since: datetime):
    return [
        Transaction.user_id == user_id,
        Transaction.txn_type.in_(DEBIT_TYPES),
        Transaction.status == TransactionStatus.COMPLETED.value,
        Transaction.occurred_at >= since,
    ]


def _category_breakdown(db: Session, user_id: int, since: datetime) -> list[CategoryBreakdown]:
    rows = db.execute(
        select(
            Transaction.merchant_category,
            func.coalesce(func.sum(Transaction.amount), 0),
            func.count(Transaction.id),
        )
        .where(*_spend_window(user_id, since))
        .group_by(Transaction.merchant_category)
        .order_by(func.sum(Transaction.amount).desc())
    ).all()

    grand_total = sum(float(r[1] or 0) for r in rows) or 1.0
    return [
        CategoryBreakdown(
            category=str(cat),
            total=quantize(total or 0),
            count=int(count or 0),
            percentage=round(float(total or 0) / grand_total * 100, 2),
            avg_amount=quantize(float(total or 0) / count if count else 0),
        )
        for cat, total, count in rows
    ]


def _daily_spend(db: Session, user_id: int, since: datetime) -> list[DailySpend]:
    rows = db.execute(
        select(Transaction.occurred_at, Transaction.amount).where(*_spend_window(user_id, since))
    ).all()

    buckets: dict[str, list[float]] = defaultdict(list)
    for occurred_at, amount in rows:
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        buckets[occurred_at.date().isoformat()].append(float(amount or 0))

    # Emit a zero row for every day in the window so charts do not skip gaps.
    out: list[DailySpend] = []
    day = since.date()
    today = datetime.now(UTC).date()
    while day <= today:
        key = day.isoformat()
        vals = buckets.get(key, [])
        out.append(DailySpend(date=key, amount=quantize(sum(vals)), count=len(vals)))
        day += timedelta(days=1)
    return out


def _monthly_trends(db: Session, user_id: int, months: int = 6) -> list[MonthlyTrend]:
    since = datetime.now(UTC) - timedelta(days=months * 31)
    rows = db.execute(
        select(Transaction.occurred_at, Transaction.signed_amount, Transaction.txn_type).where(
            Transaction.user_id == user_id,
            Transaction.status == TransactionStatus.COMPLETED.value,
            Transaction.occurred_at >= since,
        )
    ).all()

    agg: dict[str, dict[str, Any]] = {}
    for occurred_at, signed, txn_type in rows:
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        key = occurred_at.strftime("%Y-%m")
        entry = agg.setdefault(key, {"inflow": 0.0, "outflow": 0.0, "count": 0})
        amount = float(signed or 0)
        if amount >= 0:
            entry["inflow"] += amount
        else:
            entry["outflow"] += -amount
        entry["count"] += 1

    return [
        MonthlyTrend(
            month=key,
            inflow=quantize(v["inflow"]),
            outflow=quantize(v["outflow"]),
            net=quantize(v["inflow"] - v["outflow"]),
            txn_count=v["count"],
        )
        for key, v in sorted(agg.items())
    ]


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


@router.get("/dashboard", response_model=DashboardResponse, summary="Dashboard home data")
def dashboard(
    user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> DashboardResponse:
    since = datetime.now(UTC) - timedelta(days=30)

    accounts = (
        db.execute(
            select(Account)
            .where(Account.user_id == user.id, Account.status != AccountStatus.CLOSED)
            .order_by(Account.is_primary.desc(), Account.id)
        )
        .scalars()
        .all()
    )
    recent = (
        db.execute(
            select(Transaction)
            .where(Transaction.user_id == user.id)
            .order_by(Transaction.occurred_at.desc())
            .limit(8)
        )
        .scalars()
        .all()
    )

    spent = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(*_spend_window(user.id, since))
    ).scalar_one()
    received = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user.id,
            Transaction.txn_type.in_(CREDIT_TYPES),
            Transaction.status == TransactionStatus.COMPLETED.value,
            Transaction.occurred_at >= since,
        )
    ).scalar_one()

    open_alerts = int(
        db.execute(
            select(func.count(FraudAlert.id)).where(
                FraudAlert.user_id == user.id,
                FraudAlert.status.in_([AlertStatus.OPEN.value, AlertStatus.DISPUTED.value]),
            )
        ).scalar_one()
        or 0
    )
    active_loans = int(
        db.execute(
            select(func.count(Loan.id)).where(
                Loan.user_id == user.id,
                Loan.status.in_([LoanStatus.DISBURSED.value, LoanStatus.APPROVED.value]),
            )
        ).scalar_one()
        or 0
    )
    latest_score = db.execute(
        select(CreditScore.score)
        .where(CreditScore.user_id == user.id)
        .order_by(CreditScore.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    return DashboardResponse(
        total_balance=banking.total_balance(db, user.id),
        accounts=[
            {
                **{
                    k: getattr(a, k)
                    for k in (
                        "id", "account_number", "ifsc_code", "nickname", "account_type",
                        "status", "currency", "balance", "hold_amount", "overdraft_limit",
                        "interest_rate", "is_primary", "opened_on", "created_at",
                    )
                },
                "available_balance": a.available_balance,
            }
            for a in accounts
        ],
        recent_transactions=[TransactionResponse.model_validate(t) for t in recent],
        spend_last_30d=quantize(spent or 0),
        received_last_30d=quantize(received or 0),
        open_fraud_alerts=open_alerts,
        unread_notifications=notif.unread_count(db, user.id),
        active_loans=active_loans,
        category_breakdown=_category_breakdown(db, user.id, since)[:8],
        daily_spend=_daily_spend(db, user.id, since),
        latest_credit_score=int(latest_score) if latest_score else None,
    )


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #


@router.get("/insights", response_model=InsightsResponse, summary="Spending insights and anomalies")
def insights(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    days: Annotated[int, Query(ge=7, le=365)] = 30,
) -> InsightsResponse:
    since = datetime.now(UTC) - timedelta(days=days)

    spent, txn_count, largest = db.execute(
        select(
            func.coalesce(func.sum(Transaction.amount), 0),
            func.count(Transaction.id),
            func.coalesce(func.max(Transaction.amount), 0),
        ).where(*_spend_window(user.id, since))
    ).one()

    received = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user.id,
            Transaction.txn_type.in_(CREDIT_TYPES),
            Transaction.status == TransactionStatus.COMPLETED.value,
            Transaction.occurred_at >= since,
        )
    ).scalar_one()

    merchant_rows = db.execute(
        select(
            Transaction.merchant_name,
            func.coalesce(func.sum(Transaction.amount), 0),
            func.count(Transaction.id),
        )
        .where(*_spend_window(user.id, since), Transaction.merchant_name.is_not(None))
        .group_by(Transaction.merchant_name)
        .order_by(func.sum(Transaction.amount).desc())
        .limit(8)
    ).all()

    anomalies = (
        db.execute(
            select(AnomalyAlert)
            .where(AnomalyAlert.user_id == user.id, AnomalyAlert.acknowledged.is_(False))
            .order_by(AnomalyAlert.created_at.desc())
            .limit(10)
        )
        .scalars()
        .all()
    )

    total_spent = quantize(spent or 0)
    count = int(txn_count or 0)
    return InsightsResponse(
        period_days=days,
        total_spent=total_spent,
        total_received=quantize(received or 0),
        net_change=quantize(float(received or 0) - float(spent or 0)),
        txn_count=count,
        avg_transaction=quantize(float(spent or 0) / count if count else 0),
        largest_transaction=quantize(largest or 0),
        category_breakdown=_category_breakdown(db, user.id, since),
        monthly_trends=_monthly_trends(db, user.id),
        daily_spend=_daily_spend(db, user.id, since),
        top_merchants=[
            {"merchant": str(m), "total": float(t or 0), "count": int(c or 0)}
            for m, t, c in merchant_rows
        ],
        anomaly_alerts=[AnomalyAlertResponse.model_validate(a) for a in anomalies],
    )


@router.post("/insights/anomalies/{alert_id}/acknowledge", response_model=MessageResponse)
def acknowledge_anomaly(
    alert_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> MessageResponse:
    alert = db.get(AnomalyAlert, alert_id)
    if alert is None or alert.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(UTC)
    db.commit()
    return MessageResponse(message="Insight dismissed")


# --------------------------------------------------------------------------- #
# Fraud & Security Center
# --------------------------------------------------------------------------- #


@router.get(
    "/fraud/alerts",
    response_model=Page[FraudAlertResponse],
    summary="Own fraud alerts, newest first",
)
def list_fraud_alerts(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    alert_status: Annotated[str | None, Query(alias="status")] = None,
) -> Page[FraudAlertResponse]:
    stmt = (
        select(FraudAlert)
        .options(selectinload(FraudAlert.transaction))
        .where(FraudAlert.user_id == user.id)
    )
    if alert_status:
        stmt = stmt.where(FraudAlert.status == alert_status)

    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(FraudAlert.created_at.desc())
            .offset(pagination.offset)
            .limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[FraudAlertResponse].model_validate(
        pagination.envelope([FraudAlertResponse.model_validate(a) for a in rows], total)
    )


@router.post(
    "/fraud/alerts/{alert_id}/respond",
    response_model=FraudAlertResponse,
    summary="Confirm a transaction was yours, or dispute it as fraud",
)
def respond_to_alert(
    alert_id: int,
    payload: FraudRespondRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> FraudAlertResponse:
    """Record the customer's verdict.

    A customer confirming "this was me" is treated as a *signal*, not as final
    ground truth: only an admin review writes the training label, because a
    fraudster with account access could otherwise self-clear their own alerts.
    """
    alert = db.get(FraudAlert, alert_id)
    if alert is None or alert.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    if alert.status not in (AlertStatus.OPEN, AlertStatus.DISPUTED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This alert has already been resolved"
        )

    alert.customer_response = payload.response
    alert.customer_responded_at = datetime.now(UTC)
    alert.customer_note = payload.note
    alert.status = (
        AlertStatus.CONFIRMED_FRAUD if payload.response == "disputed" else AlertStatus.DISPUTED
    )

    txn = db.get(Transaction, alert.transaction_id)
    if payload.response == "disputed" and txn is not None:
        # Customer says it was not them: freeze the money if it is still held.
        if txn.status == TransactionStatus.HELD:
            banking.release_held_transfer(db, txn, approve=False)
        alert.severity = "critical"
        notif.notify(
            db,
            user,
            notif_type=NotificationType.SECURITY,
            title="Dispute received",
            body="We have escalated this transaction to our fraud team and held the funds.",
            action_url="/app/fraud-center",
            respect_preferences=False,
        )

    write_audit(
        db,
        action=f"fraud.customer_{payload.response}",
        actor=user,
        request=request,
        entity_type="fraud_alert",
        entity_id=alert.id,
        summary=f"Customer marked alert {alert.alert_ref} as {payload.response}",
    )
    db.commit()
    db.refresh(alert)
    return FraudAlertResponse.model_validate(alert)


@router.get("/fraud/summary", summary="Security posture summary")
def fraud_summary(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    total = int(
        db.execute(
            select(func.count(FraudAlert.id)).where(FraudAlert.user_id == user.id)
        ).scalar_one()
        or 0
    )
    open_count = int(
        db.execute(
            select(func.count(FraudAlert.id)).where(
                FraudAlert.user_id == user.id, FraudAlert.status == AlertStatus.OPEN.value
            )
        ).scalar_one()
        or 0
    )
    blocked = int(
        db.execute(
            select(func.count(Transaction.id)).where(
                Transaction.user_id == user.id,
                Transaction.status == TransactionStatus.BLOCKED.value,
            )
        ).scalar_one()
        or 0
    )
    confirmed = int(
        db.execute(
            select(func.count(FraudAlert.id)).where(
                FraudAlert.user_id == user.id, FraudAlert.final_label.is_(True)
            )
        ).scalar_one()
        or 0
    )
    return {
        "total_alerts": total,
        "open_alerts": open_count,
        "blocked_transactions": blocked,
        "confirmed_fraud": confirmed,
        "two_factor_enabled": user.two_factor_enabled,
        "kyc_status": user.kyc_status,
    }


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #


@router.get("/notifications", response_model=Page[NotificationResponse])
def list_notifications(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    unread_only: Annotated[bool, Query()] = False,
) -> Page[NotificationResponse]:
    stmt = select(Notification).where(Notification.user_id == user.id)
    if unread_only:
        stmt = stmt.where(Notification.is_read.is_(False))

    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(Notification.created_at.desc())
            .offset(pagination.offset)
            .limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[NotificationResponse].model_validate(
        pagination.envelope([NotificationResponse.model_validate(n) for n in rows], total)
    )


@router.post("/notifications/{notification_id}/read", response_model=MessageResponse)
def mark_read(
    notification_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> MessageResponse:
    row = db.get(Notification, notification_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    row.is_read = True
    row.read_at = datetime.now(UTC)
    db.commit()
    return MessageResponse(message="Marked as read")


@router.post("/notifications/read-all", response_model=MessageResponse)
def mark_all_read(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> MessageResponse:
    now = datetime.now(UTC)
    count = 0
    for row in db.execute(
        select(Notification).where(
            Notification.user_id == user.id, Notification.is_read.is_(False)
        )
    ).scalars():
        row.is_read = True
        row.read_at = now
        count += 1
    db.commit()
    return MessageResponse(message=f"Marked {count} notification(s) as read")
