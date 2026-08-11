"""Admin panel: platform stats, user management, review queues and model monitoring.

The fraud review queue is where the retraining loop closes: an admin verdict
writes ``Transaction.is_fraud_label``, which is the ground truth the next training
run consumes. Drift is measured with PSI against the score distribution captured
at training time.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import AdminUser, PageParams, write_audit
from app.core.database import get_db
from app.ml.inference import models_status, reload_models
from app.ml.metrics import drift_status, psi_against_baseline
from app.ml.registry import read_metrics
from app.ml.train_anomaly import MODEL_NAME as ANOMALY_MODEL
from app.ml.train_credit import MODEL_NAME as CREDIT_MODEL
from app.ml.train_fraud import MODEL_NAME as FRAUD_MODEL
from app.models.banking import Account, Transaction
from app.models.enums import (
    AccountStatus,
    AlertSeverity,
    AlertStatus,
    KycStatus,
    LoanStatus,
    NotificationType,
    TransactionStatus,
    UserRole,
    UserStatus,
)
from app.models.lending import CreditScore, Loan
from app.models.mixins import quantize
from app.models.risk import AnomalyAlert, FraudAlert
from app.models.system import AuditLog, ModelMetricSnapshot
from app.models.user import User
from app.schemas import (
    AdminStatsResponse,
    AuditLogResponse,
    FraudAlertResponse,
    FraudReviewRequest,
    KycDecisionRequest,
    LoanBookRow,
    LoanDecisionRequest,
    LoanResponse,
    MessageResponse,
    ModelPerformanceResponse,
    Page,
    UserResponse,
    UserStatusRequest,
    UserSummary,
)
from app.services import banking, notifications as notif
from app.services.email import render_basic_email, send_email

router = APIRouter(prefix="/admin", tags=["admin"])


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


@router.get("/stats", response_model=AdminStatsResponse, summary="Platform-wide statistics")
def admin_stats(admin: AdminUser, db: Annotated[Session, Depends(get_db)]) -> AdminStatsResponse:
    now = datetime.now(UTC)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    d30 = now - timedelta(days=30)

    def count(model, *where) -> int:
        return int(db.execute(select(func.count(model.id)).where(*where)).scalar_one() or 0)

    def total(col, *where) -> Decimal:
        return quantize(
            db.execute(select(func.coalesce(func.sum(col), 0)).where(*where)).scalar_one() or 0
        )

    completed = Transaction.status == TransactionStatus.COMPLETED.value

    return AdminStatsResponse(
        total_users=count(User, User.role == UserRole.CUSTOMER.value),
        active_users=count(User, User.role == UserRole.CUSTOMER.value, User.status == UserStatus.ACTIVE.value),
        pending_kyc=count(User, User.kyc_status.in_([KycStatus.NOT_STARTED.value, KycStatus.SUBMITTED.value])),
        frozen_users=count(User, User.status == UserStatus.FROZEN.value),
        total_accounts=count(Account, Account.status != AccountStatus.CLOSED.value),
        total_balance=total(Account.balance, Account.status != AccountStatus.CLOSED.value),
        txn_count_today=count(Transaction, Transaction.occurred_at >= day_start),
        txn_volume_today=total(Transaction.amount, completed, Transaction.occurred_at >= day_start),
        txn_count_30d=count(Transaction, Transaction.occurred_at >= d30),
        txn_volume_30d=total(Transaction.amount, completed, Transaction.occurred_at >= d30),
        fraud_alerts_open=count(FraudAlert, FraudAlert.status == AlertStatus.OPEN.value),
        fraud_alerts_total=count(FraudAlert),
        fraud_confirmed=count(FraudAlert, FraudAlert.final_label.is_(True)),
        blocked_transactions=count(Transaction, Transaction.status == TransactionStatus.BLOCKED.value),
        loans_pending=count(Loan, Loan.status.in_([LoanStatus.SUBMITTED.value, LoanStatus.UNDER_REVIEW.value])),
        loans_approved=count(Loan, Loan.status.in_([LoanStatus.APPROVED.value, LoanStatus.DISBURSED.value])),
        loans_disbursed_value=total(Loan.approved_amount, Loan.status == LoanStatus.DISBURSED.value),
        model_status=models_status(),
    )


# --------------------------------------------------------------------------- #
# User management
# --------------------------------------------------------------------------- #


@router.get("/users", response_model=Page[UserSummary], summary="Search and list users")
def list_users(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    search: Annotated[str | None, Query(max_length=120)] = None,
    user_status: Annotated[str | None, Query(alias="status")] = None,
    kyc_status: Annotated[str | None, Query()] = None,
    role: Annotated[str | None, Query()] = None,
) -> Page[UserSummary]:
    stmt = select(User)
    if search:
        like = f"%{search.strip()}%"
        stmt = stmt.where(or_(User.email.ilike(like), User.full_name.ilike(like), User.phone.ilike(like)))
    if user_status:
        stmt = stmt.where(User.status == user_status)
    if kyc_status:
        stmt = stmt.where(User.kyc_status == kyc_status)
    if role:
        stmt = stmt.where(User.role == role)

    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(User.created_at.desc()).offset(pagination.offset).limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[UserSummary].model_validate(
        pagination.envelope([UserSummary.model_validate(u) for u in rows], total)
    )


@router.get("/users/{user_id}", response_model=dict, summary="Full user detail")
def get_user_detail(
    user_id: int, admin: AdminUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    accounts = db.execute(select(Account).where(Account.user_id == user_id)).scalars().all()
    latest_score = db.execute(
        select(CreditScore).where(CreditScore.user_id == user_id).order_by(CreditScore.created_at.desc()).limit(1)
    ).scalar_one_or_none()

    return {
        "user": UserResponse.model_validate(user).model_dump(mode="json"),
        "accounts": [
            {
                "id": a.id,
                "account_number": a.account_number,
                "account_type": a.account_type,
                "status": a.status,
                "balance": float(a.balance),
            }
            for a in accounts
        ],
        "total_balance": float(banking.total_balance(db, user_id)),
        "txn_count": int(
            db.execute(select(func.count(Transaction.id)).where(Transaction.user_id == user_id)).scalar_one() or 0
        ),
        "open_fraud_alerts": int(
            db.execute(
                select(func.count(FraudAlert.id)).where(
                    FraudAlert.user_id == user_id, FraudAlert.status == AlertStatus.OPEN.value
                )
            ).scalar_one()
            or 0
        ),
        "loans": int(db.execute(select(func.count(Loan.id)).where(Loan.user_id == user_id)).scalar_one() or 0),
        "latest_credit_score": (
            {
                "score": latest_score.score,
                "risk_band": latest_score.risk_band,
                "probability_of_default": latest_score.probability_of_default,
                "created_at": latest_score.created_at.isoformat(),
            }
            if latest_score
            else None
        ),
    }


@router.patch("/users/{user_id}/status", response_model=UserResponse, summary="Freeze or reactivate")
def set_user_status(
    user_id: int,
    payload: UserStatusRequest,
    request: Request,
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot change your own status"
        )
    if user.role == UserRole.ADMIN and payload.status != UserStatus.ACTIVE:
        # Prevents an admin from locking every operator out of the platform.
        remaining = int(
            db.execute(
                select(func.count(User.id)).where(
                    User.role == UserRole.ADMIN.value,
                    User.status == UserStatus.ACTIVE.value,
                    User.id != user.id,
                )
            ).scalar_one()
            or 0
        )
        if remaining == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot disable the last active administrator",
            )

    before = user.status
    user.status = payload.status

    # Freezing the customer freezes their accounts too, otherwise scheduled
    # activity could still move money on a disabled profile.
    for account in db.execute(select(Account).where(Account.user_id == user.id)).scalars():
        if payload.status == UserStatus.ACTIVE and account.status == AccountStatus.FROZEN:
            account.status = AccountStatus.ACTIVE
        elif payload.status != UserStatus.ACTIVE and account.status == AccountStatus.ACTIVE:
            account.status = AccountStatus.FROZEN

    notif.notify(
        db,
        user,
        notif_type=NotificationType.ACCOUNT_UPDATE,
        title=f"Account {payload.status}",
        body=payload.reason or f"Your account status changed to {payload.status}.",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="admin.user_status",
        actor=admin,
        request=request,
        entity_type="user",
        entity_id=user.id,
        target_user_id=user.id,
        summary=f"{before} -> {payload.status}: {payload.reason or 'no reason given'}",
        before_state={"status": before},
        after_state={"status": payload.status},
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}/kyc", response_model=UserResponse, summary="Verify or reject KYC")
def decide_kyc(
    user_id: int,
    payload: KycDecisionRequest,
    request: Request,
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    before = user.kyc_status
    now = datetime.now(UTC)
    if payload.decision == "verify":
        user.kyc_status = KycStatus.VERIFIED
        user.kyc_verified_at = now
        user.kyc_rejection_reason = None
        if user.status == UserStatus.PENDING:
            user.status = UserStatus.ACTIVE
        has_account = db.execute(select(Account.id).where(Account.user_id == user.id).limit(1)).first()
        if not has_account:
            banking.create_account(db, user, nickname="Primary Savings")
    else:
        user.kyc_status = KycStatus.REJECTED
        user.kyc_rejection_reason = payload.reason

    notif.notify(
        db,
        user,
        notif_type=NotificationType.ACCOUNT_UPDATE,
        title=f"KYC {user.kyc_status}",
        body=payload.reason or f"Your KYC verification is now {user.kyc_status}.",
        action_url="/app/settings",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="admin.kyc_decision",
        actor=admin,
        request=request,
        entity_type="user",
        entity_id=user.id,
        target_user_id=user.id,
        summary=f"KYC {before} -> {user.kyc_status}",
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


# --------------------------------------------------------------------------- #
# Fraud review queue — the retraining feedback loop
# --------------------------------------------------------------------------- #


@router.get(
    "/fraud/queue",
    response_model=Page[FraudAlertResponse],
    summary="ML-flagged transactions awaiting review",
)
def fraud_queue(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    alert_status: Annotated[str | None, Query(alias="status")] = None,
    severity: Annotated[str | None, Query()] = None,
    min_score: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> Page[FraudAlertResponse]:
    stmt = select(FraudAlert).options(selectinload(FraudAlert.transaction))
    if alert_status:
        stmt = stmt.where(FraudAlert.status == alert_status)
    else:
        # Default view is the actionable queue, not the full history.
        stmt = stmt.where(
            FraudAlert.status.in_([AlertStatus.OPEN.value, AlertStatus.CONFIRMED_FRAUD.value, AlertStatus.DISPUTED.value])
        )
    if severity:
        stmt = stmt.where(FraudAlert.severity == severity)
    if min_score is not None:
        stmt = stmt.where(FraudAlert.risk_score >= min_score)

    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(FraudAlert.risk_score.desc(), FraudAlert.created_at.desc())
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
    "/fraud/queue/{alert_id}/review",
    response_model=FraudAlertResponse,
    summary="Resolve an alert and write the training label",
)
def review_alert(
    alert_id: int,
    payload: FraudReviewRequest,
    request: Request,
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
) -> FraudAlertResponse:
    """Record the analyst verdict.

    This is the only path that writes ``is_fraud_label``. That label is the
    ground truth for the next retraining run, which is why it requires an admin
    and is captured with the reviewer's identity and timestamp.
    """
    alert = db.get(FraudAlert, alert_id)
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    txn = db.get(Transaction, alert.transaction_id)
    now = datetime.now(UTC)

    alert.reviewed_by_id = admin.id
    alert.reviewed_at = now
    alert.review_note = payload.note
    alert.decision_source = "manual"

    if payload.decision == "fraud":
        alert.status = AlertStatus.RESOLVED_FRAUD
        alert.final_label = True
        if txn is not None:
            txn.is_fraud_label = True
            txn.labelled_at = now
            txn.labelled_by_id = admin.id
            if txn.status == TransactionStatus.HELD:
                banking.release_held_transfer(db, txn, approve=False)
            elif payload.reverse_transaction and txn.status == TransactionStatus.COMPLETED:
                # Post a compensating credit rather than mutating the original row:
                # a ledger must stay append-only to remain auditable.
                account = db.get(Account, txn.account_id)
                owner = db.get(User, txn.user_id)
                if account is not None and owner is not None:
                    banking.record_deposit(
                        db,
                        owner,
                        account,
                        amount=txn.amount,
                        description=f"Fraud reversal for {txn.reference}",
                    )
                    txn.status = TransactionStatus.REVERSED
    elif payload.decision == "legitimate":
        alert.status = AlertStatus.RESOLVED_LEGIT
        alert.final_label = False
        if txn is not None:
            txn.is_fraud_label = False
            txn.labelled_at = now
            txn.labelled_by_id = admin.id
            txn.is_flagged = False
            if txn.status == TransactionStatus.HELD:
                banking.release_held_transfer(db, txn, approve=True)
    else:
        # Dismissed without a verdict: deliberately leaves the label NULL so an
        # uncertain case never becomes noisy training data.
        alert.status = AlertStatus.DISMISSED
        alert.final_label = None

    owner = db.get(User, alert.user_id)
    if owner is not None:
        outcome = {
            "fraud": "confirmed as fraud and reversed",
            "legitimate": "cleared and released",
            "dismiss": "closed without action",
        }[payload.decision]
        notif.notify(
            db,
            owner,
            notif_type=NotificationType.FRAUD_ALERT,
            title=f"Security review complete: {alert.alert_ref}",
            body=f"Your flagged transaction was {outcome}.",
            action_url="/app/fraud-center",
            respect_preferences=False,
        )

    write_audit(
        db,
        action="admin.fraud_review",
        actor=admin,
        request=request,
        entity_type="fraud_alert",
        entity_id=alert.id,
        target_user_id=alert.user_id,
        summary=f"Alert {alert.alert_ref} reviewed as {payload.decision}",
        after_state={"final_label": alert.final_label, "status": alert.status},
    )
    db.commit()
    db.refresh(alert)
    return FraudAlertResponse.model_validate(alert)


# --------------------------------------------------------------------------- #
# Loan approval queue
# --------------------------------------------------------------------------- #


@router.get("/loans/queue", response_model=Page[LoanResponse], summary="Credit-scored applications")
def loan_queue(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    loan_status: Annotated[str | None, Query(alias="status")] = None,
) -> Page[LoanResponse]:
    stmt = select(Loan)
    if loan_status:
        stmt = stmt.where(Loan.status == loan_status)
    else:
        stmt = stmt.where(
            Loan.status.in_([LoanStatus.SUBMITTED.value, LoanStatus.UNDER_REVIEW.value])
        )

    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(Loan.created_at.desc()).offset(pagination.offset).limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[LoanResponse].model_validate(
        pagination.envelope([LoanResponse.model_validate(r) for r in rows], total)
    )


@router.post(
    "/loans/queue/{loan_id}/decide",
    response_model=LoanResponse,
    summary="Approve or reject, with optional manual override of model pricing",
)
def decide_loan(
    loan_id: int,
    payload: LoanDecisionRequest,
    request: Request,
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
) -> LoanResponse:
    loan = db.get(Loan, loan_id)
    if loan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Loan not found")
    if loan.status in (LoanStatus.DISBURSED, LoanStatus.CLOSED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This application is already settled"
        )

    before = loan.status
    now = datetime.now(UTC)
    loan.decided_at = now
    loan.decided_by_id = admin.id
    loan.decision_source = "manual"
    loan.manual_override = payload.override_model
    loan.override_note = payload.note

    if payload.decision == "approve":
        loan.status = LoanStatus.APPROVED
        if payload.approved_amount is not None:
            if payload.approved_amount > loan.requested_amount:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Approved amount cannot exceed the requested amount",
                )
            loan.approved_amount = quantize(payload.approved_amount)
        if payload.interest_rate is not None:
            loan.interest_rate = payload.interest_rate

        # Recompute the schedule so a manual override cannot leave stale pricing.
        from app.ml.features import emi_for

        principal = float(loan.approved_amount or 0)
        rate = float(loan.interest_rate or 0)
        emi = emi_for(principal, rate, loan.tenure_months)
        loan.emi_amount = quantize(emi)
        loan.total_payable = quantize(emi * loan.tenure_months)
        loan.processing_fee = quantize(principal * 0.01)
    else:
        loan.status = LoanStatus.REJECTED
        loan.approved_amount = None
        loan.emi_amount = None
        loan.total_payable = None

    loan.decision_reason = payload.note or f"Manually {payload.decision}d by administrator"

    owner = db.get(User, loan.user_id)
    if owner is not None:
        notif.notify(
            db,
            owner,
            notif_type=NotificationType.LOAN_UPDATE,
            title=f"Loan {loan.application_ref} {loan.status}",
            body=(
                f"Approved for {loan.approved_amount:,.2f} at {loan.interest_rate}%."
                if payload.decision == "approve"
                else (payload.note or "Your application was not approved.")
            ),
            action_url="/app/loans",
            respect_preferences=False,
        )

    write_audit(
        db,
        action="admin.loan_decision",
        actor=admin,
        request=request,
        entity_type="loan",
        entity_id=loan.id,
        target_user_id=loan.user_id,
        summary=f"{before} -> {loan.status}" + (" (model overridden)" if payload.override_model else ""),
    )
    db.commit()
    db.refresh(loan)
    return LoanResponse.model_validate(loan)


# --------------------------------------------------------------------------- #
# Loan book (active/disbursed portfolio)
#
# Distinct from the approval queue above: that one decides applications, this one
# tracks repayment on loans already disbursed. Kept separate so neither view has
# to carry filters that only make sense for the other.
# --------------------------------------------------------------------------- #

# The repayment position is derived, not stored: the schema keeps a schedule
# origin (`first_emi_date`) and a counter (`emis_paid`) rather than a row per
# instalment. Month arithmetic is not portable across SQLite and Postgres, so the
# derivation happens in Python and the scan is bounded instead.
_LOAN_BOOK_SCAN_CAP = 5000


def _repayment_position(loan: Loan, today: date) -> tuple[date | None, int]:
    """Return ``(next_due_date, days_overdue)`` for a disbursed loan.

    Single definition on purpose: the reminder endpoint must judge "overdue" by
    exactly the same rule the list endpoint displayed, otherwise an admin could
    click a button the server then rejects.

    A loan with no ``first_emi_date`` has no schedule yet, so it cannot be
    overdue -- returns ``(None, 0)`` rather than guessing an origin date.
    """
    if loan.first_emi_date is None:
        return None, 0

    origin = loan.first_emi_date
    if isinstance(origin, datetime):  # tolerate a datetime if the driver returns one
        origin = origin.date()

    next_due = origin + relativedelta(months=max(0, loan.emis_paid))
    overdue = (today - next_due).days
    return next_due, overdue if overdue > 0 else 0


@router.get(
    "/loans/book",
    response_model=Page[LoanBookRow],
    summary="Disbursed loans with EMI position and overdue status",
)
def loan_book(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    overdue_only: Annotated[bool, Query(description="Only loans past their next due date")] = False,
) -> Page[LoanBookRow]:
    """List the active loan portfolio, most overdue first.

    Sorting and filtering both key off a computed field, so the rows are ordered
    in Python after derivation. The SQL side still does the selective work:
    only disbursed loans are fetched, the borrower is eager-loaded to avoid an
    N+1, and ``overdue_only`` pushes down the one condition that *is* portable
    (a loan whose schedule has not started cannot be overdue).
    """
    today = datetime.now(UTC).date()

    stmt = (
        select(Loan)
        .options(selectinload(Loan.user))
        .where(Loan.status == LoanStatus.DISBURSED.value)
    )
    if overdue_only:
        # Necessary-but-not-sufficient pre-filter: next_due_date is always >=
        # first_emi_date, so anything scheduled to start in the future is safe to
        # discard before the exact check below.
        stmt = stmt.where(Loan.first_emi_date.is_not(None), Loan.first_emi_date <= today)

    loans = db.execute(stmt.limit(_LOAN_BOOK_SCAN_CAP)).scalars().all()

    rows: list[LoanBookRow] = []
    for loan in loans:
        next_due, days_overdue = _repayment_position(loan, today)
        if overdue_only and days_overdue <= 0:
            continue
        borrower = loan.user
        rows.append(
            LoanBookRow(
                id=loan.id,
                application_ref=loan.application_ref,
                borrower_id=loan.user_id,
                borrower_name=borrower.full_name if borrower else "Unknown",
                borrower_email=borrower.email if borrower else "",
                loan_type=loan.loan_type,
                approved_amount=loan.approved_amount,
                interest_rate=loan.interest_rate,
                tenure_months=loan.tenure_months,
                emi_amount=loan.emi_amount,
                outstanding_principal=loan.outstanding_principal,
                emis_paid=loan.emis_paid,
                emis_missed=loan.emis_missed,
                first_emi_date=loan.first_emi_date,
                next_due_date=next_due,
                days_overdue=days_overdue,
                disbursed_at=loan.disbursed_at,
            )
        )

    # Most overdue first; `id` breaks ties so pagination is stable across calls
    # rather than depending on whatever order the database returned.
    rows.sort(key=lambda r: (-r.days_overdue, r.id))

    total = len(rows)
    window = rows[pagination.offset : pagination.offset + pagination.page_size]
    return Page[LoanBookRow].model_validate(pagination.envelope(window, total))


@router.post(
    "/loans/{loan_id}/remind",
    response_model=MessageResponse,
    summary="Email an overdue borrower a payment reminder",
)
def remind_borrower(
    loan_id: int,
    request: Request,
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    """Send a payment reminder for one overdue loan.

    The overdue check is re-derived server-side rather than trusted from the
    client, so a stale page cannot trigger a reminder on a loan that was paid in
    the meantime.

    Three things happen together: an email, an in-app notification, and an audit
    entry. The email is *simulated* until a provider is configured, and a
    delivery failure does not fail the request -- the notification and audit
    record are the durable outcome, so losing the email must not roll them back.
    """
    loan = db.get(Loan, loan_id)
    if loan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Loan not found")

    borrower = db.get(User, loan.user_id)
    if borrower is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Borrower not found")

    if loan.status != LoanStatus.DISBURSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reminders apply only to disbursed loans",
        )

    today = datetime.now(UTC).date()
    next_due, days_overdue = _repayment_position(loan, today)
    if days_overdue <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This loan is not currently overdue",
        )

    emi = loan.emi_amount or Decimal("0")
    outstanding = loan.outstanding_principal or Decimal("0")
    due_label = next_due.isoformat() if next_due else "unknown"
    day_word = "day" if days_overdue == 1 else "days"

    subject = f"Payment reminder: loan {loan.application_ref}"
    html = render_basic_email(
        heading="Payment reminder",
        intro=(
            f"Hello {borrower.full_name}, our records show the EMI for your loan "
            f"{loan.application_ref} has not yet been received. It was due on "
            f"{due_label}, {days_overdue} {day_word} ago."
        ),
        rows=[
            ("Loan reference", loan.application_ref),
            ("EMI amount", f"{emi:,.2f}"),
            ("Due date", due_label),
            ("Days overdue", f"{days_overdue}"),
            ("Outstanding principal", f"{outstanding:,.2f}"),
        ],
        outro=(
            "Please arrange the payment at your earliest convenience. If you have "
            "already paid, or if you would like to discuss your repayment schedule, "
            "reply to this message and our team will follow up."
        ),
    )
    email_sent = send_email(borrower.email, subject, html)

    notif.notify(
        db,
        borrower,
        notif_type=NotificationType.LOAN_UPDATE,
        severity=AlertSeverity.HIGH if days_overdue >= 30 else AlertSeverity.MEDIUM,
        title=f"Payment overdue: {loan.application_ref}",
        body=(
            f"Your EMI of {emi:,.2f} was due on {due_label} and is {days_overdue} "
            f"{day_word} overdue. Outstanding principal is {outstanding:,.2f}."
        ),
        action_url="/app/loans",
        meta={
            "loan_ref": loan.application_ref,
            "days_overdue": days_overdue,
            "emi_amount": str(emi),
            "next_due_date": due_label,
        },
        # A repayment reminder is account-critical, not marketing: a borrower
        # should not be able to mute it via notification preferences.
        respect_preferences=False,
    )

    write_audit(
        db,
        action="admin.loan_reminder",
        actor=admin,
        request=request,
        entity_type="loan",
        entity_id=loan.id,
        target_user_id=loan.user_id,
        summary=(
            f"Reminder for {loan.application_ref}: {days_overdue} {day_word} overdue, "
            f"EMI {emi:,.2f}" + ("" if email_sent else " (email delivery failed)")
        ),
    )
    db.commit()

    detail = (
        f"Reminder sent to {borrower.email}."
        if email_sent
        else "Reminder recorded in-app; email delivery is unavailable."
    )
    return MessageResponse(message=detail)


# --------------------------------------------------------------------------- #
# Model performance & drift
# --------------------------------------------------------------------------- #


def _score_histogram(scores: list[float], bins: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        n = sum(1 for s in scores if (lo <= s < hi) or (b == bins - 1 and s == 1.0))
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": n})
    return out


@router.get(
    "/models",
    response_model=list[ModelPerformanceResponse],
    summary="Training baselines vs live behaviour, with PSI drift",
)
def model_performance(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> list[ModelPerformanceResponse]:
    since = datetime.now(UTC) - timedelta(days=days)
    status_map = models_status()
    out: list[ModelPerformanceResponse] = []

    # ---- fraud: live scores come off scored transactions ----
    fraud_scores = [
        float(s)
        for s in db.execute(
            select(Transaction.fraud_score).where(
                Transaction.fraud_score.is_not(None), Transaction.occurred_at >= since
            )
        ).scalars().all()
    ]
    fraud_latencies = [
        float(v)
        for v in db.execute(
            select(Transaction.scoring_latency_ms).where(
                Transaction.scoring_latency_ms.is_not(None), Transaction.occurred_at >= since
            )
        ).scalars().all()
    ]
    labelled = db.execute(
        select(Transaction.fraud_score, Transaction.is_fraud_label).where(
            Transaction.is_fraud_label.is_not(None),
            Transaction.fraud_score.is_not(None),
            Transaction.occurred_at >= since,
        )
    ).all()

    fraud_meta = read_metrics(FRAUD_MODEL) or {}
    threshold = float(fraud_meta.get("threshold") or 0.5)
    tp = sum(1 for s, y in labelled if float(s) >= threshold and y)
    fp = sum(1 for s, y in labelled if float(s) >= threshold and not y)
    fn = sum(1 for s, y in labelled if float(s) < threshold and y)

    psi_value = None
    baseline_edges = fraud_meta.get("psi_baseline_bins") or []
    # PSI needs a reasonable sample; on a handful of scores it is pure noise.
    if len(fraud_scores) >= 50 and baseline_edges:
        psi_value = psi_against_baseline(
            fraud_scores,
            baseline_edges,
            expected_pct=fraud_meta.get("psi_baseline_expected_pct"),
        )

    out.append(
        ModelPerformanceResponse(
            model_name=FRAUD_MODEL,
            model_version=status_map["fraud"]["version"],
            loaded=status_map["fraud"]["loaded"],
            training_metrics=status_map["fraud"]["metrics"],
            training_latency=status_map["fraud"]["latency_benchmark"],
            live_inference_count=len(fraud_scores),
            live_flagged_count=sum(1 for s in fraud_scores if s >= threshold),
            live_labelled_count=len(labelled),
            live_mean_score=round(sum(fraud_scores) / len(fraud_scores), 6) if fraud_scores else None,
            live_p95_score=(
                round(sorted(fraud_scores)[int(len(fraud_scores) * 0.95) - 1], 6)
                if len(fraud_scores) > 1
                else None
            ),
            live_mean_latency_ms=(
                round(sum(fraud_latencies) / len(fraud_latencies), 3) if fraud_latencies else None
            ),
            live_p95_latency_ms=(
                round(sorted(fraud_latencies)[int(len(fraud_latencies) * 0.95) - 1], 3)
                if len(fraud_latencies) > 1
                else None
            ),
            realised_precision=round(tp / (tp + fp), 4) if (tp + fp) else None,
            realised_recall=round(tp / (tp + fn), 4) if (tp + fn) else None,
            psi=psi_value,
            drift_status=drift_status(psi_value) if psi_value is not None else None,
            score_histogram=_score_histogram(fraud_scores),
        )
    )

    # ---- credit ----
    credit_rows = db.execute(
        select(CreditScore.probability_of_default, CreditScore.inference_latency_ms).where(
            CreditScore.created_at >= since
        )
    ).all()
    credit_scores = [float(r[0]) for r in credit_rows]
    credit_lat = [float(r[1]) for r in credit_rows if r[1] is not None]
    credit_meta = read_metrics(CREDIT_MODEL) or {}
    credit_psi = None
    if len(credit_scores) >= 50 and credit_meta.get("psi_baseline_bins"):
        credit_psi = psi_against_baseline(
            credit_scores,
            credit_meta["psi_baseline_bins"],
            expected_pct=credit_meta.get("psi_baseline_expected_pct"),
        )

    out.append(
        ModelPerformanceResponse(
            model_name=CREDIT_MODEL,
            model_version=status_map["credit"]["version"],
            loaded=status_map["credit"]["loaded"],
            training_metrics=status_map["credit"]["metrics"],
            training_latency=status_map["credit"]["latency_benchmark"],
            live_inference_count=len(credit_scores),
            live_flagged_count=sum(1 for s in credit_scores if s >= 0.25),
            live_labelled_count=0,
            live_mean_score=round(sum(credit_scores) / len(credit_scores), 6) if credit_scores else None,
            live_p95_score=(
                round(sorted(credit_scores)[int(len(credit_scores) * 0.95) - 1], 6)
                if len(credit_scores) > 1
                else None
            ),
            live_mean_latency_ms=round(sum(credit_lat) / len(credit_lat), 3) if credit_lat else None,
            live_p95_latency_ms=(
                round(sorted(credit_lat)[int(len(credit_lat) * 0.95) - 1], 3) if len(credit_lat) > 1 else None
            ),
            psi=credit_psi,
            drift_status=drift_status(credit_psi) if credit_psi is not None else None,
            score_histogram=_score_histogram(credit_scores),
        )
    )

    # ---- anomaly ----
    anomaly_rows = db.execute(
        select(AnomalyAlert.anomaly_score, AnomalyAlert.inference_latency_ms).where(
            AnomalyAlert.created_at >= since
        )
    ).all()
    a_scores = [float(r[0]) for r in anomaly_rows]
    a_lat = [float(r[1]) for r in anomaly_rows if r[1] is not None]

    out.append(
        ModelPerformanceResponse(
            model_name=ANOMALY_MODEL,
            model_version=status_map["anomaly"]["version"],
            loaded=status_map["anomaly"]["loaded"],
            training_metrics=status_map["anomaly"]["metrics"],
            training_latency=status_map["anomaly"]["latency_benchmark"],
            live_inference_count=len(a_scores),
            live_flagged_count=len(a_scores),
            live_labelled_count=int(
                db.execute(
                    select(func.count(AnomalyAlert.id)).where(AnomalyAlert.acknowledged.is_(True))
                ).scalar_one()
                or 0
            ),
            live_mean_score=round(sum(a_scores) / len(a_scores), 6) if a_scores else None,
            live_p95_score=(
                round(sorted(a_scores)[int(len(a_scores) * 0.95) - 1], 6) if len(a_scores) > 1 else None
            ),
            live_mean_latency_ms=round(sum(a_lat) / len(a_lat), 3) if a_lat else None,
            live_p95_latency_ms=(
                round(sorted(a_lat)[int(len(a_lat) * 0.95) - 1], 3) if len(a_lat) > 1 else None
            ),
            score_histogram=_score_histogram(a_scores),
        )
    )
    return out


@router.post("/models/reload", response_model=MessageResponse, summary="Hot-reload artifacts")
def reload_model_artifacts(
    request: Request, admin: AdminUser, db: Annotated[Session, Depends(get_db)]
) -> MessageResponse:
    """Pick up retrained artifacts without restarting the API."""
    status_map = reload_models()
    write_audit(
        db,
        action="admin.models_reload",
        actor=admin,
        request=request,
        entity_type="model",
        summary=f"Reloaded: {status_map}",
    )
    db.commit()
    loaded = sum(1 for v in status_map.values() if v)
    return MessageResponse(
        message=f"Reloaded {loaded}/{len(status_map)} model artifacts",
        detail=", ".join(f"{k}={'ok' if v else 'missing'}" for k, v in status_map.items()),
    )


# --------------------------------------------------------------------------- #
# Analytics & audit
# --------------------------------------------------------------------------- #


@router.get("/analytics", summary="System-wide charts")
def analytics(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    days: Annotated[int, Query(ge=7, le=365)] = 30,
) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(days=days)

    rows = db.execute(
        select(Transaction.occurred_at, Transaction.amount, Transaction.is_flagged).where(
            Transaction.occurred_at >= since
        )
    ).all()

    daily: dict[str, dict[str, float]] = {}
    for occurred_at, amount, flagged in rows:
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        key = occurred_at.date().isoformat()
        entry = daily.setdefault(key, {"volume": 0.0, "count": 0, "flagged": 0})
        entry["volume"] += float(amount or 0)
        entry["count"] += 1
        if flagged:
            entry["flagged"] += 1

    category_rows = db.execute(
        select(
            Transaction.merchant_category,
            func.coalesce(func.sum(Transaction.amount), 0),
            func.count(Transaction.id),
        )
        .where(Transaction.occurred_at >= since)
        .group_by(Transaction.merchant_category)
        .order_by(func.sum(Transaction.amount).desc())
    ).all()

    band_rows = db.execute(
        select(CreditScore.risk_band, func.count(CreditScore.id))
        .where(CreditScore.created_at >= since)
        .group_by(CreditScore.risk_band)
        .order_by(CreditScore.risk_band)
    ).all()

    severity_rows = db.execute(
        select(FraudAlert.severity, func.count(FraudAlert.id))
        .where(FraudAlert.created_at >= since)
        .group_by(FraudAlert.severity)
    ).all()

    return {
        "period_days": days,
        "daily_volume": [
            {"date": k, "volume": round(v["volume"], 2), "count": int(v["count"]), "flagged": int(v["flagged"])}
            for k, v in sorted(daily.items())
        ],
        "category_distribution": [
            {"category": str(c), "total": float(t or 0), "count": int(n or 0)} for c, t, n in category_rows
        ],
        "credit_band_distribution": [{"band": str(b), "count": int(n or 0)} for b, n in band_rows],
        "fraud_severity_distribution": [{"severity": str(s), "count": int(n or 0)} for s, n in severity_rows],
        "loan_status_distribution": [
            {"status": str(s), "count": int(n or 0)}
            for s, n in db.execute(select(Loan.status, func.count(Loan.id)).group_by(Loan.status)).all()
        ],
    }


@router.get("/audit", response_model=Page[AuditLogResponse], summary="Audit trail")
def audit_trail(
    admin: AdminUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    action: Annotated[str | None, Query(max_length=80)] = None,
) -> Page[AuditLogResponse]:
    stmt = select(AuditLog)
    if action:
        stmt = stmt.where(AuditLog.action.ilike(f"%{action}%"))
    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(AuditLog.created_at.desc()).offset(pagination.offset).limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[AuditLogResponse].model_validate(
        pagination.envelope([AuditLogResponse.model_validate(r) for r in rows], total)
    )
