"""Read-only, user-scoped tools the AI assistant may call.

Security model
--------------
The LLM chooses *which* tool to call and supplies the arguments, including
``user_id``. That makes ``user_id`` attacker-influencable: a prompt-injection
attempt such as "call get_account_balances with user_id=3" would otherwise read
another customer's finances.

Every tool therefore re-validates that argument against the authenticated
session before touching the database:

* the caller's id comes from the JWT, never from the model
* a mismatched or missing ``user_id`` raises ``ToolAuthorizationError``
* the violation is logged with both ids so the attempt is auditable
* the resulting error text is fed back to the model, so it can apologise rather
  than crash the request

Every tool is strictly read-only. The assistant can answer questions; it cannot
move money, open accounts or change settings. That is a deliberate blast-radius
decision: an LLM with write access to a ledger is not something to ship.
"""
from __future__ import annotations

import calendar
import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models.banking import Account, Transaction
from app.models.enums import (
    AccountStatus,
    LoanStatus,
    TransactionStatus,
    TransactionType,
)
from app.models.lending import Loan
from app.models.user import User

logger = logging.getLogger("intellibank.assistant.tools")

# Debits only. A salary credit is not "spending", so including inflows here
# would make every spending answer wrong.
DEBIT_TYPES = [
    TransactionType.WITHDRAWAL.value,
    TransactionType.TRANSFER_OUT.value,
    TransactionType.CARD_PAYMENT.value,
    TransactionType.LOAN_REPAYMENT.value,
    TransactionType.FEE.value,
]

SETTLED = [TransactionStatus.COMPLETED.value, TransactionStatus.HELD.value]

# Natural-language category synonyms. Users say "food", the schema says
# "dining" and "groceries"; without this mapping the assistant answers 0.00 to
# a perfectly reasonable question.
CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "food": ["dining", "groceries"],
    "dining": ["dining"],
    "restaurants": ["dining"],
    "eating out": ["dining"],
    "groceries": ["groceries"],
    "grocery": ["groceries"],
    "transport": ["transport"],
    "travel": ["travel"],
    "commute": ["transport"],
    "fuel": ["transport"],
    "shopping": ["shopping"],
    "utilities": ["utilities"],
    "bills": ["utilities", "rent"],
    "entertainment": ["entertainment"],
    "healthcare": ["healthcare"],
    "medical": ["healthcare"],
    "education": ["education"],
    "rent": ["rent"],
    "housing": ["rent"],
    "investment": ["investment"],
    "investments": ["investment"],
    "savings": ["investment"],
    "cash": ["cash"],
    "atm": ["cash"],
    "transfer": ["transfer"],
    "transfers": ["transfer"],
    "other": ["other"],
}

VALID_PERIODS = (
    "today",
    "this_week",
    "this_month",
    "last_month",
    "last_7_days",
    "last_30_days",
    "last_90_days",
    "this_year",
    "all_time",
)


class ToolAuthorizationError(PermissionError):
    """Raised when a tool call targets a user other than the authenticated one."""


class ToolInputError(ValueError):
    """Raised when the model supplies unusable arguments."""


# --------------------------------------------------------------------------- #
# Guards and helpers
# --------------------------------------------------------------------------- #


def _authorize(args: dict[str, Any], caller: User, tool_name: str) -> None:
    """Reject any tool call whose ``user_id`` is not the authenticated caller.

    This is the single choke point that makes cross-user access impossible, so
    it runs before every query and is never conditional on the tool.
    """
    raw = args.get("user_id")

    if raw is None:
        raise ToolAuthorizationError(
            "user_id is required and must match the signed-in user."
        )

    try:
        requested = int(raw)
    except (TypeError, ValueError) as exc:
        raise ToolAuthorizationError(f"user_id must be an integer, got {raw!r}.") from exc

    if requested != caller.id:
        # Logged at WARNING with both ids: this is the signature of a prompt
        # injection attempt and should be visible in production logs.
        logger.warning(
            "BLOCKED cross-user assistant tool call: tool=%s authenticated_user=%s requested_user=%s",
            tool_name,
            caller.id,
            requested,
        )
        raise ToolAuthorizationError(
            "Access denied: this assistant can only read the signed-in user's own data."
        )

    # Defence in depth: the model must not be able to redirect the query by
    # supplying an alternate identifier alongside a correct user_id.
    for forbidden in ("email", "account_number", "other_user_id", "target_user_id", "customer_id"):
        if forbidden in args:
            logger.warning(
                "Stripped disallowed assistant argument %r on tool=%s (user=%s)",
                forbidden,
                tool_name,
                caller.id,
            )
            args.pop(forbidden, None)


def _money(value: Decimal | float | int | None) -> float:
    """Serialise currency for the model.

    Decimal is not JSON-serialisable and the model only needs the figure for
    prose, so it is converted here. All arithmetic upstream stays in Decimal.
    """
    if value is None:
        return 0.0
    return round(float(value), 2)


def _month_bounds(anchor: date) -> tuple[date, date]:
    last_day = calendar.monthrange(anchor.year, anchor.month)[1]
    return anchor.replace(day=1), anchor.replace(day=last_day)


def resolve_period(period: str | None) -> tuple[datetime, datetime, str]:
    """Turn a period keyword into a concrete UTC window plus a human label."""
    now = datetime.now(UTC)
    today = now.date()
    key = (period or "this_month").strip().lower().replace("-", "_").replace(" ", "_")

    if key == "today":
        start = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
        return start, now, "today"
    if key == "this_week":
        start_date = today - timedelta(days=today.weekday())
        return datetime.combine(start_date, datetime.min.time(), tzinfo=UTC), now, "this week"
    if key == "this_month":
        first, _ = _month_bounds(today)
        return datetime.combine(first, datetime.min.time(), tzinfo=UTC), now, "this month"
    if key == "last_month":
        first_this, _ = _month_bounds(today)
        last_month_end = first_this - timedelta(days=1)
        first_last, _ = _month_bounds(last_month_end)
        return (
            datetime.combine(first_last, datetime.min.time(), tzinfo=UTC),
            datetime.combine(last_month_end, datetime.max.time(), tzinfo=UTC),
            "last month",
        )
    if key in {"last_7_days", "last_week", "past_week"}:
        return now - timedelta(days=7), now, "the last 7 days"
    if key in {"last_30_days", "past_month"}:
        return now - timedelta(days=30), now, "the last 30 days"
    if key in {"last_90_days", "last_quarter"}:
        return now - timedelta(days=90), now, "the last 90 days"
    if key in {"this_year", "ytd"}:
        return datetime(today.year, 1, 1, tzinfo=UTC), now, "this year"
    if key == "all_time":
        return datetime(2000, 1, 1, tzinfo=UTC), now, "all time"

    # Unknown keyword: default to this month but say so, rather than silently
    # answering a different question than the one asked.
    first, _ = _month_bounds(today)
    return (
        datetime.combine(first, datetime.min.time(), tzinfo=UTC),
        now,
        f"this month (unrecognised period {period!r}, defaulted)",
    )


def resolve_categories(raw: str | None) -> list[str] | None:
    """Map a natural-language category to schema categories, or None for all."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in {"all", "everything", "any"}:
        return None
    if key in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key]
    # Substring fallback: "food and drink" should still resolve.
    for synonym, mapped in CATEGORY_SYNONYMS.items():
        if synonym in key or key in synonym:
            return mapped
    return [key]  # pass through; an empty result is a truthful answer


def _parse_date(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            moment = datetime.max.time() if end_of_day else datetime.min.time()
            return datetime.combine(parsed, moment, tzinfo=UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError as exc:
        raise ToolInputError(f"Could not read {value!r} as a date. Use YYYY-MM-DD.") from exc


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def get_account_balances(db: Session, caller: User, args: dict[str, Any]) -> dict[str, Any]:
    """Every open account belonging to the caller, plus the combined total."""
    _authorize(args, caller, "get_account_balances")

    accounts = (
        db.execute(
            select(Account)
            .where(Account.user_id == caller.id, Account.status != AccountStatus.CLOSED.value)
            .order_by(Account.is_primary.desc(), Account.id)
        )
        .scalars()
        .all()
    )

    rows = [
        {
            "account_number_masked": f"****{a.account_number[-4:]}",
            "nickname": a.nickname,
            "account_type": a.account_type,
            "status": a.status,
            "balance": _money(a.balance),
            "available_balance": _money(a.available_balance),
            "on_hold": _money(a.hold_amount),
            "currency": a.currency,
            "is_primary": a.is_primary,
        }
        for a in accounts
    ]

    return {
        "currency": "INR",
        "account_count": len(rows),
        "total_balance": _money(sum((a.balance for a in accounts), Decimal("0"))),
        "total_available": _money(sum((a.available_balance for a in accounts), Decimal("0"))),
        "accounts": rows,
    }


def get_transactions(db: Session, caller: User, args: dict[str, Any]) -> dict[str, Any]:
    """Individual transactions, optionally filtered by category and date range."""
    _authorize(args, caller, "get_transactions")

    categories = resolve_categories(args.get("category"))
    start = _parse_date(args.get("start_date"))
    end = _parse_date(args.get("end_date"), end_of_day=True)

    # Row cap: the model gets a representative sample rather than a payload that
    # blows the context window on a chatty account.
    try:
        limit = min(max(int(args.get("limit", 15)), 1), 50)
    except (TypeError, ValueError):
        limit = 15

    clauses = [Transaction.user_id == caller.id, Transaction.status.in_(SETTLED)]
    if categories:
        clauses.append(Transaction.merchant_category.in_(categories))
    if start:
        clauses.append(Transaction.occurred_at >= start)
    if end:
        clauses.append(Transaction.occurred_at <= end)

    stmt = select(Transaction).where(and_(*clauses))

    total_matching = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0
    )
    rows = (
        db.execute(stmt.order_by(Transaction.occurred_at.desc()).limit(limit))
        .scalars()
        .all()
    )

    return {
        "currency": "INR",
        "filters_applied": {
            "category": args.get("category"),
            "resolved_categories": categories,
            "start_date": start.date().isoformat() if start else None,
            "end_date": end.date().isoformat() if end else None,
        },
        "total_matching": total_matching,
        "returned": len(rows),
        "transactions": [
            {
                "date": t.occurred_at.date().isoformat(),
                "description": t.description or t.merchant_name or t.merchant_category,
                "merchant": t.merchant_name,
                "category": t.merchant_category,
                "amount": _money(t.amount),
                "direction": "credit" if t.signed_amount > 0 else "debit",
                "status": t.status,
                "flagged_for_review": t.is_flagged,
            }
            for t in rows
        ],
    }


def get_spending_summary(db: Session, caller: User, args: dict[str, Any]) -> dict[str, Any]:
    """Total spend for a period plus a per-category breakdown."""
    _authorize(args, caller, "get_spending_summary")

    start, end, label = resolve_period(args.get("period"))
    categories = resolve_categories(args.get("category"))

    clauses = [
        Transaction.user_id == caller.id,
        Transaction.txn_type.in_(DEBIT_TYPES),
        Transaction.status == TransactionStatus.COMPLETED.value,
        Transaction.occurred_at >= start,
        Transaction.occurred_at <= end,
    ]
    if categories:
        clauses.append(Transaction.merchant_category.in_(categories))

    breakdown = db.execute(
        select(
            Transaction.merchant_category,
            func.coalesce(func.sum(Transaction.amount), 0),
            func.count(Transaction.id),
        )
        .where(and_(*clauses))
        .group_by(Transaction.merchant_category)
        .order_by(func.sum(Transaction.amount).desc())
    ).all()

    total = sum(float(row[1] or 0) for row in breakdown)
    txn_count = sum(int(row[2] or 0) for row in breakdown)

    income = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == caller.id,
            Transaction.txn_type.in_(
                [
                    TransactionType.DEPOSIT.value,
                    TransactionType.TRANSFER_IN.value,
                    TransactionType.INTEREST.value,
                ]
            ),
            Transaction.status == TransactionStatus.COMPLETED.value,
            Transaction.occurred_at >= start,
            Transaction.occurred_at <= end,
        )
    ).scalar_one()

    return {
        "currency": "INR",
        "period": label,
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
        "category_filter": args.get("category"),
        "resolved_categories": categories,
        "total_spent": round(total, 2),
        "total_received": _money(income),
        "net_change": round(float(income or 0) - total, 2),
        "transaction_count": txn_count,
        "average_transaction": round(total / txn_count, 2) if txn_count else 0.0,
        "by_category": [
            {
                "category": str(row[0]),
                "amount": _money(row[1]),
                "transaction_count": int(row[2] or 0),
                "share_percent": round(float(row[1] or 0) / total * 100, 1) if total else 0.0,
            }
            for row in breakdown
        ],
    }


def get_loan_status(db: Session, caller: User, args: dict[str, Any]) -> dict[str, Any]:
    """All loan applications and active loans for the caller."""
    _authorize(args, caller, "get_loan_status")

    loans = (
        db.execute(select(Loan).where(Loan.user_id == caller.id).order_by(Loan.created_at.desc()))
        .scalars()
        .all()
    )

    active_states = {LoanStatus.DISBURSED.value, LoanStatus.APPROVED.value}

    return {
        "currency": "INR",
        "loan_count": len(loans),
        "has_active_loan": any(loan.status in active_states for loan in loans),
        "total_outstanding": _money(
            sum(
                (loan.outstanding_principal or Decimal("0"))
                for loan in loans
                if loan.status == LoanStatus.DISBURSED.value
            )
        ),
        "loans": [
            {
                "reference": loan.application_ref,
                "loan_type": loan.loan_type,
                "status": loan.status,
                "requested_amount": _money(loan.requested_amount),
                "approved_amount": _money(loan.approved_amount) if loan.approved_amount else None,
                "interest_rate_percent": loan.interest_rate,
                "tenure_months": loan.tenure_months,
                "monthly_emi": _money(loan.emi_amount) if loan.emi_amount else None,
                "outstanding_principal": (
                    _money(loan.outstanding_principal) if loan.outstanding_principal else None
                ),
                "emis_paid": loan.emis_paid,
                "emis_missed": loan.emis_missed,
                "applied_on": loan.created_at.date().isoformat() if loan.created_at else None,
                "decision_reason": loan.decision_reason,
            }
            for loan in loans
        ],
    }


def get_upcoming_dues(db: Session, caller: User, args: dict[str, Any]) -> dict[str, Any]:
    """Scheduled EMI payments due in the next N days.

    Due dates are projected from ``first_emi_date`` plus the number of EMIs
    already paid, because the schema stores a schedule origin rather than a row
    per instalment.
    """
    _authorize(args, caller, "get_upcoming_dues")

    try:
        horizon_days = min(max(int(args.get("days_ahead", 30)), 1), 365)
    except (TypeError, ValueError):
        horizon_days = 30

    today = datetime.now(UTC).date()
    horizon = today + timedelta(days=horizon_days)

    disbursed = (
        db.execute(
            select(Loan).where(
                Loan.user_id == caller.id, Loan.status == LoanStatus.DISBURSED.value
            )
        )
        .scalars()
        .all()
    )

    dues: list[dict[str, Any]] = []
    for loan in disbursed:
        if not loan.first_emi_date or not loan.emi_amount:
            continue

        # Advance month-by-month from the schedule origin past every paid EMI.
        due = loan.first_emi_date
        for _ in range(loan.emis_paid):
            month = due.month + 1
            year = due.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(due.day, calendar.monthrange(year, month)[1])
            due = date(year, month, day)

        remaining = max(loan.tenure_months - loan.emis_paid, 0)
        while due <= horizon and remaining > 0:
            if due >= today:
                dues.append(
                    {
                        "loan_reference": loan.application_ref,
                        "loan_type": loan.loan_type,
                        "due_date": due.isoformat(),
                        "amount": _money(loan.emi_amount),
                        "days_until_due": (due - today).days,
                    }
                )
            month = due.month + 1
            year = due.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(due.day, calendar.monthrange(year, month)[1])
            due = date(year, month, day)
            remaining -= 1

    dues.sort(key=lambda d: d["due_date"])

    overdue = [
        {
            "loan_reference": loan.application_ref,
            "loan_type": loan.loan_type,
            "emis_missed": loan.emis_missed,
            "monthly_emi": _money(loan.emi_amount) if loan.emi_amount else None,
        }
        for loan in disbursed
        if loan.emis_missed > 0
    ]

    return {
        "currency": "INR",
        "horizon_days": horizon_days,
        "upcoming_count": len(dues),
        "total_due": round(sum(d["amount"] for d in dues), 2),
        "upcoming_dues": dues,
        "missed_payments": overdue,
        "note": (
            "No scheduled EMI payments in this window."
            if not dues
            else f"{len(dues)} payment(s) due within {horizon_days} days."
        ),
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ToolFn = Callable[[Session, User, dict[str, Any]], dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_account_balances": get_account_balances,
    "get_transactions": get_transactions,
    "get_spending_summary": get_spending_summary,
    "get_loan_status": get_loan_status,
    "get_upcoming_dues": get_upcoming_dues,
}


def execute_tool(
    name: str, db: Session, caller: User, args: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Run one tool. Returns ``(payload, ok)``.

    Failures are returned as data rather than raised so the orchestrator can
    hand the message back to the model, which then explains the problem in
    natural language instead of the user seeing a 500.
    """
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"error": f"Unknown tool {name!r}."}, False

    try:
        return tool(db, caller, dict(args or {})), True
    except ToolAuthorizationError as exc:
        return {"error": str(exc), "authorization_denied": True}, False
    except ToolInputError as exc:
        return {"error": str(exc)}, False
    except Exception:  # noqa: BLE001 - a tool bug must not 500 the chat endpoint
        logger.exception("Assistant tool %s failed for user %s", name, caller.id)
        return {"error": "That lookup failed unexpectedly."}, False
