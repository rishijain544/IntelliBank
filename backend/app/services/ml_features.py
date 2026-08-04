"""Assemble ML feature inputs from live database state.

This is the serving half of the train/serve contract. ``app/ml/datasets.py``
replays a simulated stream through ``_StreamState`` to produce
``UserHistoryStats``; this module produces the *same* dataclass from SQL
aggregates. Both then call the identical builders in ``app/ml/features.py``, so
any divergence is confined to how the aggregates are computed — which is why the
window definitions below mirror the offline tracker field for field.

Confirmed fraud is excluded from every behavioural aggregate: a bank must not let
a stolen-card spree teach the model what a customer's "normal" looks like.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session

from app.ml.features import CreditProfile, TxnContext, UserHistoryStats
from app.models.banking import Account, Card, Transaction
from app.models.enums import (
    AccountStatus,
    CardStatus,
    LoanStatus,
    TransactionStatus,
    TransactionType,
)
from app.models.lending import Loan
from app.models.user import User, UserDevice

# Debits only: an incoming salary should not shape the spending baseline.
DEBIT_TYPES = (
    TransactionType.WITHDRAWAL,
    TransactionType.TRANSFER_OUT,
    TransactionType.CARD_PAYMENT,
    TransactionType.LOAN_REPAYMENT,
    TransactionType.FEE,
)


def _f(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _baseline_filter(user_id: int) -> list:
    """Rows eligible to define a user's normal behaviour."""
    return [
        Transaction.user_id == user_id,
        Transaction.txn_type.in_([t.value for t in DEBIT_TYPES]),
        Transaction.status.in_([TransactionStatus.COMPLETED.value, TransactionStatus.HELD.value]),
        # Exclude admin-confirmed fraud from the behavioural baseline.
        or_(Transaction.is_fraud_label.is_(None), Transaction.is_fraud_label.is_(False)),
    ]


def build_user_history(
    db: Session, user: User, *, now: datetime | None = None, lookback_days: int = 180
) -> UserHistoryStats:
    """Aggregate a user's recent behaviour into the shared history contract."""
    now = now or datetime.now(UTC)
    window_start = now - timedelta(days=lookback_days)
    h1, h24, d7 = now - timedelta(hours=1), now - timedelta(hours=24), now - timedelta(days=7)

    base = _baseline_filter(user.id)
    scoped = [*base, Transaction.occurred_at >= window_start, Transaction.occurred_at <= now]

    # ---- overall amount statistics ----
    agg = db.execute(
        select(
            func.count(Transaction.id),
            func.avg(Transaction.amount),
            func.min(Transaction.occurred_at),
        ).where(and_(*scoped))
    ).one()
    total_count = int(agg[0] or 0)
    mean_amount = _f(agg[1])
    first_seen: datetime | None = agg[2]

    # SQLite lacks STDDEV, so the sample is pulled and reduced in Python. Capped
    # at 500 rows: enough for a stable mean/std, bounded work per scoring call.
    recent_rows = db.execute(
        select(Transaction.amount)
        .where(and_(*scoped))
        .order_by(Transaction.occurred_at.desc())
        .limit(500)
    ).scalars().all()
    recent_amounts = [_f(a) for a in recent_rows]
    std_amount = 0.0
    if len(recent_amounts) > 1:
        m = sum(recent_amounts) / len(recent_amounts)
        std_amount = (sum((x - m) ** 2 for x in recent_amounts) / (len(recent_amounts) - 1)) ** 0.5

    # ---- velocity windows ----
    def _count_since(ts: datetime) -> int:
        return int(
            db.execute(
                select(func.count(Transaction.id)).where(
                    and_(*base, Transaction.occurred_at >= ts, Transaction.occurred_at <= now)
                )
            ).scalar_one()
            or 0
        )

    count_1h = _count_since(h1)
    count_24h = _count_since(h24)
    sum_24h = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                and_(*base, Transaction.occurred_at >= h24, Transaction.occurred_at <= now)
            )
        ).scalar_one()
    )

    prev_at = db.execute(
        select(func.max(Transaction.occurred_at)).where(and_(*base, Transaction.occurred_at <= now))
    ).scalar_one()
    mins_since_prev = 1440.0
    if prev_at:
        if prev_at.tzinfo is None:
            prev_at = prev_at.replace(tzinfo=UTC)
        mins_since_prev = max(0.0, (now - prev_at).total_seconds() / 60.0)

    # ---- known devices / cities / merchants ----
    devices = set(
        db.execute(select(UserDevice.fingerprint).where(UserDevice.user_id == user.id))
        .scalars()
        .all()
    )
    devices.update(
        d
        for d in db.execute(
            select(Transaction.device_fingerprint.distinct()).where(and_(*scoped))
        ).scalars().all()
        if d
    )
    cities = {
        c
        for c in db.execute(
            select(Transaction.location_city.distinct()).where(and_(*scoped))
        ).scalars().all()
        if c
    }
    if user.city:
        cities.add(user.city)
    merchants = {
        m
        for m in db.execute(
            select(Transaction.merchant_name.distinct()).where(and_(*scoped))
        ).scalars().all()
        if m
    }

    # ---- category distribution and per-category baselines ----
    category_counts: dict[str, int] = {}
    category_mean: dict[str, float] = {}
    for cat, cnt, avg in db.execute(
        select(
            Transaction.merchant_category,
            func.count(Transaction.id),
            func.avg(Transaction.amount),
        )
        .where(and_(*scoped))
        .group_by(Transaction.merchant_category)
    ).all():
        category_counts[str(cat)] = int(cnt or 0)
        category_mean[str(cat)] = _f(avg)

    distinct_merchants_7d = int(
        db.execute(
            select(func.count(func.distinct(Transaction.merchant_name))).where(
                and_(*base, Transaction.occurred_at >= d7, Transaction.occurred_at <= now)
            )
        ).scalar_one()
        or 0
    )
    distinct_categories_7d = int(
        db.execute(
            select(func.count(func.distinct(Transaction.merchant_category))).where(
                and_(*base, Transaction.occurred_at >= d7, Transaction.occurred_at <= now)
            )
        ).scalar_one()
        or 0
    )

    # Declines are a risk signal, so they are counted from failed/blocked rows
    # rather than from the (successful-only) baseline set.
    failed_24h = int(
        db.execute(
            select(func.count(Transaction.id)).where(
                Transaction.user_id == user.id,
                Transaction.status.in_(
                    [TransactionStatus.FAILED.value, TransactionStatus.BLOCKED.value]
                ),
                Transaction.occurred_at >= h24,
                Transaction.occurred_at <= now,
            )
        ).scalar_one()
        or 0
    )

    # ---- habitual hour ----
    hour_rows = db.execute(
        select(Transaction.occurred_at)
        .where(and_(*scoped))
        .order_by(Transaction.occurred_at.desc())
        .limit(200)
    ).scalars().all()
    usual_hour = 13.0
    if hour_rows:
        usual_hour = sum(t.hour for t in hour_rows) / len(hour_rows)

    span_days = max((now - first_seen).days, 1) if first_seen else 1
    avg_daily = total_count / span_days if total_count else 1.0

    week_total = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                and_(*base, Transaction.occurred_at >= d7, Transaction.occurred_at <= now)
            )
        ).scalar_one()
    )
    weeks_span = max(span_days / 7.0, 1.0)
    all_time_total = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(and_(*scoped))
        ).scalar_one()
    )
    week_median = all_time_total / weeks_span if all_time_total else 0.0

    return UserHistoryStats(
        txn_count_total=total_count,
        mean_amount=mean_amount,
        std_amount=std_amount,
        recent_amounts=recent_amounts[:200],
        txn_count_1h=count_1h,
        txn_count_24h=count_24h,
        amount_sum_24h=sum_24h,
        mins_since_prev_txn=mins_since_prev,
        avg_daily_txn_count=avg_daily,
        known_devices=frozenset(devices),
        known_cities=frozenset(cities),
        known_merchants=frozenset(merchants),
        home_country=user.country or "IN",
        category_counts=category_counts,
        distinct_merchants_7d=distinct_merchants_7d,
        distinct_categories_7d=distinct_categories_7d,
        failed_txn_24h=failed_24h,
        usual_hour=usual_hour,
        category_mean_amount=category_mean,
        category_std_amount={},
        week_spend_total=week_total,
        week_spend_median=week_median,
    )


def enrich_for_category(
    db: Session, user: User, hist: UserHistoryStats, category: str, *, now: datetime | None = None
) -> UserHistoryStats:
    """Add the category-specific windows the anomaly model needs.

    Split out from ``build_user_history`` because these three aggregates depend on
    the category of the transaction being scored, and running them for every
    category on every call would be wasted work.
    """
    now = now or datetime.now(UTC)
    d7 = now - timedelta(days=7)
    base = _baseline_filter(user.id)
    cat_filter = [*base, Transaction.merchant_category == category]

    hist.category_week_spend = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                and_(*cat_filter, Transaction.occurred_at >= d7, Transaction.occurred_at <= now)
            )
        ).scalar_one()
    )

    total_cat, first_cat = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0), func.min(Transaction.occurred_at)).where(
            and_(*cat_filter, Transaction.occurred_at <= now)
        )
    ).one()
    if first_cat:
        if first_cat.tzinfo is None:
            first_cat = first_cat.replace(tzinfo=UTC)
        weeks = max((now - first_cat).days / 7.0, 1.0)
        hist.category_week_baseline = _f(total_cat) / weeks
    else:
        hist.category_week_baseline = 0.0

    last_cat = db.execute(
        select(func.max(Transaction.occurred_at)).where(
            and_(*cat_filter, Transaction.occurred_at <= now)
        )
    ).scalar_one()
    if last_cat:
        if last_cat.tzinfo is None:
            last_cat = last_cat.replace(tzinfo=UTC)
        hist.days_since_category = max(0.0, (now - last_cat).total_seconds() / 86400.0)
    else:
        hist.days_since_category = 400.0

    # Per-category std, needed for category_zscore.
    amounts = [
        _f(a)
        for a in db.execute(
            select(Transaction.amount)
            .where(and_(*cat_filter, Transaction.occurred_at <= now))
            .order_by(Transaction.occurred_at.desc())
            .limit(200)
        ).scalars().all()
    ]
    if len(amounts) > 1:
        m = sum(amounts) / len(amounts)
        hist.category_std_amount[category] = (
            sum((x - m) ** 2 for x in amounts) / (len(amounts) - 1)
        ) ** 0.5
    return hist


def build_txn_context(
    *,
    amount: Decimal | float,
    account: Account,
    occurred_at: datetime,
    category: str,
    channel: str,
    merchant_name: str | None = None,
    device_fingerprint: str | None = None,
    location_city: str | None = None,
    location_country: str = "IN",
) -> TxnContext:
    opened = account.opened_on
    if opened:
        age_days = max(
            (datetime.now(UTC).date() - opened).days,
            0,
        )
    else:
        created = account.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = max((datetime.now(UTC) - created).days, 0) if created else 365

    return TxnContext(
        amount=_f(amount),
        occurred_at=occurred_at,
        category=category,
        channel=channel,
        merchant_name=merchant_name,
        device_fingerprint=device_fingerprint,
        location_city=location_city,
        location_country=location_country,
        account_balance=_f(account.available_balance),
        account_age_days=float(age_days),
    )


def build_credit_profile(
    db: Session,
    user: User,
    *,
    requested_amount: Decimal | float,
    tenure_months: int,
    loan_type: str,
    declared_income: Decimal | float | None = None,
    now: datetime | None = None,
) -> CreditProfile:
    """Assemble a loan applicant snapshot from the user's real banking history.

    Everything except the declared income is derived from observed account
    behaviour, which is the point of scoring inside a bank rather than from a
    form: the applicant cannot inflate their transaction record.
    """
    now = now or datetime.now(UTC)
    d90 = now - timedelta(days=90)

    accounts = db.execute(
        select(Account).where(
            Account.user_id == user.id, Account.status != AccountStatus.CLOSED.value
        )
    ).scalars().all()

    balances = [_f(a.balance) for a in accounts]
    avg_balance = sum(balances) / len(balances) if balances else 0.0

    oldest = min((a.opened_on for a in accounts if a.opened_on), default=None)
    if oldest:
        account_age_months = max((now.date() - oldest).days / 30.44, 0.0)
    else:
        created = user.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        account_age_months = max((now - created).days / 30.44, 0.0) if created else 1.0

    inflow = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.user_id == user.id,
                Transaction.txn_type.in_(
                    [
                        TransactionType.DEPOSIT.value,
                        TransactionType.TRANSFER_IN.value,
                        TransactionType.INTEREST.value,
                    ]
                ),
                Transaction.status == TransactionStatus.COMPLETED.value,
                Transaction.occurred_at >= d90,
            )
        ).scalar_one()
    )
    outflow = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.user_id == user.id,
                Transaction.txn_type.in_([t.value for t in DEBIT_TYPES]),
                Transaction.status == TransactionStatus.COMPLETED.value,
                Transaction.occurred_at >= d90,
            )
        ).scalar_one()
    )
    txn_count_90d = int(
        db.execute(
            select(func.count(Transaction.id)).where(
                Transaction.user_id == user.id,
                Transaction.status == TransactionStatus.COMPLETED.value,
                Transaction.occurred_at >= d90,
            )
        ).scalar_one()
        or 0
    )

    # Monthly closing-balance volatility, computed from balance_after snapshots.
    monthly = db.execute(
        select(Transaction.balance_after)
        .where(
            Transaction.user_id == user.id,
            Transaction.balance_after.is_not(None),
            Transaction.occurred_at >= now - timedelta(days=180),
        )
        .order_by(Transaction.occurred_at.desc())
        .limit(400)
    ).scalars().all()
    snapshots = [_f(b) for b in monthly if b is not None]
    balance_volatility = 0.4
    min_balance_ratio = 0.5
    if len(snapshots) > 2:
        m = sum(snapshots) / len(snapshots)
        if m > 0:
            sd = (sum((x - m) ** 2 for x in snapshots) / (len(snapshots) - 1)) ** 0.5
            balance_volatility = min(sd / m, 4.0)
            min_balance_ratio = max(min(min(snapshots) / m, 1.6), 0.0)

    overdrafts = sum(1 for b in snapshots if b < 0)

    # Card utilisation across issued virtual cards.
    cards = db.execute(
        select(Card).where(Card.user_id == user.id, Card.status == CardStatus.ACTIVE.value)
    ).scalars().all()
    card_limit = sum(_f(c.monthly_limit) for c in cards)
    card_spend = _f(
        db.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.user_id == user.id,
                Transaction.txn_type == TransactionType.CARD_PAYMENT.value,
                Transaction.status == TransactionStatus.COMPLETED.value,
                Transaction.occurred_at >= now - timedelta(days=30),
            )
        ).scalar_one()
    )
    credit_utilisation = min(card_spend / card_limit, 3.0) if card_limit > 0 else 0.0

    prior_loans = int(
        db.execute(
            select(func.count(Loan.id)).where(
                Loan.user_id == user.id,
                Loan.status.in_(
                    [LoanStatus.DISBURSED.value, LoanStatus.CLOSED.value, LoanStatus.APPROVED.value]
                ),
            )
        ).scalar_one()
        or 0
    )
    emis_missed = int(
        db.execute(
            select(func.coalesce(func.sum(Loan.emis_missed), 0)).where(Loan.user_id == user.id)
        ).scalar_one()
        or 0
    )
    # A loan is treated as defaulted once three or more EMIs are missed.
    prior_defaults = int(
        db.execute(
            select(func.count(Loan.id)).where(Loan.user_id == user.id, Loan.emis_missed >= 3)
        ).scalar_one()
        or 0
    )

    existing_emi = _f(
        db.execute(
            select(func.coalesce(func.sum(Loan.emi_amount), 0)).where(
                Loan.user_id == user.id, Loan.status == LoanStatus.DISBURSED.value
            )
        ).scalar_one()
    ) or _f(user.existing_emi)

    age = 35.0
    if user.date_of_birth:
        age = max((now.date() - user.date_of_birth).days / 365.25, 18.0)

    income = _f(declared_income if declared_income is not None else user.annual_income) or 600_000.0

    return CreditProfile(
        age=age,
        annual_income=income,
        employment_status=user.employment_status or "salaried",
        employment_years=float(user.employment_years or 3.0),
        requested_amount=_f(requested_amount),
        tenure_months=tenure_months,
        loan_type=loan_type,
        existing_emi=existing_emi,
        dependents=int(user.dependents or 0),
        housing_status=user.housing_status or "rent",
        account_age_months=account_age_months,
        num_accounts=len(accounts),
        avg_balance=avg_balance,
        balance_volatility=balance_volatility,
        min_balance_ratio=min_balance_ratio,
        inflow_90d=max(inflow, 1.0),
        outflow_90d=max(outflow, 1.0),
        txn_count_90d=txn_count_90d,
        credit_utilisation=credit_utilisation,
        prior_loans=prior_loans,
        prior_defaults=prior_defaults,
        emis_missed=emis_missed,
        overdraft_events_90d=overdrafts,
    )
