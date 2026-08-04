"""Core banking domain logic: accounts, the ledger, transfers and cards.

The transfer engine is the heart of the app and the place where the fraud model
is wired into real money movement:

1. build features from live history,
2. score the transaction,
3. **allow / hold / block** based on that score,
4. write both ledger legs plus an alert atomically.

A single database transaction covers debit, credit, alert and notification, so a
failure mid-flight cannot leave money debited without a matching credit.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import random_reference
from app.ml.inference import FraudDecision, score_anomaly, score_fraud
from app.models.banking import Account, Beneficiary, Card, Transaction
from app.models.enums import (
    AccountStatus,
    AccountType,
    AlertSeverity,
    CardStatus,
    CardType,
    MerchantCategory,
    TransactionChannel,
    TransactionStatus,
    TransactionType,
)
from app.models.mixins import quantize
from app.models.risk import AnomalyAlert, FraudAlert
from app.models.user import User
from app.services import notifications as notif
from app.services.ml_features import (
    build_txn_context,
    build_user_history,
    enrich_for_category,
)

IFSC_CODE = "SMRT0000001"

# Per-type account defaults: interest, overdraft and minimum balance.
ACCOUNT_DEFAULTS: dict[str, dict[str, object]] = {
    AccountType.SAVINGS: {"interest_rate": 3.5, "overdraft": Decimal("0.00")},
    AccountType.CURRENT: {
        "interest_rate": 0.0,
        "overdraft": Decimal(str(settings.CURRENT_ACCOUNT_OVERDRAFT)),
    },
    AccountType.FIXED_DEPOSIT: {"interest_rate": 7.1, "overdraft": Decimal("0.00")},
    AccountType.SALARY: {"interest_rate": 3.0, "overdraft": Decimal("5000.00")},
}

TRANSFER_FEES: dict[str, Decimal] = {
    TransactionChannel.INTERNAL: Decimal("0.00"),
    TransactionChannel.UPI: Decimal("0.00"),
    TransactionChannel.IMPS: Decimal("5.00"),
    TransactionChannel.NEFT: Decimal("2.50"),
}


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


def generate_account_number(db: Session) -> str:
    """Unique 14-digit account number. Retries on the (unlikely) collision."""
    for _ in range(12):
        candidate = "5" + "".join(secrets.choice("0123456789") for _ in range(13))
        exists = db.execute(
            select(Account.id).where(Account.account_number == candidate)
        ).first()
        if not exists:
            return candidate
    raise RuntimeError("could not allocate an account number")


def create_account(
    db: Session,
    user: User,
    *,
    account_type: str = AccountType.SAVINGS,
    nickname: str | None = None,
    initial_deposit: Decimal = Decimal("0.00"),
) -> Account:
    defaults = ACCOUNT_DEFAULTS.get(account_type, ACCOUNT_DEFAULTS[AccountType.SAVINGS])
    is_first = not db.execute(
        select(Account.id).where(Account.user_id == user.id).limit(1)
    ).first()

    account = Account(
        user_id=user.id,
        account_number=generate_account_number(db),
        ifsc_code=IFSC_CODE,
        nickname=nickname,
        account_type=account_type,
        status=AccountStatus.ACTIVE,
        balance=Decimal("0.00"),
        overdraft_limit=defaults["overdraft"],
        interest_rate=float(defaults["interest_rate"]),
        is_primary=is_first,
        opened_on=date.today(),
    )
    db.add(account)
    db.flush()

    if initial_deposit > 0:
        record_deposit(
            db,
            user,
            account,
            amount=initial_deposit,
            description="Initial deposit",
            channel=TransactionChannel.SYSTEM,
        )
    return account


def get_owned_account(db: Session, user: User, account_id: int) -> Account:
    account = db.get(Account, account_id)
    if account is None or account.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    return account


def assert_account_usable(account: Account) -> None:
    if account.status == AccountStatus.FROZEN:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account is frozen and cannot be used for transactions",
        )
    if account.status == AccountStatus.CLOSED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This account is closed")


def total_balance(db: Session, user_id: int) -> Decimal:
    total = db.execute(
        select(func.coalesce(func.sum(Account.balance), 0)).where(
            Account.user_id == user_id, Account.status != AccountStatus.CLOSED
        )
    ).scalar_one()
    return quantize(total or 0)


# --------------------------------------------------------------------------- #
# Ledger primitives
# --------------------------------------------------------------------------- #


def _new_transaction(
    *,
    user_id: int,
    account: Account,
    txn_type: str,
    amount: Decimal,
    signed: Decimal,
    channel: str,
    status_value: str,
    occurred_at: datetime,
    description: str | None = None,
    category: str = MerchantCategory.OTHER,
    merchant_name: str | None = None,
    counterparty_name: str | None = None,
    counterparty_account_number: str | None = None,
    counterparty_ifsc: str | None = None,
    counterparty_account_id: int | None = None,
    fee: Decimal = Decimal("0.00"),
    device_fingerprint: str | None = None,
    ip_address: str | None = None,
    location_city: str | None = None,
    location_country: str = "IN",
    balance_after: Decimal | None = None,
    card_id: int | None = None,
) -> Transaction:
    return Transaction(
        reference=random_reference("TXN"),
        user_id=user_id,
        account_id=account.id,
        txn_type=txn_type,
        channel=channel,
        status=status_value,
        amount=quantize(amount),
        fee=quantize(fee),
        signed_amount=quantize(signed),
        balance_after=quantize(balance_after) if balance_after is not None else None,
        description=description,
        merchant_category=category,
        merchant_name=merchant_name,
        counterparty_name=counterparty_name,
        counterparty_account_number=counterparty_account_number,
        counterparty_ifsc=counterparty_ifsc,
        counterparty_account_id=counterparty_account_id,
        device_fingerprint=device_fingerprint,
        ip_address=ip_address,
        location_city=location_city,
        location_country=location_country,
        is_foreign=location_country != "IN",
        occurred_at=occurred_at,
        card_id=card_id,
    )


def record_deposit(
    db: Session,
    user: User,
    account: Account,
    *,
    amount: Decimal,
    description: str = "Deposit",
    channel: str = TransactionChannel.SYSTEM,
    category: str = MerchantCategory.OTHER,
    occurred_at: datetime | None = None,
) -> Transaction:
    """Credit an account. Used for demo funding, salary credits and disbursements."""
    assert_account_usable(account)
    amount = quantize(amount)
    account.balance = quantize(account.balance + amount)

    txn = _new_transaction(
        user_id=user.id,
        account=account,
        txn_type=TransactionType.DEPOSIT,
        amount=amount,
        signed=amount,
        channel=channel,
        status_value=TransactionStatus.COMPLETED,
        occurred_at=occurred_at or datetime.now(UTC),
        description=description,
        category=category,
        balance_after=account.balance,
    )
    db.add(txn)
    db.flush()
    return txn


def _assert_sufficient_funds(account: Account, total_debit: Decimal) -> None:
    if account.available_balance < total_debit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Insufficient funds. Available balance is "
                f"{account.available_balance:,.2f} but {total_debit:,.2f} is required."
            ),
        )
    # Savings accounts must retain a minimum balance.
    if account.account_type == AccountType.SAVINGS:
        remaining = account.balance - total_debit
        floor = Decimal(str(settings.MIN_SAVINGS_BALANCE))
        if remaining < floor:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Savings accounts must retain a minimum balance of {floor:,.2f}",
            )


def _assert_daily_limit(db: Session, user_id: int, amount: Decimal) -> None:
    since = datetime.now(UTC) - timedelta(hours=24)
    spent = db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.txn_type == TransactionType.TRANSFER_OUT,
            Transaction.status.in_(
                [TransactionStatus.COMPLETED.value, TransactionStatus.HELD.value]
            ),
            Transaction.occurred_at >= since,
        )
    ).scalar_one()
    limit = Decimal(str(settings.DAILY_TRANSFER_LIMIT))
    if quantize(spent or 0) + amount > limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"This transfer would exceed your 24-hour transfer limit of {limit:,.2f}",
        )


# --------------------------------------------------------------------------- #
# Fraud + anomaly integration
# --------------------------------------------------------------------------- #


def assess_transaction(
    db: Session,
    user: User,
    account: Account,
    *,
    amount: Decimal,
    category: str,
    channel: str,
    merchant_name: str | None,
    device_fingerprint: str | None,
    ip_address: str | None,
    location_city: str | None,
    location_country: str = "IN",
    occurred_at: datetime | None = None,
) -> tuple[FraudDecision, object]:
    """Score a pending transaction with both risk models."""
    now = occurred_at or datetime.now(UTC)
    ctx = build_txn_context(
        amount=amount,
        account=account,
        occurred_at=now,
        category=category,
        channel=channel,
        merchant_name=merchant_name,
        device_fingerprint=device_fingerprint,
        location_city=location_city,
        location_country=location_country,
    )
    hist = build_user_history(db, user, now=now)
    fraud = score_fraud(ctx, hist)

    hist = enrich_for_category(db, user, hist, category, now=now)
    anomaly = score_anomaly(ctx, hist)
    return fraud, anomaly


def _raise_alert(
    db: Session,
    user: User,
    txn: Transaction,
    decision: FraudDecision,
) -> FraudAlert:
    alert = FraudAlert(
        alert_ref=random_reference("FRD", 8),
        user_id=user.id,
        transaction_id=txn.id,
        risk_score=decision.risk_score,
        severity=decision.severity,
        auto_blocked=decision.auto_blocked,
        decision_source=decision.decision_source,
        reasons=decision.reasons,
        features=decision.features,
        top_factors=decision.top_factors,
        triggered_rules=decision.triggered_rules,
        model_name=decision.model_name,
        model_version=decision.model_version,
        inference_latency_ms=decision.latency_ms,
    )
    db.add(alert)
    db.flush()
    notif.notify_fraud_alert(
        db, user, alert_ref=alert.alert_ref, amount=txn.amount, blocked=decision.auto_blocked
    )
    return alert


def _record_anomaly(db: Session, user: User, txn: Transaction, result) -> AnomalyAlert | None:
    if not result.is_anomaly:
        return None
    alert = AnomalyAlert(
        user_id=user.id,
        transaction_id=txn.id,
        anomaly_score=result.anomaly_score,
        severity=result.severity,
        anomaly_type=result.anomaly_type,
        title=result.title,
        message=result.message,
        category=txn.merchant_category,
        baseline_value=result.baseline_value,
        observed_value=result.observed_value,
        deviation_ratio=result.deviation_ratio,
        features=result.features,
        model_name=result.model_name,
        model_version=result.model_version,
        inference_latency_ms=result.latency_ms,
    )
    db.add(alert)
    return alert


# --------------------------------------------------------------------------- #
# Transfers
# --------------------------------------------------------------------------- #


def _resolve_internal_destination(db: Session, account_number: str) -> Account:
    dest = db.execute(
        select(Account).where(Account.account_number == account_number)
    ).scalar_one_or_none()
    if dest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Destination account not found"
        )
    if dest.status != AccountStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The destination account cannot receive funds at this time",
        )
    return dest


def execute_transfer(
    db: Session,
    user: User,
    *,
    from_account_id: int,
    amount: Decimal,
    to_account_number: str | None = None,
    beneficiary: Beneficiary | None = None,
    channel: str = TransactionChannel.INTERNAL,
    description: str | None = None,
    category: str = MerchantCategory.TRANSFER,
    device_fingerprint: str | None = None,
    ip_address: str | None = None,
    location_city: str | None = None,
    location_country: str = "IN",
) -> tuple[Transaction, FraudDecision, str]:
    """Move money, gated by the fraud model.

    Returns ``(transaction, fraud_decision, message)``. Outcomes:

    * ``block``  -> nothing is debited; a BLOCKED row is written for the audit trail
    * ``review`` -> funds are debited and *held*, credited on approval
    * ``allow``  -> both legs post immediately
    """
    source = get_owned_account(db, user, from_account_id)
    assert_account_usable(source)

    amount = quantize(amount)
    fee = TRANSFER_FEES.get(channel, Decimal("0.00"))
    total_debit = quantize(amount + fee)

    # Resolve destination before touching balances.
    dest_account: Account | None = None
    counterparty_name: str | None = None
    counterparty_number: str | None = None
    counterparty_ifsc: str | None = None

    if beneficiary is not None:
        counterparty_name = beneficiary.name
        counterparty_number = beneficiary.account_number
        counterparty_ifsc = beneficiary.ifsc_code
        if beneficiary.is_internal:
            dest_account = _resolve_internal_destination(db, beneficiary.account_number)
        if beneficiary.transfer_limit and amount > beneficiary.transfer_limit:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Amount exceeds the limit set for this beneficiary ({beneficiary.transfer_limit:,.2f})",
            )
    elif to_account_number:
        dest_account = _resolve_internal_destination(db, to_account_number)
        counterparty_number = dest_account.account_number
        counterparty_ifsc = dest_account.ifsc_code
        owner = db.get(User, dest_account.user_id)
        counterparty_name = owner.full_name if owner else "IntelliBank customer"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a destination account number or a beneficiary is required",
        )

    if dest_account is not None and dest_account.id == source.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source and destination accounts must differ",
        )

    _assert_sufficient_funds(source, total_debit)
    _assert_daily_limit(db, user.id, amount)

    # ---- risk assessment happens BEFORE any balance mutation ----
    fraud, anomaly = assess_transaction(
        db,
        user,
        source,
        amount=amount,
        category=category,
        channel=channel,
        merchant_name=counterparty_name,
        device_fingerprint=device_fingerprint,
        ip_address=ip_address,
        location_city=location_city,
        location_country=location_country,
    )

    now = datetime.now(UTC)
    label = description or (
        f"Transfer to {counterparty_name}" if counterparty_name else "Transfer"
    )

    # ------------------------------------------------ blocked
    if fraud.action == "block":
        txn = _new_transaction(
            user_id=user.id,
            account=source,
            txn_type=TransactionType.TRANSFER_OUT,
            amount=amount,
            signed=-amount,
            channel=channel,
            status_value=TransactionStatus.BLOCKED,
            occurred_at=now,
            description=label,
            category=category,
            counterparty_name=counterparty_name,
            counterparty_account_number=counterparty_number,
            counterparty_ifsc=counterparty_ifsc,
            counterparty_account_id=dest_account.id if dest_account else None,
            fee=fee,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            location_city=location_city,
            location_country=location_country,
            balance_after=source.balance,
        )
        txn.fraud_score = fraud.risk_score
        txn.anomaly_score = anomaly.anomaly_score
        txn.is_flagged = True
        txn.scoring_latency_ms = fraud.latency_ms
        txn.failure_reason = "Blocked by fraud detection"
        db.add(txn)
        db.flush()
        _raise_alert(db, user, txn, fraud)
        return (
            txn,
            fraud,
            "This transfer was blocked by our fraud detection system. Review it in the Security Center.",
        )

    # ------------------------------------------------ held for review
    if fraud.action == "review":
        source.balance = quantize(source.balance - total_debit)
        source.hold_amount = quantize(source.hold_amount + amount)

        txn = _new_transaction(
            user_id=user.id,
            account=source,
            txn_type=TransactionType.TRANSFER_OUT,
            amount=amount,
            signed=-amount,
            channel=channel,
            status_value=TransactionStatus.HELD,
            occurred_at=now,
            description=label,
            category=category,
            counterparty_name=counterparty_name,
            counterparty_account_number=counterparty_number,
            counterparty_ifsc=counterparty_ifsc,
            counterparty_account_id=dest_account.id if dest_account else None,
            fee=fee,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            location_city=location_city,
            location_country=location_country,
            balance_after=source.balance,
        )
        txn.fraud_score = fraud.risk_score
        txn.anomaly_score = anomaly.anomaly_score
        txn.is_flagged = True
        txn.scoring_latency_ms = fraud.latency_ms
        db.add(txn)
        db.flush()
        _raise_alert(db, user, txn, fraud)
        _record_anomaly(db, user, txn, anomaly)
        return (
            txn,
            fraud,
            "This transfer is on hold pending a security review. You will be notified once it completes.",
        )

    # ------------------------------------------------ allowed
    source.balance = quantize(source.balance - total_debit)
    debit = _new_transaction(
        user_id=user.id,
        account=source,
        txn_type=TransactionType.TRANSFER_OUT,
        amount=amount,
        signed=-amount,
        channel=channel,
        status_value=TransactionStatus.COMPLETED,
        occurred_at=now,
        description=label,
        category=category,
        counterparty_name=counterparty_name,
        counterparty_account_number=counterparty_number,
        counterparty_ifsc=counterparty_ifsc,
        counterparty_account_id=dest_account.id if dest_account else None,
        fee=fee,
        device_fingerprint=device_fingerprint,
        ip_address=ip_address,
        location_city=location_city,
        location_country=location_country,
        balance_after=source.balance,
    )
    debit.fraud_score = fraud.risk_score
    debit.anomaly_score = anomaly.anomaly_score
    debit.scoring_latency_ms = fraud.latency_ms
    # A transfer can be allowed through and still warrant an analyst look. The
    # money moves, but the alert is raised so the case can be labelled and fed
    # back into retraining.
    debit.is_flagged = fraud.is_flagged
    db.add(debit)
    db.flush()

    if fraud.is_flagged:
        _raise_alert(db, user, debit, fraud)

    # Credit leg: only for internal destinations. External transfers are
    # simulated as leaving the bank, which is why no counter-credit is written.
    if dest_account is not None:
        dest_account.balance = quantize(dest_account.balance + amount)
        credit = _new_transaction(
            user_id=dest_account.user_id,
            account=dest_account,
            txn_type=TransactionType.TRANSFER_IN,
            amount=amount,
            signed=amount,
            channel=channel,
            status_value=TransactionStatus.COMPLETED,
            occurred_at=now,
            description=f"Transfer from {user.full_name}",
            category=MerchantCategory.TRANSFER,
            counterparty_name=user.full_name,
            counterparty_account_number=source.account_number,
            counterparty_ifsc=source.ifsc_code,
            counterparty_account_id=source.id,
            balance_after=dest_account.balance,
        )
        db.add(credit)
        db.flush()
        debit.linked_transaction_id = credit.id
        credit.linked_transaction_id = debit.id

        recipient = db.get(User, dest_account.user_id)
        if recipient:
            notif.notify(
                db,
                recipient,
                notif_type="account_update",
                title=f"Money received: {amount:,.2f}",
                body=f"{user.full_name} sent you {amount:,.2f}.",
                action_url="/app/transactions",
            )

    if beneficiary is not None:
        beneficiary.usage_count += 1
        beneficiary.last_used_at = now

    _record_anomaly(db, user, debit, anomaly)
    notif.notify_large_transaction(
        db, user, amount=amount, reference=debit.reference, description=label
    )
    return debit, fraud, "Transfer completed successfully"


def release_held_transfer(db: Session, txn: Transaction, *, approve: bool) -> Transaction:
    """Settle a held transfer after review.

    Approving posts the credit leg and clears the hold; rejecting refunds the
    debit. Either way the hold is released, so funds are never stranded.
    """
    if txn.status != TransactionStatus.HELD:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only transactions on hold can be settled",
        )

    source = db.get(Account, txn.account_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source account missing")

    source.hold_amount = quantize(max(Decimal("0.00"), source.hold_amount - txn.amount))

    if not approve:
        # Refund: the debit is reversed in full, including any fee.
        source.balance = quantize(source.balance + txn.amount + txn.fee)
        txn.status = TransactionStatus.REVERSED
        txn.failure_reason = "Reversed after fraud review"
        txn.balance_after = source.balance
        db.flush()
        return txn

    txn.status = TransactionStatus.COMPLETED
    if txn.counterparty_account_id:
        dest = db.get(Account, txn.counterparty_account_id)
        if dest is not None:
            dest.balance = quantize(dest.balance + txn.amount)
            owner = db.get(User, dest.user_id)
            credit = _new_transaction(
                user_id=dest.user_id,
                account=dest,
                txn_type=TransactionType.TRANSFER_IN,
                amount=txn.amount,
                signed=txn.amount,
                channel=txn.channel,
                status_value=TransactionStatus.COMPLETED,
                occurred_at=datetime.now(UTC),
                description=f"Transfer from {txn.counterparty_name or 'IntelliBank customer'}",
                category=MerchantCategory.TRANSFER,
                counterparty_account_id=source.id,
                counterparty_account_number=source.account_number,
                balance_after=dest.balance,
            )
            db.add(credit)
            db.flush()
            txn.linked_transaction_id = credit.id
            if owner:
                notif.notify(
                    db,
                    owner,
                    notif_type="account_update",
                    title=f"Money received: {txn.amount:,.2f}",
                    body="A held transfer to your account has been approved and credited.",
                    action_url="/app/transactions",
                )
    db.flush()
    return txn


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


def issue_card(
    db: Session,
    user: User,
    account: Account,
    *,
    card_type: str = CardType.VIRTUAL_DEBIT,
    daily_limit: Decimal = Decimal("50000.00"),
    per_txn_limit: Decimal = Decimal("25000.00"),
    monthly_limit: Decimal = Decimal("200000.00"),
) -> Card:
    """Issue a virtual card.

    Only the last four digits are retained in clear text; the full number is
    hashed, mirroring PCI-DSS handling. The plaintext number is never stored and
    is returned to the caller exactly once, at issuance.
    """
    number = "4" + "".join(secrets.choice("0123456789") for _ in range(15))
    card_hash = hashlib.sha256(number.encode()).hexdigest()
    today = date.today()

    card = Card(
        user_id=user.id,
        account_id=account.id,
        card_last4=number[-4:],
        card_hash=card_hash,
        card_network="VISA",
        card_type=card_type,
        status=CardStatus.ACTIVE,
        cardholder_name=user.full_name.upper(),
        expiry_month=today.month,
        expiry_year=today.year + 4,
        daily_limit=quantize(daily_limit),
        per_txn_limit=quantize(per_txn_limit),
        monthly_limit=quantize(monthly_limit),
        credit_limit=quantize(monthly_limit) if card_type == CardType.VIRTUAL_CREDIT else None,
    )
    db.add(card)
    db.flush()

    notif.notify(
        db,
        user,
        notif_type="card_update",
        title="Virtual card issued",
        body=f"Your new virtual card ending {card.card_last4} is ready to use.",
        action_url="/app/cards",
    )
    return card


def set_card_frozen(db: Session, card: Card, *, freeze: bool, reason: str | None = None) -> Card:
    if card.status == CardStatus.CANCELLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This card has been cancelled"
        )
    card.status = CardStatus.FROZEN if freeze else CardStatus.ACTIVE
    card.frozen_at = datetime.now(UTC) if freeze else None
    card.freeze_reason = reason if freeze else None
    db.flush()
    return card
