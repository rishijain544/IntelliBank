"""Accounts, transactions, beneficiaries and virtual cards."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import (
    AccountStatus,
    AccountType,
    CardStatus,
    CardType,
    MerchantCategory,
    TransactionChannel,
    TransactionStatus,
    TransactionType,
)
from app.models.mixins import Money, TimestampMixin, TZDateTime

if TYPE_CHECKING:
    from app.models.risk import AnomalyAlert, FraudAlert
    from app.models.user import User


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    account_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    ifsc_code: Mapped[str] = mapped_column(String(15), default="SMRT0000001", nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(80))

    account_type: Mapped[str] = mapped_column(String(20), default=AccountType.SAVINGS, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=AccountStatus.ACTIVE, nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)

    balance: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    # Funds reserved by pending/held transactions.
    hold_amount: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    overdraft_limit: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    interest_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    opened_on: Mapped[date | None] = mapped_column(Date)
    closed_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    user: Mapped["User"] = relationship(back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="account",
        cascade="all, delete-orphan",
        foreign_keys="Transaction.account_id",
    )
    cards: Mapped[list["Card"]] = relationship(back_populates="account", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_accounts_user_status", "user_id", "status"),)

    @property
    def available_balance(self) -> Decimal:
        return self.balance - self.hold_amount + self.overdraft_limit

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Account {self.account_number} {self.balance}>"


class Transaction(Base, TimestampMixin):
    """Single-entry ledger row. A transfer produces two linked rows
    (``transfer_out`` on the source, ``transfer_in`` on the destination)."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    reference: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)

    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    counterparty_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="SET NULL")
    )
    # Mirror row of the same transfer (out <-> in).
    linked_transaction_id: Mapped[int | None] = mapped_column(
        ForeignKey("transactions.id", ondelete="SET NULL")
    )
    card_id: Mapped[int | None] = mapped_column(ForeignKey("cards.id", ondelete="SET NULL"))

    txn_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(16), default=TransactionChannel.INTERNAL, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=TransactionStatus.COMPLETED, nullable=False, index=True
    )

    amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    fee: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    # Signed: credits positive, debits negative. Simplifies aggregation.
    signed_amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    balance_after: Mapped[Decimal | None] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)

    description: Mapped[str | None] = mapped_column(String(255))
    merchant_name: Mapped[str | None] = mapped_column(String(150), index=True)
    merchant_category: Mapped[str] = mapped_column(
        String(24), default=MerchantCategory.OTHER, nullable=False, index=True
    )
    counterparty_name: Mapped[str | None] = mapped_column(String(150))
    counterparty_account_number: Mapped[str | None] = mapped_column(String(34))
    counterparty_ifsc: Mapped[str | None] = mapped_column(String(15))

    # ---- Contextual signals consumed by the ML feature builders ----
    device_fingerprint: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    location_city: Mapped[str | None] = mapped_column(String(100))
    location_country: Mapped[str | None] = mapped_column(String(2))
    is_foreign: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # ---- Model outputs cached on the row for fast listing ----
    fraud_score: Mapped[float | None] = mapped_column(Float, index=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    scoring_latency_ms: Mapped[float | None] = mapped_column(Float)

    # ---- Ground truth for the retraining loop ----
    # None = unlabelled, True = confirmed fraud, False = confirmed legitimate.
    is_fraud_label: Mapped[bool | None] = mapped_column(Boolean, index=True)
    labelled_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    labelled_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    occurred_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text)

    account: Mapped["Account"] = relationship(back_populates="transactions", foreign_keys=[account_id])
    fraud_alert: Mapped["FraudAlert | None"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_txn_user_time", "user_id", "occurred_at"),
        Index("ix_txn_account_time", "account_id", "occurred_at"),
        Index("ix_txn_flagged_status", "is_flagged", "status"),
        Index("ix_txn_category_time", "merchant_category", "occurred_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Txn {self.reference} {self.txn_type} {self.amount}>"


class Beneficiary(Base, TimestampMixin):
    __tablename__ = "beneficiaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    name: Mapped[str] = mapped_column(String(150), nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(80))
    account_number: Mapped[str] = mapped_column(String(34), nullable=False)
    ifsc_code: Mapped[str] = mapped_column(String(15), nullable=False)
    bank_name: Mapped[str] = mapped_column(String(120), default="IntelliBank", nullable=False)
    is_internal: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Cooling-off period before high-value transfers are permitted (real-bank behaviour).
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    transfer_limit: Mapped[Decimal | None] = mapped_column(Money)
    last_used_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped["User"] = relationship(back_populates="beneficiaries")

    __table_args__ = (
        Index("ix_beneficiary_user_acct", "user_id", "account_number", "ifsc_code", unique=True),
    )


class Card(Base, TimestampMixin):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), index=True)

    # Only the last 4 digits are retained in clear text (PCI-DSS style handling).
    card_last4: Mapped[str] = mapped_column(String(4), nullable=False)
    card_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    card_network: Mapped[str] = mapped_column(String(20), default="VISA", nullable=False)
    card_type: Mapped[str] = mapped_column(String(20), default=CardType.VIRTUAL_DEBIT, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=CardStatus.ACTIVE, nullable=False, index=True)

    cardholder_name: Mapped[str] = mapped_column(String(150), nullable=False)
    expiry_month: Mapped[int] = mapped_column(Integer, nullable=False)
    expiry_year: Mapped[int] = mapped_column(Integer, nullable=False)

    daily_limit: Mapped[Decimal] = mapped_column(Money, default=Decimal("50000.00"), nullable=False)
    per_txn_limit: Mapped[Decimal] = mapped_column(Money, default=Decimal("25000.00"), nullable=False)
    monthly_limit: Mapped[Decimal] = mapped_column(Money, default=Decimal("200000.00"), nullable=False)
    credit_limit: Mapped[Decimal | None] = mapped_column(Money)

    online_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    international_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    contactless_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    atm_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    frozen_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    freeze_reason: Mapped[str | None] = mapped_column(String(255))

    user: Mapped["User"] = relationship(back_populates="cards")
    account: Mapped["Account"] = relationship(back_populates="cards")

    @property
    def masked_number(self) -> str:
        return f"**** **** **** {self.card_last4}"
