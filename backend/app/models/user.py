"""User identity, KYC profile and login-device tracking."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import KycStatus, UserRole, UserStatus
from app.models.mixins import Money, TimestampMixin, TZDateTime

if TYPE_CHECKING:
    from app.models.banking import Account, Beneficiary, Card
    from app.models.lending import CreditScore, Loan
    from app.models.risk import AnomalyAlert, FraudAlert
    from app.models.system import AuditLog, Notification


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date)

    role: Mapped[str] = mapped_column(String(20), default=UserRole.CUSTOMER, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default=UserStatus.PENDING, nullable=False, index=True)

    # ---- KYC (simulated) ----
    kyc_status: Mapped[str] = mapped_column(String(20), default=KycStatus.NOT_STARTED, nullable=False, index=True)
    kyc_submitted_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    kyc_verified_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    kyc_rejection_reason: Mapped[str | None] = mapped_column(Text)
    # Government IDs are stored hashed + masked. NEVER store real IDs in a demo app.
    pan_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    pan_masked: Mapped[str | None] = mapped_column(String(20))
    aadhaar_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    aadhaar_masked: Mapped[str | None] = mapped_column(String(20))
    id_document_name: Mapped[str | None] = mapped_column(String(255))
    id_document_type: Mapped[str | None] = mapped_column(String(50))

    # ---- Address ----
    address_line1: Mapped[str | None] = mapped_column(String(255))
    address_line2: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(100))
    postal_code: Mapped[str | None] = mapped_column(String(20))
    country: Mapped[str] = mapped_column(String(2), default="IN", nullable=False)

    # ---- Financial profile (feeds the credit-scoring model) ----
    annual_income: Mapped[Decimal | None] = mapped_column(Money)
    employment_status: Mapped[str | None] = mapped_column(String(50))
    employment_years: Mapped[float | None] = mapped_column()
    existing_emi: Mapped[Decimal | None] = mapped_column(Money)
    dependents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    housing_status: Mapped[str | None] = mapped_column(String(30))

    # ---- Security ----
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    two_factor_secret: Mapped[str | None] = mapped_column(String(64))
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(TZDateTime)
    last_login_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    password_changed_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    # ---- Notification preferences ----
    notify_email: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_sms: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notify_large_txn: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_login: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_marketing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # ---- Relationships ----
    accounts: Mapped[list["Account"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    cards: Mapped[list["Card"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    beneficiaries: Mapped[list["Beneficiary"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    # foreign_keys is required on both sides: Loan reaches users.id via user_id
    # *and* decided_by_id, so the join is otherwise ambiguous.
    loans: Mapped[list["Loan"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="Loan.user_id",
    )
    credit_scores: Mapped[list["CreditScore"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    # FraudAlert also links to users.id twice (user_id and reviewed_by_id).
    fraud_alerts: Mapped[list["FraudAlert"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="FraudAlert.user_id",
    )
    anomaly_alerts: Mapped[list["AnomalyAlert"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    notifications: Mapped[list["Notification"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    devices: Mapped[list["UserDevice"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="actor", foreign_keys="AuditLog.actor_id"
    )

    __table_args__ = (Index("ix_users_role_status", "role", "status"),)

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    @property
    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.id} {self.email} {self.role}>"


class UserDevice(Base, TimestampMixin):
    """Known login devices — powers "login from new device" notifications and is a
    feature source for the fraud model (device change signal)."""

    __tablename__ = "user_devices"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_agent: Mapped[str | None] = mapped_column(String(400))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    location: Mapped[str | None] = mapped_column(String(120))
    trusted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    login_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    user: Mapped["User"] = relationship(back_populates="devices")

    __table_args__ = (Index("ix_device_user_fp", "user_id", "fingerprint", unique=True),)


class RefreshToken(Base, TimestampMixin):
    """Persisted refresh-token handles, enabling server-side session revocation."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    jti: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    device_fingerprint: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(64))
