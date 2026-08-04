"""Loans and credit-score records (credit-scoring model outputs)."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import DecisionSource, LoanStatus, LoanType
from app.models.mixins import Money, TimestampMixin, TZDateTime

if TYPE_CHECKING:
    from app.models.user import User


class Loan(Base, TimestampMixin):
    __tablename__ = "loans"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_ref: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    disbursement_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="SET NULL")
    )
    credit_score_id: Mapped[int | None] = mapped_column(
        ForeignKey("credit_scores.id", ondelete="SET NULL")
    )

    loan_type: Mapped[str] = mapped_column(String(20), default=LoanType.PERSONAL, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=LoanStatus.SUBMITTED, nullable=False, index=True
    )

    requested_amount: Mapped[Decimal] = mapped_column(Money, nullable=False)
    approved_amount: Mapped[Decimal | None] = mapped_column(Money)
    tenure_months: Mapped[int] = mapped_column(Integer, nullable=False)
    purpose: Mapped[str | None] = mapped_column(String(255))

    # ---- Pricing derived from the credit model's risk band ----
    interest_rate: Mapped[float | None] = mapped_column(Float)
    emi_amount: Mapped[Decimal | None] = mapped_column(Money)
    total_payable: Mapped[Decimal | None] = mapped_column(Money)
    processing_fee: Mapped[Decimal | None] = mapped_column(Money)

    # ---- Applicant snapshot at time of application (model inputs, immutable) ----
    declared_income: Mapped[Decimal | None] = mapped_column(Money)
    existing_emi: Mapped[Decimal | None] = mapped_column(Money)
    employment_status: Mapped[str | None] = mapped_column(String(50))
    employment_years: Mapped[float | None] = mapped_column(Float)

    # ---- Decision trail ----
    decision_source: Mapped[str] = mapped_column(String(16), default=DecisionSource.MODEL, nullable=False)
    decision_reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    manual_override: Mapped[bool] = mapped_column(default=False, nullable=False)
    override_note: Mapped[str | None] = mapped_column(Text)

    # ---- Repayment ----
    disbursed_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    first_emi_date: Mapped[date | None] = mapped_column()
    outstanding_principal: Mapped[Decimal | None] = mapped_column(Money)
    emis_paid: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    emis_missed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped["User"] = relationship(back_populates="loans", foreign_keys=[user_id])
    credit_score: Mapped["CreditScore | None"] = relationship(
        back_populates="loan", foreign_keys=[credit_score_id]
    )

    __table_args__ = (Index("ix_loans_status_created", "status", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Loan {self.application_ref} {self.status}>"


class CreditScore(Base, TimestampMixin):
    """One row per scoring run. Retained for auditability and drift monitoring."""

    __tablename__ = "credit_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    # 300-900 CIBIL-style score derived from the model's default probability.
    score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    probability_of_default: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(2), nullable=False, index=True)  # A..E
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # approve/review/reject
    suggested_rate: Mapped[float] = mapped_column(Float, nullable=False)
    max_eligible_amount: Mapped[Decimal | None] = mapped_column(Money)

    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)

    # Feature vector + SHAP-style contributions for the explainability panel.
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    top_factors: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)

    loan: Mapped["Loan | None"] = relationship(
        back_populates="credit_score", foreign_keys="Loan.credit_score_id", uselist=False
    )
    user: Mapped["User"] = relationship(back_populates="credit_scores")

    __table_args__ = (Index("ix_credit_user_created", "user_id", "created_at"),)
