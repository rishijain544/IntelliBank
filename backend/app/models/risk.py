"""Fraud alerts (supervised model) and anomaly alerts (unsupervised model)."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import AlertSeverity, AlertStatus, DecisionSource
from app.models.mixins import TimestampMixin, TZDateTime

if TYPE_CHECKING:
    from app.models.banking import Transaction
    from app.models.user import User


class FraudAlert(Base, TimestampMixin):
    """Raised when the fraud classifier scores a transaction above the review threshold."""

    __tablename__ = "fraud_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    alert_ref: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), unique=True, index=True
    )

    risk_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(10), default=AlertSeverity.MEDIUM, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default=AlertStatus.OPEN, nullable=False, index=True)
    decision_source: Mapped[str] = mapped_column(String(16), default=DecisionSource.MODEL, nullable=False)

    # Whether the transaction was auto-blocked vs merely flagged for review.
    auto_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    reasons: Mapped[list[str] | None] = mapped_column(JSON)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    top_factors: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    triggered_rules: Mapped[list[str] | None] = mapped_column(JSON)

    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)

    # ---- Customer response (Fraud & Security Center) ----
    customer_response: Mapped[str | None] = mapped_column(String(20))  # confirmed | disputed
    customer_responded_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    customer_note: Mapped[str | None] = mapped_column(Text)

    # ---- Admin review (feeds the retraining loop) ----
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    review_note: Mapped[str | None] = mapped_column(Text)
    # Final ground truth written back onto the transaction for retraining.
    final_label: Mapped[bool | None] = mapped_column(Boolean, index=True)

    transaction: Mapped["Transaction"] = relationship(back_populates="fraud_alert")
    user: Mapped["User"] = relationship(back_populates="fraud_alerts", foreign_keys=[user_id])

    __table_args__ = (
        Index("ix_fraud_status_score", "status", "risk_score"),
        Index("ix_fraud_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FraudAlert {self.alert_ref} {self.risk_score:.3f} {self.status}>"


class AnomalyAlert(Base, TimestampMixin):
    """Unsupervised "this doesn't look like you" nudge surfaced on the Insights page.

    Distinct from fraud: no financial action is taken, it is purely informational.
    """

    __tablename__ = "anomaly_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    transaction_id: Mapped[int | None] = mapped_column(ForeignKey("transactions.id", ondelete="CASCADE"))

    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), default=AlertSeverity.LOW, nullable=False)
    # e.g. "category_spike" | "unusual_amount" | "new_merchant" | "velocity" | "time_of_day"
    anomaly_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(160), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(24))

    baseline_value: Mapped[float | None] = mapped_column(Float)
    observed_value: Mapped[float | None] = mapped_column(Float)
    deviation_ratio: Mapped[float | None] = mapped_column(Float)

    features: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)

    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    period_start: Mapped[datetime | None] = mapped_column(TZDateTime)
    period_end: Mapped[datetime | None] = mapped_column(TZDateTime)

    user: Mapped["User"] = relationship(back_populates="anomaly_alerts")

    __table_args__ = (Index("ix_anomaly_user_ack", "user_id", "acknowledged", "created_at"),)
