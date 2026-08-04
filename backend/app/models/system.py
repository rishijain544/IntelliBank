"""Notifications, audit logging and ML model-performance/drift telemetry."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import AlertSeverity, NotificationType
from app.models.mixins import TimestampMixin, TZDateTime

if TYPE_CHECKING:
    from app.models.user import User


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    notif_type: Mapped[str] = mapped_column(
        String(30), default=NotificationType.GENERAL, nullable=False, index=True
    )
    severity: Mapped[str] = mapped_column(String(10), default=AlertSeverity.LOW, nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Deep-link target in the SPA, e.g. "/app/fraud-center".
    action_url: Mapped[str | None] = mapped_column(String(255))
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    read_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    # Email delivery is simulated: we record intent rather than sending mail.
    email_queued: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="notifications")

    __table_args__ = (Index("ix_notif_user_read_created", "user_id", "is_read", "created_at"),)


class AuditLog(Base, TimestampMixin):
    """Append-only trail of privileged and security-relevant actions."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    actor_email: Mapped[str | None] = mapped_column(String(255))
    actor_role: Mapped[str | None] = mapped_column(String(20))

    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(50), index=True)
    entity_id: Mapped[str | None] = mapped_column(String(50))
    target_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    summary: Mapped[str | None] = mapped_column(String(500))
    before_state: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after_state: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    actor: Mapped["User | None"] = relationship(back_populates="audit_logs", foreign_keys=[actor_id])

    __table_args__ = (Index("ix_audit_action_created", "action", "created_at"),)


class ModelMetricSnapshot(Base, TimestampMixin):
    """Periodic snapshot of live model behaviour, powering the admin drift dashboard.

    Training-time metrics live in the artifact JSON; this table captures *production*
    behaviour (score distribution, flag rate, realised precision/recall from
    admin-reviewed labels) so drift can be compared against the training baseline.
    """

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)

    window_start: Mapped[datetime | None] = mapped_column(TZDateTime)
    window_end: Mapped[datetime | None] = mapped_column(TZDateTime)

    inference_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    flagged_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    labelled_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    mean_score: Mapped[float | None] = mapped_column(Float)
    p95_score: Mapped[float | None] = mapped_column(Float)
    mean_latency_ms: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)

    realised_precision: Mapped[float | None] = mapped_column(Float)
    realised_recall: Mapped[float | None] = mapped_column(Float)
    # Population Stability Index vs the training score distribution.
    psi: Mapped[float | None] = mapped_column(Float)
    drift_status: Mapped[str | None] = mapped_column(String(20))  # stable | watch | drifting

    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    __table_args__ = (Index("ix_metrics_model_created", "model_name", "created_at"),)
