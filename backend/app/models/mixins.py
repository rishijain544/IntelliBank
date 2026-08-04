"""Shared column types and mixins.

``Money`` keeps exact decimal semantics on PostgreSQL while remaining usable on
SQLite (which has no native NUMERIC affinity) by round-tripping through Decimal.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import DateTime, Numeric, TypeDecorator, func
from sqlalchemy.orm import Mapped, mapped_column

TWO_PLACES = Decimal("0.01")


def utcnow() -> datetime:
    return datetime.now(UTC)


def quantize(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


class Money(TypeDecorator):
    """NUMERIC(18,2) that always yields a 2dp ``Decimal`` regardless of backend."""

    impl = Numeric(18, 2)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return quantize(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return quantize(value)


class TZDateTime(TypeDecorator):
    """Timezone-aware datetime that stores UTC on every backend."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=utcnow, server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=utcnow, onupdate=utcnow, server_default=func.now(), nullable=False
    )
