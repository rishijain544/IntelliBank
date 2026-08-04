"""Pure feature builders shared by offline training and online inference.

The single most common failure in ML-backed products is train/serve skew: the
trainer computes a feature one way, production computes it another, and the
model silently degrades. Everything here is a pure function over plain
dataclasses, so the offline simulator and the live SQL path produce byte-identical
feature vectors from equivalent inputs.

Data flow::

    simulator stream ---> UserHistoryStats ---\\
                                              >--- build_*_features() ---> FeatureSet.vectorise()
    SQL aggregates   ---> UserHistoryStats ---/
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from app.ml.schema import (
    CHANNEL_RISK,
    EMPLOYMENT_STABILITY,
    HIGH_RISK_CATEGORIES,
    HOUSING_STABILITY,
    LOAN_PURPOSE_RISK,
)

# --------------------------------------------------------------------------- #
# Input containers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TxnContext:
    """The transaction being scored."""

    amount: float
    occurred_at: datetime
    category: str = "other"
    channel: str = "internal"
    merchant_name: str | None = None
    device_fingerprint: str | None = None
    location_city: str | None = None
    location_country: str = "IN"
    account_balance: float = 0.0
    account_age_days: float = 365.0


@dataclass(slots=True)
class UserHistoryStats:
    """Rolling aggregates of a user's prior behaviour.

    Every field has a neutral default so a brand-new user with zero history
    still produces a valid vector (cold start returns mid-range priors rather
    than a crash or a garbage extreme).
    """

    txn_count_total: int = 0
    mean_amount: float = 0.0
    std_amount: float = 0.0
    recent_amounts: list[float] = field(default_factory=list)

    txn_count_1h: int = 0
    txn_count_24h: int = 0
    amount_sum_24h: float = 0.0
    mins_since_prev_txn: float = 1440.0
    avg_daily_txn_count: float = 1.0

    known_devices: frozenset[str] = frozenset()
    known_cities: frozenset[str] = frozenset()
    known_merchants: frozenset[str] = frozenset()
    home_country: str = "IN"

    category_counts: dict[str, int] = field(default_factory=dict)
    distinct_merchants_7d: int = 0
    distinct_categories_7d: int = 0
    failed_txn_24h: int = 0
    usual_hour: float = 13.0

    # Anomaly-specific baselines
    category_mean_amount: dict[str, float] = field(default_factory=dict)
    category_std_amount: dict[str, float] = field(default_factory=dict)
    category_week_spend: float = 0.0
    category_week_baseline: float = 0.0
    days_since_category: float = 7.0
    week_spend_total: float = 0.0
    week_spend_median: float = 0.0


@dataclass(slots=True)
class CreditProfile:
    """Applicant snapshot assembled at loan-application time."""

    age: float = 35.0
    annual_income: float = 600_000.0
    employment_status: str = "salaried"
    employment_years: float = 3.0
    requested_amount: float = 300_000.0
    tenure_months: int = 36
    loan_type: str = "personal"
    existing_emi: float = 0.0
    dependents: int = 1
    housing_status: str = "rent"

    account_age_months: float = 24.0
    num_accounts: int = 1
    avg_balance: float = 25_000.0
    balance_volatility: float = 0.4
    min_balance_ratio: float = 0.5
    inflow_90d: float = 150_000.0
    outflow_90d: float = 140_000.0
    txn_count_90d: int = 60
    credit_utilisation: float = 0.3
    prior_loans: int = 0
    prior_defaults: int = 0
    emis_missed: int = 0
    overdraft_events_90d: int = 0


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def _percentile_of(value: float, sample: list[float]) -> float:
    if not sample:
        return 0.5
    below = sum(1 for s in sample if s <= value)
    return below / len(sample)


def _cyclical_hour(hour: float) -> tuple[float, float]:
    radians = 2 * math.pi * (hour / 24.0)
    return math.sin(radians), math.cos(radians)


def _is_round_amount(amount: float) -> bool:
    """Round figures (10000, 50000) correlate with card-testing and mule activity."""
    if amount <= 0:
        return False
    return amount % 10_000 == 0 or (amount >= 1000 and amount % 5_000 == 0)


def emi_for(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard reducing-balance EMI."""
    if months <= 0:
        return principal
    r = annual_rate_pct / 12.0 / 100.0
    if r <= 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


# --------------------------------------------------------------------------- #
# 1. Fraud features
# --------------------------------------------------------------------------- #


def build_fraud_features(ctx: TxnContext, hist: UserHistoryStats) -> dict[str, float]:
    """Assemble the 24-dimensional fraud feature vector."""
    hour = ctx.occurred_at.hour + ctx.occurred_at.minute / 60.0
    hour_sin, hour_cos = _cyclical_hour(hour)

    mean_amt = hist.mean_amount if hist.mean_amount > 0 else ctx.amount or 1.0
    std_amt = hist.std_amount if hist.std_amount > 0 else max(mean_amt * 0.5, 1.0)

    total_cat = sum(hist.category_counts.values())
    cat_share = _safe_div(hist.category_counts.get(ctx.category, 0), total_cat, 0.0)

    available = max(ctx.account_balance, 1.0)

    return {
        "amount_log": _log1p(ctx.amount),
        "amount_to_user_mean": _safe_div(ctx.amount, mean_amt, 1.0),
        "amount_zscore": _safe_div(ctx.amount - mean_amt, std_amt, 0.0),
        "amount_to_balance": _safe_div(ctx.amount, available, 0.05),
        "amount_percentile": _percentile_of(ctx.amount, hist.recent_amounts),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_night": 1.0 if ctx.occurred_at.hour < 5 else 0.0,
        "is_weekend": 1.0 if ctx.occurred_at.weekday() >= 5 else 0.0,
        "txn_count_1h": float(hist.txn_count_1h),
        "txn_count_24h": float(hist.txn_count_24h),
        "amount_sum_24h_log": _log1p(hist.amount_sum_24h),
        "mins_since_prev_txn": float(min(hist.mins_since_prev_txn, 10_080)),
        "device_is_new": 0.0
        if (ctx.device_fingerprint and ctx.device_fingerprint in hist.known_devices)
        else 1.0,
        "location_is_new": 0.0
        if (ctx.location_city and ctx.location_city in hist.known_cities)
        else 1.0,
        "is_foreign": 1.0 if ctx.location_country != hist.home_country else 0.0,
        "merchant_is_new": 0.0
        if (ctx.merchant_name and ctx.merchant_name in hist.known_merchants)
        else 1.0,
        "category_is_rare": 1.0 - cat_share,
        "distinct_merchants_7d": float(hist.distinct_merchants_7d),
        "channel_risk": CHANNEL_RISK.get(ctx.channel, 0.3),
        "account_age_days": float(ctx.account_age_days),
        "failed_txn_24h": float(hist.failed_txn_24h),
        "is_round_amount": 1.0 if _is_round_amount(ctx.amount) else 0.0,
        "high_risk_category": 1.0 if ctx.category in HIGH_RISK_CATEGORIES else 0.0,
    }


# --------------------------------------------------------------------------- #
# 2. Credit features
# --------------------------------------------------------------------------- #


def build_credit_features(
    profile: CreditProfile, *, assumed_rate_pct: float = 14.0
) -> dict[str, float]:
    """Assemble the 26-dimensional credit feature vector.

    ``assumed_rate_pct`` prices the projected EMI before a risk band is known;
    using a fixed assumption keeps the feature deterministic and prevents the
    circularity of pricing depending on the score that depends on pricing.
    """
    monthly_income = max(profile.annual_income / 12.0, 1.0)
    projected_emi = emi_for(profile.requested_amount, assumed_rate_pct, profile.tenure_months)

    return {
        "age": profile.age,
        "annual_income_log": _log1p(profile.annual_income),
        "employment_years": profile.employment_years,
        "employment_stability": EMPLOYMENT_STABILITY.get(profile.employment_status, 0.5),
        "loan_amount_log": _log1p(profile.requested_amount),
        "loan_to_income": _safe_div(profile.requested_amount, profile.annual_income, 1.0),
        "tenure_months": float(profile.tenure_months),
        "emi_to_income": _safe_div(projected_emi, monthly_income, 0.3),
        "debt_to_income": _safe_div(projected_emi + profile.existing_emi, monthly_income, 0.3),
        "existing_emi_log": _log1p(profile.existing_emi),
        "dependents": float(profile.dependents),
        "housing_stability": HOUSING_STABILITY.get(profile.housing_status, 0.5),
        "account_age_months": profile.account_age_months,
        "num_accounts": float(profile.num_accounts),
        "avg_balance_log": _log1p(profile.avg_balance),
        "balance_volatility": profile.balance_volatility,
        "min_balance_ratio": profile.min_balance_ratio,
        "savings_rate": _safe_div(
            profile.inflow_90d - profile.outflow_90d, max(profile.inflow_90d, 1.0), 0.0
        ),
        "inflow_outflow_ratio": _safe_div(profile.inflow_90d, max(profile.outflow_90d, 1.0), 1.0),
        "txn_count_90d": float(profile.txn_count_90d),
        "credit_utilisation": profile.credit_utilisation,
        "prior_loans": float(profile.prior_loans),
        "prior_defaults": float(profile.prior_defaults),
        "emis_missed": float(profile.emis_missed),
        "overdraft_events_90d": float(profile.overdraft_events_90d),
        "loan_purpose_risk": LOAN_PURPOSE_RISK.get(profile.loan_type, 0.5),
    }


# --------------------------------------------------------------------------- #
# 3. Anomaly features
# --------------------------------------------------------------------------- #


def build_anomaly_features(ctx: TxnContext, hist: UserHistoryStats) -> dict[str, float]:
    """Assemble the 14-dimensional per-user behavioural vector."""
    hour = ctx.occurred_at.hour + ctx.occurred_at.minute / 60.0
    hour_sin, hour_cos = _cyclical_hour(hour)

    mean_amt = hist.mean_amount if hist.mean_amount > 0 else ctx.amount or 1.0
    std_amt = hist.std_amount if hist.std_amount > 0 else max(mean_amt * 0.5, 1.0)

    cat_mean = hist.category_mean_amount.get(ctx.category, mean_amt)
    cat_std = hist.category_std_amount.get(ctx.category, std_amt)
    cat_std = cat_std if cat_std > 0 else max(cat_mean * 0.5, 1.0)

    # Circular distance between this hour and the user's habitual hour, in [0, 1].
    raw_gap = abs(hour - hist.usual_hour)
    hour_gap = min(raw_gap, 24 - raw_gap) / 12.0

    return {
        "amount_log": _log1p(ctx.amount),
        "amount_to_user_mean": _safe_div(ctx.amount, mean_amt, 1.0),
        "amount_zscore": _safe_div(ctx.amount - mean_amt, std_amt, 0.0),
        "category_spend_ratio": _safe_div(
            hist.category_week_spend, max(hist.category_week_baseline, 1.0), 1.0
        ),
        "category_zscore": _safe_div(ctx.amount - cat_mean, cat_std, 0.0),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "hour_deviation": hour_gap,
        "txn_count_24h_ratio": _safe_div(
            hist.txn_count_24h, max(hist.avg_daily_txn_count, 0.5), 1.0
        ),
        "days_since_category": float(hist.days_since_category),
        "merchant_is_new": 0.0
        if (ctx.merchant_name and ctx.merchant_name in hist.known_merchants)
        else 1.0,
        "distinct_categories_7d": float(hist.distinct_categories_7d),
        "weekly_spend_ratio": _safe_div(
            hist.week_spend_total, max(hist.week_spend_median, 1.0), 1.0
        ),
        "is_weekend": 1.0 if ctx.occurred_at.weekday() >= 5 else 0.0,
    }
