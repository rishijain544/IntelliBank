"""Feature contracts shared by training and inference.

Defining the feature order in one place is what prevents train/serve skew: the
trainer builds matrices from these specs, and the runtime feature builders emit
dicts validated against the same specs before vectorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["numeric", "cyclical", "binary", "categorical"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    description: str
    # Value used when the signal cannot be computed (e.g. brand-new user with no history).
    default: float = 0.0
    # Clipping bounds guard against absurd inference-time values poisoning the model.
    lo: float | None = None
    hi: float | None = None


@dataclass(frozen=True, slots=True)
class FeatureSet:
    name: str
    specs: tuple[FeatureSpec, ...]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def defaults(self) -> dict[str, float]:
        return {s.name: s.default for s in self.specs}

    def by_name(self, name: str) -> FeatureSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def vectorise(self, values: dict[str, float]) -> list[float]:
        """Order, default-fill and clip a feature dict into a model-ready row."""
        row: list[float] = []
        for spec in self.specs:
            v = values.get(spec.name, spec.default)
            if v is None:
                v = spec.default
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = spec.default
            if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guard
                v = spec.default
            if spec.lo is not None:
                v = max(spec.lo, v)
            if spec.hi is not None:
                v = min(spec.hi, v)
            row.append(v)
        return row

    def __len__(self) -> int:
        return len(self.specs)


# --------------------------------------------------------------------------- #
# 1. Fraud detection — supervised binary classification
# --------------------------------------------------------------------------- #
FRAUD_FEATURES = FeatureSet(
    name="fraud",
    specs=(
        FeatureSpec("amount_log", "numeric", "log1p of transaction amount", 0.0, 0, 20),
        FeatureSpec("amount_to_user_mean", "numeric", "amount / user's mean txn amount", 1.0, 0, 200),
        FeatureSpec("amount_zscore", "numeric", "z-score of amount vs user history", 0.0, -20, 20),
        FeatureSpec("amount_to_balance", "numeric", "amount / available balance", 0.05, 0, 10),
        FeatureSpec("amount_percentile", "numeric", "percentile of amount in user history", 0.5, 0, 1),
        FeatureSpec("hour_sin", "cyclical", "sin(2*pi*hour/24)", 0.0, -1, 1),
        FeatureSpec("hour_cos", "cyclical", "cos(2*pi*hour/24)", 1.0, -1, 1),
        FeatureSpec("is_night", "binary", "txn between 00:00 and 05:00", 0.0, 0, 1),
        FeatureSpec("is_weekend", "binary", "Saturday or Sunday", 0.0, 0, 1),
        FeatureSpec("txn_count_1h", "numeric", "user txn count in trailing 1h", 0.0, 0, 200),
        FeatureSpec("txn_count_24h", "numeric", "user txn count in trailing 24h", 1.0, 0, 500),
        FeatureSpec("amount_sum_24h_log", "numeric", "log1p of user's 24h debit volume", 0.0, 0, 20),
        FeatureSpec("mins_since_prev_txn", "numeric", "minutes since previous txn (capped 7d)", 1440.0, 0, 10080),
        FeatureSpec("device_is_new", "binary", "device fingerprint never seen before", 0.0, 0, 1),
        FeatureSpec("location_is_new", "binary", "city never seen before", 0.0, 0, 1),
        FeatureSpec("is_foreign", "binary", "country differs from user's home country", 0.0, 0, 1),
        FeatureSpec("merchant_is_new", "binary", "merchant never transacted with before", 0.0, 0, 1),
        FeatureSpec("category_is_rare", "numeric", "1 - share of this category in user history", 0.5, 0, 1),
        FeatureSpec("distinct_merchants_7d", "numeric", "distinct merchants in trailing 7d", 1.0, 0, 200),
        FeatureSpec("channel_risk", "numeric", "prior risk weight of the channel used", 0.2, 0, 1),
        FeatureSpec("account_age_days", "numeric", "age of the debited account", 365.0, 0, 20000),
        FeatureSpec("failed_txn_24h", "numeric", "failed/declined attempts in trailing 24h", 0.0, 0, 100),
        FeatureSpec("is_round_amount", "binary", "amount is a suspiciously round figure", 0.0, 0, 1),
        FeatureSpec("high_risk_category", "binary", "cash/investment/travel style category", 0.0, 0, 1),
    ),
)

# --------------------------------------------------------------------------- #
# 2. Credit scoring — supervised default prediction
# --------------------------------------------------------------------------- #
CREDIT_FEATURES = FeatureSet(
    name="credit",
    specs=(
        FeatureSpec("age", "numeric", "applicant age in years", 35.0, 18, 100),
        FeatureSpec("annual_income_log", "numeric", "log1p of declared annual income", 13.0, 0, 25),
        FeatureSpec("employment_years", "numeric", "years in current employment", 3.0, 0, 50),
        FeatureSpec("employment_stability", "numeric", "encoded employment status (0 unstable..1 stable)", 0.6, 0, 1),
        FeatureSpec("loan_amount_log", "numeric", "log1p of requested amount", 12.0, 0, 25),
        FeatureSpec("loan_to_income", "numeric", "requested amount / annual income", 0.5, 0, 20),
        FeatureSpec("tenure_months", "numeric", "requested tenure", 36.0, 3, 360),
        FeatureSpec("emi_to_income", "numeric", "projected EMI / monthly income", 0.2, 0, 5),
        FeatureSpec("debt_to_income", "numeric", "(existing EMI + new EMI) / monthly income", 0.25, 0, 5),
        FeatureSpec("existing_emi_log", "numeric", "log1p of existing monthly obligations", 0.0, 0, 20),
        FeatureSpec("dependents", "numeric", "number of dependents", 1.0, 0, 15),
        FeatureSpec("housing_stability", "numeric", "encoded housing status (rent..own)", 0.5, 0, 1),
        FeatureSpec("account_age_months", "numeric", "months since oldest account opened", 24.0, 0, 600),
        FeatureSpec("num_accounts", "numeric", "count of active accounts held", 1.0, 0, 20),
        FeatureSpec("avg_balance_log", "numeric", "log1p of mean balance across accounts", 10.0, 0, 25),
        FeatureSpec("balance_volatility", "numeric", "std/mean of monthly closing balance", 0.4, 0, 10),
        FeatureSpec("min_balance_ratio", "numeric", "lowest monthly balance / mean balance", 0.5, 0, 5),
        FeatureSpec("savings_rate", "numeric", "(inflow - outflow) / inflow over 90d", 0.15, -5, 1),
        FeatureSpec("inflow_outflow_ratio", "numeric", "90d credits / 90d debits", 1.05, 0, 20),
        FeatureSpec("txn_count_90d", "numeric", "transaction count in trailing 90d", 60.0, 0, 5000),
        FeatureSpec("credit_utilisation", "numeric", "card spend / card limit", 0.3, 0, 3),
        FeatureSpec("prior_loans", "numeric", "number of previous loans taken", 0.0, 0, 30),
        FeatureSpec("prior_defaults", "numeric", "previous loans that defaulted", 0.0, 0, 20),
        FeatureSpec("emis_missed", "numeric", "historical missed EMI count", 0.0, 0, 60),
        FeatureSpec("overdraft_events_90d", "numeric", "times balance went negative in 90d", 0.0, 0, 100),
        FeatureSpec("loan_purpose_risk", "numeric", "prior risk weight of the loan type", 0.4, 0, 1),
    ),
)

# --------------------------------------------------------------------------- #
# 3. Anomaly detection — unsupervised, per-user behavioural drift
# --------------------------------------------------------------------------- #
ANOMALY_FEATURES = FeatureSet(
    name="anomaly",
    specs=(
        FeatureSpec("amount_log", "numeric", "log1p of transaction amount", 0.0, 0, 20),
        FeatureSpec("amount_to_user_mean", "numeric", "amount / user's mean txn amount", 1.0, 0, 200),
        FeatureSpec("amount_zscore", "numeric", "z-score of amount vs user history", 0.0, -20, 20),
        FeatureSpec("category_spend_ratio", "numeric", "week spend in category / weekly baseline", 1.0, 0, 100),
        FeatureSpec("category_zscore", "numeric", "z-score of amount within this category", 0.0, -20, 20),
        FeatureSpec("hour_sin", "cyclical", "sin(2*pi*hour/24)", 0.0, -1, 1),
        FeatureSpec("hour_cos", "cyclical", "cos(2*pi*hour/24)", 1.0, -1, 1),
        FeatureSpec("hour_deviation", "numeric", "|hour - user's usual hour| normalised", 0.0, 0, 1),
        FeatureSpec("txn_count_24h_ratio", "numeric", "24h txn count / user's daily average", 1.0, 0, 100),
        FeatureSpec("days_since_category", "numeric", "days since last txn in this category", 7.0, 0, 400),
        FeatureSpec("merchant_is_new", "binary", "merchant never transacted with before", 0.0, 0, 1),
        FeatureSpec("distinct_categories_7d", "numeric", "distinct categories in trailing 7d", 3.0, 0, 20),
        FeatureSpec("weekly_spend_ratio", "numeric", "this week total / median weekly total", 1.0, 0, 100),
        FeatureSpec("is_weekend", "binary", "Saturday or Sunday", 0.0, 0, 1),
    ),
)

FEATURE_SETS: dict[str, FeatureSet] = {
    "fraud": FRAUD_FEATURES,
    "credit": CREDIT_FEATURES,
    "anomaly": ANOMALY_FEATURES,
}


# --------------------------------------------------------------------------- #
# Encoding tables shared by the synthetic generator and the runtime builders
# --------------------------------------------------------------------------- #
CHANNEL_RISK: dict[str, float] = {
    "internal": 0.05,
    "neft": 0.25,
    "imps": 0.35,
    "upi": 0.30,
    "card": 0.55,
    "atm": 0.45,
    "system": 0.02,
}

HIGH_RISK_CATEGORIES = frozenset({"cash", "investment", "travel", "other"})

EMPLOYMENT_STABILITY: dict[str, float] = {
    "unemployed": 0.0,
    "student": 0.15,
    "gig": 0.30,
    "self_employed": 0.50,
    "contract": 0.60,
    "salaried": 0.85,
    "government": 1.00,
    "retired": 0.55,
}

HOUSING_STABILITY: dict[str, float] = {
    "rent": 0.25,
    "family": 0.45,
    "mortgage": 0.70,
    "own": 1.00,
}

LOAN_PURPOSE_RISK: dict[str, float] = {
    "home": 0.15,
    "auto": 0.30,
    "education": 0.40,
    "personal": 0.65,
    "business": 0.80,
}

RISK_BANDS: tuple[tuple[str, float, float, float], ...] = (
    # (band, max_probability_of_default, interest_rate_pct, income_multiple_cap)
    ("A", 0.05, 9.5, 12.0),
    ("B", 0.12, 12.5, 9.0),
    ("C", 0.25, 16.0, 6.0),
    ("D", 0.45, 21.0, 3.0),
    ("E", 1.01, 26.0, 1.0),
)


def band_for_probability(pd_value: float) -> tuple[str, float, float]:
    """Map a default probability to ``(band, suggested_rate, income_multiple)``."""
    for band, threshold, rate, multiple in RISK_BANDS:
        if pd_value <= threshold:
            return band, rate, multiple
    return "E", 26.0, 1.0


def score_from_probability(pd_value: float) -> int:
    """Map default probability to a 300-900 CIBIL-style score."""
    pd_value = min(max(pd_value, 1e-6), 1 - 1e-6)
    # Monotonic decreasing: PD 0.01 -> ~860, PD 0.5 -> ~570, PD 0.9 -> ~380
    scaled = 900 - 600 * (pd_value**0.6)
    return int(min(900, max(300, round(scaled))))
