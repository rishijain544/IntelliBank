"""Inference-layer tests.

Focus is on the properties that would silently break the product rather than on
line coverage: cold-start behaviour, rule/model precedence, monotonic credit
ranking, and the single-call latency budget.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.ml.features import CreditProfile, TxnContext, UserHistoryStats
from app.ml.inference import (
    COLD_START_MIN_TXNS,
    score_anomaly,
    score_credit,
    score_fraud,
    warm_up,
)


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warm_up()


@pytest.fixture
def established_history() -> UserHistoryStats:
    """A user with a settled behavioural baseline."""
    return UserHistoryStats(
        txn_count_total=220,
        mean_amount=900.0,
        std_amount=400.0,
        recent_amounts=[700, 850, 900, 1100, 950] * 20,
        txn_count_24h=3,
        amount_sum_24h=2600.0,
        mins_since_prev_txn=300.0,
        avg_daily_txn_count=3.0,
        known_devices=frozenset({"d1"}),
        known_cities=frozenset({"Mumbai"}),
        known_merchants=frozenset({"Swiggy", "BigBasket"}),
        category_counts={"dining": 80, "groceries": 60, "transport": 40},
        distinct_merchants_7d=5,
        distinct_categories_7d=3,
        usual_hour=13.0,
    )


@pytest.fixture
def routine_txn() -> TxnContext:
    return TxnContext(
        amount=850.0,
        occurred_at=datetime.now(UTC).replace(hour=13),
        category="dining",
        channel="card",
        merchant_name="Swiggy",
        device_fingerprint="d1",
        location_city="Mumbai",
        location_country="IN",
        account_balance=45_000.0,
        account_age_days=800,
    )


@pytest.fixture
def attack_txn() -> TxnContext:
    return TxnContext(
        amount=95_000.0,
        occurred_at=datetime.now(UTC).replace(hour=3),
        category="shopping",
        channel="card",
        merchant_name="Intl Merchant",
        device_fingerprint="attacker-device",
        location_city="Lagos",
        location_country="NG",
        account_balance=100_000.0,
        account_age_days=800,
    )


# --------------------------------------------------------------------------- fraud


def test_routine_transaction_is_allowed(routine_txn, established_history):
    d = score_fraud(routine_txn, established_history)
    assert d.action == "allow"
    assert not d.is_flagged
    assert d.risk_score < 0.5


def test_attack_outranks_routine(routine_txn, attack_txn, established_history):
    hostile = replace(established_history, failed_txn_24h=4, txn_count_1h=7)
    normal = score_fraud(routine_txn, established_history)
    attack = score_fraud(attack_txn, hostile)

    assert attack.risk_score > normal.risk_score
    assert attack.is_flagged
    assert attack.action in {"review", "block"}
    assert attack.triggered_rules, "hard policy breaches should surface as rules"


def test_cold_start_user_is_not_auto_blocked(routine_txn):
    """A first transaction must never be auto-blocked.

    Regression guard: the model treats "new device + new city + new merchant" as
    near-certain fraud because every training user had warm-up history.
    """
    fresh = UserHistoryStats()
    d = score_fraud(routine_txn, fresh)

    assert not d.auto_blocked
    assert d.action != "block"
    assert any("Limited history" in r for r in d.reasons)


def test_cold_start_damping_scales_with_history(routine_txn):
    """Confidence in the model should grow monotonically with observed history."""
    scores = []
    for n in (0, 5, COLD_START_MIN_TXNS):
        hist = UserHistoryStats(
            txn_count_total=n,
            mean_amount=900.0,
            std_amount=400.0,
            recent_amounts=[900.0] * max(n, 1),
        )
        scores.append(score_fraud(routine_txn, hist).risk_score)
    assert scores[0] <= scores[1] <= scores[2]


def test_rules_escalate_even_when_model_is_calm(established_history):
    """Foreign transactions must be flagged on policy regardless of model output."""
    foreign = TxnContext(
        amount=900.0,
        occurred_at=datetime.now(UTC).replace(hour=13),
        category="dining",
        channel="card",
        merchant_name="Swiggy",
        device_fingerprint="d1",
        location_city="Paris",
        location_country="FR",
        account_balance=45_000.0,
    )
    d = score_fraud(foreign, established_history)
    assert any("Foreign" in r for r in d.triggered_rules)
    assert d.risk_score > 0.0


def test_scoring_never_raises_on_degenerate_input():
    """Money movement must not depend on well-formed feature inputs."""
    weird = TxnContext(amount=0.0, occurred_at=datetime.now(UTC), category="unknown-cat", channel="unknown")
    d = score_fraud(weird, UserHistoryStats())
    assert 0.0 <= d.risk_score <= 1.0


# -------------------------------------------------------------------------- credit


def _strong() -> CreditProfile:
    return CreditProfile(
        age=38, annual_income=1_800_000, employment_status="government",
        employment_years=10, requested_amount=1_500_000, tenure_months=60,
        loan_type="auto", avg_balance=400_000, min_balance_ratio=0.8,
        prior_defaults=0, credit_utilisation=0.15, account_age_months=120,
    )


def _weak() -> CreditProfile:
    return CreditProfile(
        age=24, annual_income=240_000, employment_status="gig",
        employment_years=0.5, requested_amount=900_000, tenure_months=36,
        loan_type="business", existing_emi=9_000, prior_defaults=2,
        emis_missed=8, credit_utilisation=1.1, avg_balance=3_000,
        min_balance_ratio=0.05, overdraft_events_90d=6, account_age_months=6,
    )


def test_credit_ranks_strong_above_weak():
    strong, weak = score_credit(_strong()), score_credit(_weak())
    assert strong.score > weak.score
    assert strong.probability_of_default < weak.probability_of_default
    assert strong.suggested_rate <= weak.suggested_rate


def test_credit_score_stays_in_cibil_range():
    for profile in (_strong(), _weak(), CreditProfile()):
        d = score_credit(profile)
        assert 300 <= d.score <= 900
        assert d.risk_band in {"A", "B", "C", "D", "E"}
        assert d.decision in {"approve", "review", "reject"}


def test_approved_amount_never_exceeds_request():
    d = score_credit(_strong())
    assert d.approved_amount <= _strong().requested_amount


def test_rejected_application_prices_nothing():
    d = score_credit(_weak())
    if d.decision == "reject":
        assert d.approved_amount == 0
        assert d.emi_amount == 0


def test_prior_defaults_force_human_review():
    profile = replace(_strong(), prior_defaults=1)
    assert score_credit(profile).decision in {"review", "reject"}


def test_emi_affordability_is_enforced():
    """A large request on a small income must be capped, not approved in full."""
    overreach = CreditProfile(
        age=30, annual_income=300_000, employment_status="salaried",
        requested_amount=5_000_000, tenure_months=36, loan_type="personal",
    )
    d = score_credit(overreach)
    assert d.approved_amount < overreach.requested_amount


# ------------------------------------------------------------------------- anomaly


def test_category_spike_is_detected_and_explained(established_history):
    spiked = replace(established_history, category_week_spend=28_000.0, category_week_baseline=4_000.0)
    a = score_anomaly(
        TxnContext(amount=9_000.0, occurred_at=datetime.now(UTC), category="dining", merchant_name="Swiggy"),
        spiked,
    )
    assert a.anomaly_type == "category_spike"
    assert "dining" in a.message
    assert 0.0 <= a.anomaly_score <= 1.0


def test_routine_activity_is_not_anomalous(routine_txn, established_history):
    a = score_anomaly(routine_txn, established_history)
    assert not a.is_anomaly


# ------------------------------------------------------------------------- latency


def test_fraud_latency_within_budget(routine_txn, established_history):
    """p95 single-call latency must stay well inside the 200ms budget."""
    for _ in range(10):
        score_fraud(routine_txn, established_history)

    samples = []
    for _ in range(150):
        t0 = time.perf_counter()
        score_fraud(routine_txn, established_history)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()

    p95 = samples[int(len(samples) * 0.95)]
    assert p95 < 200, f"p95 latency {p95:.1f}ms exceeds the 200ms budget"


def test_credit_latency_within_budget():
    profile = _strong()
    for _ in range(5):
        score_credit(profile)
    samples = []
    for _ in range(60):
        t0 = time.perf_counter()
        score_credit(profile)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    assert samples[int(len(samples) * 0.95)] < 200
