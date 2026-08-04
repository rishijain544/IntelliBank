"""Runtime inference for the three models.

Responsibilities kept deliberately in one place:

* **Load each artifact once**, thread-safely, and reuse the estimator across
  requests. Re-reading a joblib file per request is the usual reason a "<200ms"
  claim quietly becomes 2s under load.
* **Vectorise through the shared ``FeatureSet``**, so serving applies the exact
  ordering, defaults and clipping used in training.
* **Explain every decision.** Tree SHAP comes free with XGBoost via
  ``pred_contribs``, so no extra dependency is needed to answer "why was this
  flagged" — which the Fraud Center and loan-decision UIs both require.
* **Degrade instead of failing.** If an artifact is missing the fraud path falls
  back to the deterministic rule engine and reports ``model_available=False``
  rather than 500-ing a transfer. Money movement must not depend on a model file.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from app.core.config import settings
from app.ml.features import (
    CreditProfile,
    TxnContext,
    UserHistoryStats,
    build_anomaly_features,
    build_credit_features,
    build_fraud_features,
    emi_for,
)
from app.ml.registry import ModelArtifact, artifact_exists, load_artifact
from app.ml.schema import (
    ANOMALY_FEATURES,
    CREDIT_FEATURES,
    FRAUD_FEATURES,
    band_for_probability,
    score_from_probability,
)
from app.ml.train_anomaly import MODEL_NAME as ANOMALY_MODEL
from app.ml.train_anomaly import normalise_scores
from app.ml.train_credit import MODEL_NAME as CREDIT_MODEL
from app.ml.train_fraud import MODEL_NAME as FRAUD_MODEL

# Number of prior transactions before the fraud model's output is trusted at full
# weight. Below this the account is "thin file" and scoring leans on policy rules.
COLD_START_MIN_TXNS = 15

# --------------------------------------------------------------------------- #
# Artifact cache
# --------------------------------------------------------------------------- #

_cache: dict[str, ModelArtifact | None] = {}
_lock = threading.RLock()


def get_model(name: str) -> ModelArtifact | None:
    """Return a cached artifact, or ``None`` when it has not been trained yet."""
    with _lock:
        if name not in _cache:
            try:
                _cache[name] = load_artifact(name) if artifact_exists(name) else None
            except Exception:  # noqa: BLE001 - a corrupt artifact must not break the API
                _cache[name] = None
        return _cache[name]


def reload_models() -> dict[str, bool]:
    """Drop the cache so a retrained artifact is picked up without a restart."""
    with _lock:
        _cache.clear()
    return {
        name: get_model(name) is not None
        for name in (FRAUD_MODEL, CREDIT_MODEL, ANOMALY_MODEL)
    }


def models_status() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, name in (
        ("fraud", FRAUD_MODEL),
        ("credit", CREDIT_MODEL),
        ("anomaly", ANOMALY_MODEL),
    ):
        art = get_model(name)
        out[key] = {
            "loaded": art is not None,
            "name": name,
            "version": art.version if art else None,
            "trained_at": art.trained_at if art else None,
            "threshold": art.threshold if art else None,
            "n_features": art.n_features if art else None,
            "metrics": art.metrics.get("test", {}) if art else {},
            "latency_benchmark": art.metrics.get("latency", {}) if art else {},
        }
    return out


# --------------------------------------------------------------------------- #
# Tree SHAP explanations
# --------------------------------------------------------------------------- #


def _unwrap_booster(estimator: Any):
    """Reach the XGBoost booster through a calibration wrapper, if present."""
    if hasattr(estimator, "get_booster"):
        return estimator.get_booster()
    inner = getattr(estimator, "calibrated_classifiers_", None)
    if inner:
        base = getattr(inner[0], "estimator", None)
        if base is not None and hasattr(base, "get_booster"):
            return base.get_booster()
    return None


def _shap_factors(
    artifact: ModelArtifact,
    row: np.ndarray,
    *,
    top_n: int = 5,
    friendly: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-feature SHAP contributions for a single prediction.

    Exact for tree ensembles (XGBoost ships TreeSHAP), so the explanation is the
    model's real reasoning rather than a post-hoc approximation.
    """
    booster = _unwrap_booster(artifact.estimator)
    if booster is None:
        return []
    try:
        import xgboost as xgb

        dm = xgb.DMatrix(row, feature_names=artifact.feature_names)
        contribs = booster.predict(dm, pred_contribs=True)[0]
    except Exception:  # noqa: BLE001 - explanations are best-effort, never fatal
        return []

    # Final element is the bias/base term, not a feature.
    values = contribs[: len(artifact.feature_names)]
    order = np.argsort(-np.abs(values))[:top_n]
    factors: list[dict[str, Any]] = []
    for i in order:
        name = artifact.feature_names[int(i)]
        contribution = float(values[int(i)])
        if abs(contribution) < 1e-6:
            continue
        factors.append(
            {
                "feature": name,
                "label": (friendly or {}).get(name, name.replace("_", " ")),
                "value": round(float(row[0][int(i)]), 4),
                "contribution": round(contribution, 5),
                "direction": "increases risk" if contribution > 0 else "reduces risk",
            }
        )
    return factors


FRAUD_LABELS: dict[str, str] = {
    "amount_log": "Transaction amount",
    "amount_to_user_mean": "Amount vs your usual",
    "amount_zscore": "Amount deviation",
    "amount_to_balance": "Share of available balance",
    "amount_percentile": "Amount percentile",
    "is_night": "Late-night activity",
    "txn_count_1h": "Transactions in last hour",
    "txn_count_24h": "Transactions in last 24h",
    "amount_sum_24h_log": "24h spend volume",
    "mins_since_prev_txn": "Time since previous transaction",
    "device_is_new": "Unrecognised device",
    "location_is_new": "New location",
    "is_foreign": "Foreign country",
    "merchant_is_new": "First time at this merchant",
    "category_is_rare": "Unusual spending category",
    "channel_risk": "Payment channel risk",
    "account_age_days": "Account age",
    "failed_txn_24h": "Recent declined attempts",
    "is_round_amount": "Round-figure amount",
    "high_risk_category": "High-risk category",
    "distinct_merchants_7d": "Merchant variety (7d)",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "is_weekend": "Weekend activity",
}

CREDIT_LABELS: dict[str, str] = {
    "annual_income_log": "Declared income",
    "debt_to_income": "Total debt-to-income",
    "emi_to_income": "EMI affordability",
    "loan_to_income": "Loan size vs income",
    "prior_defaults": "Previous defaults",
    "emis_missed": "Missed EMI history",
    "employment_stability": "Employment stability",
    "employment_years": "Years employed",
    "credit_utilisation": "Credit utilisation",
    "savings_rate": "Savings rate",
    "min_balance_ratio": "Lowest balance held",
    "balance_volatility": "Balance volatility",
    "avg_balance_log": "Average balance",
    "account_age_months": "Banking relationship length",
    "overdraft_events_90d": "Overdraft events",
    "existing_emi_log": "Existing obligations",
    "tenure_months": "Requested tenure",
    "dependents": "Dependents",
    "age": "Age",
    "housing_stability": "Housing stability",
    "loan_purpose_risk": "Loan purpose",
    "num_accounts": "Accounts held",
    "txn_count_90d": "Recent account activity",
    "inflow_outflow_ratio": "Income vs spending",
    "loan_amount_log": "Requested amount",
}


# --------------------------------------------------------------------------- #
# 1. Fraud scoring — rule/ML hybrid
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FraudDecision:
    risk_score: float
    action: str  # allow | review | block
    is_flagged: bool
    auto_blocked: bool
    severity: str
    reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    model_name: str = FRAUD_MODEL
    model_version: str = "0.0.0"
    model_available: bool = True
    decision_source: str = "hybrid"
    latency_ms: float = 0.0


# Deterministic rules run alongside the model. They exist for two reasons: they
# catch patterns with too few historical examples for the model to have learned,
# and they give compliance a human-readable trail that a probability cannot.
def _evaluate_rules(ctx: TxnContext, hist: UserHistoryStats, feats: dict[str, float]) -> tuple[list[str], float]:
    rules: list[str] = []
    escalation = 0.0

    if feats["is_foreign"] >= 1.0:
        rules.append("Foreign-country transaction")
        escalation = max(escalation, 0.35)
    if feats["device_is_new"] >= 1.0 and feats["amount_to_user_mean"] > 3:
        rules.append("Large amount from an unrecognised device")
        escalation = max(escalation, 0.45)
    if feats["txn_count_1h"] >= 6:
        rules.append(f"Velocity spike: {int(feats['txn_count_1h'])} transactions in an hour")
        escalation = max(escalation, 0.40)
    if feats["failed_txn_24h"] >= 3:
        rules.append(f"{int(feats['failed_txn_24h'])} declined attempts in 24h")
        escalation = max(escalation, 0.50)
    if feats["is_night"] >= 1.0 and feats["amount_to_user_mean"] > 5:
        rules.append("High-value transaction outside normal hours")
        escalation = max(escalation, 0.30)
    if feats["amount_to_balance"] > 0.9:
        rules.append("Transaction drains almost the entire balance")
        escalation = max(escalation, 0.35)
    if feats["device_is_new"] >= 1.0 and feats["location_is_new"] >= 1.0 and feats["is_night"] >= 1.0:
        rules.append("New device, new location, late at night")
        escalation = max(escalation, 0.55)
    return rules, escalation


def _severity_for(score: float) -> str:
    if score >= 0.90:
        return "critical"
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def score_fraud(ctx: TxnContext, hist: UserHistoryStats) -> FraudDecision:
    """Score one transaction. Never raises: a scoring failure must not block money."""
    t0 = time.perf_counter()
    feats = build_fraud_features(ctx, hist)
    rules, rule_escalation = _evaluate_rules(ctx, hist, feats)

    artifact = get_model(FRAUD_MODEL)
    reasons: list[str] = list(rules)
    top_factors: list[dict[str, Any]] = []

    if artifact is None:
        # Rules-only fallback keeps transfers working before the first training run.
        score = min(rule_escalation, 0.95)
        latency = (time.perf_counter() - t0) * 1000
        return FraudDecision(
            risk_score=round(score, 6),
            action="review" if score >= settings.FRAUD_REVIEW_THRESHOLD else "allow",
            is_flagged=score >= settings.FRAUD_REVIEW_THRESHOLD,
            auto_blocked=False,
            severity=_severity_for(score),
            reasons=reasons or ["No risk indicators detected"],
            triggered_rules=rules,
            features=feats,
            model_available=False,
            decision_source="rule",
            latency_ms=round(latency, 3),
        )

    row = np.asarray([FRAUD_FEATURES.vectorise(feats)], dtype=np.float32)
    try:
        model_score = float(artifact.estimator.predict_proba(row)[0][1])
    except Exception:  # noqa: BLE001
        model_score = min(rule_escalation, 0.9)

    # ---- cold-start guard ----------------------------------------------------
    # Every simulated training user has warm-up history, so the model never saw a
    # *legitimate* "new device + new city + new merchant" row and treats that
    # combination as near-certain fraud. Real customers genuinely start with no
    # history, so trusting the model here would block every first transaction.
    # Below the confidence floor the score is damped toward the rule signal, which
    # mirrors how banks underwrite new accounts on policy rules until a
    # behavioural baseline exists.
    cold_start = hist.txn_count_total < COLD_START_MIN_TXNS
    if cold_start:
        confidence = hist.txn_count_total / COLD_START_MIN_TXNS
        model_score = model_score * confidence
        reasons.append(
            f"Limited history ({hist.txn_count_total} prior transactions): "
            "scored conservatively on policy rules"
        )

    # Rules can only escalate, never suppress: a model saying "safe" must not
    # override a hard policy breach such as a foreign card-not-present charge.
    risk = max(model_score, rule_escalation * 0.9)

    if model_score >= artifact.threshold:
        reasons.insert(0, f"Model risk score {model_score:.0%} exceeds the review threshold")
    top_factors = _shap_factors(artifact, row, friendly=FRAUD_LABELS)

    # ---- alerting vs. money movement are separate decisions -------------------
    # The model's own threshold (~0.15) is tuned for precision on the *alert
    # queue* — it answers "should an analyst look at this?". It must not decide
    # whether to freeze a customer's payment, or every first transfer to a new
    # payee gets held, which is what happens when the two are conflated.
    #
    # Holding money follows the configured business policy instead:
    #   risk >= FRAUD_BLOCK_THRESHOLD (0.90) -> block outright
    #   risk >= FRAUD_REVIEW_THRESHOLD (0.55) -> hold for review
    #   otherwise                             -> allow, alerting if the model flags it
    flagged = risk >= min(settings.FRAUD_REVIEW_THRESHOLD, artifact.threshold)
    block = risk >= settings.FRAUD_BLOCK_THRESHOLD
    hold = risk >= settings.FRAUD_REVIEW_THRESHOLD

    # A thin-file customer is never auto-blocked; ambiguity goes to a human.
    if cold_start and block:
        block = False
        hold = True

    action = "block" if block else ("review" if hold else "allow")

    latency = (time.perf_counter() - t0) * 1000
    return FraudDecision(
        risk_score=round(risk, 6),
        action=action,
        # Still raise an alert on a moderate score even when the payment is
        # allowed through: the analyst queue is how mislabelled cases get caught
        # and fed back into retraining.
        is_flagged=flagged,
        auto_blocked=block,
        severity=_severity_for(risk),
        reasons=reasons or ["No risk indicators detected"],
        triggered_rules=rules,
        top_factors=top_factors,
        features=feats,
        model_version=artifact.version,
        model_available=True,
        decision_source="hybrid" if rules else "model",
        latency_ms=round(latency, 3),
    )


# --------------------------------------------------------------------------- #
# 2. Credit scoring
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CreditDecision:
    score: int
    probability_of_default: float
    risk_band: str
    decision: str  # approve | review | reject
    suggested_rate: float
    max_eligible_amount: float
    approved_amount: float
    emi_amount: float
    total_payable: float
    processing_fee: float
    reasons: list[str] = field(default_factory=list)
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    model_name: str = CREDIT_MODEL
    model_version: str = "0.0.0"
    model_available: bool = True
    latency_ms: float = 0.0


def score_credit(profile: CreditProfile) -> CreditDecision:
    """Price a loan application from the model's calibrated default probability."""
    t0 = time.perf_counter()
    feats = build_credit_features(profile)
    artifact = get_model(CREDIT_MODEL)

    if artifact is None:
        # Transparent heuristic stand-in so the loan flow is demoable untrained.
        dti = feats["debt_to_income"]
        pd_value = float(min(0.85, max(0.02, 0.10 + 0.35 * max(0.0, dti - 0.4) + 0.25 * profile.prior_defaults)))
        model_available = False
        version = "0.0.0"
        top_factors: list[dict[str, Any]] = []
    else:
        row = np.asarray([CREDIT_FEATURES.vectorise(feats)], dtype=np.float32)
        try:
            pd_value = float(artifact.estimator.predict_proba(row)[0][1])
        except Exception:  # noqa: BLE001
            pd_value = 0.35
        model_available = True
        version = artifact.version
        top_factors = _shap_factors(artifact, row, top_n=6, friendly=CREDIT_LABELS)

    band, rate, income_multiple = band_for_probability(pd_value)
    score = score_from_probability(pd_value)

    # Eligibility is capped by both a band-specific income multiple and residual
    # affordability, so a high score alone cannot approve an unaffordable EMI.
    monthly_income = max(profile.annual_income / 12.0, 1.0)
    disposable = max(monthly_income * 0.55 - profile.existing_emi, 0.0)
    affordable_emi = disposable
    r = rate / 12.0 / 100.0
    n = max(profile.tenure_months, 1)
    factor = (1 + r) ** n
    affordable_principal = affordable_emi * (factor - 1) / (r * factor) if r > 0 else affordable_emi * n
    max_eligible = float(max(0.0, min(profile.annual_income * income_multiple, affordable_principal)))

    reasons: list[str] = []
    if band in ("A", "B"):
        decision = "approve"
    elif band == "C":
        decision = "approve" if pd_value <= 0.18 else "review"
    elif band == "D":
        decision = "review"
    else:
        decision = "reject"
        reasons.append("Default risk is above our lending appetite")

    if profile.prior_defaults > 0:
        reasons.append(f"{profile.prior_defaults} prior default(s) on record")
        if decision == "approve":
            decision = "review"
    if feats["debt_to_income"] > 0.60:
        reasons.append("Debt-to-income above the 60% policy limit")
        if decision == "approve":
            decision = "review"
    if max_eligible < profile.requested_amount * 0.5 and decision != "reject":
        reasons.append("Requested amount exceeds affordability; a lower amount may be offered")

    approved = 0.0 if decision == "reject" else float(min(profile.requested_amount, max_eligible))
    emi = emi_for(approved, rate, profile.tenure_months) if approved > 0 else 0.0
    total = emi * profile.tenure_months if approved > 0 else 0.0
    fee = round(approved * 0.01, 2) if approved > 0 else 0.0

    if not reasons:
        reasons.append(f"Risk band {band}: default probability {pd_value:.1%}")

    latency = (time.perf_counter() - t0) * 1000
    return CreditDecision(
        score=score,
        probability_of_default=round(pd_value, 6),
        risk_band=band,
        decision=decision,
        suggested_rate=rate,
        max_eligible_amount=round(max_eligible, 2),
        approved_amount=round(approved, 2),
        emi_amount=round(emi, 2),
        total_payable=round(total, 2),
        processing_fee=fee,
        reasons=reasons,
        top_factors=top_factors,
        features=feats,
        model_version=version,
        model_available=model_available,
        latency_ms=round(latency, 3),
    )


# --------------------------------------------------------------------------- #
# 3. Anomaly scoring
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AnomalyResult:
    anomaly_score: float
    is_anomaly: bool
    severity: str
    anomaly_type: str
    title: str
    message: str
    features: dict[str, float] = field(default_factory=dict)
    model_name: str = ANOMALY_MODEL
    model_version: str = "0.0.0"
    model_available: bool = True
    latency_ms: float = 0.0
    baseline_value: float | None = None
    observed_value: float | None = None
    deviation_ratio: float | None = None


def _classify_anomaly(feats: dict[str, float], ctx: TxnContext) -> tuple[str, str, str]:
    """Turn the dominant deviation into plain-language copy for the Insights feed.

    The score alone is not actionable; users need "you spent 3x more on dining
    this week", which is why the type is derived from whichever signal deviates most.
    """
    cat = ctx.category.replace("_", " ")
    ratio = feats.get("category_spend_ratio", 1.0)
    if ratio >= 2.5:
        return (
            "category_spike",
            f"Unusual {cat} spending",
            f"You spent about {ratio:.1f}x your usual weekly amount on {cat}.",
        )
    if feats.get("amount_to_user_mean", 1.0) >= 5:
        return (
            "unusual_amount",
            "Unusually large transaction",
            f"This {cat} transaction is roughly {feats['amount_to_user_mean']:.1f}x your typical amount.",
        )
    if feats.get("txn_count_24h_ratio", 1.0) >= 3:
        return (
            "velocity",
            "Higher activity than usual",
            f"Your transaction count today is about {feats['txn_count_24h_ratio']:.1f}x your daily average.",
        )
    if feats.get("hour_deviation", 0.0) >= 0.5:
        return (
            "time_of_day",
            "Activity at an unusual hour",
            f"This {cat} transaction happened well outside your normal spending hours.",
        )
    if feats.get("merchant_is_new", 0.0) >= 1.0:
        return (
            "new_merchant",
            "First transaction with a new merchant",
            f"This is your first {cat} transaction with {ctx.merchant_name or 'this merchant'}.",
        )
    return (
        "pattern_shift",
        "Spending pattern shift",
        f"This {cat} transaction does not match your recent habits.",
    )


def score_anomaly(ctx: TxnContext, hist: UserHistoryStats) -> AnomalyResult:
    t0 = time.perf_counter()
    feats = build_anomaly_features(ctx, hist)
    artifact = get_model(ANOMALY_MODEL)
    a_type, title, message = _classify_anomaly(feats, ctx)

    if artifact is None:
        # Without the model, only flag on an unambiguous statistical deviation.
        proxy = min(1.0, max(0.0, (abs(feats.get("amount_zscore", 0.0)) - 2.0) / 4.0))
        latency = (time.perf_counter() - t0) * 1000
        return AnomalyResult(
            anomaly_score=round(proxy, 6),
            is_anomaly=proxy >= 0.5,
            severity="low" if proxy < 0.6 else "medium",
            anomaly_type=a_type,
            title=title,
            message=message,
            features=feats,
            model_available=False,
            latency_ms=round(latency, 3),
        )

    row = np.asarray([ANOMALY_FEATURES.vectorise(feats)], dtype=np.float64)
    try:
        scaled = artifact.scaler.transform(row) if artifact.scaler is not None else row
        raw = artifact.estimator.score_samples(scaled)
        norm = artifact.metrics.get("score_normalisation", {})
        score = float(
            normalise_scores(
                raw,
                float(norm.get("ref_min", -0.6)),
                float(norm.get("ref_max", -0.3)),
            )[0]
        )
    except Exception:  # noqa: BLE001
        score = 0.0

    threshold = max(artifact.threshold, settings.ANOMALY_ALERT_THRESHOLD * 0.5)
    is_anom = score >= threshold
    severity = "high" if score >= threshold + 0.15 else ("medium" if is_anom else "low")

    latency = (time.perf_counter() - t0) * 1000
    return AnomalyResult(
        anomaly_score=round(score, 6),
        is_anomaly=is_anom,
        severity=severity,
        anomaly_type=a_type,
        title=title,
        message=message,
        features=feats,
        model_version=artifact.version,
        model_available=True,
        latency_ms=round(latency, 3),
        baseline_value=round(hist.category_week_baseline, 2),
        observed_value=round(hist.category_week_spend, 2),
        deviation_ratio=round(feats.get("category_spend_ratio", 1.0), 3),
    )


def warm_up() -> dict[str, Any]:
    """Load artifacts and run one throwaway prediction each at startup.

    First-call cost in XGBoost/sklearn (thread pools, lazy imports) is large
    enough to blow a latency budget, so it is paid during boot rather than by
    whichever customer happens to transact first.
    """
    from datetime import UTC

    status = reload_models()
    ctx = TxnContext(amount=1000.0, occurred_at=datetime.now(UTC), category="dining", channel="card")
    hist = UserHistoryStats()
    try:
        score_fraud(ctx, hist)
        score_anomaly(ctx, hist)
        score_credit(CreditProfile())
    except Exception:  # noqa: BLE001 - warm-up must never prevent boot
        pass
    return status
