"""Pydantic request/response schemas.

Validation lives here rather than in the routers so that every endpoint shares
the same money, ID and password rules. Amounts are typed as ``Decimal`` with a
2dp constraint: using ``float`` for currency is how rounding drift gets into a
ledger.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

from app.core.security import password_strength_issues


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _validate_amount(v: Decimal) -> Decimal:
    if v <= 0:
        raise ValueError("amount must be greater than zero")
    if v > Decimal("100000000"):
        raise ValueError("amount exceeds the per-transaction ceiling")
    if v.as_tuple().exponent < -2:
        raise ValueError("amount cannot have more than 2 decimal places")
    return v


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=2, max_length=150)
    phone: str | None = Field(default=None, max_length=20)
    date_of_birth: date | None = None

    @field_validator("password")
    @classmethod
    def _strong_password(cls, v: str) -> str:
        issues = password_strength_issues(v)
        if issues:
            raise ValueError("password " + "; ".join(issues))
        return v

    @field_validator("phone")
    @classmethod
    def _clean_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        digits = "".join(ch for ch in v if ch.isdigit() or ch == "+")
        if len(digits) < 8:
            raise ValueError("phone number looks too short")
        return digits

    @field_validator("date_of_birth")
    @classmethod
    def _adult(cls, v: date | None) -> date | None:
        if v is None:
            return None
        age = (date.today() - v).days / 365.25
        if age < 18:
            raise ValueError("account holders must be at least 18 years old")
        if age > 120:
            raise ValueError("date of birth is not plausible")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    totp_code: str | None = Field(default=None, max_length=10)
    device_name: str | None = Field(default=None, max_length=120)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class TwoFactorChallenge(BaseModel):
    """Returned instead of tokens when the account has 2FA enabled."""

    requires_2fa: Literal[True] = True
    detail: str = "Enter the 6-digit code from your authenticator app"


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _strong(cls, v: str) -> str:
        issues = password_strength_issues(v)
        if issues:
            raise ValueError("password " + "; ".join(issues))
        return v

    @model_validator(mode="after")
    def _different(self):
        if self.current_password == self.new_password:
            raise ValueError("new password must differ from the current one")
        return self


class TwoFactorSetupResponse(BaseModel):
    secret: str
    provisioning_uri: str
    detail: str = "Scan the QR code, then confirm with a generated code to enable 2FA"


class TwoFactorVerifyRequest(BaseModel):
    code: str = Field(min_length=6, max_length=10)


# --------------------------------------------------------------------------- #
# KYC
# --------------------------------------------------------------------------- #


class KycSubmitRequest(BaseModel):
    """Simulated KYC. IDs are hashed server-side and never stored in the clear."""

    pan: str = Field(min_length=10, max_length=10, description="PAN-format ID, e.g. ABCDE1234F")
    aadhaar: str = Field(min_length=12, max_length=12, description="12-digit Aadhaar-style ID")
    document_type: Literal["passport", "driving_licence", "voter_id", "aadhaar_card"]
    document_name: str = Field(min_length=1, max_length=255)
    address_line1: str = Field(min_length=3, max_length=255)
    address_line2: str | None = Field(default=None, max_length=255)
    city: str = Field(min_length=2, max_length=100)
    state: str = Field(min_length=2, max_length=100)
    postal_code: str = Field(min_length=4, max_length=20)
    annual_income: Decimal = Field(gt=0, le=Decimal("1000000000"))
    employment_status: Literal[
        "salaried", "self_employed", "government", "contract", "gig", "student", "unemployed", "retired"
    ]
    employment_years: float = Field(ge=0, le=60)
    dependents: int = Field(default=0, ge=0, le=15)
    housing_status: Literal["rent", "own", "mortgage", "family"] = "rent"

    @field_validator("pan")
    @classmethod
    def _pan_format(cls, v: str) -> str:
        v = v.upper().strip()
        if not (v[:5].isalpha() and v[5:9].isdigit() and v[9].isalpha()):
            raise ValueError("PAN must match the pattern AAAAA9999A")
        return v

    @field_validator("aadhaar")
    @classmethod
    def _aadhaar_digits(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if not v.isdigit() or len(v) != 12:
            raise ValueError("Aadhaar-style ID must be exactly 12 digits")
        return v


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #


class UserResponse(ORMModel):
    id: int
    email: EmailStr
    full_name: str
    phone: str | None = None
    role: str
    status: str
    kyc_status: str
    two_factor_enabled: bool
    city: str | None = None
    country: str
    annual_income: Decimal | None = None
    employment_status: str | None = None
    pan_masked: str | None = None
    created_at: datetime
    last_login_at: datetime | None = None


class UserSummary(ORMModel):
    """Compact projection for admin tables."""

    id: int
    email: EmailStr
    full_name: str
    role: str
    status: str
    kyc_status: str
    created_at: datetime
    last_login_at: datetime | None = None


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=150)
    phone: str | None = Field(default=None, max_length=20)
    address_line1: str | None = Field(default=None, max_length=255)
    city: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)
    postal_code: str | None = Field(default=None, max_length=20)
    annual_income: Decimal | None = Field(default=None, gt=0)
    employment_status: str | None = None
    dependents: int | None = Field(default=None, ge=0, le=15)


class NotificationPrefsRequest(BaseModel):
    notify_email: bool | None = None
    notify_sms: bool | None = None
    notify_large_txn: bool | None = None
    notify_login: bool | None = None
    notify_marketing: bool | None = None


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


class AccountCreateRequest(BaseModel):
    account_type: Literal["savings", "current", "fixed_deposit", "salary"] = "savings"
    nickname: str | None = Field(default=None, max_length=80)
    initial_deposit: Decimal = Field(default=Decimal("0"), ge=0)

    @field_validator("initial_deposit")
    @classmethod
    def _dp(cls, v: Decimal) -> Decimal:
        if v.as_tuple().exponent < -2:
            raise ValueError("initial deposit cannot have more than 2 decimal places")
        return v


class AccountResponse(ORMModel):
    id: int
    account_number: str
    ifsc_code: str
    nickname: str | None = None
    account_type: str
    status: str
    currency: str
    balance: Decimal
    hold_amount: Decimal
    available_balance: Decimal
    overdraft_limit: Decimal
    interest_rate: float
    is_primary: bool
    opened_on: date | None = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #


class TransactionResponse(ORMModel):
    id: int
    reference: str
    account_id: int
    txn_type: str
    channel: str
    status: str
    amount: Decimal
    fee: Decimal
    signed_amount: Decimal
    balance_after: Decimal | None = None
    currency: str
    description: str | None = None
    merchant_name: str | None = None
    merchant_category: str
    counterparty_name: str | None = None
    counterparty_account_number: str | None = None
    location_city: str | None = None
    # Exposed because a foreign transaction is one of the strongest fraud
    # signals, and the review queue needs it to judge an alert.
    location_country: str | None = None
    is_foreign: bool = False
    fraud_score: float | None = None
    anomaly_score: float | None = None
    is_flagged: bool
    occurred_at: datetime
    created_at: datetime


class TransactionFilters(BaseModel):
    account_id: int | None = None
    txn_type: str | None = None
    status: str | None = None
    category: str | None = None
    channel: str | None = None
    min_amount: Decimal | None = Field(default=None, ge=0)
    max_amount: Decimal | None = Field(default=None, ge=0)
    date_from: datetime | None = None
    date_to: datetime | None = None
    search: str | None = Field(default=None, max_length=120)
    flagged_only: bool = False

    @model_validator(mode="after")
    def _ranges(self):
        if self.min_amount is not None and self.max_amount is not None:
            if self.min_amount > self.max_amount:
                raise ValueError("min_amount cannot exceed max_amount")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be after date_to")
        return self


class DepositRequest(BaseModel):
    """Simulated cash/salary credit, used to fund demo accounts."""

    account_id: int
    amount: Decimal = Field(gt=0)
    description: str | None = Field(default=None, max_length=255)

    _check = field_validator("amount")(_validate_amount)


# --------------------------------------------------------------------------- #
# Transfers
# --------------------------------------------------------------------------- #


class InternalTransferRequest(BaseModel):
    """Transfer between two accounts, either own-to-own or to another customer."""

    from_account_id: int
    to_account_number: str = Field(min_length=6, max_length=34)
    amount: Decimal = Field(gt=0)
    description: str | None = Field(default=None, max_length=255)
    category: str = "transfer"

    _check = field_validator("amount")(_validate_amount)


class ExternalTransferRequest(BaseModel):
    """Simulated NEFT/IMPS transfer to a beneficiary at another bank."""

    from_account_id: int
    beneficiary_id: int
    amount: Decimal = Field(gt=0)
    channel: Literal["neft", "imps", "upi"] = "imps"
    description: str | None = Field(default=None, max_length=255)

    _check = field_validator("amount")(_validate_amount)


class TransferResponse(BaseModel):
    """Transfer outcome, including the fraud decision that gated it."""

    transaction: TransactionResponse
    status: str
    message: str
    fraud: "FraudAssessment"


class FraudAssessment(BaseModel):
    risk_score: float
    action: str
    severity: str
    is_flagged: bool
    auto_blocked: bool
    reasons: list[str] = []
    triggered_rules: list[str] = []
    top_factors: list[dict[str, Any]] = []
    model_name: str
    model_version: str
    model_available: bool
    decision_source: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Beneficiaries
# --------------------------------------------------------------------------- #


class BeneficiaryCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=150)
    nickname: str | None = Field(default=None, max_length=80)
    account_number: str = Field(min_length=6, max_length=34)
    ifsc_code: str = Field(min_length=6, max_length=15)
    bank_name: str = Field(default="IntelliBank", max_length=120)
    transfer_limit: Decimal | None = Field(default=None, gt=0)

    @field_validator("account_number")
    @classmethod
    def _digits(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if not v.isalnum():
            raise ValueError("account number must be alphanumeric")
        return v

    @field_validator("ifsc_code")
    @classmethod
    def _ifsc(cls, v: str) -> str:
        return v.strip().upper()


class BeneficiaryResponse(ORMModel):
    id: int
    name: str
    nickname: str | None = None
    account_number: str
    ifsc_code: str
    bank_name: str
    is_internal: bool
    is_verified: bool
    transfer_limit: Decimal | None = None
    last_used_at: datetime | None = None
    usage_count: int
    created_at: datetime


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


class CardCreateRequest(BaseModel):
    account_id: int
    card_type: Literal["virtual_debit", "virtual_credit"] = "virtual_debit"
    daily_limit: Decimal = Field(default=Decimal("50000"), gt=0)
    per_txn_limit: Decimal = Field(default=Decimal("25000"), gt=0)
    monthly_limit: Decimal = Field(default=Decimal("200000"), gt=0)

    @model_validator(mode="after")
    def _limit_hierarchy(self):
        if self.per_txn_limit > self.daily_limit:
            raise ValueError("per-transaction limit cannot exceed the daily limit")
        if self.daily_limit > self.monthly_limit:
            raise ValueError("daily limit cannot exceed the monthly limit")
        return self


class CardResponse(ORMModel):
    id: int
    account_id: int
    card_last4: str
    masked_number: str
    card_network: str
    card_type: str
    status: str
    cardholder_name: str
    expiry_month: int
    expiry_year: int
    daily_limit: Decimal
    per_txn_limit: Decimal
    monthly_limit: Decimal
    credit_limit: Decimal | None = None
    online_enabled: bool
    international_enabled: bool
    contactless_enabled: bool
    atm_enabled: bool
    freeze_reason: str | None = None
    created_at: datetime


class CardLimitsRequest(BaseModel):
    daily_limit: Decimal | None = Field(default=None, gt=0)
    per_txn_limit: Decimal | None = Field(default=None, gt=0)
    monthly_limit: Decimal | None = Field(default=None, gt=0)
    online_enabled: bool | None = None
    international_enabled: bool | None = None
    contactless_enabled: bool | None = None
    atm_enabled: bool | None = None


class CardFreezeRequest(BaseModel):
    freeze: bool
    reason: str | None = Field(default=None, max_length=255)


# --------------------------------------------------------------------------- #
# Loans
# --------------------------------------------------------------------------- #


class LoanEligibilityRequest(BaseModel):
    """Dry-run scoring: returns a decision without creating an application."""

    loan_type: Literal["personal", "home", "auto", "education", "business"] = "personal"
    amount: Decimal = Field(gt=0)
    tenure_months: int = Field(ge=3, le=360)
    declared_income: Decimal | None = Field(default=None, gt=0)

    _check = field_validator("amount")(_validate_amount)


class LoanApplyRequest(LoanEligibilityRequest):
    purpose: str | None = Field(default=None, max_length=255)
    disbursement_account_id: int | None = None


class CreditScoreResponse(BaseModel):
    score: int = Field(ge=300, le=900)
    probability_of_default: float
    risk_band: str
    decision: str
    suggested_rate: float
    max_eligible_amount: Decimal
    approved_amount: Decimal
    emi_amount: Decimal
    total_payable: Decimal
    processing_fee: Decimal
    reasons: list[str] = []
    top_factors: list[dict[str, Any]] = []
    model_name: str
    model_version: str
    model_available: bool
    latency_ms: float


class LoanResponse(ORMModel):
    id: int
    application_ref: str
    loan_type: str
    status: str
    requested_amount: Decimal
    approved_amount: Decimal | None = None
    tenure_months: int
    purpose: str | None = None
    interest_rate: float | None = None
    emi_amount: Decimal | None = None
    total_payable: Decimal | None = None
    processing_fee: Decimal | None = None
    decision_source: str
    decision_reason: str | None = None
    decided_at: datetime | None = None
    manual_override: bool
    disbursed_at: datetime | None = None
    outstanding_principal: Decimal | None = None
    emis_paid: int
    emis_missed: int
    created_at: datetime


class LoanApplicationResult(BaseModel):
    loan: LoanResponse
    credit: CreditScoreResponse


# --------------------------------------------------------------------------- #
# Fraud & anomaly
# --------------------------------------------------------------------------- #


class FraudAlertResponse(ORMModel):
    id: int
    alert_ref: str
    transaction_id: int
    risk_score: float
    severity: str
    status: str
    decision_source: str
    auto_blocked: bool
    reasons: list[str] | None = None
    top_factors: list[dict[str, Any]] | None = None
    triggered_rules: list[str] | None = None
    model_name: str
    model_version: str
    inference_latency_ms: float | None = None
    customer_response: str | None = None
    customer_note: str | None = None
    review_note: str | None = None
    final_label: bool | None = None
    created_at: datetime
    transaction: TransactionResponse | None = None


class FraudRespondRequest(BaseModel):
    """Customer verdict from the Fraud & Security Center."""

    response: Literal["confirmed", "disputed"]
    note: str | None = Field(default=None, max_length=1000)


class FraudReviewRequest(BaseModel):
    """Admin verdict; writes the ground-truth label used for retraining."""

    decision: Literal["fraud", "legitimate", "dismiss"]
    note: str | None = Field(default=None, max_length=1000)
    reverse_transaction: bool = False


class AnomalyAlertResponse(ORMModel):
    id: int
    transaction_id: int | None = None
    anomaly_score: float
    severity: str
    anomaly_type: str
    title: str
    message: str
    category: str | None = None
    baseline_value: float | None = None
    observed_value: float | None = None
    deviation_ratio: float | None = None
    model_name: str
    model_version: str
    acknowledged: bool
    created_at: datetime


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #


class CategoryBreakdown(BaseModel):
    category: str
    total: Decimal
    count: int
    percentage: float
    avg_amount: Decimal


class MonthlyTrend(BaseModel):
    month: str
    inflow: Decimal
    outflow: Decimal
    net: Decimal
    txn_count: int


class DailySpend(BaseModel):
    date: str
    amount: Decimal
    count: int


class InsightsResponse(BaseModel):
    period_days: int
    total_spent: Decimal
    total_received: Decimal
    net_change: Decimal
    txn_count: int
    avg_transaction: Decimal
    largest_transaction: Decimal
    category_breakdown: list[CategoryBreakdown]
    monthly_trends: list[MonthlyTrend]
    daily_spend: list[DailySpend]
    top_merchants: list[dict[str, Any]]
    anomaly_alerts: list[AnomalyAlertResponse]


class DashboardResponse(BaseModel):
    total_balance: Decimal
    accounts: list[AccountResponse]
    recent_transactions: list[TransactionResponse]
    spend_last_30d: Decimal
    received_last_30d: Decimal
    open_fraud_alerts: int
    unread_notifications: int
    active_loans: int
    category_breakdown: list[CategoryBreakdown]
    daily_spend: list[DailySpend]
    latest_credit_score: int | None = None


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #


class NotificationResponse(ORMModel):
    id: int
    notif_type: str
    severity: str
    title: str
    body: str
    action_url: str | None = None
    meta: dict[str, Any] | None = None
    is_read: bool
    created_at: datetime


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #


class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    pending_kyc: int
    frozen_users: int
    total_accounts: int
    total_balance: Decimal
    txn_count_today: int
    txn_volume_today: Decimal
    txn_count_30d: int
    txn_volume_30d: Decimal
    fraud_alerts_open: int
    fraud_alerts_total: int
    fraud_confirmed: int
    blocked_transactions: int
    loans_pending: int
    loans_approved: int
    loans_disbursed_value: Decimal
    model_status: dict[str, Any]


class UserStatusRequest(BaseModel):
    status: Literal["active", "frozen", "suspended"]
    reason: str | None = Field(default=None, max_length=500)


class KycDecisionRequest(BaseModel):
    decision: Literal["verify", "reject"]
    reason: str | None = Field(default=None, max_length=500)


class LoanDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approved_amount: Decimal | None = Field(default=None, gt=0)
    interest_rate: float | None = Field(default=None, gt=0, le=60)
    note: str | None = Field(default=None, max_length=1000)
    override_model: bool = False


class AuditLogResponse(ORMModel):
    id: int
    actor_id: int | None = None
    actor_email: str | None = None
    actor_role: str | None = None
    action: str
    entity_type: str | None = None
    entity_id: str | None = None
    summary: str | None = None
    ip_address: str | None = None
    success: bool
    created_at: datetime


class ModelPerformanceResponse(BaseModel):
    """Training baseline vs observed production behaviour, per model."""

    model_name: str
    model_version: str | None = None
    loaded: bool
    training_metrics: dict[str, Any]
    training_latency: dict[str, Any]
    live_inference_count: int
    live_flagged_count: int
    live_labelled_count: int
    live_mean_score: float | None = None
    live_p95_score: float | None = None
    live_mean_latency_ms: float | None = None
    live_p95_latency_ms: float | None = None
    realised_precision: float | None = None
    realised_recall: float | None = None
    psi: float | None = None
    drift_status: str | None = None
    score_histogram: list[dict[str, Any]] = []


# --------------------------------------------------------------------------- #
# AI assistant
# --------------------------------------------------------------------------- #


class AssistantTurn(BaseModel):
    """One prior message, replayed so follow-up questions have context."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    # Client-held history: the server stays stateless, and a user cannot be
    # shown another session's messages because nothing is stored server-side.
    history: list[AssistantTurn] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned


class AssistantToolCall(BaseModel):
    """A tool the model invoked, surfaced so the answer can be audited."""

    name: str
    arguments: dict[str, Any] = {}
    ok: bool
    duration_ms: float


class AssistantChatResponse(BaseModel):
    message: str
    tool_calls: list[AssistantToolCall] = []
    # "gemini" or "fallback" — the UI labels degraded answers rather than
    # passing templated prose off as model output.
    engine: str
    model: str | None = None
    latency_ms: float
    degraded_reason: str | None = None


class AssistantStatusResponse(BaseModel):
    enabled: bool
    engine: str
    model: str | None = None
    api_key_configured: bool
    tools: list[str]
    capabilities: str


# --------------------------------------------------------------------------- #
# Generic envelopes
# --------------------------------------------------------------------------- #


class Page[T](BaseModel):
    """Cursor-free pagination envelope: simple, and adequate at this scale."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class MessageResponse(BaseModel):
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    database: str
    cache_backend: str
    models: dict[str, bool]
    timestamp: datetime


# Resolve forward references declared above.
TokenResponse.model_rebuild()
TransferResponse.model_rebuild()
