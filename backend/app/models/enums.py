"""Enumerations shared across models, schemas and services."""
from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    CUSTOMER = "customer"
    ADMIN = "admin"


class UserStatus(StrEnum):
    PENDING = "pending"        # registered, KYC not submitted
    ACTIVE = "active"
    FROZEN = "frozen"          # admin-frozen
    SUSPENDED = "suspended"


class KycStatus(StrEnum):
    NOT_STARTED = "not_started"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"


class AccountType(StrEnum):
    SAVINGS = "savings"
    CURRENT = "current"
    FIXED_DEPOSIT = "fixed_deposit"
    SALARY = "salary"


class AccountStatus(StrEnum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


class TransactionType(StrEnum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    FEE = "fee"
    INTEREST = "interest"
    LOAN_DISBURSEMENT = "loan_disbursement"
    LOAN_REPAYMENT = "loan_repayment"
    CARD_PAYMENT = "card_payment"


class TransactionChannel(StrEnum):
    INTERNAL = "internal"
    NEFT = "neft"
    IMPS = "imps"
    UPI = "upi"
    CARD = "card"
    ATM = "atm"
    SYSTEM = "system"


class TransactionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"
    BLOCKED = "blocked"        # stopped by fraud model
    HELD = "held"              # awaiting manual review


class MerchantCategory(StrEnum):
    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    TRAVEL = "travel"
    RENT = "rent"
    INVESTMENT = "investment"
    CASH = "cash"
    TRANSFER = "transfer"
    OTHER = "other"


class CardType(StrEnum):
    VIRTUAL_DEBIT = "virtual_debit"
    VIRTUAL_CREDIT = "virtual_credit"


class CardStatus(StrEnum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class LoanType(StrEnum):
    PERSONAL = "personal"
    HOME = "home"
    AUTO = "auto"
    EDUCATION = "education"
    BUSINESS = "business"


class LoanStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISBURSED = "disbursed"
    CLOSED = "closed"


class AlertStatus(StrEnum):
    OPEN = "open"                 # awaiting customer or admin action
    CONFIRMED_FRAUD = "confirmed_fraud"
    DISPUTED = "disputed"         # customer says "this was me"
    RESOLVED_LEGIT = "resolved_legit"
    RESOLVED_FRAUD = "resolved_fraud"
    DISMISSED = "dismissed"


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(StrEnum):
    LARGE_TRANSACTION = "large_transaction"
    NEW_DEVICE_LOGIN = "new_device_login"
    FRAUD_ALERT = "fraud_alert"
    ANOMALY = "anomaly"
    LOAN_UPDATE = "loan_update"
    CARD_UPDATE = "card_update"
    ACCOUNT_UPDATE = "account_update"
    SECURITY = "security"
    GENERAL = "general"


class DecisionSource(StrEnum):
    MODEL = "model"
    RULE = "rule"
    HYBRID = "hybrid"
    MANUAL = "manual"
