"""SQLAlchemy model registry. Importing this package registers every mapper."""
from app.models.banking import Account, Beneficiary, Card, Transaction
from app.models.enums import (
    AccountStatus,
    AccountType,
    AlertSeverity,
    AlertStatus,
    CardStatus,
    CardType,
    DecisionSource,
    KycStatus,
    LoanStatus,
    LoanType,
    MerchantCategory,
    NotificationType,
    TransactionChannel,
    TransactionStatus,
    TransactionType,
    UserRole,
    UserStatus,
)
from app.models.lending import CreditScore, Loan
from app.models.risk import AnomalyAlert, FraudAlert
from app.models.system import AuditLog, ModelMetricSnapshot, Notification
from app.models.user import RefreshToken, User, UserDevice

__all__ = [
    "Account",
    "AccountStatus",
    "AccountType",
    "AlertSeverity",
    "AlertStatus",
    "AnomalyAlert",
    "AuditLog",
    "Beneficiary",
    "Card",
    "CardStatus",
    "CardType",
    "CreditScore",
    "DecisionSource",
    "FraudAlert",
    "KycStatus",
    "Loan",
    "LoanStatus",
    "LoanType",
    "MerchantCategory",
    "ModelMetricSnapshot",
    "Notification",
    "NotificationType",
    "RefreshToken",
    "Transaction",
    "TransactionChannel",
    "TransactionStatus",
    "TransactionType",
    "User",
    "UserDevice",
    "UserRole",
    "UserStatus",
]
