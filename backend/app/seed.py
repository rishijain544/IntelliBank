"""Database seeding: demo users with realistic history, alerts and loans.

Run with::

    python -m app.seed              # create demo data (idempotent-ish)
    python -m app.seed --reset      # drop and recreate all tables first

The seeded transaction history is generated to look like real spending — habitual
merchants, log-normal amounts, salary credits, a few injected fraud episodes — so
the dashboards, charts and ML feature aggregates all have something meaningful to
work with. Transactions are inserted with historical timestamps and then scored,
which is what populates the fraud/anomaly alert queues.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.database import Base, SessionLocal, engine, init_db
from app.core.security import hash_password, random_reference
from app.ml.features import TxnContext
from app.ml.inference import score_anomaly, score_fraud, warm_up
from app.models.banking import Account, Beneficiary, Card, Transaction
from app.models.enums import (
    AccountType,
    AlertSeverity,
    AlertStatus,
    CardType,
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
from app.models.mixins import quantize
from app.models.risk import AnomalyAlert, FraudAlert
from app.models.system import AuditLog, Notification
from app.models.user import RefreshToken, User, UserDevice
from app.services import banking, notifications as notif
from app.services.ml_features import build_user_history, enrich_for_category

DEMO_PASSWORD = "Demo@Pass123"
ADMIN_PASSWORD = "Admin@Pass123"

# (name, category, mean_amount, coefficient_of_variation)
MERCHANTS = [
    ("BigBasket", MerchantCategory.GROCERIES, 2200, 0.45),
    ("DMart", MerchantCategory.GROCERIES, 1800, 0.50),
    ("Swiggy", MerchantCategory.DINING, 480, 0.60),
    ("Zomato", MerchantCategory.DINING, 520, 0.65),
    ("Starbucks", MerchantCategory.DINING, 650, 0.40),
    ("Uber", MerchantCategory.TRANSPORT, 320, 0.70),
    ("Indian Oil", MerchantCategory.TRANSPORT, 2500, 0.35),
    ("Amazon", MerchantCategory.SHOPPING, 1900, 0.90),
    ("Flipkart", MerchantCategory.SHOPPING, 1700, 0.95),
    ("Myntra", MerchantCategory.SHOPPING, 2300, 0.70),
    ("Airtel", MerchantCategory.UTILITIES, 799, 0.20),
    ("Tata Power", MerchantCategory.UTILITIES, 1900, 0.40),
    ("Netflix", MerchantCategory.ENTERTAINMENT, 649, 0.10),
    ("PVR Cinemas", MerchantCategory.ENTERTAINMENT, 900, 0.45),
    ("Apollo Pharmacy", MerchantCategory.HEALTHCARE, 850, 0.60),
    ("MakeMyTrip", MerchantCategory.TRAVEL, 18000, 0.85),
    ("Landlord Rent", MerchantCategory.RENT, 25000, 0.25),
    ("Zerodha", MerchantCategory.INVESTMENT, 15000, 1.00),
    ("ATM Withdrawal", MerchantCategory.CASH, 5000, 0.60),
]

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"]

# Balance floor kept in every seeded account. A seeded ledger that goes negative
# would poison the ML features (amount_to_balance saturates) and misrepresent the
# credit model's balance-volatility inputs.
MIN_FLOOR = Decimal("2000")

DEMO_USERS = [
    {
        "email": "priya@intellibank.dev",
        "full_name": "Priya Ramanathan",
        "city": "Bengaluru",
        "income": Decimal("1850000"),
        "employment": "salaried",
        "employment_years": 7.5,
        "dependents": 1,
        "housing": "mortgage",
        "dob": date(1990, 4, 12),
        "profile": "affluent",
    },
    {
        "email": "arjun@intellibank.dev",
        "full_name": "Arjun Mehta",
        "city": "Mumbai",
        "income": Decimal("780000"),
        "employment": "salaried",
        "employment_years": 3.0,
        "dependents": 0,
        "housing": "rent",
        "dob": date(1996, 9, 3),
        "profile": "average",
    },
    {
        "email": "kavya@intellibank.dev",
        "full_name": "Kavya Nair",
        "city": "Pune",
        "income": Decimal("420000"),
        "employment": "gig",
        "employment_years": 1.5,
        "dependents": 2,
        "housing": "rent",
        "dob": date(1999, 1, 22),
        "profile": "thin_file",
    },
    {
        "email": "rohan@intellibank.dev",
        "full_name": "Rohan Gupta",
        "city": "Delhi",
        "income": Decimal("1150000"),
        "employment": "self_employed",
        "employment_years": 5.0,
        "dependents": 3,
        "housing": "own",
        "dob": date(1987, 6, 30),
        "profile": "volatile",
    },
]


def _draw(rng: random.Random, mean: float, cv: float) -> Decimal:
    """Log-normal draw: spending is right-skewed and never negative."""
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(max(mean, 1.0)) - 0.5 * sigma**2
    return quantize(min(max(rng.lognormvariate(mu, sigma), 15.0), 2_000_000.0))


def reset_database() -> None:
    print("dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("schema recreated")


def _clear_demo_data(db: Session) -> None:
    """Remove previously seeded rows so re-running does not duplicate history."""
    emails = [u["email"] for u in DEMO_USERS] + ["admin@intellibank.dev"]
    users = db.execute(select(User).where(User.email.in_(emails))).scalars().all()
    if not users:
        return
    print(f"removing {len(users)} existing demo user(s) and their data...")
    for user in users:
        # Explicit deletes for tables whose FKs are SET NULL rather than CASCADE.
        db.execute(delete(AuditLog).where(AuditLog.actor_id == user.id))
        db.execute(delete(AuditLog).where(AuditLog.target_user_id == user.id))
        db.delete(user)
    db.commit()


def create_admin(db: Session) -> User:
    admin = User(
        email="admin@intellibank.dev",
        hashed_password=hash_password(ADMIN_PASSWORD),
        full_name="System Administrator",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        kyc_status=KycStatus.VERIFIED,
        kyc_verified_at=datetime.now(UTC),
        city="Mumbai",
        country="IN",
        password_changed_at=datetime.now(UTC),
    )
    db.add(admin)
    db.flush()
    print(f"  admin: {admin.email} / {ADMIN_PASSWORD}")
    return admin


def create_customer(db: Session, spec: dict, rng: random.Random) -> User:
    user = User(
        email=spec["email"],
        hashed_password=hash_password(DEMO_PASSWORD),
        full_name=spec["full_name"],
        phone=f"+9198{rng.randint(10000000, 99999999)}",
        date_of_birth=spec["dob"],
        role=UserRole.CUSTOMER,
        status=UserStatus.ACTIVE,
        kyc_status=KycStatus.VERIFIED,
        kyc_submitted_at=datetime.now(UTC) - timedelta(days=200),
        kyc_verified_at=datetime.now(UTC) - timedelta(days=199),
        pan_masked="ABCXXXXF",
        aadhaar_masked=f"XXXX-XXXX-{rng.randint(1000, 9999)}",
        id_document_type="passport",
        address_line1=f"{rng.randint(1, 200)} Demo Street",
        city=spec["city"],
        state="Karnataka" if spec["city"] == "Bengaluru" else "Maharashtra",
        postal_code=f"{rng.randint(400001, 600001)}",
        country="IN",
        annual_income=spec["income"],
        employment_status=spec["employment"],
        employment_years=spec["employment_years"],
        dependents=spec["dependents"],
        housing_status=spec["housing"],
        password_changed_at=datetime.now(UTC) - timedelta(days=100),
    )
    db.add(user)
    db.flush()
    return user


def seed_history(
    db: Session, user: User, account: Account, spec: dict, rng: random.Random, *, days: int = 180
) -> None:
    """Insert ~6 months of realistic activity, oldest first.

    Balances are advanced row by row so ``balance_after`` forms a coherent series,
    which is what the credit model's volatility features read.
    """
    profile = spec["profile"]
    txn_per_day = {"affluent": 2.6, "average": 1.9, "thin_file": 0.9, "volatile": 2.2}[profile]
    # Fraction of take-home pay this persona spends each month.
    spend_ratio = {"affluent": 0.55, "average": 0.72, "thin_file": 0.88, "volatile": 0.80}[profile]

    monthly_salary = quantize(spec["income"] / 12)
    start = datetime.now(UTC) - timedelta(days=days)
    home_city = spec["city"]
    device = f"seed-device-{user.id}"

    # Habitual users concentrate spending on a small merchant set.
    preferred = rng.sample(MERCHANTS, k=rng.randint(7, 12))

    # Scale merchant amounts so expected monthly outflow lands on spend_ratio of
    # salary. Drawing amounts independently of income is what drains a seeded
    # ledger to zero and then distorts every balance-derived ML feature.
    avg_txn_mean = sum(m[2] for m in preferred) / len(preferred)
    expected_monthly_txns = txn_per_day * 30.44
    target_monthly_spend = float(monthly_salary) * spend_ratio
    scale = target_monthly_spend / (avg_txn_mean * expected_monthly_txns)
    scale = min(max(scale, 0.15), 12.0)

    events: list[tuple[datetime, str, object]] = []

    # Salary credits on the 1st of each month.
    cursor = start.replace(day=1, hour=10, minute=0, second=0, microsecond=0)
    while cursor < datetime.now(UTC):
        if cursor >= start:
            events.append((cursor, "salary", None))
        cursor = (cursor.replace(day=28) + timedelta(days=8)).replace(day=1, hour=10)

    # Regular spending.
    for _ in range(int(txn_per_day * days)):
        offset_days = rng.random() * days
        hour = min(max(int(rng.gauss(14, 3.5)), 6), 23)
        ts = start + timedelta(days=offset_days, hours=hour - 12)
        events.append((ts, "spend", rng.choice(preferred)))

    # A couple of fraud episodes on the primary demo user so the queues are populated.
    fraud_events: list[tuple[datetime, str, object]] = []
    if profile in ("affluent", "volatile"):
        for _ in range(2):
            ts = datetime.now(UTC) - timedelta(days=rng.randint(2, 25), hours=rng.randint(0, 4))
            fraud_events.append((ts, "fraud", None))
    events.extend(fraud_events)

    events.sort(key=lambda e: e[0])
    # Open with roughly two months of salary as a buffer.
    balance = quantize(monthly_salary * Decimal("2"))
    account.balance = balance

    created: list[Transaction] = []
    for ts, kind, payload in events:
        if kind == "salary":
            amount = quantize(monthly_salary * Decimal(str(rng.uniform(0.97, 1.03))))
            balance = quantize(balance + amount)
            txn = Transaction(
                reference=random_reference("TXN"),
                user_id=user.id,
                account_id=account.id,
                txn_type=TransactionType.DEPOSIT,
                channel=TransactionChannel.NEFT,
                status=TransactionStatus.COMPLETED,
                amount=amount,
                signed_amount=amount,
                balance_after=balance,
                description="Salary credit",
                merchant_name="Employer Payroll",
                merchant_category=MerchantCategory.OTHER,
                device_fingerprint=device,
                location_city=home_city,
                location_country="IN",
                occurred_at=ts,
            )
            db.add(txn)
            created.append(txn)
            continue

        if kind == "fraud":
            # Foreign card-not-present charge from an unrecognised device.
            amount = _draw(rng, 42_000 * scale, 0.8)
            # Even injected fraud must respect the ledger: a debit that overdraws
            # the account would leave a negative balance, which then makes every
            # later legitimate transaction look like it is draining the account.
            if amount > balance - MIN_FLOOR:
                amount = quantize(max((balance - MIN_FLOOR) * Decimal("0.6"), Decimal("0")))
            if amount <= 0:
                continue
            city, country = rng.choice([("Lagos", "NG"), ("Kyiv", "UA"), ("Caracas", "VE")])
            txn = Transaction(
                reference=random_reference("TXN"),
                user_id=user.id,
                account_id=account.id,
                txn_type=TransactionType.CARD_PAYMENT,
                channel=TransactionChannel.CARD,
                status=TransactionStatus.COMPLETED,
                amount=amount,
                signed_amount=-amount,
                balance_after=quantize(balance - amount),
                description="Intl Merchant purchase",
                merchant_name="Intl Merchant",
                merchant_category=MerchantCategory.SHOPPING,
                device_fingerprint=f"attacker-{rng.randint(1000, 9999)}",
                location_city=city,
                location_country=country,
                is_foreign=True,
                occurred_at=ts,
            )
            balance = quantize(balance - amount)
            db.add(txn)
            created.append(txn)
            continue

        name, category, mean, cv = payload  # type: ignore[misc]
        amount = _draw(rng, mean * scale, cv)
        if amount > balance - MIN_FLOOR:
            continue  # skip rather than overdraw

        is_card = category not in (MerchantCategory.RENT, MerchantCategory.CASH)
        balance = quantize(balance - amount)
        txn = Transaction(
            reference=random_reference("TXN"),
            user_id=user.id,
            account_id=account.id,
            txn_type=(
                TransactionType.CARD_PAYMENT
                if is_card
                else (
                    TransactionType.WITHDRAWAL
                    if category == MerchantCategory.CASH
                    else TransactionType.TRANSFER_OUT
                )
            ),
            channel=(
                TransactionChannel.CARD
                if is_card
                else (
                    TransactionChannel.ATM
                    if category == MerchantCategory.CASH
                    else TransactionChannel.IMPS
                )
            ),
            status=TransactionStatus.COMPLETED,
            amount=amount,
            signed_amount=-amount,
            balance_after=balance,
            description=f"{name} payment",
            merchant_name=name,
            merchant_category=category,
            device_fingerprint=device,
            location_city=home_city,
            location_country="IN",
            occurred_at=ts,
        )
        db.add(txn)
        created.append(txn)

    account.balance = balance
    db.flush()
    print(f"    {len(created)} transactions, closing balance {balance:,.2f}")

    # Register the habitual device so the fraud model has a known-device baseline.
    db.add(
        UserDevice(
            user_id=user.id,
            fingerprint=device,
            user_agent="Mozilla/5.0 (seed demo device)",
            ip_address="203.0.113.10",
            trusted=True,
            login_count=len(created) // 4 + 1,
            last_seen_at=datetime.now(UTC),
        )
    )
    db.flush()
    _score_recent(db, user, created, rng)


def _score_recent(db: Session, user: User, txns: list[Transaction], rng: random.Random) -> None:
    """Score the most recent transactions to populate the alert queues.

    Only the recent tail is scored: each call builds rolling aggregates from SQL,
    so scoring six months of rows would be needlessly slow for a demo seed.
    """
    recent = sorted(txns, key=lambda t: t.occurred_at)[-45:]
    debits = [t for t in recent if t.signed_amount < 0]
    fraud_count = anomaly_count = 0

    for txn in debits:
        hist = build_user_history(db, user, now=txn.occurred_at)
        hist = enrich_for_category(db, user, hist, txn.merchant_category, now=txn.occurred_at)
        ctx = TxnContext(
            amount=float(txn.amount),
            occurred_at=txn.occurred_at,
            category=txn.merchant_category,
            channel=txn.channel,
            merchant_name=txn.merchant_name,
            device_fingerprint=txn.device_fingerprint,
            location_city=txn.location_city,
            location_country=txn.location_country or "IN",
            account_balance=float(txn.balance_after or 0),
            account_age_days=400.0,
        )
        fraud = score_fraud(ctx, hist)
        anomaly = score_anomaly(ctx, hist)

        txn.fraud_score = fraud.risk_score
        txn.anomaly_score = anomaly.anomaly_score
        txn.scoring_latency_ms = fraud.latency_ms
        txn.is_flagged = fraud.is_flagged

        if fraud.is_flagged:
            alert = FraudAlert(
                alert_ref=random_reference("FRD", 8),
                user_id=user.id,
                transaction_id=txn.id,
                risk_score=fraud.risk_score,
                severity=fraud.severity,
                status=AlertStatus.OPEN,
                decision_source=fraud.decision_source,
                auto_blocked=fraud.auto_blocked,
                reasons=fraud.reasons,
                features=fraud.features,
                top_factors=fraud.top_factors,
                triggered_rules=fraud.triggered_rules,
                model_name=fraud.model_name,
                model_version=fraud.model_version,
                inference_latency_ms=fraud.latency_ms,
            )
            db.add(alert)
            fraud_count += 1
            notif.notify_fraud_alert(
                db, user, alert_ref=alert.alert_ref, amount=txn.amount, blocked=fraud.auto_blocked
            )

        if anomaly.is_anomaly and anomaly_count < 5:
            db.add(
                AnomalyAlert(
                    user_id=user.id,
                    transaction_id=txn.id,
                    anomaly_score=anomaly.anomaly_score,
                    severity=anomaly.severity,
                    anomaly_type=anomaly.anomaly_type,
                    title=anomaly.title,
                    message=anomaly.message,
                    category=txn.merchant_category,
                    baseline_value=anomaly.baseline_value,
                    observed_value=anomaly.observed_value,
                    deviation_ratio=anomaly.deviation_ratio,
                    features=anomaly.features,
                    model_name=anomaly.model_name,
                    model_version=anomaly.model_version,
                    inference_latency_ms=anomaly.latency_ms,
                )
            )
            anomaly_count += 1

    db.flush()
    print(f"    scored {len(debits)} debits -> {fraud_count} fraud alert(s), {anomaly_count} insight(s)")


def seed_extras(db: Session, user: User, account: Account, spec: dict, rng: random.Random) -> None:
    """Cards, beneficiaries and a loan history."""
    banking.issue_card(db, user, account, card_type=CardType.VIRTUAL_DEBIT)
    if spec["profile"] == "affluent":
        card = banking.issue_card(
            db,
            user,
            account,
            card_type=CardType.VIRTUAL_CREDIT,
            daily_limit=Decimal("150000"),
            per_txn_limit=Decimal("75000"),
            monthly_limit=Decimal("500000"),
        )
        card.international_enabled = True

    for name, bank in (("Anil Kumar", "HDFC Bank"), ("Sneha Patel", "ICICI Bank")):
        db.add(
            Beneficiary(
                user_id=user.id,
                name=name,
                nickname=name.split()[0],
                account_number=str(rng.randint(10**11, 10**12 - 1)),
                ifsc_code="HDFC0001234" if bank == "HDFC Bank" else "ICIC0005678",
                bank_name=bank,
                is_internal=False,
                is_verified=True,
                activated_at=datetime.now(UTC) - timedelta(days=60),
            )
        )

    # A repaid loan for the affluent user gives the credit model real history.
    if spec["profile"] in ("affluent", "volatile"):
        missed = 0 if spec["profile"] == "affluent" else 3
        principal = Decimal("400000")
        db.add(
            Loan(
                application_ref=random_reference("LN", 8),
                user_id=user.id,
                disbursement_account_id=account.id,
                loan_type=LoanType.AUTO,
                status=LoanStatus.DISBURSED,
                requested_amount=principal,
                approved_amount=principal,
                tenure_months=48,
                purpose="Vehicle purchase",
                interest_rate=11.5,
                emi_amount=quantize(Decimal("10430")),
                total_payable=quantize(Decimal("500640")),
                processing_fee=quantize(principal * Decimal("0.01")),
                declared_income=spec["income"],
                employment_status=spec["employment"],
                employment_years=spec["employment_years"],
                decision_source="model",
                decision_reason="Seeded loan history",
                decided_at=datetime.now(UTC) - timedelta(days=150),
                disbursed_at=datetime.now(UTC) - timedelta(days=150),
                outstanding_principal=quantize(principal * Decimal("0.7")),
                emis_paid=5,
                emis_missed=missed,
            )
        )
    db.flush()


def seed(reset: bool = False) -> None:
    if reset:
        reset_database()
    else:
        init_db()

    print("\nwarming ML models...")
    status = warm_up()
    for name, loaded in status.items():
        print(f"  {name}: {'loaded' if loaded else 'NOT TRAINED (rules fallback)'}")

    rng = random.Random(20260731)
    db: Session = SessionLocal()
    try:
        _clear_demo_data(db)

        print("\ncreating users...")
        create_admin(db)

        for spec in DEMO_USERS:
            user = create_customer(db, spec, rng)
            print(f"  customer: {user.email} / {DEMO_PASSWORD}  ({spec['profile']})")

            account = banking.create_account(
                db, user, account_type=AccountType.SAVINGS, nickname="Primary Savings"
            )
            if spec["profile"] in ("affluent", "volatile"):
                banking.create_account(
                    db,
                    user,
                    account_type=AccountType.CURRENT,
                    nickname="Business Current",
                    initial_deposit=Decimal("85000"),
                )

            seed_history(db, user, account, spec, rng)
            seed_extras(db, user, account, spec, rng)
            db.commit()

        # Cross-customer beneficiary so internal transfers are demoable.
        priya = db.execute(select(User).where(User.email == "priya@intellibank.dev")).scalar_one()
        arjun = db.execute(select(User).where(User.email == "arjun@intellibank.dev")).scalar_one()
        arjun_account = db.execute(
            select(Account).where(Account.user_id == arjun.id).limit(1)
        ).scalar_one()
        db.add(
            Beneficiary(
                user_id=priya.id,
                name=arjun.full_name,
                nickname="Arjun (IntelliBank)",
                account_number=arjun_account.account_number,
                ifsc_code=arjun_account.ifsc_code,
                bank_name="IntelliBank",
                is_internal=True,
                is_verified=True,
                activated_at=datetime.now(UTC) - timedelta(days=30),
            )
        )
        db.commit()

        # ---- summary ----
        from sqlalchemy import func

        counts = {
            "users": db.execute(select(func.count(User.id))).scalar_one(),
            "accounts": db.execute(select(func.count(Account.id))).scalar_one(),
            "transactions": db.execute(select(func.count(Transaction.id))).scalar_one(),
            "fraud_alerts": db.execute(select(func.count(FraudAlert.id))).scalar_one(),
            "anomaly_alerts": db.execute(select(func.count(AnomalyAlert.id))).scalar_one(),
            "cards": db.execute(select(func.count(Card.id))).scalar_one(),
            "loans": db.execute(select(func.count(Loan.id))).scalar_one(),
            "notifications": db.execute(select(func.count(Notification.id))).scalar_one(),
        }
        print("\n" + "=" * 60)
        print("SEED COMPLETE")
        print("=" * 60)
        for key, value in counts.items():
            print(f"  {key:16s} {value:>6,}")
        print("\nsign in with:")
        print(f"  admin    : admin@intellibank.dev / {ADMIN_PASSWORD}")
        print(f"  customer : priya@intellibank.dev / {DEMO_PASSWORD}")
        print("=" * 60)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.seed", description="Seed demo data")
    parser.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    args = parser.parse_args(argv)
    seed(reset=args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
