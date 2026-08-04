"""Shared pytest fixtures.

Each test gets an isolated in-memory SQLite database and its own TestClient, so
tests cannot leak state into one another or into the developer's dev database.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal

import pytest

# Must be set before app.core.config is imported anywhere.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
# >=32 bytes: PyJWT warns below the RFC 7518 minimum for HS256.
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-only-32b")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.cache import reset_kv  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.enums import KycStatus, UserRole, UserStatus  # noqa: E402
from app.models.user import User  # noqa: E402
from app.core.security import hash_password  # noqa: E402

TEST_PASSWORD = "Test@Pass123"


@pytest.fixture
def db_session() -> Iterator:
    """A fresh in-memory schema per test.

    StaticPool keeps every connection pointed at the same in-memory database;
    without it each new connection would see an empty schema.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(db_session) -> Iterator[TestClient]:
    """TestClient wired to the per-test session, with rate limiters reset."""
    reset_kv()  # otherwise limits carry across tests and cause flaky 429s

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass  # lifecycle owned by the db_session fixture

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_user(
    db,
    *,
    email: str,
    role: str = UserRole.CUSTOMER,
    status: str = UserStatus.ACTIVE,
    kyc: str = KycStatus.VERIFIED,
    income: Decimal = Decimal("900000"),
) -> User:
    user = User(
        email=email,
        hashed_password=hash_password(TEST_PASSWORD),
        full_name=email.split("@")[0].title(),
        role=role,
        status=status,
        kyc_status=kyc,
        city="Mumbai",
        country="IN",
        annual_income=income,
        employment_status="salaried",
        employment_years=5.0,
        dependents=1,
        housing_status="rent",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def customer(db_session) -> User:
    return _make_user(db_session, email="customer@test.dev")


@pytest.fixture
def admin(db_session) -> User:
    return _make_user(db_session, email="admin@test.dev", role=UserRole.ADMIN)


@pytest.fixture
def pending_user(db_session) -> User:
    """Registered but not KYC-verified: must be blocked from banking features."""
    return _make_user(
        db_session,
        email="pending@test.dev",
        status=UserStatus.PENDING,
        kyc=KycStatus.NOT_STARTED,
    )


def auth_headers(client: TestClient, email: str) -> dict[str, str]:
    resp = client.post(
        "/api/v1/auth/login", json={"email": email, "password": TEST_PASSWORD}
    )
    assert resp.status_code == 200, f"login failed: {resp.text}"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def customer_headers(client, customer) -> dict[str, str]:
    return auth_headers(client, customer.email)


@pytest.fixture
def admin_headers(client, admin) -> dict[str, str]:
    return auth_headers(client, admin.email)


@pytest.fixture
def funded_account(client, customer_headers) -> dict:
    """An active savings account with a working balance."""
    created = client.post(
        "/api/v1/accounts",
        json={"account_type": "savings", "nickname": "Test Savings"},
        headers=customer_headers,
    )
    assert created.status_code == 201, created.text
    account = created.json()

    funded = client.post(
        f"/api/v1/accounts/{account['id']}/deposit",
        json={"account_id": account["id"], "amount": "250000.00", "description": "Test funding"},
        headers=customer_headers,
    )
    assert funded.status_code == 200, funded.text
    return client.get(f"/api/v1/accounts/{account['id']}", headers=customer_headers).json()
