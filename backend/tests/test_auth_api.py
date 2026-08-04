"""Authentication, registration, KYC and session-security tests."""
from __future__ import annotations

from tests.conftest import TEST_PASSWORD, _make_user, auth_headers

BASE = "/api/v1"


# ------------------------------------------------------------------ registration


def test_register_creates_pending_user(client):
    resp = client.post(
        f"{BASE}/auth/register",
        json={
            "email": "new@test.dev",
            "password": "Str0ng!Passw0rd",
            "full_name": "New Customer",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["kyc_status"] == "not_started"
    # A password must never appear in a response payload.
    assert "password" not in body and "hashed_password" not in body


def test_register_rejects_weak_password(client):
    resp = client.post(
        f"{BASE}/auth/register",
        json={"email": "weak@test.dev", "password": "password", "full_name": "Weak"},
    )
    assert resp.status_code == 422
    assert "fields" in resp.json()


def test_register_rejects_duplicate_email(client, customer):
    resp = client.post(
        f"{BASE}/auth/register",
        json={
            "email": customer.email,
            "password": "Str0ng!Passw0rd",
            "full_name": "Duplicate",
        },
    )
    assert resp.status_code == 409


def test_register_rejects_minor(client):
    resp = client.post(
        f"{BASE}/auth/register",
        json={
            "email": "kid@test.dev",
            "password": "Str0ng!Passw0rd",
            "full_name": "Too Young",
            "date_of_birth": "2015-01-01",
        },
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------------- login


def test_login_returns_token_pair(client, customer):
    resp = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"] and body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == customer.email


def test_login_wrong_password_is_generic(client, customer):
    """Error text must not reveal whether the email exists."""
    wrong_pw = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": "Wr0ng!Password"}
    )
    unknown = client.post(
        f"{BASE}/auth/login", json={"email": "nobody@test.dev", "password": "Wr0ng!Password"}
    )
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"]


def test_account_lockout_after_repeated_failures(client, customer):
    for _ in range(5):
        client.post(
            f"{BASE}/auth/login", json={"email": customer.email, "password": "Wr0ng!Password"}
        )
    # The 6th attempt should be locked out even with the correct password.
    resp = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    )
    assert resp.status_code == 423
    assert "locked" in resp.json()["detail"].lower()


def test_frozen_user_cannot_log_in(client, db_session, customer):
    customer.status = "frozen"
    db_session.commit()
    resp = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    )
    assert resp.status_code == 403


# ------------------------------------------------------------------------- /me


def test_me_requires_token(client):
    assert client.get(f"{BASE}/auth/me").status_code == 401


def test_me_rejects_garbage_token(client):
    resp = client.get(f"{BASE}/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert resp.status_code == 401


def test_me_returns_profile(client, customer, customer_headers):
    resp = client.get(f"{BASE}/auth/me", headers=customer_headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == customer.email


def test_refresh_token_cannot_be_used_as_access_token(client, customer):
    """Token type must be enforced, or a long-lived refresh token becomes an
    unrevokable access token."""
    login = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    ).json()
    resp = client.get(
        f"{BASE}/auth/me", headers={"Authorization": f"Bearer {login['refresh_token']}"}
    )
    assert resp.status_code == 401


# ----------------------------------------------------------------- refresh flow


def test_refresh_rotates_tokens(client, customer):
    login = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    ).json()
    resp = client.post(f"{BASE}/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert resp.status_code == 200
    assert resp.json()["refresh_token"] != login["refresh_token"]


def test_refresh_reuse_revokes_all_sessions(client, customer):
    """Replaying a rotated refresh token is treated as theft: every session dies."""
    login = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    ).json()
    old_refresh = login["refresh_token"]

    rotated = client.post(f"{BASE}/auth/refresh", json={"refresh_token": old_refresh})
    assert rotated.status_code == 200
    new_refresh = rotated.json()["refresh_token"]

    replay = client.post(f"{BASE}/auth/refresh", json={"refresh_token": old_refresh})
    assert replay.status_code == 401
    assert "reuse" in replay.json()["detail"].lower()

    # The legitimately rotated token must also be dead after the breach response.
    assert client.post(f"{BASE}/auth/refresh", json={"refresh_token": new_refresh}).status_code == 401


def test_logout_revokes_refresh_token(client, customer):
    login = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    ).json()
    headers = {"Authorization": f"Bearer {login['access_token']}"}

    out = client.post(
        f"{BASE}/auth/logout", json={"refresh_token": login["refresh_token"]}, headers=headers
    )
    assert out.status_code == 200
    assert (
        client.post(f"{BASE}/auth/refresh", json={"refresh_token": login["refresh_token"]}).status_code
        == 401
    )


# --------------------------------------------------------------------------- KYC


def test_kyc_activates_account_and_opens_savings(client, pending_user):
    headers = auth_headers(client, pending_user.email)
    resp = client.post(
        f"{BASE}/auth/kyc",
        json={
            "pan": "ABCDE1234F",
            "aadhaar": "123456789012",
            "document_type": "passport",
            "document_name": "passport.pdf",
            "address_line1": "12 Test Road",
            "city": "Mumbai",
            "state": "Maharashtra",
            "postal_code": "400001",
            "annual_income": "850000",
            "employment_status": "salaried",
            "employment_years": 4,
            "dependents": 1,
            "housing_status": "rent",
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kyc_status"] == "verified"
    assert body["status"] == "active"
    # Raw government IDs must never be echoed back.
    assert "123456789012" not in resp.text
    assert body["pan_masked"] and "ABCDE1234F" not in body["pan_masked"]

    accounts = client.get(f"{BASE}/accounts", headers=headers).json()
    assert len(accounts) == 1


def test_kyc_rejects_malformed_pan(client, pending_user):
    headers = auth_headers(client, pending_user.email)
    resp = client.post(
        f"{BASE}/auth/kyc",
        json={
            "pan": "1234567890",
            "aadhaar": "123456789012",
            "document_type": "passport",
            "document_name": "d.pdf",
            "address_line1": "12 Test Road",
            "city": "Mumbai",
            "state": "MH",
            "postal_code": "400001",
            "annual_income": "850000",
            "employment_status": "salaried",
            "employment_years": 4,
        },
        headers=headers,
    )
    assert resp.status_code == 422


def test_unverified_user_can_open_account_and_fund_it(client, pending_user):
    """Onboarding is deliberately open to unverified users.

    Gating account creation behind KYC also disables the ML features, because a
    customer with no account generates no transactions to score and no history to
    analyse. These actions carry no external compliance exposure.
    """
    headers = auth_headers(client, pending_user.email)

    created = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=headers
    )
    assert created.status_code == 201, created.text
    account = created.json()

    funded = client.post(
        f"{BASE}/accounts/{account['id']}/deposit",
        json={"account_id": account["id"], "amount": "25000.00"},
        headers=headers,
    )
    assert funded.status_code == 200, funded.text


def test_unverified_user_can_transfer_internally(client, db_session, pending_user):
    """Internal transfers stay open; only interbank payments require KYC."""
    headers = auth_headers(client, pending_user.email)
    source = client.post(
        f"{BASE}/accounts", json={"account_type": "savings", "initial_deposit": "50000.00"},
        headers=headers,
    ).json()

    recipient = _make_user(db_session, email="internal-target@test.dev")
    r_headers = auth_headers(client, recipient.email)
    destination = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=r_headers
    ).json()

    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": source["id"],
            "to_account_number": destination["account_number"],
            "amount": "1000.00",
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    # Still scored, even though the sender is unverified.
    assert "fraud" in resp.json()
    assert resp.json()["fraud"]["model_name"]


def test_unverified_user_can_check_loan_eligibility(client, pending_user):
    """A dry-run quote creates no application, so there is nothing to verify."""
    headers = auth_headers(client, pending_user.email)
    resp = client.post(
        f"{BASE}/loans/eligibility",
        json={"loan_type": "personal", "amount": "200000", "tenure_months": 24},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert 300 <= resp.json()["score"] <= 900


def test_unverified_user_blocked_from_compliance_gated_features(client, pending_user):
    """External transfers, cards and binding applications still require KYC."""
    headers = auth_headers(client, pending_user.email)
    account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings", "initial_deposit": "50000.00"},
        headers=headers,
    ).json()

    card = client.post(
        f"{BASE}/cards", json={"account_id": account["id"]}, headers=headers
    )
    assert card.status_code == 403

    application = client.post(
        f"{BASE}/loans/apply",
        json={"loan_type": "personal", "amount": "200000", "tenure_months": 24},
        headers=headers,
    )
    assert application.status_code == 403
    assert "kyc" in application.json()["detail"].lower()

    external = client.post(
        f"{BASE}/transfers/external",
        json={
            "from_account_id": account["id"],
            "beneficiary_id": 999_999,
            "amount": "1000.00",
            "channel": "imps",
        },
        headers=headers,
    )
    assert external.status_code == 403


# ---------------------------------------------------------------- password & 2FA


def test_password_change_requires_current_password(client, customer_headers):
    resp = client.post(
        f"{BASE}/auth/change-password",
        json={"current_password": "Wr0ng!Password", "new_password": "Newer@Pass456"},
        headers=customer_headers,
    )
    assert resp.status_code == 400


def test_password_change_revokes_sessions(client, customer):
    login = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    ).json()
    headers = {"Authorization": f"Bearer {login['access_token']}"}

    changed = client.post(
        f"{BASE}/auth/change-password",
        json={"current_password": TEST_PASSWORD, "new_password": "Newer@Pass456"},
        headers=headers,
    )
    assert changed.status_code == 200
    assert (
        client.post(f"{BASE}/auth/refresh", json={"refresh_token": login["refresh_token"]}).status_code
        == 401
    )
    # The new password must work.
    assert (
        client.post(
            f"{BASE}/auth/login", json={"email": customer.email, "password": "Newer@Pass456"}
        ).status_code
        == 200
    )


def test_two_factor_setup_and_enable(client, customer_headers):
    import pyotp

    setup = client.post(f"{BASE}/auth/2fa/setup", headers=customer_headers)
    assert setup.status_code == 200
    secret = setup.json()["secret"]
    assert setup.json()["provisioning_uri"].startswith("otpauth://")

    bad = client.post(
        f"{BASE}/auth/2fa/enable", json={"code": "000000"}, headers=customer_headers
    )
    assert bad.status_code == 400

    good = client.post(
        f"{BASE}/auth/2fa/enable",
        json={"code": pyotp.TOTP(secret).now()},
        headers=customer_headers,
    )
    assert good.status_code == 200


def test_login_requires_totp_once_enabled(client, customer, customer_headers):
    import pyotp

    secret = client.post(f"{BASE}/auth/2fa/setup", headers=customer_headers).json()["secret"]
    client.post(
        f"{BASE}/auth/2fa/enable",
        json={"code": pyotp.TOTP(secret).now()},
        headers=customer_headers,
    )

    without_code = client.post(
        f"{BASE}/auth/login", json={"email": customer.email, "password": TEST_PASSWORD}
    )
    assert without_code.status_code == 428  # credentials ok, second factor needed

    with_code = client.post(
        f"{BASE}/auth/login",
        json={
            "email": customer.email,
            "password": TEST_PASSWORD,
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert with_code.status_code == 200


# ------------------------------------------------------------------------- RBAC


def test_customer_cannot_reach_admin_endpoints(client, customer_headers):
    for path in ("/admin/stats", "/admin/users", "/admin/fraud/queue", "/admin/models"):
        resp = client.get(f"{BASE}{path}", headers=customer_headers)
        assert resp.status_code == 403, f"{path} leaked to a customer"


def test_admin_can_reach_admin_endpoints(client, admin_headers):
    resp = client.get(f"{BASE}/admin/stats", headers=admin_headers)
    assert resp.status_code == 200
    assert "total_users" in resp.json()
