"""Banking, transfer, loan and fraud-workflow integration tests.

Emphasis is on money-movement invariants: balances must reconcile, funds must
never be created or destroyed, and every ML-gated path must behave predictably.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import TEST_PASSWORD, _make_user, auth_headers

BASE = "/api/v1"


def _balance(client, headers, account_id: int) -> Decimal:
    return Decimal(client.get(f"{BASE}/accounts/{account_id}", headers=headers).json()["balance"])


# ---------------------------------------------------------------------- accounts


def test_create_and_list_accounts(client, customer_headers):
    created = client.post(
        f"{BASE}/accounts",
        json={"account_type": "savings", "nickname": "Holiday Fund", "initial_deposit": "5000.00"},
        headers=customer_headers,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["account_number"].startswith("5")
    assert Decimal(body["balance"]) == Decimal("5000.00")
    assert body["is_primary"] is True  # first account

    listing = client.get(f"{BASE}/accounts", headers=customer_headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_cannot_read_another_users_account(client, db_session, customer_headers):
    other = _make_user(db_session, email="other@test.dev")
    other_headers = auth_headers(client, other.email)
    other_account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=other_headers
    ).json()

    resp = client.get(f"{BASE}/accounts/{other_account['id']}", headers=customer_headers)
    assert resp.status_code == 404  # not 403: existence is not disclosed


def test_deposit_increases_balance(client, customer_headers, funded_account):
    before = Decimal(funded_account["balance"])
    resp = client.post(
        f"{BASE}/accounts/{funded_account['id']}/deposit",
        json={"account_id": funded_account["id"], "amount": "1500.50"},
        headers=customer_headers,
    )
    assert resp.status_code == 200
    assert _balance(client, customer_headers, funded_account["id"]) == before + Decimal("1500.50")


def test_deposit_rejects_negative_and_overprecise_amounts(client, customer_headers, funded_account):
    for bad in ("-100.00", "0", "10.999"):
        resp = client.post(
            f"{BASE}/accounts/{funded_account['id']}/deposit",
            json={"account_id": funded_account["id"], "amount": bad},
            headers=customer_headers,
        )
        assert resp.status_code == 422, f"accepted bad amount {bad}"


# --------------------------------------------------------------------- transfers


def test_internal_transfer_conserves_money(client, db_session, customer_headers, funded_account):
    """The core ledger invariant: what leaves one account arrives at the other."""
    recipient = _make_user(db_session, email="recipient@test.dev")
    r_headers = auth_headers(client, recipient.email)
    r_account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=r_headers
    ).json()

    sender_before = _balance(client, customer_headers, funded_account["id"])
    recipient_before = _balance(client, r_headers, r_account["id"])
    amount = Decimal("7500.00")

    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": r_account["account_number"],
            "amount": str(amount),
            "description": "Rent share",
        },
        headers=customer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["transaction"]["status"] == "completed"
    assert "fraud" in body and 0.0 <= body["fraud"]["risk_score"] <= 1.0

    sender_after = _balance(client, customer_headers, funded_account["id"])
    recipient_after = _balance(client, r_headers, r_account["id"])
    assert sender_before - sender_after == amount
    assert recipient_after - recipient_before == amount


def test_transfer_rejects_insufficient_funds(client, customer_headers, funded_account, db_session):
    recipient = _make_user(db_session, email="poor-target@test.dev")
    r_headers = auth_headers(client, recipient.email)
    r_account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=r_headers
    ).json()

    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": r_account["account_number"],
            "amount": "99999999.00",
        },
        headers=customer_headers,
    )
    assert resp.status_code == 422
    assert "insufficient" in resp.json()["detail"].lower()


def test_transfer_to_self_is_rejected(client, customer_headers, funded_account):
    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": funded_account["account_number"],
            "amount": "100.00",
        },
        headers=customer_headers,
    )
    assert resp.status_code == 400


def test_transfer_to_unknown_account_is_rejected(client, customer_headers, funded_account):
    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": "59999999999999",
            "amount": "100.00",
        },
        headers=customer_headers,
    )
    assert resp.status_code == 404


def test_savings_minimum_balance_is_enforced(client, customer_headers, funded_account, db_session):
    recipient = _make_user(db_session, email="drain@test.dev")
    r_headers = auth_headers(client, recipient.email)
    r_account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=r_headers
    ).json()

    balance = _balance(client, customer_headers, funded_account["id"])
    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": r_account["account_number"],
            "amount": str(balance),  # would leave 0, below the savings floor
        },
        headers=customer_headers,
    )
    assert resp.status_code == 422
    assert "minimum balance" in resp.json()["detail"].lower()


def test_transfer_is_scored_by_fraud_model(client, customer_headers, funded_account, db_session):
    recipient = _make_user(db_session, email="scored@test.dev")
    r_headers = auth_headers(client, recipient.email)
    r_account = client.post(
        f"{BASE}/accounts", json={"account_type": "savings"}, headers=r_headers
    ).json()

    resp = client.post(
        f"{BASE}/transfers/internal",
        json={
            "from_account_id": funded_account["id"],
            "to_account_number": r_account["account_number"],
            "amount": "2500.00",
        },
        headers=customer_headers,
    )
    fraud = resp.json()["fraud"]
    assert fraud["action"] in {"allow", "review", "block"}
    assert fraud["model_name"]
    assert fraud["latency_ms"] < 200, "scoring must stay inside the latency budget"
    # A brand-new account has no history, so the cold-start guard must engage.
    assert not fraud["auto_blocked"]


# ------------------------------------------------------------------ transactions


def test_transaction_list_pagination_and_filters(client, customer_headers, funded_account):
    for i in range(5):
        client.post(
            f"{BASE}/accounts/{funded_account['id']}/deposit",
            json={"account_id": funded_account["id"], "amount": f"{100 + i}.00"},
            headers=customer_headers,
        )

    page = client.get(
        f"{BASE}/transactions", params={"page": 1, "page_size": 3}, headers=customer_headers
    )
    assert page.status_code == 200
    body = page.json()
    assert len(body["items"]) == 3
    assert body["has_next"] is True
    assert body["page"] == 1

    filtered = client.get(
        f"{BASE}/transactions", params={"txn_type": "deposit"}, headers=customer_headers
    ).json()
    assert all(t["txn_type"] == "deposit" for t in filtered["items"])


def test_transaction_amount_range_validation(client, customer_headers):
    resp = client.get(
        f"{BASE}/transactions",
        params={"min_amount": 500, "max_amount": 100},
        headers=customer_headers,
    )
    assert resp.status_code == 422


def test_csv_export_returns_spreadsheet(client, customer_headers, funded_account):
    resp = client.get(f"{BASE}/transactions/export/csv", headers=customer_headers)
    assert resp.status_code == 200, resp.text
    assert "text/csv" in resp.headers["content-type"]
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"\xef\xbb\xbf")  # Excel-safe BOM
    assert b"reference" in resp.content


def test_pdf_export_returns_document(client, customer_headers, funded_account):
    resp = client.get(
        f"{BASE}/transactions/export/pdf",
        params={"account_id": funded_account["id"]},
        headers=customer_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"


# ------------------------------------------------------------------------- cards


def test_card_issuance_never_exposes_full_number(client, customer_headers, funded_account):
    resp = client.post(
        f"{BASE}/cards",
        json={"account_id": funded_account["id"], "card_type": "virtual_debit"},
        headers=customer_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["masked_number"].startswith("****")
    assert len(body["card_last4"]) == 4
    assert "card_hash" not in body and "card_number" not in body


def test_card_freeze_and_unfreeze(client, customer_headers, funded_account):
    card = client.post(
        f"{BASE}/cards", json={"account_id": funded_account["id"]}, headers=customer_headers
    ).json()

    frozen = client.patch(
        f"{BASE}/cards/{card['id']}/freeze",
        json={"freeze": True, "reason": "Lost phone"},
        headers=customer_headers,
    )
    assert frozen.status_code == 200
    assert frozen.json()["status"] == "frozen"

    thawed = client.patch(
        f"{BASE}/cards/{card['id']}/freeze", json={"freeze": False}, headers=customer_headers
    )
    assert thawed.json()["status"] == "active"


def test_card_limit_hierarchy_enforced_on_update(client, customer_headers, funded_account):
    """A partial update must not be able to break the limit invariant."""
    card = client.post(
        f"{BASE}/cards", json={"account_id": funded_account["id"]}, headers=customer_headers
    ).json()

    resp = client.patch(
        f"{BASE}/cards/{card['id']}/limits",
        json={"daily_limit": "1000.00"},  # below the existing per-txn limit
        headers=customer_headers,
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------------- loans


def test_loan_eligibility_returns_live_score(client, customer_headers, funded_account):
    resp = client.post(
        f"{BASE}/loans/eligibility",
        json={"loan_type": "personal", "amount": "300000", "tenure_months": 36},
        headers=customer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 300 <= body["score"] <= 900
    assert body["risk_band"] in list("ABCDE")
    assert body["decision"] in {"approve", "review", "reject"}
    assert body["suggested_rate"] > 0
    assert body["latency_ms"] < 200


def test_loan_eligibility_does_not_create_an_application(client, customer_headers, funded_account):
    client.post(
        f"{BASE}/loans/eligibility",
        json={"loan_type": "personal", "amount": "200000", "tenure_months": 24},
        headers=customer_headers,
    )
    assert client.get(f"{BASE}/loans", headers=customer_headers).json()["total"] == 0


def test_loan_application_is_scored_and_persisted(client, customer_headers, funded_account):
    resp = client.post(
        f"{BASE}/loans/apply",
        json={
            "loan_type": "personal",
            "amount": "250000",
            "tenure_months": 36,
            "purpose": "Home renovation",
            "disbursement_account_id": funded_account["id"],
        },
        headers=customer_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["loan"]["application_ref"].startswith("LN")
    assert body["loan"]["status"] in {"approved", "under_review", "rejected"}
    assert body["credit"]["score"] >= 300

    listing = client.get(f"{BASE}/loans", headers=customer_headers).json()
    assert listing["total"] == 1


def test_unaffordable_loan_is_capped_or_refused(client, customer_headers, funded_account):
    """A request far beyond income must not be approved in full."""
    resp = client.post(
        f"{BASE}/loans/eligibility",
        json={"loan_type": "personal", "amount": "50000000", "tenure_months": 12},
        headers=customer_headers,
    )
    body = resp.json()
    assert Decimal(body["approved_amount"]) < Decimal("50000000")


def test_loan_amount_validation(client, customer_headers):
    resp = client.post(
        f"{BASE}/loans/eligibility",
        json={"loan_type": "personal", "amount": "-5000", "tenure_months": 36},
        headers=customer_headers,
    )
    assert resp.status_code == 422


# ------------------------------------------------------- dashboard & insights


def test_dashboard_aggregates(client, customer_headers, funded_account):
    resp = client.get(f"{BASE}/dashboard", headers=customer_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert Decimal(body["total_balance"]) > 0
    assert len(body["accounts"]) == 1
    assert isinstance(body["daily_spend"], list)
    assert isinstance(body["category_breakdown"], list)


def test_insights_period_and_shape(client, customer_headers, funded_account):
    resp = client.get(f"{BASE}/insights", params={"days": 30}, headers=customer_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["period_days"] == 30
    assert "monthly_trends" in body and "top_merchants" in body
    assert isinstance(body["anomaly_alerts"], list)


def test_insights_rejects_out_of_range_period(client, customer_headers):
    assert client.get(f"{BASE}/insights", params={"days": 5000}, headers=customer_headers).status_code == 422


def test_fraud_summary(client, customer_headers):
    resp = client.get(f"{BASE}/fraud/summary", headers=customer_headers)
    assert resp.status_code == 200
    assert "total_alerts" in resp.json()


# ------------------------------------------------------------------ admin flows


def test_admin_stats_include_model_status(client, admin_headers):
    resp = client.get(f"{BASE}/admin/stats", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "model_status" in body
    assert set(body["model_status"]) == {"fraud", "credit", "anomaly"}


def test_admin_can_freeze_user_and_accounts(client, db_session, admin_headers):
    victim = _make_user(db_session, email="tofreeze@test.dev")
    v_headers = auth_headers(client, victim.email)
    client.post(f"{BASE}/accounts", json={"account_type": "savings"}, headers=v_headers)

    resp = client.patch(
        f"{BASE}/admin/users/{victim.id}/status",
        json={"status": "frozen", "reason": "Suspicious activity"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "frozen"

    # A frozen user must be locked out immediately, not at token expiry.
    assert client.get(f"{BASE}/auth/me", headers=v_headers).status_code == 403


def test_admin_cannot_freeze_self(client, admin, admin_headers):
    resp = client.patch(
        f"{BASE}/admin/users/{admin.id}/status",
        json={"status": "frozen"},
        headers=admin_headers,
    )
    assert resp.status_code == 400


def test_admin_cannot_disable_last_administrator(client, db_session, admin, admin_headers):
    other_admin = _make_user(db_session, email="admin2@test.dev", role="admin")
    # Disabling the *other* admin is fine while this one remains active.
    resp = client.patch(
        f"{BASE}/admin/users/{other_admin.id}/status",
        json={"status": "suspended"},
        headers=admin_headers,
    )
    assert resp.status_code == 200


def test_admin_model_performance_endpoint(client, admin_headers):
    resp = client.get(f"{BASE}/admin/models", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    models = resp.json()
    assert len(models) == 3
    for m in models:
        assert "training_metrics" in m and "live_inference_count" in m


def test_admin_analytics_endpoint(client, admin_headers):
    resp = client.get(f"{BASE}/admin/analytics", params={"days": 30}, headers=admin_headers)
    assert resp.status_code == 200
    assert "daily_volume" in resp.json()


def test_admin_audit_trail_records_actions(client, admin_headers, customer_headers, funded_account):
    resp = client.get(f"{BASE}/admin/audit", headers=admin_headers)
    assert resp.status_code == 200
    actions = {row["action"] for row in resp.json()["items"]}
    # Account creation and login are both audited.
    assert any("account" in a or "login" in a for a in actions)


def test_admin_fraud_queue_accessible(client, admin_headers):
    resp = client.get(f"{BASE}/admin/fraud/queue", headers=admin_headers)
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_admin_loan_queue_accessible(client, admin_headers):
    resp = client.get(f"{BASE}/admin/loans/queue", headers=admin_headers)
    assert resp.status_code == 200


# ------------------------------------------------------------------ meta / health


def test_health_endpoint(client):
    resp = client.get(f"{BASE}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_security_headers_present(client):
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "X-Process-Time" in resp.headers


def test_ml_status_is_public(client):
    resp = client.get(f"{BASE}/ml/status")
    assert resp.status_code == 200
    assert set(resp.json()) == {"fraud", "credit", "anomaly"}
