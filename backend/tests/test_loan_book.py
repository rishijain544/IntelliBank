"""Loan book: repayment position, overdue filtering and payment reminders.

The interesting behaviour here is that ``next_due_date`` and ``days_overdue`` are
*derived* on every read rather than stored, so these tests pin the derivation
rule itself. The reminder endpoint re-derives it with the same helper, which is
what stops a stale admin page from triggering a reminder on a loan that has since
been paid -- so that agreement is asserted directly.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from dateutil.relativedelta import relativedelta

from app.models.banking import Account
from app.models.enums import (
    AccountStatus,
    AccountType,
    LoanStatus,
    LoanType,
    NotificationType,
)
from app.models.lending import Loan
from app.models.system import Notification
from app.services import email as email_service

from .conftest import _make_user

BASE = "/api/v1"


def _seed_account(db, user, *, number: str, balance: str = "50000.00") -> Account:
    account = Account(
        account_number=number,
        user_id=user.id,
        account_type=AccountType.SAVINGS.value,
        status=AccountStatus.ACTIVE.value,
        balance=Decimal(balance),
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _seed_loan(
    db,
    user,
    *,
    ref: str,
    status: str = LoanStatus.DISBURSED.value,
    emis_paid: int = 0,
    first_emi_date: date | None = None,
    emi_amount: str = "10000.00",
    outstanding: str = "300000.00",
) -> Loan:
    """A disbursed loan whose schedule origin is set explicitly.

    ``first_emi_date`` is the knob every test in this module turns: combined with
    ``emis_paid`` it fully determines whether the loan reads as overdue.
    """
    account = _seed_account(db, user, number=f"9{user.id:013d}")
    loan = Loan(
        application_ref=ref,
        user_id=user.id,
        disbursement_account_id=account.id,
        loan_type=LoanType.PERSONAL.value,
        status=status,
        requested_amount=Decimal("400000.00"),
        approved_amount=Decimal("400000.00"),
        tenure_months=36,
        interest_rate=12.5,
        emi_amount=Decimal(emi_amount),
        outstanding_principal=Decimal(outstanding),
        emis_paid=emis_paid,
        emis_missed=0,
        decision_source="model",
        disbursed_at=datetime.now(UTC) - timedelta(days=200),
        first_emi_date=first_emi_date,
    )
    db.add(loan)
    db.commit()
    db.refresh(loan)
    return loan


@pytest.fixture(autouse=True)
def _clear_outbox():
    """The outbox is process-global; reset it so counts are per-test."""
    email_service.outbox.clear()
    yield
    email_service.outbox.clear()


@pytest.fixture
def overdue_loan(db_session):
    """A loan whose next EMI fell due 45 days ago."""
    user = _make_user(db_session, email="overdue@test.dev")
    today = datetime.now(UTC).date()
    # 3 paid EMIs, so the schedule origin is 3 months before the missed due date.
    return _seed_loan(
        db_session,
        user,
        ref="LNOVERDUE1",
        emis_paid=3,
        first_emi_date=today - relativedelta(months=3) - timedelta(days=45),
    )


@pytest.fixture
def current_loan(db_session):
    """A loan whose next EMI is not due for another ~15 days."""
    user = _make_user(db_session, email="current@test.dev")
    today = datetime.now(UTC).date()
    return _seed_loan(
        db_session,
        user,
        ref="LNCURRENT1",
        emis_paid=2,
        first_emi_date=today - relativedelta(months=2) + timedelta(days=15),
    )


# ------------------------------------------------------------------ listing


def test_loan_book_requires_admin(client, customer_headers):
    resp = client.get(f"{BASE}/admin/loans/book", headers=customer_headers)
    assert resp.status_code == 403


def test_loan_book_lists_disbursed_loan_with_borrower(client, admin_headers, overdue_loan):
    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["total"] == 1
    row = body["items"][0]
    assert row["application_ref"] == "LNOVERDUE1"
    # An admin chasing a payment needs to know who to contact.
    assert row["borrower_email"] == "overdue@test.dev"
    assert row["borrower_id"] == overdue_loan.user_id
    assert row["days_overdue"] == 45


def test_next_due_date_steps_by_calendar_month(client, admin_headers, overdue_loan):
    """Month arithmetic must be calendar-correct, not 30-day approximation."""
    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    row = resp.json()["items"][0]

    expected = overdue_loan.first_emi_date + relativedelta(months=overdue_loan.emis_paid)
    assert row["next_due_date"] == expected.isoformat()


def test_current_loan_reports_zero_overdue(client, admin_headers, current_loan):
    """Not-yet-due must read as 0, never as a negative day count."""
    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    row = resp.json()["items"][0]
    assert row["days_overdue"] == 0
    assert row["next_due_date"] > datetime.now(UTC).date().isoformat()


def test_overdue_only_filter_excludes_current_loans(
    client, admin_headers, overdue_loan, current_loan
):
    unfiltered = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    assert unfiltered.json()["total"] == 2

    filtered = client.get(
        f"{BASE}/admin/loans/book", params={"overdue_only": True}, headers=admin_headers
    )
    assert filtered.status_code == 200
    items = filtered.json()["items"]
    assert [r["application_ref"] for r in items] == ["LNOVERDUE1"]


def test_loan_without_schedule_is_never_overdue(client, admin_headers, db_session):
    """A disbursed loan with no first_emi_date has no schedule yet."""
    user = _make_user(db_session, email="noschedule@test.dev")
    _seed_loan(db_session, user, ref="LNNOSCHED1", first_emi_date=None)

    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    row = resp.json()["items"][0]
    assert row["next_due_date"] is None
    assert row["days_overdue"] == 0

    filtered = client.get(
        f"{BASE}/admin/loans/book", params={"overdue_only": True}, headers=admin_headers
    )
    assert filtered.json()["total"] == 0


def test_loan_book_excludes_undisbursed_applications(client, admin_headers, db_session):
    """The book tracks repayment; pending applications belong to the queue."""
    user = _make_user(db_session, email="pendingloan@test.dev")
    _seed_loan(
        db_session,
        user,
        ref="LNPENDING1",
        status=LoanStatus.SUBMITTED.value,
        first_emi_date=datetime.now(UTC).date() - timedelta(days=90),
    )

    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    assert resp.json()["total"] == 0


def test_loan_book_sorts_most_overdue_first(client, admin_headers, db_session):
    today = datetime.now(UTC).date()
    for name, days in (("mild", 5), ("severe", 120), ("moderate", 40)):
        user = _make_user(db_session, email=f"{name}@test.dev")
        _seed_loan(
            db_session,
            user,
            ref=f"LN{name.upper()}",
            emis_paid=1,
            first_emi_date=today - relativedelta(months=1) - timedelta(days=days),
        )

    resp = client.get(f"{BASE}/admin/loans/book", headers=admin_headers)
    overdue = [r["days_overdue"] for r in resp.json()["items"]]
    assert overdue == sorted(overdue, reverse=True)
    assert overdue[0] == 120


def test_existing_loan_queue_is_unchanged(client, admin_headers, overdue_loan):
    """The approval queue must not start reporting disbursed loans."""
    resp = client.get(f"{BASE}/admin/loans/queue", headers=admin_headers)
    assert resp.status_code == 200
    refs = [row["application_ref"] for row in resp.json()["items"]]
    assert "LNOVERDUE1" not in refs


# ------------------------------------------------------------------ reminders


def test_reminder_requires_admin(client, customer_headers, overdue_loan):
    resp = client.post(
        f"{BASE}/admin/loans/{overdue_loan.id}/remind", headers=customer_headers
    )
    assert resp.status_code == 403


def test_reminder_sends_email_and_notification(
    client, admin_headers, db_session, overdue_loan
):
    resp = client.post(f"{BASE}/admin/loans/{overdue_loan.id}/remind", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    assert "overdue@test.dev" in resp.json()["message"]

    sent = email_service.outbox.recent()
    assert len(sent) == 1
    message = sent[0]
    assert message.to == "overdue@test.dev"
    assert "LNOVERDUE1" in message.subject
    # The facts an overdue borrower needs, and nothing dressed up.
    assert "45" in message.text
    assert "10,000.00" in message.text
    assert "300,000.00" in message.text

    notif = (
        db_session.query(Notification)
        .filter(Notification.user_id == overdue_loan.user_id)
        .one()
    )
    assert notif.notif_type == NotificationType.LOAN_UPDATE
    assert notif.meta["loan_ref"] == "LNOVERDUE1"
    assert notif.meta["days_overdue"] == 45
    assert notif.action_url == "/app/loans"


def test_reminder_rejects_loan_that_is_not_overdue(client, admin_headers, current_loan):
    """Server re-derives overdue status; a stale page cannot force a reminder."""
    resp = client.post(f"{BASE}/admin/loans/{current_loan.id}/remind", headers=admin_headers)
    assert resp.status_code == 400
    assert "not currently overdue" in resp.json()["detail"]
    assert email_service.outbox.recent() == []


def test_reminder_rejects_undisbursed_loan(client, admin_headers, db_session):
    user = _make_user(db_session, email="submitted@test.dev")
    loan = _seed_loan(
        db_session,
        user,
        ref="LNSUBMIT01",
        status=LoanStatus.SUBMITTED.value,
        first_emi_date=datetime.now(UTC).date() - timedelta(days=60),
    )

    resp = client.post(f"{BASE}/admin/loans/{loan.id}/remind", headers=admin_headers)
    assert resp.status_code == 400
    assert "disbursed" in resp.json()["detail"]


def test_reminder_on_missing_loan_returns_404(client, admin_headers):
    resp = client.post(f"{BASE}/admin/loans/999999/remind", headers=admin_headers)
    assert resp.status_code == 404


def test_reminder_is_audited(client, admin_headers, overdue_loan):
    client.post(f"{BASE}/admin/loans/{overdue_loan.id}/remind", headers=admin_headers)

    audit = client.get(f"{BASE}/admin/audit", headers=admin_headers)
    entries = [row for row in audit.json()["items"] if row["action"] == "admin.loan_reminder"]
    assert len(entries) == 1
    assert "LNOVERDUE1" in entries[0]["summary"]


def test_reminder_survives_email_transport_failure(
    client, admin_headers, db_session, overdue_loan, monkeypatch
):
    """A dead transport must not roll back the notification or the audit row."""
    monkeypatch.setattr(email_service, "_deliver", lambda to, subject, html: False)

    resp = client.post(f"{BASE}/admin/loans/{overdue_loan.id}/remind", headers=admin_headers)
    assert resp.status_code == 200
    assert "email delivery is unavailable" in resp.json()["message"]

    # The durable outcome still happened.
    assert (
        db_session.query(Notification)
        .filter(Notification.user_id == overdue_loan.user_id)
        .count()
        == 1
    )


def test_reminder_ignores_notification_preferences(
    client, admin_headers, db_session, overdue_loan
):
    """Repayment reminders are account-critical, not marketing."""
    borrower = db_session.get(Loan, overdue_loan.id).user
    borrower.notify_marketing = False
    borrower.notify_large_txn = False
    db_session.commit()

    resp = client.post(f"{BASE}/admin/loans/{overdue_loan.id}/remind", headers=admin_headers)
    assert resp.status_code == 200
    assert (
        db_session.query(Notification).filter(Notification.user_id == borrower.id).count() == 1
    )


# ------------------------------------------------------------------ email template


def test_email_escapes_borrower_controlled_text():
    """Names are user-supplied, so the template must not emit raw markup."""
    html = email_service.render_basic_email(
        heading="Payment reminder",
        intro="Hello <script>alert(1)</script>,",
        rows=[("Loan reference", "<b>LN1</b>")],
        outro="Thanks",
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>LN1</b>" not in html


def test_send_email_rejects_invalid_address_without_raising():
    assert email_service.send_email("not-an-email", "Subject", "<p>Body</p>") is False
    assert email_service.outbox.stats()["failed"] == 1
