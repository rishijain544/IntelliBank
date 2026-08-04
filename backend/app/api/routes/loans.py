"""Loan applications with live credit scoring.

The eligibility endpoint is a dry run: it scores the applicant and returns pricing
without persisting an application, so the UI can show a live quote as the user
adjusts amount and tenure. ``/apply`` scores again server-side rather than
trusting a client-supplied quote, because pricing must never be client-controlled.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import ActiveUser, CurrentUser, OnboardingUser, PageParams, write_audit
from app.core.database import get_db
from app.core.security import random_reference
from app.ml.inference import CreditDecision, score_credit
from app.models.enums import AlertSeverity, LoanStatus, NotificationType
from app.models.lending import CreditScore, Loan
from app.models.mixins import quantize
from app.schemas import (
    CreditScoreResponse,
    LoanApplicationResult,
    LoanApplyRequest,
    LoanEligibilityRequest,
    LoanResponse,
    MessageResponse,
    Page,
)
from app.services import banking, notifications as notif
from app.services.ml_features import build_credit_profile

router = APIRouter(prefix="/loans", tags=["loans"])

MAX_OPEN_APPLICATIONS = 3


def _credit_payload(d: CreditDecision) -> CreditScoreResponse:
    return CreditScoreResponse(
        score=d.score,
        probability_of_default=d.probability_of_default,
        risk_band=d.risk_band,
        decision=d.decision,
        suggested_rate=d.suggested_rate,
        max_eligible_amount=quantize(d.max_eligible_amount),
        approved_amount=quantize(d.approved_amount),
        emi_amount=quantize(d.emi_amount),
        total_payable=quantize(d.total_payable),
        processing_fee=quantize(d.processing_fee),
        reasons=d.reasons,
        top_factors=d.top_factors,
        model_name=d.model_name,
        model_version=d.model_version,
        model_available=d.model_available,
        latency_ms=d.latency_ms,
    )


def _score_applicant(
    db: Session, user, payload: LoanEligibilityRequest
) -> CreditDecision:
    profile = build_credit_profile(
        db,
        user,
        requested_amount=payload.amount,
        tenure_months=payload.tenure_months,
        loan_type=payload.loan_type,
        declared_income=payload.declared_income,
    )
    return score_credit(profile)


def _persist_score(db: Session, user, decision: CreditDecision) -> CreditScore:
    row = CreditScore(
        user_id=user.id,
        score=decision.score,
        probability_of_default=decision.probability_of_default,
        risk_band=decision.risk_band,
        decision=decision.decision,
        suggested_rate=decision.suggested_rate,
        max_eligible_amount=quantize(decision.max_eligible_amount),
        model_name=decision.model_name,
        model_version=decision.model_version,
        inference_latency_ms=decision.latency_ms,
        features=decision.features,
        top_factors=decision.top_factors,
    )
    db.add(row)
    db.flush()
    return row


@router.post(
    "/eligibility",
    response_model=CreditScoreResponse,
    summary="Live credit score and indicative pricing (no application created)",
)
def check_eligibility(
    payload: LoanEligibilityRequest,
    # Ungated: a dry-run quote. It creates no application and writes nothing to
    # the applicant's record, so there is nothing to verify identity against.
    # The binding /apply endpoint below still requires KYC.
    user: OnboardingUser,
    db: Annotated[Session, Depends(get_db)],
) -> CreditScoreResponse:
    return _credit_payload(_score_applicant(db, user, payload))


@router.post(
    "/apply",
    response_model=LoanApplicationResult,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a loan application (scored server-side)",
)
def apply_for_loan(
    payload: LoanApplyRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> LoanApplicationResult:
    open_count = (
        db.execute(
            select(func.count(Loan.id)).where(
                Loan.user_id == user.id,
                Loan.status.in_(
                    [
                        LoanStatus.SUBMITTED.value,
                        LoanStatus.UNDER_REVIEW.value,
                        LoanStatus.APPROVED.value,
                    ]
                ),
            )
        ).scalar_one()
        or 0
    )
    if open_count >= MAX_OPEN_APPLICATIONS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You already have {open_count} applications in progress",
        )

    disbursement_account = None
    if payload.disbursement_account_id is not None:
        disbursement_account = banking.get_owned_account(
            db, user, payload.disbursement_account_id
        )

    decision = _score_applicant(db, user, payload)
    score_row = _persist_score(db, user, decision)

    # Band A/B auto-approve; C/D go to the manual queue; E is declined outright.
    if decision.decision == "approve":
        loan_status = LoanStatus.APPROVED
    elif decision.decision == "reject":
        loan_status = LoanStatus.REJECTED
    else:
        loan_status = LoanStatus.UNDER_REVIEW

    loan = Loan(
        application_ref=random_reference("LN", 8),
        user_id=user.id,
        disbursement_account_id=disbursement_account.id if disbursement_account else None,
        credit_score_id=score_row.id,
        loan_type=payload.loan_type,
        status=loan_status,
        requested_amount=quantize(payload.amount),
        approved_amount=quantize(decision.approved_amount) if loan_status != LoanStatus.REJECTED else None,
        tenure_months=payload.tenure_months,
        purpose=payload.purpose,
        interest_rate=decision.suggested_rate,
        emi_amount=quantize(decision.emi_amount) if decision.emi_amount else None,
        total_payable=quantize(decision.total_payable) if decision.total_payable else None,
        processing_fee=quantize(decision.processing_fee) if decision.processing_fee else None,
        declared_income=quantize(payload.declared_income) if payload.declared_income else user.annual_income,
        existing_emi=user.existing_emi,
        employment_status=user.employment_status,
        employment_years=user.employment_years,
        decision_source="model" if decision.model_available else "rule",
        decision_reason="; ".join(decision.reasons[:3]),
        decided_at=datetime.now(UTC) if loan_status != LoanStatus.UNDER_REVIEW else None,
    )
    db.add(loan)
    db.flush()

    severity = AlertSeverity.LOW if loan_status == LoanStatus.APPROVED else AlertSeverity.MEDIUM
    body = {
        LoanStatus.APPROVED: (
            f"Approved for {decision.approved_amount:,.2f} at {decision.suggested_rate}% "
            f"over {payload.tenure_months} months (EMI {decision.emi_amount:,.2f})."
        ),
        LoanStatus.REJECTED: "We are unable to approve this application at present.",
        LoanStatus.UNDER_REVIEW: "Your application is being reviewed by our credit team.",
    }[loan_status]

    notif.notify(
        db,
        user,
        notif_type=NotificationType.LOAN_UPDATE,
        severity=severity,
        title=f"Loan application {loan.application_ref}: {loan_status.replace('_', ' ')}",
        body=body,
        action_url="/app/loans",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="loan.apply",
        actor=user,
        request=request,
        entity_type="loan",
        entity_id=loan.id,
        summary=f"{payload.loan_type} {payload.amount} -> {loan_status} (score {decision.score})",
        after_state={"band": decision.risk_band, "pd": decision.probability_of_default},
    )
    db.commit()
    db.refresh(loan)
    return LoanApplicationResult(
        loan=LoanResponse.model_validate(loan), credit=_credit_payload(decision)
    )


@router.get("", response_model=Page[LoanResponse], summary="Own loan applications")
def list_loans(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
) -> Page[LoanResponse]:
    stmt = select(Loan).where(Loan.user_id == user.id)
    total = int(db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(Loan.created_at.desc())
            .offset(pagination.offset)
            .limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[LoanResponse].model_validate(
        pagination.envelope([LoanResponse.model_validate(r) for r in rows], total)
    )


@router.get("/{loan_id}", response_model=LoanApplicationResult)
def get_loan(
    loan_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> LoanApplicationResult:
    loan = db.get(Loan, loan_id)
    if loan is None or loan.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Loan not found")

    score = db.get(CreditScore, loan.credit_score_id) if loan.credit_score_id else None
    credit = CreditScoreResponse(
        score=score.score if score else 0,
        probability_of_default=score.probability_of_default if score else 0.0,
        risk_band=score.risk_band if score else "E",
        decision=score.decision if score else "review",
        suggested_rate=score.suggested_rate if score else 0.0,
        max_eligible_amount=score.max_eligible_amount or Decimal("0") if score else Decimal("0"),
        approved_amount=loan.approved_amount or Decimal("0"),
        emi_amount=loan.emi_amount or Decimal("0"),
        total_payable=loan.total_payable or Decimal("0"),
        processing_fee=loan.processing_fee or Decimal("0"),
        reasons=[loan.decision_reason] if loan.decision_reason else [],
        top_factors=score.top_factors or [] if score else [],
        model_name=score.model_name if score else "n/a",
        model_version=score.model_version if score else "n/a",
        model_available=bool(score),
        latency_ms=score.inference_latency_ms or 0.0 if score else 0.0,
    )
    return LoanApplicationResult(loan=LoanResponse.model_validate(loan), credit=credit)


@router.post("/{loan_id}/accept", response_model=LoanResponse, summary="Accept and disburse")
def accept_loan(
    loan_id: int,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> LoanResponse:
    loan = db.get(Loan, loan_id)
    if loan is None or loan.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Loan not found")
    if loan.status != LoanStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only approved applications can be disbursed",
        )

    account_id = loan.disbursement_account_id
    if account_id is None:
        primary = db.execute(
            select(banking.Account)
            .where(banking.Account.user_id == user.id)
            .order_by(banking.Account.is_primary.desc())
            .limit(1)
        ).scalar_one_or_none()
        if primary is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="No account available for disbursement"
            )
        account_id = primary.id
        loan.disbursement_account_id = account_id

    account = banking.get_owned_account(db, user, account_id)
    net = quantize((loan.approved_amount or Decimal("0")) - (loan.processing_fee or Decimal("0")))

    banking.record_deposit(
        db,
        user,
        account,
        amount=net,
        description=f"Loan disbursement {loan.application_ref} (net of fees)",
    )

    now = datetime.now(UTC)
    loan.status = LoanStatus.DISBURSED
    loan.disbursed_at = now
    loan.outstanding_principal = loan.approved_amount
    loan.first_emi_date = (now + timedelta(days=30)).date()

    notif.notify(
        db,
        user,
        notif_type=NotificationType.LOAN_UPDATE,
        title="Loan disbursed",
        body=f"{net:,.2f} has been credited to account {account.account_number}.",
        action_url="/app/loans",
        respect_preferences=False,
    )
    write_audit(
        db,
        action="loan.disburse",
        actor=user,
        request=request,
        entity_type="loan",
        entity_id=loan.id,
        summary=f"Disbursed {net} to {account.account_number}",
    )
    db.commit()
    db.refresh(loan)
    return LoanResponse.model_validate(loan)


@router.post("/{loan_id}/cancel", response_model=MessageResponse)
def cancel_loan(
    loan_id: int,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    loan = db.get(Loan, loan_id)
    if loan is None or loan.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Loan not found")
    if loan.status in (LoanStatus.DISBURSED, LoanStatus.CLOSED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="A disbursed loan cannot be cancelled"
        )
    loan.status = LoanStatus.CLOSED
    loan.decision_reason = "Withdrawn by applicant"
    write_audit(
        db,
        action="loan.cancel",
        actor=user,
        request=request,
        entity_type="loan",
        entity_id=loan.id,
    )
    db.commit()
    return MessageResponse(message=f"Application {loan.application_ref} withdrawn")
