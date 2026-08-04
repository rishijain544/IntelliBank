"""Accounts, transactions, transfers, beneficiaries, cards and statement exports."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import ActiveUser, CurrentUser, OnboardingUser, PageParams, write_audit
from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import RateLimiter
from app.core.security import device_fingerprint
from app.models.banking import Account, Beneficiary, Card, Transaction
from app.models.enums import (
    AccountStatus,
    CardStatus,
    MerchantCategory,
    TransactionChannel,
    TransactionStatus,
    TransactionType,
)
from app.models.mixins import quantize
from app.schemas import (
    AccountCreateRequest,
    AccountResponse,
    BeneficiaryCreateRequest,
    BeneficiaryResponse,
    CardCreateRequest,
    CardFreezeRequest,
    CardLimitsRequest,
    CardResponse,
    DepositRequest,
    ExternalTransferRequest,
    FraudAssessment,
    InternalTransferRequest,
    MessageResponse,
    Page,
    TransactionResponse,
    TransferResponse,
)
from app.services import banking
from app.services.exports import (
    export_filename,
    transactions_to_csv,
    transactions_to_pdf,
)

router = APIRouter(tags=["banking"])

MAX_EXPORT_ROWS = 5000


def _account_payload(account: Account) -> AccountResponse:
    """Include the derived available balance, which is not a mapped column."""
    return AccountResponse(
        **{
            **{
                k: getattr(account, k)
                for k in (
                    "id",
                    "account_number",
                    "ifsc_code",
                    "nickname",
                    "account_type",
                    "status",
                    "currency",
                    "balance",
                    "hold_amount",
                    "overdraft_limit",
                    "interest_rate",
                    "is_primary",
                    "opened_on",
                    "created_at",
                )
            },
            "available_balance": account.available_balance,
        }
    )


def _card_payload(card: Card) -> CardResponse:
    return CardResponse(
        **{
            **{
                k: getattr(card, k)
                for k in (
                    "id",
                    "account_id",
                    "card_last4",
                    "card_network",
                    "card_type",
                    "status",
                    "cardholder_name",
                    "expiry_month",
                    "expiry_year",
                    "daily_limit",
                    "per_txn_limit",
                    "monthly_limit",
                    "credit_limit",
                    "online_enabled",
                    "international_enabled",
                    "contactless_enabled",
                    "atm_enabled",
                    "freeze_reason",
                    "created_at",
                )
            },
            "masked_number": card.masked_number,
        }
    )


def _request_context(request: Request) -> dict[str, Any]:
    """Device/IP/location signals attached to every money movement for scoring."""
    ua = request.headers.get("user-agent", "")
    ip = request.client.host if request.client else "unknown"
    return {
        "device_fingerprint": device_fingerprint(ua, ip),
        "ip_address": ip,
        # Real deployments resolve these from a geo-IP provider or the card network.
        "location_city": request.headers.get("x-client-city"),
        "location_country": request.headers.get("x-client-country", "IN"),
    }


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


@router.get("/accounts", response_model=list[AccountResponse], summary="List own accounts")
def list_accounts(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    include_closed: Annotated[bool, Query()] = False,
) -> list[AccountResponse]:
    stmt = select(Account).where(Account.user_id == user.id)
    if not include_closed:
        stmt = stmt.where(Account.status != AccountStatus.CLOSED)
    accounts = db.execute(stmt.order_by(Account.is_primary.desc(), Account.id)).scalars().all()
    return [_account_payload(a) for a in accounts]


@router.post(
    "/accounts",
    response_model=AccountResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open an additional account",
)
def create_account(
    payload: AccountCreateRequest,
    request: Request,
    # Ungated: a customer needs a funded account before any ML feature has data
    # to work with. Opening one carries no external compliance exposure.
    user: OnboardingUser,
    db: Annotated[Session, Depends(get_db)],
) -> AccountResponse:
    open_count = (
        db.execute(
            select(func.count(Account.id)).where(
                Account.user_id == user.id, Account.status == AccountStatus.ACTIVE
            )
        ).scalar_one()
        or 0
    )
    if open_count >= 6:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You have reached the maximum of 6 open accounts",
        )

    account = banking.create_account(
        db,
        user,
        account_type=payload.account_type,
        nickname=payload.nickname,
        initial_deposit=payload.initial_deposit,
    )
    write_audit(
        db,
        action="account.create",
        actor=user,
        request=request,
        entity_type="account",
        entity_id=account.id,
        summary=f"Opened {payload.account_type} account {account.account_number}",
    )
    db.commit()
    db.refresh(account)
    return _account_payload(account)


@router.get("/accounts/{account_id}", response_model=AccountResponse)
def get_account(
    account_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> AccountResponse:
    return _account_payload(banking.get_owned_account(db, user, account_id))


@router.post("/accounts/{account_id}/deposit", response_model=TransactionResponse)
def deposit(
    account_id: int,
    payload: DepositRequest,
    request: Request,
    # Ungated for the same reason as account opening: without a balance there is
    # nothing for the fraud model to score.
    user: OnboardingUser,
    db: Annotated[Session, Depends(get_db)],
) -> TransactionResponse:
    """Simulated credit, used to fund demo accounts."""
    if payload.account_id != account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Account ID mismatch"
        )
    account = banking.get_owned_account(db, user, account_id)
    txn = banking.record_deposit(
        db,
        user,
        account,
        amount=payload.amount,
        description=payload.description or "Simulated deposit",
    )
    write_audit(
        db,
        action="account.deposit",
        actor=user,
        request=request,
        entity_type="transaction",
        entity_id=txn.id,
        summary=f"Deposited {payload.amount} into {account.account_number}",
    )
    db.commit()
    db.refresh(txn)
    return TransactionResponse.model_validate(txn)


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #


def _transaction_query(
    user_id: int,
    *,
    account_id: int | None = None,
    txn_type: str | None = None,
    txn_status: str | None = None,
    category: str | None = None,
    channel: str | None = None,
    min_amount: Decimal | None = None,
    max_amount: Decimal | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    search: str | None = None,
    flagged_only: bool = False,
) -> Select:
    stmt = select(Transaction).where(Transaction.user_id == user_id)
    if account_id is not None:
        stmt = stmt.where(Transaction.account_id == account_id)
    if txn_type:
        stmt = stmt.where(Transaction.txn_type == txn_type)
    if txn_status:
        stmt = stmt.where(Transaction.status == txn_status)
    if category:
        stmt = stmt.where(Transaction.merchant_category == category)
    if channel:
        stmt = stmt.where(Transaction.channel == channel)
    if min_amount is not None:
        stmt = stmt.where(Transaction.amount >= min_amount)
    if max_amount is not None:
        stmt = stmt.where(Transaction.amount <= max_amount)
    if date_from is not None:
        stmt = stmt.where(Transaction.occurred_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Transaction.occurred_at <= date_to)
    if flagged_only:
        stmt = stmt.where(Transaction.is_flagged.is_(True))
    if search:
        like = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Transaction.description.ilike(like),
                Transaction.merchant_name.ilike(like),
                Transaction.reference.ilike(like),
                Transaction.counterparty_name.ilike(like),
            )
        )
    return stmt


@router.get(
    "/transactions",
    response_model=Page[TransactionResponse],
    summary="Transaction history with filters and pagination",
)
def list_transactions(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    pagination: PageParams,
    account_id: Annotated[int | None, Query()] = None,
    txn_type: Annotated[str | None, Query()] = None,
    txn_status: Annotated[str | None, Query(alias="status")] = None,
    category: Annotated[str | None, Query()] = None,
    channel: Annotated[str | None, Query()] = None,
    min_amount: Annotated[Decimal | None, Query(ge=0)] = None,
    max_amount: Annotated[Decimal | None, Query(ge=0)] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=120)] = None,
    flagged_only: Annotated[bool, Query()] = False,
) -> Page[TransactionResponse]:
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_amount cannot exceed max_amount",
        )

    stmt = _transaction_query(
        user.id,
        account_id=account_id,
        txn_type=txn_type,
        txn_status=txn_status,
        category=category,
        channel=channel,
        min_amount=min_amount,
        max_amount=max_amount,
        date_from=date_from,
        date_to=date_to,
        search=search,
        flagged_only=flagged_only,
    )

    total = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0
    )
    rows = (
        db.execute(
            stmt.order_by(Transaction.occurred_at.desc(), Transaction.id.desc())
            .offset(pagination.offset)
            .limit(pagination.page_size)
        )
        .scalars()
        .all()
    )
    return Page[TransactionResponse].model_validate(
        pagination.envelope([TransactionResponse.model_validate(r) for r in rows], total)
    )


# NOTE: the /transactions/export/* routes MUST be declared before
# /transactions/{transaction_id}. Starlette matches in registration order, so a
# dynamic segment declared first would swallow "export" and fail int parsing.
@router.get("/transactions/export/csv", summary="Export transactions as CSV")
def export_csv(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    account_id: Annotated[int | None, Query()] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
    category: Annotated[str | None, Query()] = None,
) -> Response:
    stmt = _transaction_query(
        user.id,
        account_id=account_id,
        date_from=date_from,
        date_to=date_to,
        category=category,
    )
    rows = (
        db.execute(stmt.order_by(Transaction.occurred_at.desc()).limit(MAX_EXPORT_ROWS))
        .scalars()
        .all()
    )
    account = db.get(Account, account_id) if account_id else None
    filename = export_filename(
        "transactions", "csv", account.account_number if account else None
    )
    return Response(
        content=transactions_to_csv(rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/transactions/export/pdf", summary="Download a PDF statement")
def export_pdf(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    account_id: Annotated[int | None, Query()] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
) -> Response:
    account = banking.get_owned_account(db, user, account_id) if account_id else None
    stmt = _transaction_query(
        user.id, account_id=account_id, date_from=date_from, date_to=date_to
    )
    rows = (
        db.execute(stmt.order_by(Transaction.occurred_at.desc()).limit(MAX_EXPORT_ROWS))
        .scalars()
        .all()
    )
    pdf = transactions_to_pdf(
        rows, user=user, account=account, period_from=date_from, period_to=date_to
    )
    filename = export_filename("statement", "pdf", account.account_number if account else None)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Declared after the static /transactions/export/* paths on purpose (see note above).
@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> TransactionResponse:
    txn = db.get(Transaction, transaction_id)
    if txn is None or txn.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return TransactionResponse.model_validate(txn)


# --------------------------------------------------------------------------- #
# Transfers
# --------------------------------------------------------------------------- #


@router.post(
    "/transfers/internal",
    response_model=TransferResponse,
    dependencies=[Depends(RateLimiter(settings.RATE_LIMIT_TRANSFER, scope="transfer"))],
    summary="Transfer to a IntelliBank account (own or another customer)",
)
def transfer_internal(
    payload: InternalTransferRequest,
    request: Request,
    # Ungated: simulated money moving between accounts inside this platform.
    # External transfers below stay KYC-gated, mirroring how real banks apply
    # stricter verification to interbank payments than to on-us transfers.
    user: OnboardingUser,
    db: Annotated[Session, Depends(get_db)],
) -> TransferResponse:
    ctx = _request_context(request)
    txn, fraud, message = banking.execute_transfer(
        db,
        user,
        from_account_id=payload.from_account_id,
        to_account_number=payload.to_account_number,
        amount=payload.amount,
        channel=TransactionChannel.INTERNAL,
        description=payload.description,
        category=payload.category or MerchantCategory.TRANSFER,
        **ctx,
    )
    write_audit(
        db,
        action="transfer.internal",
        actor=user,
        request=request,
        entity_type="transaction",
        entity_id=txn.id,
        summary=f"{payload.amount} to {payload.to_account_number}: {txn.status}",
        after_state={"risk_score": fraud.risk_score, "action": fraud.action},
        success=txn.status != TransactionStatus.BLOCKED,
    )
    db.commit()
    db.refresh(txn)
    return TransferResponse(
        transaction=TransactionResponse.model_validate(txn),
        status=txn.status,
        message=message,
        fraud=FraudAssessment(**_fraud_dict(fraud)),
    )


def _fraud_dict(fraud) -> dict[str, Any]:
    """``FraudDecision`` uses ``slots=True``, so build the payload explicitly."""
    return {
        "risk_score": fraud.risk_score,
        "action": fraud.action,
        "severity": fraud.severity,
        "is_flagged": fraud.is_flagged,
        "auto_blocked": fraud.auto_blocked,
        "reasons": fraud.reasons,
        "triggered_rules": fraud.triggered_rules,
        "top_factors": fraud.top_factors,
        "model_name": fraud.model_name,
        "model_version": fraud.model_version,
        "model_available": fraud.model_available,
        "decision_source": fraud.decision_source,
        "latency_ms": fraud.latency_ms,
    }


@router.post(
    "/transfers/external",
    response_model=TransferResponse,
    dependencies=[Depends(RateLimiter(settings.RATE_LIMIT_TRANSFER, scope="transfer"))],
    summary="Simulated NEFT/IMPS/UPI transfer to a saved beneficiary",
)
def transfer_external(
    payload: ExternalTransferRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> TransferResponse:
    beneficiary = db.get(Beneficiary, payload.beneficiary_id)
    if beneficiary is None or beneficiary.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Beneficiary not found")
    if not beneficiary.is_verified:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This beneficiary is still in the activation window",
        )

    ctx = _request_context(request)
    txn, fraud, message = banking.execute_transfer(
        db,
        user,
        from_account_id=payload.from_account_id,
        beneficiary=beneficiary,
        amount=payload.amount,
        channel=payload.channel,
        description=payload.description,
        **ctx,
    )
    write_audit(
        db,
        action="transfer.external",
        actor=user,
        request=request,
        entity_type="transaction",
        entity_id=txn.id,
        summary=f"{payload.amount} via {payload.channel} to {beneficiary.name}: {txn.status}",
        after_state={"risk_score": fraud.risk_score, "action": fraud.action},
        success=txn.status != TransactionStatus.BLOCKED,
    )
    db.commit()
    db.refresh(txn)
    return TransferResponse(
        transaction=TransactionResponse.model_validate(txn),
        status=txn.status,
        message=message,
        fraud=FraudAssessment(**_fraud_dict(fraud)),
    )


# --------------------------------------------------------------------------- #
# Beneficiaries
# --------------------------------------------------------------------------- #


@router.get("/beneficiaries", response_model=list[BeneficiaryResponse])
def list_beneficiaries(
    user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[BeneficiaryResponse]:
    rows = (
        db.execute(
            select(Beneficiary)
            .where(Beneficiary.user_id == user.id)
            .order_by(Beneficiary.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [BeneficiaryResponse.model_validate(b) for b in rows]


@router.post(
    "/beneficiaries",
    response_model=BeneficiaryResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_beneficiary(
    payload: BeneficiaryCreateRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> BeneficiaryResponse:
    duplicate = db.execute(
        select(Beneficiary.id).where(
            Beneficiary.user_id == user.id,
            Beneficiary.account_number == payload.account_number,
            Beneficiary.ifsc_code == payload.ifsc_code,
        )
    ).first()
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This beneficiary is already saved"
        )

    own = db.execute(
        select(Account.id).where(
            Account.user_id == user.id, Account.account_number == payload.account_number
        )
    ).first()
    if own:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use own-account transfer for your own accounts",
        )

    is_internal = payload.ifsc_code.upper() == banking.IFSC_CODE
    if is_internal:
        exists = db.execute(
            select(Account.id).where(Account.account_number == payload.account_number)
        ).first()
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No IntelliBank account matches that number",
            )

    beneficiary = Beneficiary(
        user_id=user.id,
        name=payload.name.strip(),
        nickname=payload.nickname,
        account_number=payload.account_number,
        ifsc_code=payload.ifsc_code,
        bank_name=payload.bank_name if not is_internal else "IntelliBank",
        is_internal=is_internal,
        transfer_limit=payload.transfer_limit,
        # Auto-verified so the demo flow is not blocked; real banks impose a
        # cooling-off window before a new payee can receive high-value transfers.
        is_verified=True,
        activated_at=datetime.now(UTC),
    )
    db.add(beneficiary)
    write_audit(
        db,
        action="beneficiary.create",
        actor=user,
        request=request,
        entity_type="beneficiary",
        summary=f"Added beneficiary {payload.name}",
    )
    db.commit()
    db.refresh(beneficiary)
    return BeneficiaryResponse.model_validate(beneficiary)


@router.delete("/beneficiaries/{beneficiary_id}", response_model=MessageResponse)
def delete_beneficiary(
    beneficiary_id: int,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    beneficiary = db.get(Beneficiary, beneficiary_id)
    if beneficiary is None or beneficiary.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Beneficiary not found")
    name = beneficiary.name
    db.delete(beneficiary)
    write_audit(
        db,
        action="beneficiary.delete",
        actor=user,
        request=request,
        entity_type="beneficiary",
        entity_id=beneficiary_id,
        summary=f"Removed beneficiary {name}",
    )
    db.commit()
    return MessageResponse(message=f"Removed {name}")


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


@router.get("/cards", response_model=list[CardResponse])
def list_cards(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> list[CardResponse]:
    rows = (
        db.execute(select(Card).where(Card.user_id == user.id).order_by(Card.id.desc()))
        .scalars()
        .all()
    )
    return [_card_payload(c) for c in rows]


@router.post("/cards", response_model=CardResponse, status_code=status.HTTP_201_CREATED)
def create_card(
    payload: CardCreateRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> CardResponse:
    account = banking.get_owned_account(db, user, payload.account_id)
    banking.assert_account_usable(account)

    active = (
        db.execute(
            select(func.count(Card.id)).where(
                Card.user_id == user.id, Card.status == CardStatus.ACTIVE
            )
        ).scalar_one()
        or 0
    )
    if active >= 5:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Maximum of 5 active cards reached"
        )

    card = banking.issue_card(
        db,
        user,
        account,
        card_type=payload.card_type,
        daily_limit=payload.daily_limit,
        per_txn_limit=payload.per_txn_limit,
        monthly_limit=payload.monthly_limit,
    )
    write_audit(
        db,
        action="card.issue",
        actor=user,
        request=request,
        entity_type="card",
        entity_id=card.id,
        summary=f"Issued {payload.card_type} ending {card.card_last4}",
    )
    db.commit()
    db.refresh(card)
    return _card_payload(card)


def _get_owned_card(db: Session, user, card_id: int) -> Card:
    card = db.get(Card, card_id)
    if card is None or card.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Card not found")
    return card


@router.patch("/cards/{card_id}/freeze", response_model=CardResponse)
def freeze_card(
    card_id: int,
    payload: CardFreezeRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> CardResponse:
    card = _get_owned_card(db, user, card_id)
    banking.set_card_frozen(db, card, freeze=payload.freeze, reason=payload.reason)
    write_audit(
        db,
        action="card.freeze" if payload.freeze else "card.unfreeze",
        actor=user,
        request=request,
        entity_type="card",
        entity_id=card.id,
        summary=f"Card ending {card.card_last4} {'frozen' if payload.freeze else 'unfrozen'}",
    )
    db.commit()
    db.refresh(card)
    return _card_payload(card)


@router.patch("/cards/{card_id}/limits", response_model=CardResponse)
def update_card_limits(
    card_id: int,
    payload: CardLimitsRequest,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> CardResponse:
    card = _get_owned_card(db, user, card_id)
    if card.status == CardStatus.CANCELLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This card has been cancelled"
        )

    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in updates.items():
        setattr(card, field, value)

    # Re-validate the hierarchy after a partial update, since any single field
    # can break the invariant the create-time validator enforced.
    if card.per_txn_limit > card.daily_limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Per-transaction limit cannot exceed the daily limit",
        )
    if card.daily_limit > card.monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Daily limit cannot exceed the monthly limit",
        )

    write_audit(
        db,
        action="card.update_limits",
        actor=user,
        request=request,
        entity_type="card",
        entity_id=card.id,
        summary=f"Updated limits on card ending {card.card_last4}",
        after_state={k: str(v) for k, v in updates.items()},
    )
    db.commit()
    db.refresh(card)
    return _card_payload(card)


@router.delete("/cards/{card_id}", response_model=MessageResponse)
def cancel_card(
    card_id: int,
    request: Request,
    user: ActiveUser,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    card = _get_owned_card(db, user, card_id)
    card.status = CardStatus.CANCELLED
    write_audit(
        db,
        action="card.cancel",
        actor=user,
        request=request,
        entity_type="card",
        entity_id=card.id,
        summary=f"Cancelled card ending {card.card_last4}",
    )
    db.commit()
    return MessageResponse(message=f"Card ending {card.card_last4} cancelled")
