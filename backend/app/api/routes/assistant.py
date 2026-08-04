"""AI banking assistant endpoints.

The assistant is read-only and hard-scoped to the caller. See
``app/services/assistant_tools.py`` for the authorisation choke point and
``app/services/assistant.py`` for the Gemini function-calling loop.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, write_audit
from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import RateLimiter
from app.schemas import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantStatusResponse,
    AssistantToolCall,
)
from app.services.assistant import ask_assistant, assistant_status

logger = logging.getLogger("intellibank.assistant.api")

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post(
    "/chat",
    response_model=AssistantChatResponse,
    # Rate limited per user: each message can fan out into several LLM calls,
    # so this is both an abuse control and a cost control.
    dependencies=[Depends(RateLimiter(settings.RATE_LIMIT_ASSISTANT, scope="assistant", by_user=True))],
    summary="Ask the banking assistant a question about your own data",
)
def chat(
    payload: AssistantChatRequest,
    request: Request,
    # CurrentUser, not ActiveUser: the assistant is read-only and answering
    # "what's my balance" for an unverified user leaks nothing.
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> AssistantChatResponse:
    reply = ask_assistant(
        db,
        user,
        payload.message,
        history=[turn.model_dump() for turn in payload.history],
    )

    # Audit the interaction. The question is recorded (truncated) because an
    # assistant that can read financial data should leave a trail; the answer is
    # not stored, to avoid duplicating account data into the audit table.
    write_audit(
        db,
        action="assistant.query",
        actor=user,
        request=request,
        entity_type="assistant",
        summary=f"{payload.message[:180]} -> {reply.engine}"
        + (f" ({reply.degraded_reason})" if reply.degraded_reason else ""),
        after_state={
            "tools_called": [call.name for call in reply.tool_calls],
            "engine": reply.engine,
            "latency_ms": reply.latency_ms,
        },
    )
    db.commit()

    return AssistantChatResponse(
        message=reply.message,
        tool_calls=[
            AssistantToolCall(
                name=call.name,
                arguments=call.arguments,
                ok=call.ok,
                duration_ms=call.duration_ms,
            )
            for call in reply.tool_calls
        ],
        engine=reply.engine,
        model=reply.model,
        latency_ms=reply.latency_ms,
        degraded_reason=reply.degraded_reason,
    )


@router.get(
    "/status",
    response_model=AssistantStatusResponse,
    summary="Assistant availability and the tools it can call",
)
def status(user: CurrentUser) -> AssistantStatusResponse:
    return AssistantStatusResponse(**assistant_status())
