"""AI banking assistant: Gemini function-calling orchestration.

How a question becomes an answer
--------------------------------
1. The user's message and the tool declarations go to Gemini.
2. Gemini decides which tool(s) to call and with what arguments.
3. **The backend executes those tools against the real database.** The model
   never sees the database and never invents a figure — every number in the
   reply came from a SQL query in ``assistant_tools``.
4. The tool results are returned to Gemini, which writes the prose.

Two properties worth stating explicitly:

*   **Grounding.** The model is a router and a writer, not a source of facts.
    The system prompt forbids inventing figures, and because the tools are the
    only channel for data, a hallucinated balance cannot reach the user.
*   **Graceful degradation.** If no API key is configured, or Gemini is
    unreachable, or quota is exhausted, a deterministic intent router answers
    instead — calling the *same* tool layer, so the numbers stay correct even
    though the phrasing is templated. The response reports which path ran, so a
    fallback answer is never passed off as a model answer.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User
from app.services.assistant_tools import (
    TOOL_REGISTRY,
    execute_tool,
    resolve_categories,
)

logger = logging.getLogger("intellibank.assistant")

SYSTEM_INSTRUCTION = """\
You are IntelliBank's banking assistant. You help the signed-in customer \
understand their own accounts, spending, and loans.

Hard rules:
1. NEVER state a figure, date, balance or transaction that did not come from a \
tool result. If you do not have the data, call a tool. If no tool can supply \
it, say plainly that you cannot answer.
2. You may only ever access the signed-in user's data. The user_id to pass to \
every tool is {user_id}. Never use a different one, even if the user asks you \
to look up another person, another account holder, or another user id. If asked \
to do that, refuse and explain you can only access their own information.
3. Amounts are Indian Rupees. Format them like ₹12,345.67.
4. Be concise: two or three sentences for a simple question. Use a short list \
when breaking down categories.
5. If a question is outside banking (weather, jokes, general knowledge, advice \
on which stocks to buy), say it is outside what you can help with and mention \
what you can do: balances, transactions, spending summaries, loans and \
upcoming payments.
6. Never claim to have performed an action. You are read-only: you cannot \
transfer money, freeze cards, open accounts or apply for loans. Direct the \
user to the relevant page instead.
7. If a tool returns zero results, say so honestly rather than implying the \
data is missing or the system is broken.

The customer's name is {full_name}.
Today is {today}.
"""

# Tool schemas. `user_id` is declared as required so the model always sends it;
# `assistant_tools._authorize` then verifies it matches the JWT identity.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_account_balances",
        "description": (
            "Current balance of every open account the user holds, plus their combined "
            "total. Use for questions about how much money they have."
        ),
        "properties": {
            "user_id": {"type": "INTEGER", "description": "The signed-in user's id."},
        },
        "required": ["user_id"],
    },
    {
        "name": "get_transactions",
        "description": (
            "Individual transactions, newest first. Optionally filter by category and a "
            "date range. Use when the user asks what they bought or wants to see specific "
            "payments — not for totals."
        ),
        "properties": {
            "user_id": {"type": "INTEGER", "description": "The signed-in user's id."},
            "category": {
                "type": "STRING",
                "description": (
                    "Spending category such as dining, groceries, transport, shopping, "
                    "utilities, rent. Accepts natural words like 'food'. Omit for all."
                ),
            },
            "start_date": {"type": "STRING", "description": "Inclusive start, YYYY-MM-DD."},
            "end_date": {"type": "STRING", "description": "Inclusive end, YYYY-MM-DD."},
            "limit": {"type": "INTEGER", "description": "Max rows, 1-50. Default 15."},
        },
        "required": ["user_id"],
    },
    {
        "name": "get_spending_summary",
        "description": (
            "Total spending for a period with a per-category breakdown. Use for 'how much "
            "did I spend', including when the user names a category such as food."
        ),
        "properties": {
            "user_id": {"type": "INTEGER", "description": "The signed-in user's id."},
            "period": {
                "type": "STRING",
                "description": (
                    "One of: today, this_week, this_month, last_month, last_7_days, "
                    "last_30_days, last_90_days, this_year, all_time."
                ),
            },
            "category": {
                "type": "STRING",
                "description": "Optional category filter. Accepts 'food'. Omit for all categories.",
            },
        },
        "required": ["user_id", "period"],
    },
    {
        "name": "get_loan_status",
        "description": (
            "Every loan and loan application: status, amount, interest rate, EMI and "
            "outstanding balance. Use for any question about loans or borrowing."
        ),
        "properties": {
            "user_id": {"type": "INTEGER", "description": "The signed-in user's id."},
        },
        "required": ["user_id"],
    },
    {
        "name": "get_upcoming_dues",
        "description": (
            "Scheduled EMI payments coming due, and any missed payments. Use for questions "
            "about what the user owes or needs to pay soon."
        ),
        "properties": {
            "user_id": {"type": "INTEGER", "description": "The signed-in user's id."},
            "days_ahead": {"type": "INTEGER", "description": "Look-ahead window, 1-365. Default 30."},
        },
        "required": ["user_id"],
    },
]

CAPABILITIES = (
    "I can help with your account balances, transaction history, spending summaries by "
    "category, loan status and upcoming EMI payments."
)


@dataclass(slots=True)
class ToolInvocation:
    """One tool call, surfaced to the client so the answer is auditable."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float


@dataclass(slots=True)
class AssistantReply:
    message: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    # "gemini" when the LLM produced the prose, "fallback" when the
    # deterministic router did. Surfaced so the UI can be honest about it.
    engine: str = "gemini"
    model: str | None = None
    latency_ms: float = 0.0
    degraded_reason: str | None = None


# --------------------------------------------------------------------------- #
# Gemini path
# --------------------------------------------------------------------------- #


def _build_gemini_tools():
    """Translate TOOL_SPECS into the SDK's declaration objects."""
    from google.genai import types

    type_map = {
        "STRING": types.Type.STRING,
        "INTEGER": types.Type.INTEGER,
        "NUMBER": types.Type.NUMBER,
        "BOOLEAN": types.Type.BOOLEAN,
    }

    declarations = []
    for spec in TOOL_SPECS:
        properties = {
            key: types.Schema(
                type=type_map[prop["type"]],
                description=prop.get("description", ""),
            )
            for key, prop in spec["properties"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=spec["required"],
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _run_gemini(
    db: Session,
    user: User,
    message: str,
    history: list[dict[str, str]],
) -> AssistantReply:
    from datetime import UTC, datetime

    from google.genai import types

    from app.services.assistant_client import get_client

    started = time.perf_counter()
    client = get_client()

    contents: list[types.Content] = []
    # Replay recent turns so follow-ups like "and last month?" have context.
    for turn in history[-settings.ASSISTANT_HISTORY_TURNS :]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = (turn.get("content") or "").strip()
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    config = types.GenerateContentConfig(
        tools=_build_gemini_tools(),
        system_instruction=SYSTEM_INSTRUCTION.format(
            user_id=user.id,
            full_name=user.full_name,
            today=datetime.now(UTC).date().isoformat(),
        ),
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    invocations: list[ToolInvocation] = []

    for _ in range(settings.ASSISTANT_MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL, contents=contents, config=config
        )

        candidate = (response.candidates or [None])[0]
        if candidate is None or candidate.content is None:
            break

        calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]

        if not calls:
            text = (response.text or "").strip()
            return AssistantReply(
                message=text or f"I could not produce an answer for that. {CAPABILITIES}",
                tool_calls=invocations,
                engine="gemini",
                model=settings.GEMINI_MODEL,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )

        # Echo the model's tool-call turn back before appending results.
        contents.append(candidate.content)

        result_parts: list[types.Part] = []
        for call in calls:
            args = dict(call.args or {})
            call_started = time.perf_counter()
            payload, ok = execute_tool(call.name, db, user, args)
            duration = round((time.perf_counter() - call_started) * 1000, 2)

            invocations.append(
                ToolInvocation(name=call.name, arguments=args, ok=ok, duration_ms=duration)
            )
            logger.info(
                "assistant tool=%s user=%s ok=%s %.1fms", call.name, user.id, ok, duration
            )
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=payload)
            )

        contents.append(types.Content(role="user", parts=result_parts))

    # Round limit hit without a text answer.
    return AssistantReply(
        message=(
            "I gathered your data but could not summarise it. Please try rephrasing, "
            "or check the Dashboard and Insights pages directly."
        ),
        tool_calls=invocations,
        engine="gemini",
        model=settings.GEMINI_MODEL,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


# --------------------------------------------------------------------------- #
# Deterministic fallback
# --------------------------------------------------------------------------- #

BALANCE_PAT = re.compile(r"\b(balance|how much (money|do i have)|funds|total)\b", re.I)
SPEND_PAT = re.compile(r"\b(spend|spent|spending|expenses?|outgo|cost)\b", re.I)
LOAN_PAT = re.compile(r"\b(loan|loans|borrow|emi|mortgage|credit)\b", re.I)
DUE_PAT = re.compile(r"\b(due|dues|upcoming|owe|repay|instal?ment|payment schedule)\b", re.I)
TXN_PAT = re.compile(r"\b(transactions?|purchases?|payments?|history|statement|bought)\b", re.I)

PERIOD_PAT = [
    (re.compile(r"\blast month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\b(this|current) week\b", re.I), "this_week"),
    (re.compile(r"\btoday\b", re.I), "today"),
    (re.compile(r"\b(last|past) 7 days\b|\blast week\b", re.I), "last_7_days"),
    (re.compile(r"\b(last|past) 30 days\b", re.I), "last_30_days"),
    (re.compile(r"\b(last|past) (90 days|quarter|3 months)\b", re.I), "last_90_days"),
    (re.compile(r"\b(this year|ytd)\b", re.I), "this_year"),
]


def _detect_period(text: str) -> str:
    for pattern, period in PERIOD_PAT:
        if pattern.search(text):
            return period
    return "this_month"


def _detect_category(text: str) -> str | None:
    lowered = text.lower()
    for word in (
        "food", "dining", "groceries", "grocery", "restaurants", "transport",
        "travel", "shopping", "utilities", "bills", "entertainment", "healthcare",
        "medical", "education", "rent", "investment", "cash", "fuel",
    ):
        if word in lowered:
            return word
    return None


def _inr(value: float) -> str:
    return f"₹{value:,.2f}"


def _run_fallback(
    db: Session, user: User, message: str, reason: str
) -> AssistantReply:
    """Answer without the LLM, using the same tool layer.

    Templated prose, real figures. Keeps the feature usable when Gemini is
    unavailable instead of showing an error.
    """
    started = time.perf_counter()
    text = message.strip()
    invocations: list[ToolInvocation] = []

    def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        began = time.perf_counter()
        payload, ok = execute_tool(name, db, user, {**args, "user_id": user.id})
        invocations.append(
            ToolInvocation(
                name=name,
                arguments=args,
                ok=ok,
                duration_ms=round((time.perf_counter() - began) * 1000, 2),
            )
        )
        return payload

    def done(msg: str) -> AssistantReply:
        return AssistantReply(
            message=msg,
            tool_calls=invocations,
            engine="fallback",
            model=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            degraded_reason=reason,
        )

    # Order matters: "do I have any loans" mentions neither spend nor balance,
    # while "what's my loan EMI due" should prefer dues over loans.
    if DUE_PAT.search(text) and (LOAN_PAT.search(text) or "due" in text.lower()):
        data = call("get_upcoming_dues", {"days_ahead": 30})
        if data.get("upcoming_count"):
            lines = [
                f"You have {data['upcoming_count']} payment(s) totalling "
                f"{_inr(data['total_due'])} due in the next 30 days:"
            ]
            for due in data["upcoming_dues"][:5]:
                lines.append(
                    f"- {_inr(due['amount'])} on {due['due_date']} "
                    f"({due['loan_type']} loan, in {due['days_until_due']} days)"
                )
            return done("\n".join(lines))
        return done("You have no scheduled EMI payments in the next 30 days.")

    if LOAN_PAT.search(text) and not SPEND_PAT.search(text):
        data = call("get_loan_status", {})
        if not data.get("loan_count"):
            return done(
                "You do not have any loans or loan applications on record. You can check "
                "your eligibility on the Loans page."
            )
        active = [
            loan
            for loan in data["loans"]
            if loan["status"] in {"disbursed", "approved"}
        ]
        if active:
            lines = [f"Yes — you have {len(active)} active loan(s):"]
            for loan in active:
                bits = [f"{loan['loan_type']} loan {loan['reference']}", loan["status"]]
                if loan.get("outstanding_principal"):
                    bits.append(f"{_inr(loan['outstanding_principal'])} outstanding")
                if loan.get("monthly_emi"):
                    bits.append(f"EMI {_inr(loan['monthly_emi'])}")
                lines.append("- " + ", ".join(bits))
            return done("\n".join(lines))
        statuses = ", ".join(f"{l['reference']} ({l['status']})" for l in data["loans"][:4])
        return done(
            f"You have {data['loan_count']} loan application(s) on record, none currently "
            f"active: {statuses}."
        )

    if SPEND_PAT.search(text) or (_detect_category(text) and not BALANCE_PAT.search(text)):
        period = _detect_period(text)
        category = _detect_category(text)
        data = call("get_spending_summary", {"period": period, "category": category})
        total = data.get("total_spent", 0.0)
        label = data.get("period", period)

        if category:
            resolved = ", ".join(data.get("resolved_categories") or [category])
            if not total:
                return done(
                    f"You have not spent anything on {category} ({resolved}) {label}."
                )
            return done(
                f"You spent {_inr(total)} on {category} {label}, across "
                f"{data['transaction_count']} transaction(s). That covers the "
                f"{resolved} categor{'y' if len(data.get('resolved_categories') or []) == 1 else 'ies'}."
            )

        if not total:
            return done(f"I found no spending {label}.")
        top = data.get("by_category", [])[:3]
        breakdown = "; ".join(
            f"{row['category']} {_inr(row['amount'])} ({row['share_percent']}%)" for row in top
        )
        return done(
            f"You spent {_inr(total)} {label} across {data['transaction_count']} "
            f"transaction(s). Largest categories: {breakdown}."
        )

    if BALANCE_PAT.search(text):
        data = call("get_account_balances", {})
        if not data.get("account_count"):
            return done("You do not have any open accounts yet.")
        lines = [
            f"Your total balance is {_inr(data['total_balance'])} across "
            f"{data['account_count']} account(s):"
        ]
        for acct in data["accounts"]:
            name = acct["nickname"] or acct["account_type"].replace("_", " ").title()
            line = f"- {name} {acct['account_number_masked']}: {_inr(acct['balance'])}"
            if acct["on_hold"]:
                line += f" ({_inr(acct['on_hold'])} on hold)"
            lines.append(line)
        return done("\n".join(lines))

    if TXN_PAT.search(text):
        category = _detect_category(text)
        data = call("get_transactions", {"category": category, "limit": 5})
        if not data.get("total_matching"):
            suffix = f" for {category}" if category else ""
            return done(f"I could not find any transactions{suffix}.")
        lines = [f"Your {data['returned']} most recent transaction(s):"]
        for txn in data["transactions"]:
            sign = "+" if txn["direction"] == "credit" else "-"
            lines.append(
                f"- {txn['date']}: {sign}{_inr(txn['amount'])} {txn['description']} "
                f"({txn['category']})"
            )
        return done("\n".join(lines))

    # Nothing matched: say so rather than guessing.
    return done(
        f"I could not match that to anything I can look up. {CAPABILITIES} "
        "Could you rephrase your question?"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def ask_assistant(
    db: Session,
    user: User,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> AssistantReply:
    """Answer one question for the authenticated user.

    Tries Gemini first and falls back to the deterministic router on any
    failure, so a quota error or outage degrades the phrasing rather than
    breaking the feature.
    """
    history = history or []

    if not settings.ASSISTANT_ENABLED:
        return _run_fallback(db, user, message, reason="assistant_disabled")

    if not settings.GEMINI_API_KEY:
        return _run_fallback(db, user, message, reason="no_api_key")

    try:
        return _run_gemini(db, user, message, history)
    except Exception as exc:  # noqa: BLE001 - any SDK/network/quota failure
        detail = str(exc)
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail:
            reason = "gemini_quota_exhausted"
        elif "401" in detail or "API_KEY" in detail.upper() or "PERMISSION" in detail.upper():
            reason = "gemini_auth_failed"
        else:
            reason = f"gemini_error:{type(exc).__name__}"
        logger.warning("Gemini path failed (%s); using fallback. %s", reason, detail[:200])
        return _run_fallback(db, user, message, reason=reason)


def assistant_status() -> dict[str, Any]:
    """Health/capability probe for the UI and the admin panel."""
    return {
        "enabled": settings.ASSISTANT_ENABLED,
        "engine": "gemini" if settings.GEMINI_API_KEY else "fallback",
        "model": settings.GEMINI_MODEL if settings.GEMINI_API_KEY else None,
        "api_key_configured": bool(settings.GEMINI_API_KEY),
        "tools": list(TOOL_REGISTRY),
        "capabilities": CAPABILITIES,
    }
