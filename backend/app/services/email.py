"""Outbound email.

Delivery is **simulated by default**: messages are logged and counted rather
than sent, because a demo project should not ship SMTP credentials. This mirrors
the approach already taken in :mod:`app.services.notifications`, where rows are
marked ``email_queued`` instead of handed to a provider.

Swapping in a real provider means changing exactly one function -- ``_deliver``
-- and nothing at the call sites. That is the whole point of the indirection:
:func:`send_email` is the stable seam, so routes never learn whether the
transport is SendGrid, SES, or a log line.

Nothing here raises on failure. A reminder email that cannot be delivered must
not roll back the audit log or the in-app notification that accompanies it, so
delivery problems surface as a ``False`` return and a warning.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape as _escape
from threading import Lock
from typing import Any

logger = logging.getLogger("app.email")

# Deliberately permissive: this validates shape, not deliverability. Anything
# stricter rejects addresses that are legal under RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


@dataclass
class SentMessage:
    """A record of one simulated send, kept for tests and the admin view."""

    to: str
    subject: str
    html: str
    sent_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def text(self) -> str:
        return html_to_text(self.html)


class _Outbox:
    """Bounded in-memory record of simulated sends.

    Bounded because an unbounded list in a long-running process is a slow leak.
    Lock-guarded because FastAPI serves sync routes on a thread pool, so two
    requests can append concurrently.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._items: list[SentMessage] = []
        self._max = max_size
        self._sent = 0
        self._failed = 0
        self._lock = Lock()

    def record(self, message: SentMessage) -> None:
        with self._lock:
            self._sent += 1
            self._items.append(message)
            if len(self._items) > self._max:
                del self._items[: len(self._items) - self._max]

    def record_failure(self) -> None:
        with self._lock:
            self._failed += 1

    def recent(self, limit: int = 20) -> list[SentMessage]:
        with self._lock:
            return list(reversed(self._items[-limit:]))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sent": self._sent,
                "failed": self._failed,
                "retained": len(self._items),
                "transport": "simulated",
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._sent = 0
            self._failed = 0


outbox = _Outbox()


def html_to_text(html: str) -> str:
    """Crude HTML-to-text for the plain-text part and for log lines.

    Not a general HTML parser -- it only needs to handle the small, known
    templates in this module.
    """
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#8377;", "\u20b9")
        .replace("&rupee;", "\u20b9")
    )
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def is_valid_address(address: str | None) -> bool:
    return bool(address and _EMAIL_RE.match(address.strip()))


def _deliver(to: str, subject: str, html: str) -> bool:
    """The single seam a real provider replaces.

    A live implementation goes here -- SMTP, SendGrid, SES -- and every caller
    keeps working unchanged. Until then, log the message so the behaviour is
    observable in development without any credentials.
    """
    logger.info(
        "[SIMULATED EMAIL] to=%s subject=%s\n%s",
        to,
        subject,
        html_to_text(html),
    )
    return True


def send_email(to: str, subject: str, html: str) -> bool:
    """Send one message. Returns ``True`` when it was accepted for delivery.

    Never raises. Callers treat a ``False`` as "the in-app notification is still
    the source of truth", which keeps a transport failure from breaking a
    request that also wrote to the database.
    """
    address = (to or "").strip()
    if not is_valid_address(address):
        logger.warning("email not sent: invalid recipient %r", to)
        outbox.record_failure()
        return False
    if not subject.strip():
        logger.warning("email not sent to %s: empty subject", address)
        outbox.record_failure()
        return False

    try:
        delivered = _deliver(address, subject, html)
    except Exception:  # pragma: no cover - defensive, transport is simulated
        logger.exception("email delivery raised for %s", address)
        outbox.record_failure()
        return False

    if delivered:
        outbox.record(SentMessage(to=address, subject=subject, html=html))
    else:
        outbox.record_failure()
    return delivered


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

_BASE_STYLE = (
    "font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "font-size:14px;line-height:1.6;color:#1f2933;"
)


def render_basic_email(*, heading: str, intro: str, rows: list[tuple[str, str]], outro: str) -> str:
    """Shared shell so every message has the same structure and footer.

    The footer carries the simulated-project disclaimer, which belongs on
    anything that could otherwise be mistaken for real bank correspondence.

    Every interpolated value is escaped, because this renderer feeds both HTML
    and the plain-text log copy. A borrower name or loan reference is
    user-controlled input, so escaping is a correctness requirement, not
    decoration.
    """
    def esc(value: str) -> str:
        return _escape(str(value), quote=True)

    cells = "".join(
        f'<tr><td style="padding:6px 16px 6px 0;color:#616e7c;">{esc(label)}</td>'
        f'<td style="padding:6px 0;font-weight:600;text-align:right;">{esc(value)}</td></tr>'
        for label, value in rows
    )
    return (
        f'<div style="{_BASE_STYLE}max-width:560px;">'
        f'<h2 style="margin:0 0 12px;font-size:18px;">{esc(heading)}</h2>'
        f"<p style=\"margin:0 0 16px;\">{esc(intro)}</p>"
        f'<table role="presentation" style="border-collapse:collapse;width:100%;'
        f'margin:0 0 16px;border-top:1px solid #e4e7eb;border-bottom:1px solid #e4e7eb;">'
        f"{cells}</table>"
        f'<p style="margin:0 0 20px;">{esc(outro)}</p>'
        f'<p style="margin:0;padding-top:14px;border-top:1px solid #e4e7eb;'
        f'font-size:12px;color:#9aa5b1;">'
        f"IntelliBank is a simulated banking platform built for educational "
        f"purposes. No real accounts, money, or credit reporting are involved."
        f"</p></div>"
    )
