"""Mail skill — IMAP read / SMTP send using the provisioned venia.cloud mailbox.

Credentials live in os.environ (populated by Hook #1 from secrets.env):
  MELM_MAILBOX_EMAIL, MELM_MAILBOX_PASSWORD, MELM_MAILBOX_USERNAME,
  MELM_MAILBOX_IMAP_HOST, MELM_MAILBOX_SMTP_HOST,
  MELM_MAILBOX_IMAP_PORT (default 993), MELM_MAILBOX_SMTP_PORT (default 587).

All network I/O is bounded by _IMAP_TIMEOUT / _SMTP_TIMEOUT. Passwords are
never logged or included in returned strings.
"""

from __future__ import annotations

import email as _email_lib
import imaplib
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.header import decode_header as _decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .assistant_skill_base import SkillManifest, register_skill

MANIFEST = SkillManifest(
    family="mail",
    frames=("mail_send", "mail_read"),
    knowledge_refs=(),
    template_refs={},
)

register_skill(MANIFEST)

_IMAP_TIMEOUT = 10
_SMTP_TIMEOUT = 10


@dataclass(frozen=True)
class MailboxConfig:
    """Mailbox credentials loaded from os.environ."""

    email: str
    password: str
    username: str
    imap_host: str
    smtp_host: str
    imap_port: int
    smtp_port: int

    @classmethod
    def from_env(cls) -> "MailboxConfig | None":
        addr = os.environ.get("MELM_MAILBOX_EMAIL", "")
        pwd = os.environ.get("MELM_MAILBOX_PASSWORD", "")
        imap = os.environ.get("MELM_MAILBOX_IMAP_HOST", "")
        smtp = os.environ.get("MELM_MAILBOX_SMTP_HOST", "")
        if not (addr and pwd and imap and smtp):
            return None
        return cls(
            email=addr,
            password=pwd,
            username=os.environ.get("MELM_MAILBOX_USERNAME", addr),
            imap_host=imap,
            smtp_host=smtp,
            imap_port=int(os.environ.get("MELM_MAILBOX_IMAP_PORT", "993")),
            smtp_port=int(os.environ.get("MELM_MAILBOX_SMTP_PORT", "587")),
        )


def is_configured() -> bool:
    """True when all required mailbox env vars are present."""
    return MailboxConfig.from_env() is not None


@dataclass(frozen=True)
class MailSummary:
    uid: str
    sender: str
    subject: str
    date: str


def _header_str(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = []
    for chunk, charset in _decode_header(raw):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(str(chunk))
    return "".join(parts)


def fetch_inbox(max_messages: int = 10) -> list[MailSummary] | str:
    """Fetch the most recent unread messages from INBOX.

    Returns a list of MailSummary on success, or an error string on failure.
    """
    cfg = MailboxConfig.from_env()
    if cfg is None:
        return "Mailbox not configured."
    try:
        ctx = ssl.create_default_context()
        with imaplib.IMAP4_SSL(cfg.imap_host, cfg.imap_port, ssl_context=ctx) as imap:
            imap.socket().settimeout(_IMAP_TIMEOUT)
            imap.login(cfg.username, cfg.password)
            imap.select("INBOX", readonly=True)
            _, data = imap.search(None, "UNSEEN")
            uids = (data[0].split() if data and data[0] else [])[-max_messages:]
            summaries: list[MailSummary] = []
            for uid in reversed(uids):
                _, msg_data = imap.fetch(uid, "(RFC822.HEADER)")
                if not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else b""
                msg = _email_lib.message_from_bytes(raw)
                summaries.append(MailSummary(
                    uid=uid.decode(),
                    sender=_header_str(msg.get("From")),
                    subject=_header_str(msg.get("Subject")) or "(no subject)",
                    date=_header_str(msg.get("Date")),
                ))
        return summaries
    except (imaplib.IMAP4.error, OSError, ssl.SSLError) as exc:
        return f"Could not read inbox: {type(exc).__name__}"


def send_message(to: str, subject: str, body: str) -> bool | str:
    """Send a plain-text email via SMTP with STARTTLS.

    Returns True on success, or an error string on failure.
    """
    cfg = MailboxConfig.from_env()
    if cfg is None:
        return "Mailbox not configured."
    msg = MIMEMultipart("alternative")
    msg["From"] = cfg.email
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=_SMTP_TIMEOUT) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ctx)
            smtp.ehlo()
            smtp.login(cfg.username, cfg.password)
            smtp.sendmail(cfg.email, [to], msg.as_string())
        return True
    except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
        return f"Could not send email: {type(exc).__name__}"


def summarise_inbox(summaries: list[MailSummary]) -> str:
    """Format a MailSummary list into a natural-language answer."""
    if not summaries:
        return "Your inbox is empty — no unread messages."
    n = len(summaries)
    lines = [f"You have {n} unread message{'s' if n != 1 else ''}:"]
    for s in summaries[:5]:
        lines.append(f"  • From {s.sender} — {s.subject}")
    if n > 5:
        lines.append(f"  … and {n - 5} more.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent parsing helpers used by the router handler
# ---------------------------------------------------------------------------

def _get_mail_verb_sets():
    if not hasattr(_get_mail_verb_sets, "_cache"):
        try:
            from melm.contracts import load_mail_verb_sets
            _get_mail_verb_sets._cache = load_mail_verb_sets()
        except Exception:
            _get_mail_verb_sets._cache = (frozenset({"send", "write", "compose", "reply"}), frozenset({"check", "read", "show", "fetch", "open", "get"}), frozenset({"email", "inbox"}))
    return _get_mail_verb_sets._cache
_SEND_VERBS: frozenset[str]
_READ_VERBS: frozenset[str]
_SEND_VERBS, _READ_VERBS, _ = _get_mail_verb_sets()
_BODY_MARKERS: tuple[str, ...] = ("saying", "that", "tell", "about", "message")


def is_send_intent(tokens: tuple[str, ...]) -> bool:
    """True when the utterance is a send-mail request.

    Requires an explicit send verb (send/write/compose/reply) AND an
    email/mail context token AND a target ("to" or an inline address).
    "email" is NOT a send verb here because "check my email" would misfire.
    """
    token_set = frozenset(tokens)
    has_send_verb = bool(token_set & _SEND_VERBS)
    has_mail_context = bool(token_set & {"email", "mail"})
    has_target = "to" in token_set or any("@" in t for t in tokens)
    return has_send_verb and has_mail_context and has_target


def is_read_intent(tokens: tuple[str, ...]) -> bool:
    token_set = frozenset(tokens)
    return bool(token_set & _READ_VERBS) and bool(token_set & {"email", "inbox", "messages", "mail"})


def extract_mail_body(text: str) -> str:
    """Extract body text after a body-marker word in *text*."""
    for marker in _BODY_MARKERS:
        idx = text.find(f" {marker} ")
        if idx != -1:
            return text[idx + len(marker) + 2:].strip()
    return ""


def resolve_recipient(
    tokens: tuple[str, ...],
    contacts: dict[str, str],
) -> tuple[str, str] | None:
    """Return (display_name, email_address) for first matched contact, or None."""
    for tok in tokens:
        if tok in contacts:
            addr = contacts[tok]
            # contacts dict value is a phone number for call; for email we
            # check if it looks like an email address, else skip
            if "@" in addr:
                return tok, addr
    # Fallback: check if any token is an email address directly
    for tok in tokens:
        if "@" in tok and "." in tok:
            return tok, tok
    return None
