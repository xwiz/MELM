#!/usr/bin/env python3
"""first_boot_selfconfig.py — one-shot first-boot self-configuration for MELM.

Runs once per device. On first boot (when no ``provisioned.json`` marker
exists) it:

  1. Computes a stable, anonymous ``device_id`` (from ``/etc/machine-id`` if
     present, else a UUID persisted under the MELM config dir).
  2. Claims a ``venia.cloud`` mailbox in ONE shot by POSTing to the mailer
     service, and stores the returned credentials in ``secrets.env`` (0600).
  3. Writes identity / personality preferences (emoji, display name) to
     ``identity.json``.
  4. Writes a ``provisioned.json`` marker so subsequent boots are no-ops.

Design constraints (matching the rest of MELM):
  - **stdlib only** — ``urllib``, ``json``, ``argparse``, ``uuid``, ``hashlib``.
    No ``requests``, no third-party HTTP client.
  - **idempotent** — re-running is a no-op unless ``--force`` is passed.
  - **non-fatal email failure** — if the mailbox claim fails after retries the
    device is still marked provisioned for everything else; the email step can
    be retried later with ``--force --skip-models ...`` etc.
  - **secrets never echoed** — the mailbox password is written to ``secrets.env``
    with mode 0600 and never printed to stdout/stderr/logs.

This file is normally invoked by ``scripts/provision_device.sh`` but is a fully
standalone CLI. Run ``python3 scripts/first_boot_selfconfig.py --help``.

MELM itself remains stdlib-only and needs none of this to run offline; the
mailbox + identity prefs power optional features (agent email, persona).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

# Be resilient on consoles that cannot encode emoji (e.g. Windows cp1252).
# The Linux/Pi target is UTF-8, but help text and logs include an emoji default;
# reconfigure the streams to never crash on an un-encodable glyph.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# --------------------------------------------------------------------------- #
# Defaults (all overridable via env or flags)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_DIR = os.environ.get(
    "MELM_CONFIG_DIR", str(Path.home() / ".config" / "melm")
)

# Mailer service: base URL + path are configurable so the same code works
# against staging / prod / a local mock.
DEFAULT_MAILER_BASE_URL = os.environ.get(
    "MELM_MAILER_BASE_URL", "https://mail.venia.cloud"
)
DEFAULT_CLAIM_PATH = os.environ.get("MELM_MAILER_CLAIM_PATH", "/api/v1/agent/claim")
DEFAULT_PARTNER_TOKEN = os.environ.get("MELM_PARTNER_TOKEN", "VENMAIL-AGENT")
DEFAULT_PREFERRED_HANDLE = os.environ.get("MELM_PREFERRED_HANDLE", "")

# Identity / persona defaults (overridable via flags or env).
DEFAULT_EMOJI = os.environ.get("MELM_IDENTITY_EMOJI", "🤖")
DEFAULT_DISPLAY_NAME = os.environ.get("MELM_IDENTITY_DISPLAY_NAME", "MELM")
DEFAULT_PERSONA = os.environ.get("MELM_IDENTITY_PERSONA", "friendly_local_assistant")

# Network retry policy (mirrors the bash side).
DEFAULT_RETRY_MAX = int(os.environ.get("MELM_RETRY_MAX", "4"))
DEFAULT_RETRY_BASE_DELAY = float(os.environ.get("MELM_RETRY_BASE_DELAY", "3"))
DEFAULT_HTTP_TIMEOUT = float(os.environ.get("MELM_HTTP_TIMEOUT", "20"))


# --------------------------------------------------------------------------- #
# Logging (stderr only; stdout is reserved for optional machine output)
# --------------------------------------------------------------------------- #


def _log(level: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {level:<5} {msg}", file=sys.stderr)


def info(msg: str) -> None:
    _log("INFO", msg)


def warn(msg: str) -> None:
    _log("WARN", msg)


def error(msg: str) -> None:
    _log("ERROR", msg)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #


def ensure_dir(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
    return path


def write_private_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with 0600 perms, atomically via a temp file."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Create with restrictive perms from the start.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Secrets file (KEY=VALUE; never echo values)
# --------------------------------------------------------------------------- #


def set_secret(secrets_file: Path, key: str, value: str) -> None:
    """Upsert KEY=VALUE in secrets.env (0600). The value is NEVER logged."""
    ensure_dir(secrets_file.parent)
    lines: list[str] = []
    if secrets_file.exists():
        with secrets_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                # Drop any prior live OR commented assignment for this key.
                bare = stripped[1:].lstrip() if stripped.startswith("#") else stripped
                if bare.startswith(f"{key}="):
                    continue
                lines.append(line.rstrip("\n"))
    lines.append(f"{key}={value}")
    tmp = secrets_file.with_suffix(secrets_file.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, secrets_file)
    try:
        os.chmod(secrets_file, 0o600)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Stable device identity
# --------------------------------------------------------------------------- #


def compute_device_id(config_dir: Path) -> str:
    """Return a stable, anonymous device id.

    Preference order:
      1. /etc/machine-id  (systemd; stable across reboots, distinct per install)
      2. /var/lib/dbus/machine-id  (older systems)
      3. a UUID persisted at <config_dir>/device_id

    The raw machine-id is never sent verbatim; we hash it so the id is stable
    but not reversible to the host machine-id (privacy-preserving).
    """
    import hashlib

    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            raw = ""
        if raw:
            digest = hashlib.sha256(f"melm-device:{raw}".encode("utf-8")).hexdigest()
            return f"melm-{digest[:32]}"

    # Fallback: persist a random UUID so the id is stable on this device.
    persisted = config_dir / "device_id"
    try:
        existing = persisted.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    new_id = f"melm-{uuid.uuid4().hex}"
    ensure_dir(config_dir)
    try:
        persisted.write_text(new_id + "\n", encoding="utf-8")
        os.chmod(persisted, 0o600)
    except OSError:
        pass
    return new_id


# --------------------------------------------------------------------------- #
# Mailbox claim (one-shot POST, stdlib urllib, retry-with-backoff)
# --------------------------------------------------------------------------- #


class ClaimError(Exception):
    """Raised when the mailbox claim cannot be completed."""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "melm-first-boot-selfconfig/1",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URL)
        status = getattr(resp, "status", resp.getcode())
        raw = resp.read().decode("utf-8")
    if status != 200:
        raise ClaimError(f"unexpected status {status}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClaimError(f"invalid JSON in claim response: {exc}") from exc


def claim_mailbox(
    base_url: str,
    claim_path: str,
    partner_token: str,
    device_id: str,
    preferred_handle: str = "",
    retry_max: int = DEFAULT_RETRY_MAX,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Claim a venia.cloud mailbox in one shot.

    Contract (request)::

        POST {base_url}{claim_path}
        Content-Type: application/json
        {
          "partner_token": "VENMAIL-AGENT",
          "device_id": "<stable device id>",
          "preferred_handle": "<optional>"   # omitted if empty
        }

    Contract (response, HTTP 200)::

        {
          "email": "...@venia.cloud",
          "password": "...",
          "imap_host": "...",
          "smtp_host": "...",
          ...                                 # extra fields tolerated/stored
        }

    Retries with exponential backoff on transient (network / 5xx) errors.
    Raises ClaimError on permanent failure.
    """
    url = base_url.rstrip("/") + "/" + claim_path.lstrip("/")
    payload: dict[str, Any] = {
        "partner_token": partner_token,
        "device_id": device_id,
    }
    if preferred_handle:
        payload["preferred_handle"] = preferred_handle

    attempt = 1
    delay = retry_base_delay
    last_err: Optional[Exception] = None
    while attempt <= retry_max:
        try:
            info(f"Claiming mailbox (attempt {attempt}/{retry_max}) at {url}")
            data = _post_json(url, payload, timeout=timeout)
            if not isinstance(data, dict) or "email" not in data:
                raise ClaimError("response missing required 'email' field")
            return data
        except urllib.error.HTTPError as exc:
            last_err = exc
            # 4xx (except 429) is permanent — do not hammer the service.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise ClaimError(f"claim rejected (HTTP {exc.code})") from exc
            warn(f"claim transient error HTTP {exc.code}")
        except (urllib.error.URLError, ClaimError, TimeoutError, OSError) as exc:
            last_err = exc
            warn(f"claim transient error: {exc}")

        if attempt >= retry_max:
            break
        warn(f"retrying mailbox claim in {delay:.0f}s")
        time.sleep(delay)
        attempt += 1
        delay *= 2

    raise ClaimError(f"mailbox claim failed after {retry_max} attempt(s): {last_err}")


def store_mailbox_credentials(secrets_file: Path, claim: dict[str, Any]) -> None:
    """Persist mailbox creds to secrets.env (0600). Password never logged.

    Maps the claim response onto MELM_MAILBOX_* keys. Unknown extra fields are
    stored under MELM_MAILBOX_<UPPER> so future server fields survive.
    """
    field_map = {
        "email": "MELM_MAILBOX_EMAIL",
        "password": "MELM_MAILBOX_PASSWORD",
        "imap_host": "MELM_MAILBOX_IMAP_HOST",
        "smtp_host": "MELM_MAILBOX_SMTP_HOST",
        "imap_port": "MELM_MAILBOX_IMAP_PORT",
        "smtp_port": "MELM_MAILBOX_SMTP_PORT",
        "username": "MELM_MAILBOX_USERNAME",
    }
    for key, env_key in field_map.items():
        if key in claim and claim[key] not in (None, ""):
            set_secret(secrets_file, env_key, str(claim[key]))


# --------------------------------------------------------------------------- #
# Identity / persona preferences
# --------------------------------------------------------------------------- #


def build_identity(
    device_id: str,
    email: Optional[str],
    emoji: str,
    display_name: str,
    persona: str,
) -> dict[str, Any]:
    """Build the identity.json document.

    Schema (identity.json, v1)::

        {
          "schema": "melm.identity.v1",
          "device_id": "<stable id>",
          "display_name": "MELM",
          "emoji": "🤖",
          "persona": "friendly_local_assistant",
          "email": "...@venia.cloud" | null,
          "prefs": {
            "use_emoji": true,
            "greeting_style": "warm",
            "verbosity": "concise"
          }
        }
    """
    return {
        "schema": "melm.identity.v1",
        "device_id": device_id,
        "display_name": display_name,
        "emoji": emoji,
        "persona": persona,
        "email": email,
        "prefs": {
            "use_emoji": bool(emoji),
            "greeting_style": os.environ.get("MELM_IDENTITY_GREETING_STYLE", "warm"),
            "verbosity": os.environ.get("MELM_IDENTITY_VERBOSITY", "concise"),
        },
    }


# --------------------------------------------------------------------------- #
# Marker
# --------------------------------------------------------------------------- #


def marker_path(config_dir: Path) -> Path:
    return config_dir / "provisioned.json"


def is_provisioned(config_dir: Path) -> bool:
    return marker_path(config_dir).is_file()


def write_marker(config_dir: Path, payload: dict[str, Any]) -> None:
    write_private_json(marker_path(config_dir), payload)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir).expanduser()
    secrets_file = (
        Path(args.secrets_file).expanduser()
        if args.secrets_file
        else config_dir / "secrets.env"
    )
    identity_file = config_dir / "identity.json"
    ensure_dir(config_dir)

    if is_provisioned(config_dir) and not args.force:
        info(f"Device already provisioned (marker at {marker_path(config_dir)}).")
        info("Use --force to re-run first-boot self-config.")
        return 0

    device_id = compute_device_id(config_dir)
    info(f"device_id={device_id}")

    email: Optional[str] = None
    email_status = "skipped"

    if args.skip_email:
        info("Skipping mailbox claim (--skip-email).")
    else:
        try:
            claim = claim_mailbox(
                base_url=args.mailer_base_url,
                claim_path=args.claim_path,
                partner_token=args.partner_token,
                device_id=device_id,
                preferred_handle=args.preferred_handle,
                retry_max=args.retry_max,
                retry_base_delay=args.retry_base_delay,
                timeout=args.http_timeout,
            )
            store_mailbox_credentials(secrets_file, claim)
            email = str(claim.get("email") or "") or None
            # Never log the password; only confirm the email handle.
            info(f"Mailbox claimed: {email} (credentials stored 0600).")
            email_status = "claimed"
        except ClaimError as exc:
            # Non-fatal: provisioning continues; email can be retried later.
            warn(f"Mailbox claim failed (non-fatal): {exc}")
            email_status = "failed"

    identity = build_identity(
        device_id=device_id,
        email=email,
        emoji=args.emoji,
        display_name=args.display_name,
        persona=args.persona,
    )
    write_private_json(identity_file, identity)
    info(f"Wrote identity prefs: {identity_file}")

    write_marker(
        config_dir,
        {
            "schema": "melm.provisioned.v1",
            "device_id": device_id,
            "provisioned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "email_status": email_status,
            "email": email,
            "identity_file": str(identity_file),
            "secrets_file": str(secrets_file),
        },
    )
    info(f"Wrote provisioning marker: {marker_path(config_dir)}")
    info("First-boot self-config complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="first_boot_selfconfig.py",
        description=(
            "One-shot first-boot self-configuration for MELM: stable device id, "
            "venia.cloud mailbox claim, identity prefs, provisioning marker. "
            "stdlib-only, idempotent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        help="MELM config dir (holds secrets.env, identity.json, provisioned.json)",
    )
    parser.add_argument(
        "--secrets-file",
        default="",
        help="Secrets file path (default: <config-dir>/secrets.env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a provisioned.json marker already exists",
    )
    parser.add_argument(
        "--skip-email",
        action="store_true",
        help="Skip the venia.cloud mailbox claim entirely",
    )

    # Mailer service contract (all configurable for staging/prod/local mock).
    parser.add_argument("--mailer-base-url", default=DEFAULT_MAILER_BASE_URL,
                        help="Base URL of the mailer service")
    parser.add_argument("--claim-path", default=DEFAULT_CLAIM_PATH,
                        help="Claim endpoint path appended to the base URL")
    parser.add_argument("--partner-token", default=DEFAULT_PARTNER_TOKEN,
                        help="Partner token sent in the claim payload")
    parser.add_argument("--preferred-handle", default=DEFAULT_PREFERRED_HANDLE,
                        help="Optional preferred mailbox handle (best-effort)")

    # Identity / persona prefs.
    parser.add_argument("--emoji", default=DEFAULT_EMOJI,
                        help="Assistant face/emoji written to identity.json")
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME,
                        help="Display name written to identity.json")
    parser.add_argument("--persona", default=DEFAULT_PERSONA,
                        help="Persona key written to identity.json")

    # Retry / timeout.
    parser.add_argument("--retry-max", type=int, default=DEFAULT_RETRY_MAX,
                        help="Max attempts for the mailbox claim")
    parser.add_argument("--retry-base-delay", type=float,
                        default=DEFAULT_RETRY_BASE_DELAY,
                        help="Base backoff delay (seconds; doubles each retry)")
    parser.add_argument("--http-timeout", type=float, default=DEFAULT_HTTP_TIMEOUT,
                        help="Per-request HTTP timeout (seconds)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
