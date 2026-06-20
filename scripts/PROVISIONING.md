# MELM Device Provisioning

Robust, idempotent provisioning for MELM on a **Raspberry Pi 5** or any generic
Linux box (**arm64** or **x86_64**).

> MELM itself is **stdlib-only Python 3.11+** and runs offline with **none** of
> these dependencies. Provisioning installs the *optional* tiers (constrained
> decoding model, spell-correction, grammar lint) and performs first-boot
> self-config (device identity + agent mailbox). The base assistant works with
> zero of them: `python3 -m melm chat`.

---

## Quick start

```bash
# Full provision, prompt for Wi-Fi passphrase interactively (never echoed):
scripts/provision_device.sh --wifi-ssid HomeNet

# Ethernet box — skip Wi-Fi, just fetch the model + self-config:
scripts/provision_device.sh --skip-wifi

# Everything including the optional llama.cpp build + Harper grammar LSP:
scripts/provision_device.sh --with-llama --with-harper

# See every flag:
scripts/provision_device.sh --help
```

The script is **idempotent** — safe to re-run. Each stage detects work already
done (model present and size-valid, package installed, already online, already
provisioned) and skips it. Network operations **retry with exponential
backoff**.

---

## Components

| File | Role |
|------|------|
| `scripts/provision_device.sh` | Orchestrator (bash, `set -euo pipefail`). Wi-Fi, secrets, deps, self-config, summary. |
| `scripts/first_boot_selfconfig.py` | First-boot self-config (stdlib only): device id, mailbox claim, identity prefs, marker. |
| `scripts/PROVISIONING.md` | This document. |

### Stages (in order)

1. **Wi-Fi** (`--skip-wifi`) — nmcli **or** headless wpa_supplicant. Skipped if
   already online.
2. **API key** (secrets) — optional cloud LLM key into `secrets.env` (0600).
3. **Models** (`--skip-models`) — Qwen2.5-0.5B-Instruct GGUF (~644 MB).
4. **symspellpy** (`--skip-symspell`) — pip + bundled frequency dictionaries.
5. **llama.cpp** (`--skip-llama`, default OFF; enable with `--with-llama`) —
   prebuilt binary if available, else build from source.
6. **Harper** (`--skip-harper`, default OFF; enable with `--with-harper`) —
   `cargo install harper-ls` if `cargo` present, else skipped.
7. **First-boot self-config** (`--skip-selfconfig`) — delegates to
   `first_boot_selfconfig.py`.

A **final summary** prints what each stage did, plus the resolved config / data
/ models / secrets paths.

---

## Dependencies fetched

| Dependency | Source | Notes |
|------------|--------|-------|
| **Qwen2.5-0.5B-Instruct GGUF** | repo `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file `qwen2.5-0.5b-instruct-q8_0.gguf` (~644 MB) | `huggingface_hub` if importable, else `curl` from the HF `resolve` URL. Size floor + optional sha256 sanity. Stored in `~/.local/share/melm/models`. **SmolLM is NOT used — Qwen only.** |
| **symspellpy** | pip (`--user`) | Bundles `frequency_dictionary_en_82_765.txt` + bigram dict in-package. |
| **llama.cpp** | Homebrew prebuilt → else `git`+`cmake` build | Optional. Arch auto-detected via `uname -m`. |
| **Harper (`harper-ls`)** | `cargo install harper-ls` | Optional. Skipped if `cargo` absent. |

Arch detection: `uname -m` → `arm64` (aarch64/arm64), `x86_64` (x86_64/amd64),
`armhf` (armv7l/armv6l).

---

## Config & secrets layout

```
~/.config/melm/                 (MELM_CONFIG_DIR, mode 0700)
├── secrets.env                 KEY=VALUE secrets, mode 0600 — never committed
├── identity.json               persona / emoji / display name, mode 0600
├── provisioned.json            first-boot marker, mode 0600
└── device_id                   persisted UUID fallback (only if no machine-id)

~/.local/share/melm/            (MELM_DATA_DIR)
├── models/                     (MELM_MODELS_DIR)
│   └── qwen2.5-0.5b-instruct-q8_0.gguf
└── src/llama.cpp/              (only if --with-llama builds from source)
```

### `secrets.env` (mode 0600)

KEY=VALUE lines, loaded by MELM at runtime, never echoed by the provisioner.

```sh
# Cloud LLM API key (optional; only the gated cloud_handoff path uses it):
MELM_CLOUD_API_KEY=

# Mailbox credentials (filled by first_boot_selfconfig.py after email claim):
MELM_MAILBOX_EMAIL=
MELM_MAILBOX_PASSWORD=
MELM_MAILBOX_IMAP_HOST=
MELM_MAILBOX_SMTP_HOST=
MELM_MAILBOX_IMAP_PORT=
MELM_MAILBOX_SMTP_PORT=
```

### `identity.json` (schema `melm.identity.v1`, mode 0600)

```json
{
  "schema": "melm.identity.v1",
  "device_id": "melm-<hash-or-uuid>",
  "display_name": "MELM",
  "emoji": "🤖",
  "persona": "friendly_local_assistant",
  "email": "agent-...@venia.cloud",
  "prefs": {
    "use_emoji": true,
    "greeting_style": "warm",
    "verbosity": "concise"
  }
}
```

Defaults are overridable via `--emoji`, `--display-name`, `--persona` (or the
`MELM_IDENTITY_*` env vars).

### `provisioned.json` (schema `melm.provisioned.v1`, mode 0600)

The first-boot marker. Its presence makes self-config a no-op; pass `--force`
to redo.

```json
{
  "schema": "melm.provisioned.v1",
  "device_id": "melm-...",
  "provisioned_at": "2026-06-20T00:00:00Z",
  "email_status": "claimed | skipped | failed",
  "email": "agent-...@venia.cloud",
  "identity_file": "...",
  "secrets_file": "..."
}
```

---

## First-boot self-config contract

`first_boot_selfconfig.py` is stdlib-only (`urllib`, `json`, `argparse`,
`uuid`, `hashlib`) and idempotent. On first boot it:

1. **Device id** — hashes `/etc/machine-id` (or `/var/lib/dbus/machine-id`)
   with sha256 so the id is stable but not reversible to the host machine-id.
   If no machine-id exists, persists a random UUID at `~/.config/melm/device_id`.
2. **Mailbox claim** — one-shot `POST` to the mailer service:

   **Request**
   ```http
   POST {MELM_MAILER_BASE_URL}{MELM_MAILER_CLAIM_PATH}
   Content-Type: application/json

   {
     "partner_token": "VENMAIL-AGENT",
     "device_id": "<stable device id>",
     "preferred_handle": "<optional>"
   }
   ```

   **Response (HTTP 200)**
   ```json
   {
     "email": "...@venia.cloud",
     "password": "...",
     "imap_host": "...",
     "smtp_host": "...",
     "imap_port": 993,
     "smtp_port": 587
   }
   ```

   Retries with exponential backoff on network / 5xx / 429 errors. A 4xx
   (other than 429) fails fast (no retry). Credentials are stored in
   `secrets.env` (0600); the password is **never** printed. Failure is
   **non-fatal** — the device is still marked provisioned and the claim can be
   retried later with `--force`.
3. **Identity prefs** — writes `identity.json`.
4. **Marker** — writes `provisioned.json`.

Run standalone:

```bash
python3 scripts/first_boot_selfconfig.py --help
python3 scripts/first_boot_selfconfig.py --skip-email            # local-only
python3 scripts/first_boot_selfconfig.py --force                 # redo
python3 scripts/first_boot_selfconfig.py \
    --mailer-base-url https://mail.staging.venia.cloud \
    --preferred-handle kitchen-pi
```

---

## Environment variables

All flags have an env equivalent. Flags win over env; env wins over defaults.

### Paths
| Var | Default |
|-----|---------|
| `MELM_DATA_DIR` | `~/.local/share/melm` |
| `MELM_CONFIG_DIR` | `~/.config/melm` |
| `MELM_MODELS_DIR` | `$MELM_DATA_DIR/models` |
| `MELM_SECRETS_FILE` | `$MELM_CONFIG_DIR/secrets.env` |

### Model
| Var | Default |
|-----|---------|
| `MELM_QWEN_REPO_ID` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| `MELM_QWEN_FILENAME` | `qwen2.5-0.5b-instruct-q8_0.gguf` |
| `MELM_QWEN_MIN_BYTES` | `500000000` (size floor) |
| `MELM_QWEN_SHA256` | _unset_ (optional checksum) |

### Wi-Fi
| Var | Default |
|-----|---------|
| `MELM_WIFI_SSID` | _unset_ (prompted) |
| `MELM_WIFI_PSK` | _unset_ (prompted, `read -s`, never echoed) |
| `MELM_WIFI_CONF_FILE` | _unset_ (`ssid=` / `psk=` lines, 0600) |
| `MELM_WIFI_COUNTRY` | `US` |
| `MELM_WIFI_BACKEND` | `auto` (`nmcli` \| `wpa_supplicant`) |

### Secrets / mailer / identity
| Var | Default |
|-----|---------|
| `MELM_CLOUD_API_KEY` | _unset_ (optional) |
| `MELM_MAILER_BASE_URL` | `https://mail.venia.cloud` |
| `MELM_MAILER_CLAIM_PATH` | `/api/agent/claim` |
| `MELM_PARTNER_TOKEN` | `VENMAIL-AGENT` |
| `MELM_PREFERRED_HANDLE` | _unset_ |
| `MELM_IDENTITY_EMOJI` | `🤖` |
| `MELM_IDENTITY_DISPLAY_NAME` | `MELM` |
| `MELM_IDENTITY_PERSONA` | `friendly_local_assistant` |
| `MELM_IDENTITY_GREETING_STYLE` | `warm` |
| `MELM_IDENTITY_VERBOSITY` | `concise` |

### Retry / skip
| Var | Default |
|-----|---------|
| `MELM_RETRY_MAX` | `4` |
| `MELM_RETRY_BASE_DELAY` | `3` (seconds, doubles each attempt) |
| `MELM_HTTP_TIMEOUT` | `20` (seconds; Python side) |
| `MELM_SKIP_WIFI` / `MELM_SKIP_MODELS` / `MELM_SKIP_SYMSPELL` | `0` |
| `MELM_SKIP_HARPER` / `MELM_SKIP_LLAMA` | `1` (optional, off) |
| `MELM_SKIP_EMAIL` / `MELM_SKIP_SELFCONFIG` | `0` |
| `MELM_FORCE_SELFCONFIG` | `0` |

---

## Security notes

- **Passphrases and API keys are never printed or logged.** Wi-Fi passphrase is
  read with `read -s`; the mailbox password is written straight to disk.
- **Secrets are 0600** (`secrets.env`, `identity.json`, `provisioned.json`,
  `device_id`). The config dir is 0700.
- **wpa_supplicant** uses `wpa_passphrase` so a hashed PSK is stored, not the
  plaintext; the `#psk=` plaintext comment is stripped; the file is installed
  0600. The passphrase is fed on stdin (never in the process list).
- **nmcli** stores the PSK in NetworkManager's system connection files (root) —
  unavoidable for that backend; the provisioner does not log it.
- **device_id** is a sha256 of the machine-id, not the raw machine-id.

---

## Recommended hooks into existing MELM code (NOT implemented here)

These are intentionally left to the owners of the core modules:

1. **Load `secrets.env` at runtime.** A small loader (e.g. in
   `melm/appliance/` or the CLI bootstrap) should read `~/.config/melm/secrets.env`
   into `os.environ` on startup so the gated `cloud_handoff` path and any agent
   email feature can find `MELM_CLOUD_API_KEY` / `MELM_MAILBOX_*`. Keep it
   opt-in and never log values.
2. **Consume `identity.json`.** The synthesis / persona layer could read
   `display_name` + `emoji` + `prefs` to set the assistant face and greeting
   style (mirrors the web UI's mood-driven face). A loader contract such as
   `assistant_identity.v1.json` already exists for identity templates — wiring
   `identity.json` into that path would let provisioning override the persona
   per device.
3. **Agent mailbox skill (future).** With `MELM_MAILBOX_*` present, a radial
   skill module (e.g. `assistant_skill_mailbox.py`) could send/read agent mail
   over IMAP/SMTP (both in stdlib: `imaplib` / `smtplib`). This would be a new
   capability-manifest-gated skill, not a core change.
4. **Optional decoder wiring.** The `ConstrainedDecoder` registry
   (`assistant_decoder*.py`) should look for the provisioned GGUF at
   `$MELM_MODELS_DIR/qwen2.5-0.5b-instruct-q8_0.gguf` when selecting a backend,
   falling back to the template backend when absent — preserving the zero-dep
   default.
```
