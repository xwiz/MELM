#!/usr/bin/env bash
#
# provision_device.sh — Robust device provisioning for MELM on a Raspberry Pi 5
# or any generic Linux box (arm64 / x86_64).
#
# This script is idempotent and safe to re-run. It downloads optional runtime
# dependencies (Qwen GGUF model, symspellpy, llama.cpp, Harper), configures
# Wi-Fi securely, writes API/secret keys, and runs the first-boot self-config.
#
# MELM itself is stdlib-only Python 3.11+ and needs NONE of these to run. The
# downloads here power optional tiers (constrained decoding, spell-correction,
# grammar linting). The base assistant works with zero of them installed.
#
# Usage:
#   scripts/provision_device.sh [OPTIONS]
#
# Run `scripts/provision_device.sh --help` for the full flag list.
#
# Conventions: stdlib-only Python, SQLite, commodity hardware. No root required
# except for system Wi-Fi config (the script will sudo only where it must, and
# tells you when).
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Constants and defaults
# --------------------------------------------------------------------------- #

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# XDG-ish locations (overridable via env).
MELM_DATA_DIR="${MELM_DATA_DIR:-${HOME}/.local/share/melm}"
MELM_CONFIG_DIR="${MELM_CONFIG_DIR:-${HOME}/.config/melm}"
MELM_MODELS_DIR="${MELM_MODELS_DIR:-${MELM_DATA_DIR}/models}"
MELM_SECRETS_FILE="${MELM_SECRETS_FILE:-${MELM_CONFIG_DIR}/secrets.env}"

# Qwen GGUF model (the ONLY model MELM uses — SmolLM is NOT used).
QWEN_REPO_ID="${MELM_QWEN_REPO_ID:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
QWEN_FILENAME="${MELM_QWEN_FILENAME:-qwen2.5-0.5b-instruct-q8_0.gguf}"
# Expected size of the q8_0 build, ~644 MB. Used for a sanity floor only.
QWEN_MIN_BYTES="${MELM_QWEN_MIN_BYTES:-500000000}"   # 500 MB floor
# Optional explicit checksum (sha256). If unset, only size sanity is enforced.
QWEN_SHA256="${MELM_QWEN_SHA256:-}"

# Retry policy for network operations.
RETRY_MAX="${MELM_RETRY_MAX:-4}"
RETRY_BASE_DELAY="${MELM_RETRY_BASE_DELAY:-3}"   # seconds; doubles each attempt

# Stage skip flags (also settable via env: MELM_SKIP_WIFI=1 etc.).
SKIP_WIFI="${MELM_SKIP_WIFI:-0}"
SKIP_MODELS="${MELM_SKIP_MODELS:-0}"
SKIP_SYMSPELL="${MELM_SKIP_SYMSPELL:-0}"
SKIP_HARPER="${MELM_SKIP_HARPER:-1}"   # optional, off by default
SKIP_LLAMA="${MELM_SKIP_LLAMA:-1}"     # optional, off by default
SKIP_EMAIL="${MELM_SKIP_EMAIL:-0}"
SKIP_SELFCONFIG="${MELM_SKIP_SELFCONFIG:-0}"
FORCE_SELFCONFIG="${MELM_FORCE_SELFCONFIG:-0}"

# Wi-Fi inputs (any may be provided via env or flags; passphrase never echoed).
WIFI_SSID="${MELM_WIFI_SSID:-}"
WIFI_PSK="${MELM_WIFI_PSK:-}"
WIFI_CONF_FILE="${MELM_WIFI_CONF_FILE:-}"   # optional file: SSID / PSK key=val
WIFI_COUNTRY="${MELM_WIFI_COUNTRY:-US}"
WIFI_BACKEND="${MELM_WIFI_BACKEND:-auto}"   # auto | nmcli | wpa_supplicant

# API key (optional cloud LLM key) provided via env or prompt.
CLOUD_API_KEY="${MELM_CLOUD_API_KEY:-}"

# Detected at runtime.
ARCH=""
OS_NAME=""

# Track what we actually did for the final summary.
declare -a SUMMARY=()

# --------------------------------------------------------------------------- #
# Logging helpers (stderr for logs, stdout reserved for any machine output)
# --------------------------------------------------------------------------- #

log()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
info()  { log "INFO  $*"; }
warn()  { log "WARN  $*"; }
error() { log "ERROR $*"; }
die()   { error "$*"; exit 1; }

summarize() { SUMMARY+=("$1"); }

# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #

usage() {
  cat <<'EOF'
provision_device.sh — provision MELM on a Raspberry Pi 5 / generic Linux box.

USAGE:
  scripts/provision_device.sh [OPTIONS]

STAGE FLAGS (skip stages you do not need; all stages are idempotent):
  --skip-wifi          Skip Wi-Fi setup (already on ethernet / connected)
  --skip-models        Skip Qwen GGUF model download
  --skip-symspell      Skip symspellpy + dictionary install
  --skip-harper        Skip Harper (harper-ls) install   [default: skipped]
  --skip-llama         Skip llama.cpp install/build       [default: skipped]
  --skip-email         Skip the one-shot venia.cloud email claim
  --skip-selfconfig    Skip the first-boot self-config step entirely

  --with-harper        Enable Harper install (overrides default skip)
  --with-llama         Enable llama.cpp install/build (overrides default skip)

OPTIONS:
  --models-dir DIR     Where to store GGUF models (default: ~/.local/share/melm/models)
  --wifi-ssid SSID     Wi-Fi network name
  --wifi-psk PSK       Wi-Fi passphrase (PREFER interactive prompt or a conf file;
                       passing on the CLI may leak via process listing)
  --wifi-conf FILE     File with SSID/PSK (key=value lines; perms should be 0600)
  --wifi-backend NAME  auto | nmcli | wpa_supplicant   (default: auto)
  --wifi-country CC    Regulatory country code (default: US)
  --api-key KEY        Cloud LLM API key (PREFER prompt/env to avoid leaks)
  --force              Re-run first-boot self-config even if already provisioned
  -h, --help           Show this help and exit

ENVIRONMENT (all flags have an env equivalent):
  MELM_DATA_DIR, MELM_CONFIG_DIR, MELM_MODELS_DIR, MELM_SECRETS_FILE
  MELM_QWEN_REPO_ID, MELM_QWEN_FILENAME, MELM_QWEN_MIN_BYTES, MELM_QWEN_SHA256
  MELM_WIFI_SSID, MELM_WIFI_PSK, MELM_WIFI_CONF_FILE, MELM_WIFI_COUNTRY, MELM_WIFI_BACKEND
  MELM_CLOUD_API_KEY
  MELM_MAILER_BASE_URL, MELM_PARTNER_TOKEN, MELM_PREFERRED_HANDLE  (first-boot self-config)
  MELM_SKIP_WIFI, MELM_SKIP_MODELS, MELM_SKIP_SYMSPELL, MELM_SKIP_HARPER,
  MELM_SKIP_LLAMA, MELM_SKIP_EMAIL, MELM_SKIP_SELFCONFIG, MELM_FORCE_SELFCONFIG

SECURITY:
  - Passphrases and API keys are NEVER printed or logged.
  - Secrets are written to ~/.config/melm/secrets.env with 0600 permissions.
  - wpa_supplicant.conf is written 0600, using wpa_passphrase so the plaintext
    PSK is not stored in the clear where avoidable.

EXAMPLES:
  # Full provision, prompt for Wi-Fi passphrase interactively:
  scripts/provision_device.sh --wifi-ssid HomeNet

  # Ethernet box, just grab the model and self-config:
  scripts/provision_device.sh --skip-wifi

  # Everything including optional llama.cpp + Harper:
  scripts/provision_device.sh --with-llama --with-harper
EOF
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-wifi)        SKIP_WIFI=1 ;;
      --skip-models)      SKIP_MODELS=1 ;;
      --skip-symspell)    SKIP_SYMSPELL=1 ;;
      --skip-harper)      SKIP_HARPER=1 ;;
      --skip-llama)       SKIP_LLAMA=1 ;;
      --skip-email)       SKIP_EMAIL=1 ;;
      --skip-selfconfig)  SKIP_SELFCONFIG=1 ;;
      --with-harper)      SKIP_HARPER=0 ;;
      --with-llama)       SKIP_LLAMA=0 ;;
      --models-dir)       MELM_MODELS_DIR="${2:?--models-dir needs a value}"; shift ;;
      --wifi-ssid)        WIFI_SSID="${2:?--wifi-ssid needs a value}"; shift ;;
      --wifi-psk)         WIFI_PSK="${2:?--wifi-psk needs a value}"; shift ;;
      --wifi-conf)        WIFI_CONF_FILE="${2:?--wifi-conf needs a value}"; shift ;;
      --wifi-backend)     WIFI_BACKEND="${2:?--wifi-backend needs a value}"; shift ;;
      --wifi-country)     WIFI_COUNTRY="${2:?--wifi-country needs a value}"; shift ;;
      --api-key)          CLOUD_API_KEY="${2:?--api-key needs a value}"; shift ;;
      --force)            FORCE_SELFCONFIG=1 ;;
      -h|--help)          usage; exit 0 ;;
      *)                  die "Unknown argument: $1 (try --help)" ;;
    esac
    shift
  done
}

# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #

detect_env() {
  OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
  local raw_arch
  raw_arch="$(uname -m 2>/dev/null || echo unknown)"
  case "${raw_arch}" in
    aarch64|arm64)        ARCH="arm64" ;;
    x86_64|amd64)         ARCH="x86_64" ;;
    armv7l|armv6l)        ARCH="armhf" ;;
    *)                    ARCH="${raw_arch}" ;;
  esac
  info "Detected OS=${OS_NAME} arch=${ARCH} (raw=${raw_arch})"
  if [[ "${OS_NAME}" != "Linux" ]]; then
    warn "This provisioner targets Linux / Raspberry Pi OS. Wi-Fi and"
    warn "system stages may be no-ops or unsupported on ${OS_NAME}."
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

python_bin() {
  if have python3; then echo python3
  elif have python; then echo python
  else return 1
  fi
}

# Generic retry-with-exponential-backoff wrapper.
# Usage: retry "<human description>" <command> [args...]
retry() {
  local desc="$1"; shift
  local attempt=1 delay="${RETRY_BASE_DELAY}"
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= RETRY_MAX )); then
      error "${desc}: failed after ${attempt} attempt(s)."
      return 1
    fi
    warn "${desc}: attempt ${attempt}/${RETRY_MAX} failed; retrying in ${delay}s..."
    sleep "${delay}"
    attempt=$(( attempt + 1 ))
    delay=$(( delay * 2 ))
  done
}

ensure_dir() {
  local d="$1" mode="${2:-0755}"
  mkdir -p "${d}"
  chmod "${mode}" "${d}" 2>/dev/null || true
}

# --------------------------------------------------------------------------- #
# Stage: model download (idempotent, retries, size/checksum sanity)
# --------------------------------------------------------------------------- #

verify_model_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local size
  size="$(stat -c '%s' "${path}" 2>/dev/null || stat -f '%z' "${path}" 2>/dev/null || echo 0)"
  if (( size < QWEN_MIN_BYTES )); then
    warn "Model file too small (${size} < ${QWEN_MIN_BYTES} bytes) — treating as incomplete."
    return 1
  fi
  if [[ -n "${QWEN_SHA256}" ]]; then
    if have sha256sum; then
      local got
      got="$(sha256sum "${path}" | awk '{print $1}')"
      if [[ "${got}" != "${QWEN_SHA256}" ]]; then
        warn "Model checksum mismatch (got ${got}, expected ${QWEN_SHA256})."
        return 1
      fi
      info "Model checksum verified."
    else
      warn "sha256sum not available; skipping checksum verification."
    fi
  fi
  return 0
}

download_model_hf() {
  local py; py="$(python_bin)" || return 1
  "${py}" - "$@" <<'PYEOF'
import sys
repo_id, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from huggingface_hub import hf_hub_download
except Exception:
    sys.exit(3)  # signal "hub not available" to the caller
path = hf_hub_download(repo_id=repo_id, filename=filename,
                       local_dir=local_dir, local_dir_use_symlinks=False)
print(path)
PYEOF
}

download_model_curl() {
  local url="$1" out="$2"
  # --fail: error on HTTP >=400, -L: follow redirects, -C -: resume partial.
  curl --fail --location --retry 0 -C - -o "${out}.part" "${url}" \
    && mv -f "${out}.part" "${out}"
}

stage_models() {
  if [[ "${SKIP_MODELS}" == "1" ]]; then
    info "Skipping model download (--skip-models)."
    summarize "models: SKIPPED"
    return 0
  fi
  ensure_dir "${MELM_MODELS_DIR}"
  local target="${MELM_MODELS_DIR}/${QWEN_FILENAME}"

  if verify_model_file "${target}"; then
    info "Model already present and valid: ${target}"
    summarize "models: present (${QWEN_FILENAME})"
    return 0
  fi

  info "Downloading Qwen model ${QWEN_REPO_ID}/${QWEN_FILENAME} (~644MB) ..."

  # Prefer huggingface_hub (handles resume + integrity); fall back to curl.
  local py rc
  if py="$(python_bin)"; then
    set +e
    retry "huggingface_hub download" \
      download_model_hf "${QWEN_REPO_ID}" "${QWEN_FILENAME}" "${MELM_MODELS_DIR}"
    rc=$?
    set -e
    if [[ "${rc}" == "0" ]] && verify_model_file "${target}"; then
      info "Model downloaded via huggingface_hub."
      summarize "models: downloaded (huggingface_hub)"
      return 0
    fi
    info "huggingface_hub unavailable or failed; falling back to curl."
  fi

  if ! have curl; then
    die "Neither a working huggingface_hub nor curl is available — cannot fetch model."
  fi
  local url="https://huggingface.co/${QWEN_REPO_ID}/resolve/main/${QWEN_FILENAME}?download=true"
  if ! retry "curl model download" download_model_curl "${url}" "${target}"; then
    die "Model download failed. Check connectivity, then re-run (download resumes)."
  fi
  if ! verify_model_file "${target}"; then
    die "Downloaded model failed size/checksum sanity check: ${target}"
  fi
  info "Model downloaded via curl: ${target}"
  summarize "models: downloaded (curl)"
}

# --------------------------------------------------------------------------- #
# Stage: symspellpy (spell-correction tier) + bundled frequency dictionaries
# --------------------------------------------------------------------------- #

stage_symspell() {
  if [[ "${SKIP_SYMSPELL}" == "1" ]]; then
    info "Skipping symspellpy install (--skip-symspell)."
    summarize "symspell: SKIPPED"
    return 0
  fi
  local py; py="$(python_bin)" || { warn "No Python; skipping symspellpy."; summarize "symspell: SKIPPED (no python)"; return 0; }

  if "${py}" -c 'import symspellpy' >/dev/null 2>&1; then
    info "symspellpy already installed."
    summarize "symspell: present"
    return 0
  fi

  # Pick a pip invocation that exists.
  local pip_cmd=()
  if "${py}" -m pip --version >/dev/null 2>&1; then
    pip_cmd=("${py}" -m pip install --user --upgrade symspellpy)
  elif have pip3; then
    pip_cmd=(pip3 install --user --upgrade symspellpy)
  else
    warn "pip not available; cannot install symspellpy. Skipping."
    summarize "symspell: SKIPPED (no pip)"
    return 0
  fi

  if retry "symspellpy install" "${pip_cmd[@]}"; then
    # symspellpy ships frequency_dictionary_en_82_765.txt + bigram dict in-package.
    if "${py}" -c 'import importlib.resources as r, symspellpy; print(r.files(symspellpy))' >/dev/null 2>&1; then
      info "symspellpy installed (bundled frequency dictionaries available in-package)."
    else
      info "symspellpy installed."
    fi
    summarize "symspell: installed"
  else
    warn "symspellpy install failed; spell-correction tier will be unavailable. Continuing."
    summarize "symspell: FAILED (non-fatal)"
  fi
}

# --------------------------------------------------------------------------- #
# Stage: llama.cpp (optional) — prebuilt binary if available, else build
# --------------------------------------------------------------------------- #

stage_llama() {
  if [[ "${SKIP_LLAMA}" == "1" ]]; then
    info "Skipping llama.cpp (use --with-llama to enable)."
    summarize "llama.cpp: SKIPPED"
    return 0
  fi
  if have llama-cli || have llama-server || have main; then
    info "llama.cpp binary already on PATH."
    summarize "llama.cpp: present"
    return 0
  fi

  # 1) Prefer a distro/package-managed prebuilt where it exists.
  if have brew; then
    info "Attempting prebuilt llama.cpp via Homebrew."
    if retry "brew install llama.cpp" brew install llama.cpp; then
      summarize "llama.cpp: installed (brew)"
      return 0
    fi
  fi

  # 2) Build from source with cmake.
  if ! have git || ! have cmake; then
    warn "git and cmake are required to build llama.cpp; neither prebuilt nor"
    warn "build toolchain found. Install build-essential cmake git, or re-run"
    warn "with --skip-llama. Continuing without llama.cpp."
    summarize "llama.cpp: SKIPPED (no toolchain)"
    return 0
  fi

  local src_dir="${MELM_DATA_DIR}/src/llama.cpp"
  ensure_dir "$(dirname "${src_dir}")"
  if [[ ! -d "${src_dir}/.git" ]]; then
    if ! retry "git clone llama.cpp" git clone --depth 1 https://github.com/ggml-org/llama.cpp "${src_dir}"; then
      warn "Failed to clone llama.cpp; continuing without it."
      summarize "llama.cpp: FAILED clone (non-fatal)"
      return 0
    fi
  else
    info "llama.cpp source already cloned; reusing."
  fi

  info "Building llama.cpp from source (arch=${ARCH}) — this can take several minutes."
  if cmake -S "${src_dir}" -B "${src_dir}/build" -DCMAKE_BUILD_TYPE=Release \
     && cmake --build "${src_dir}/build" --config Release -j "$(nproc 2>/dev/null || echo 2)"; then
    info "llama.cpp built. Binaries under: ${src_dir}/build/bin"
    summarize "llama.cpp: built from source (${src_dir}/build/bin)"
  else
    warn "llama.cpp build failed; continuing without it."
    summarize "llama.cpp: FAILED build (non-fatal)"
  fi
}

# --------------------------------------------------------------------------- #
# Stage: Harper (optional grammar linter) — cargo install harper-ls
# --------------------------------------------------------------------------- #

stage_harper() {
  if [[ "${SKIP_HARPER}" == "1" ]]; then
    info "Skipping Harper (use --with-harper to enable)."
    summarize "harper: SKIPPED"
    return 0
  fi
  if have harper-ls; then
    info "harper-ls already installed."
    summarize "harper: present"
    return 0
  fi
  if ! have cargo; then
    warn "cargo not found; cannot install harper-ls. Install Rust (rustup) or"
    warn "re-run with --skip-harper. Continuing without Harper."
    summarize "harper: SKIPPED (no cargo)"
    return 0
  fi
  if retry "cargo install harper-ls" cargo install harper-ls; then
    info "harper-ls installed via cargo."
    summarize "harper: installed (cargo)"
  else
    warn "harper-ls install failed; continuing without Harper."
    summarize "harper: FAILED (non-fatal)"
  fi
}

# --------------------------------------------------------------------------- #
# Stage: secure Wi-Fi setup (NetworkManager nmcli OR headless wpa_supplicant)
# --------------------------------------------------------------------------- #

# Load SSID/PSK from a conf file if provided. File format: simple key=value
#   ssid=MyNetwork
#   psk=secretpassphrase
load_wifi_conf_file() {
  local f="$1"
  [[ -f "${f}" ]] || die "Wi-Fi conf file not found: ${f}"
  # Read without echoing the PSK.
  local line key val
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"                       # strip comments
    [[ -z "${line//[[:space:]]/}" ]] && continue
    key="${line%%=*}"; val="${line#*=}"
    key="$(echo "${key}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    # Trim surrounding whitespace from value only (preserve internal chars).
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    case "${key}" in
      ssid)     [[ -z "${WIFI_SSID}" ]] && WIFI_SSID="${val}" ;;
      psk|psk=) [[ -z "${WIFI_PSK}"  ]] && WIFI_PSK="${val}" ;;
      passphrase) [[ -z "${WIFI_PSK}" ]] && WIFI_PSK="${val}" ;;
    esac
  done < "${f}"
}

prompt_wifi_secrets() {
  if [[ -z "${WIFI_SSID}" ]]; then
    printf 'Wi-Fi SSID: ' >&2
    IFS= read -r WIFI_SSID
  fi
  if [[ -z "${WIFI_PSK}" ]]; then
    printf 'Wi-Fi passphrase (input hidden): ' >&2
    # -s: silent. Never echoed.
    IFS= read -rs WIFI_PSK
    printf '\n' >&2
  fi
}

already_online() {
  # Best-effort connectivity check; conservative (no false positives).
  if have nmcli; then
    nmcli -t -f STATE general 2>/dev/null | grep -q '^connected$' && return 0
  fi
  if have ip; then
    ip route get 1.1.1.1 >/dev/null 2>&1 && return 0
  fi
  return 1
}

wifi_via_nmcli() {
  info "Configuring Wi-Fi via NetworkManager (nmcli)."
  # Idempotent: nmcli reuses/updates a connection profile of the same SSID.
  # PSK passed as an argument here is unavoidable for nmcli; it does not get
  # logged by this script. nmcli stores it in system connection files (root).
  if sudo nmcli device wifi connect "${WIFI_SSID}" password "${WIFI_PSK}" >/dev/null 2>&1; then
    info "Connected to '${WIFI_SSID}' via nmcli."
    summarize "wifi: connected via nmcli (ssid hidden)"
    return 0
  fi
  warn "nmcli connect failed (out of range, wrong passphrase, or no Wi-Fi radio)."
  return 1
}

wifi_via_wpa_supplicant() {
  info "Configuring headless Wi-Fi via wpa_supplicant."
  if ! have wpa_passphrase; then
    warn "wpa_passphrase not found; cannot write a hashed PSK. Install wpasupplicant."
    return 1
  fi
  local conf="/etc/wpa_supplicant/wpa_supplicant.conf"
  local tmp
  tmp="$(mktemp)"
  chmod 0600 "${tmp}"
  {
    printf 'country=%s\n' "${WIFI_COUNTRY}"
    printf 'ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n'
    printf 'update_config=1\n\n'
  } > "${tmp}"
  # wpa_passphrase derives the hashed PSK so the plaintext is NOT stored.
  # Feed the passphrase on stdin so it never appears in the process list.
  if ! printf '%s' "${WIFI_PSK}" | wpa_passphrase "${WIFI_SSID}" >> "${tmp}" 2>/dev/null; then
    rm -f "${tmp}"
    warn "wpa_passphrase failed (passphrase length must be 8..63 chars)."
    return 1
  fi
  # Strip the human-readable plaintext '#psk=' comment line wpa_passphrase adds.
  if have sed; then
    sed -i '/^[[:space:]]*#psk=/d' "${tmp}" 2>/dev/null || true
  fi
  info "Installing ${conf} (0600, root). You may be prompted for sudo."
  if sudo install -m 0600 "${tmp}" "${conf}"; then
    rm -f "${tmp}"
    info "Wrote ${conf}. Reconfiguring wpa_supplicant."
    sudo wpa_cli -i "$(detect_wlan_iface)" reconfigure >/dev/null 2>&1 \
      || sudo systemctl restart wpa_supplicant 2>/dev/null \
      || warn "Could not auto-reload wpa_supplicant; a reboot will apply it."
    summarize "wifi: wpa_supplicant.conf written 0600 (hashed PSK, ssid hidden)"
    return 0
  fi
  rm -f "${tmp}"
  warn "Failed to install ${conf} (need root)."
  return 1
}

detect_wlan_iface() {
  if have iw; then
    iw dev 2>/dev/null | awk '/Interface/{print $2; exit}'
  elif have ip; then
    ip -o link show 2>/dev/null | awk -F': ' '/wlan|wlp/{print $2; exit}'
  fi
}

stage_wifi() {
  if [[ "${SKIP_WIFI}" == "1" ]]; then
    info "Skipping Wi-Fi setup (--skip-wifi)."
    summarize "wifi: SKIPPED"
    return 0
  fi
  if [[ "${OS_NAME}" != "Linux" ]]; then
    warn "Wi-Fi setup only supported on Linux; skipping on ${OS_NAME}."
    summarize "wifi: SKIPPED (non-Linux)"
    return 0
  fi
  if already_online; then
    info "Device already appears to be online (ethernet or existing Wi-Fi)."
    info "Skipping Wi-Fi setup. Use no --skip-wifi but disconnect to force reconfigure."
    summarize "wifi: already online (skipped)"
    return 0
  fi

  if [[ -n "${WIFI_CONF_FILE}" ]]; then
    load_wifi_conf_file "${WIFI_CONF_FILE}"
  fi
  prompt_wifi_secrets
  [[ -n "${WIFI_SSID}" ]] || die "No Wi-Fi SSID provided."
  [[ -n "${WIFI_PSK}"  ]] || die "No Wi-Fi passphrase provided."

  local backend="${WIFI_BACKEND}"
  if [[ "${backend}" == "auto" ]]; then
    if have nmcli; then backend="nmcli"; else backend="wpa_supplicant"; fi
  fi
  info "Wi-Fi backend: ${backend}"

  local ok=1
  case "${backend}" in
    nmcli)          wifi_via_nmcli || ok=0 ;;
    wpa_supplicant) wifi_via_wpa_supplicant || ok=0 ;;
    *)              die "Unknown --wifi-backend: ${backend}" ;;
  esac

  # Scrub the plaintext passphrase from memory ASAP.
  WIFI_PSK=""
  unset WIFI_PSK
  WIFI_PSK=""

  if [[ "${ok}" != "1" ]]; then
    warn "Wi-Fi configuration did not complete. The device may need ethernet,"
    warn "or re-run after checking SSID/passphrase. Continuing with provisioning."
    summarize "wifi: FAILED (non-fatal)"
  fi
}

# --------------------------------------------------------------------------- #
# Stage: API key / secrets file (0600)
# --------------------------------------------------------------------------- #

ensure_secrets_file() {
  ensure_dir "${MELM_CONFIG_DIR}" 0700
  if [[ ! -f "${MELM_SECRETS_FILE}" ]]; then
    umask 077
    cat > "${MELM_SECRETS_FILE}" <<'EOF'
# MELM secrets — KEEP THIS FILE PRIVATE (mode 0600).
# Lines are KEY=VALUE. Loaded by MELM at runtime; never committed to git.
#
# Cloud LLM API key (optional; only used by the gated cloud_handoff path):
# MELM_CLOUD_API_KEY=
#
# Mailbox credentials (filled by first_boot_selfconfig.py after email claim):
# MELM_MAILBOX_EMAIL=
# MELM_MAILBOX_PASSWORD=
# MELM_MAILBOX_IMAP_HOST=
# MELM_MAILBOX_SMTP_HOST=
EOF
  fi
  chmod 0600 "${MELM_SECRETS_FILE}"
}

# Upsert a KEY=VALUE into the secrets file without echoing the value.
set_secret() {
  local key="$1" val="$2"
  ensure_secrets_file
  local tmp
  tmp="$(mktemp)"
  chmod 0600 "${tmp}"
  # Drop any existing (commented or live) line for this key, then append.
  grep -v -E "^#?[[:space:]]*${key}=" "${MELM_SECRETS_FILE}" > "${tmp}" 2>/dev/null || true
  printf '%s=%s\n' "${key}" "${val}" >> "${tmp}"
  mv -f "${tmp}" "${MELM_SECRETS_FILE}"
  chmod 0600 "${MELM_SECRETS_FILE}"
}

stage_api_key() {
  ensure_secrets_file
  if [[ -z "${CLOUD_API_KEY}" ]]; then
    # Only prompt if a TTY is attached; otherwise leave it blank (optional).
    if [[ -t 0 ]]; then
      printf 'Cloud LLM API key (optional, press Enter to skip; input hidden): ' >&2
      IFS= read -rs CLOUD_API_KEY
      printf '\n' >&2
    fi
  fi
  if [[ -n "${CLOUD_API_KEY}" ]]; then
    set_secret "MELM_CLOUD_API_KEY" "${CLOUD_API_KEY}"
    CLOUD_API_KEY=""; unset CLOUD_API_KEY; CLOUD_API_KEY=""
    info "Cloud API key stored in ${MELM_SECRETS_FILE} (0600)."
    summarize "api_key: stored (value hidden)"
  else
    info "No cloud API key provided; leaving template in place (optional)."
    summarize "api_key: not set (optional)"
  fi
}

# --------------------------------------------------------------------------- #
# Stage: first-boot self-config (delegates to the Python helper)
# --------------------------------------------------------------------------- #

stage_selfconfig() {
  if [[ "${SKIP_SELFCONFIG}" == "1" ]]; then
    info "Skipping first-boot self-config (--skip-selfconfig)."
    summarize "selfconfig: SKIPPED"
    return 0
  fi
  local py; py="$(python_bin)" || die "Python 3.11+ required for self-config."
  local helper="${SCRIPT_DIR}/first_boot_selfconfig.py"
  [[ -f "${helper}" ]] || die "Missing helper: ${helper}"

  local args=(--config-dir "${MELM_CONFIG_DIR}" --secrets-file "${MELM_SECRETS_FILE}")
  [[ "${FORCE_SELFCONFIG}" == "1" ]] && args+=(--force)
  [[ "${SKIP_EMAIL}" == "1" ]]      && args+=(--skip-email)

  info "Running first-boot self-config."
  if "${py}" "${helper}" "${args[@]}"; then
    summarize "selfconfig: completed"
  else
    local rc=$?
    warn "Self-config returned non-zero (${rc}); see messages above. Continuing."
    summarize "selfconfig: PARTIAL (exit ${rc})"
  fi
}

# --------------------------------------------------------------------------- #
# Final summary
# --------------------------------------------------------------------------- #

print_summary() {
  info "==================== PROVISIONING SUMMARY ===================="
  local item
  for item in "${SUMMARY[@]}"; do
    info "  - ${item}"
  done
  info "  config dir : ${MELM_CONFIG_DIR}"
  info "  data dir   : ${MELM_DATA_DIR}"
  info "  models dir : ${MELM_MODELS_DIR}"
  info "  secrets    : ${MELM_SECRETS_FILE} (0600)"
  info "============================================================="
  info "MELM runs stdlib-only: 'python3 -m melm chat' from the repo root."
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

main() {
  parse_args "$@"
  detect_env
  ensure_dir "${MELM_DATA_DIR}"
  ensure_dir "${MELM_CONFIG_DIR}" 0700
  ensure_dir "${MELM_MODELS_DIR}"

  stage_wifi
  stage_api_key
  stage_models
  stage_symspell
  stage_llama
  stage_harper
  stage_selfconfig

  print_summary
}

main "$@"
