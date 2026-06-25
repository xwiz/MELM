"""Startup provisioning helpers — stdlib only.

Called at ``python -m melm`` entry point to inject secrets.env into os.environ
and to locate the provisioned GGUF model for the ConstrainedDecoder backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_SECRETS_PATH = Path.home() / ".config" / "melm" / "secrets.env"
_IDENTITY_PATH = Path.home() / ".config" / "melm" / "identity.json"
_DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "melm" / "models"
_QWEN_FILENAME = "qwen2.5-0.5b-instruct-q8_0.gguf"


def load_secrets_env(path: Path = _SECRETS_PATH) -> int:
    """Load KEY=VALUE pairs from *path* into ``os.environ`` (skip existing keys).

    Returns number of variables loaded. No-op if file absent or unreadable.
    Lines starting with ``#`` are comments; blank lines are skipped.
    """
    if not path.is_file():
        return 0
    loaded = 0
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value.strip()
                loaded += 1
    except Exception:
        pass
    return loaded


def load_identity_prefs(path: Path = _IDENTITY_PATH) -> dict:
    """Load ~/.config/melm/identity.json and return its parsed dict.

    Returns ``{}`` if the file is absent, unreadable, or not valid JSON.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_gguf_model_path() -> Path | None:
    """Return path to the provisioned Qwen GGUF, or ``None`` if not found.

    Checks ``$MELM_MODELS_DIR`` first, then the default models directory
    (``~/.local/share/melm/models/``).
    """
    models_dir = Path(os.environ.get("MELM_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))
    candidate = models_dir / _QWEN_FILENAME
    return candidate if candidate.is_file() else None
