"""Typed local action execution adapters for the Assistant OS MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Literal


ActionMode = Literal["dry-run", "real"]


@dataclass(frozen=True)
class DeviceActionExecutionResult:
    action_id: str
    action_type: str
    target: str
    mode: ActionMode
    status: str
    side_effect_executed: bool
    reason: str
    command: tuple[str, ...] = ()
    resolved_target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target": self.target,
            "mode": self.mode,
            "status": self.status,
            "side_effect_executed": self.side_effect_executed,
            "reason": self.reason,
            "command": list(self.command),
            "resolved_target": self.resolved_target,
            "payload": self.payload,
        }


class LocalDeviceActionExecutor:
    """Executes confirmed local actions through explicit adapters.

    The default `dry-run` mode records the action boundary without causing OS
    side effects. `real` mode requires an executable command supplied as an
    argv-style string and still refuses missing local media paths.
    """

    def __init__(
        self,
        *,
        mode: ActionMode = "dry-run",
        media_player_command: str = "",
        call_command: str = "",
        tts_command: str = "",
        timeout_seconds: float = 5.0,
    ) -> None:
        if mode not in {"dry-run", "real"}:
            raise ValueError(f"unsupported action mode: {mode!r}")
        self.mode: ActionMode = mode
        self.media_player_command = media_player_command
        self.call_command = call_command
        self.tts_command = tts_command
        self.timeout_seconds = timeout_seconds

    def execute(self, pending: dict[str, Any], *, store: Any | None = None) -> DeviceActionExecutionResult:
        action_type = str(pending.get("action_type", ""))
        if action_type == "play_media":
            return self._execute_media(pending, store=store)
        if action_type == "call_contact":
            return self._execute_call(pending, store=store)
        if action_type == "say_tts":
            return self._execute_tts(pending)
        return DeviceActionExecutionResult(
            action_id=str(pending.get("action_id", "")),
            action_type=action_type,
            target=str(pending.get("target", "")),
            mode=self.mode,
            status="blocked",
            side_effect_executed=False,
            reason="unsupported_action_type",
        )

    def _execute_media(self, pending: dict[str, Any], *, store: Any | None) -> DeviceActionExecutionResult:
        target = str(pending.get("target", ""))
        item_id = _media_item_id_from_target(target)
        payload = _media_payload(store, item_id)
        media_path = str(payload.get("path", ""))
        metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
        resolved_path = str(metadata.get("resolved_path") or media_path)
        resolved_target = resolved_path or media_path or item_id
        if self.mode == "dry-run":
            return DeviceActionExecutionResult(
                action_id=str(pending.get("action_id", "")),
                action_type="play_media",
                target=target,
                mode=self.mode,
                status="prepared",
                side_effect_executed=False,
                reason="dry_run_no_side_effect",
                resolved_target=resolved_target,
                payload={
                    "item_id": item_id,
                    "title": payload.get("title", item_id),
                    "path": media_path,
                    "resolved_path": resolved_path,
                    "path_exists": bool(payload.get("path_exists", False)),
                },
            )
        if not self.media_player_command:
            return self._blocked_real_action(pending, "play_media", target, "missing_media_player_command")
        if not media_path:
            return self._blocked_real_action(pending, "play_media", target, "missing_media_path")
        path = Path(resolved_path)
        if not path.exists():
            return self._blocked_real_action(pending, "play_media", target, "media_path_not_found")
        return self._run_command(
            pending,
            action_type="play_media",
            target=target,
            command=(*_command_argv(self.media_player_command), str(path)),
            resolved_target=str(path),
            payload={"item_id": item_id, "path": str(path), "path_exists": True},
        )

    def _execute_call(self, pending: dict[str, Any], *, store: Any | None = None) -> DeviceActionExecutionResult:
        target = str(pending.get("target", ""))
        contact_id, number = _call_contact_target(pending, store)
        resolved_target = number or target
        payload = {"contact_id": contact_id} if contact_id else {}
        if self.mode == "dry-run":
            return DeviceActionExecutionResult(
                action_id=str(pending.get("action_id", "")),
                action_type="call_contact",
                target=target,
                mode=self.mode,
                status="prepared",
                side_effect_executed=False,
                reason="dry_run_no_side_effect",
                resolved_target=resolved_target,
                payload=payload,
            )
        if not self.call_command:
            return self._blocked_real_action(pending, "call_contact", target, "missing_call_command")
        return self._run_command(
            pending,
            action_type="call_contact",
            target=target,
            command=(*_command_argv(self.call_command), resolved_target),
            resolved_target=resolved_target,
            payload=payload,
        )

    def _run_command(
        self,
        pending: dict[str, Any],
        *,
        action_type: str,
        target: str,
        command: tuple[str, ...],
        resolved_target: str,
        payload: dict[str, Any] | None = None,
    ) -> DeviceActionExecutionResult:
        if not command:
            return self._blocked_real_action(pending, action_type, target, "empty_command")
        try:
            completed = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive real-device path
            return DeviceActionExecutionResult(
                action_id=str(pending.get("action_id", "")),
                action_type=action_type,
                target=target,
                mode=self.mode,
                status="error",
                side_effect_executed=False,
                reason=type(exc).__name__,
                command=command,
                resolved_target=resolved_target,
                payload={"error": str(exc), **(payload or {})},
            )
        return DeviceActionExecutionResult(
            action_id=str(pending.get("action_id", "")),
            action_type=action_type,
            target=target,
            mode=self.mode,
            status="executed" if completed.returncode == 0 else "error",
            side_effect_executed=completed.returncode == 0,
            reason=f"returncode:{completed.returncode}",
            command=command,
            resolved_target=resolved_target,
            payload={
                "returncode": completed.returncode,
                "stdout": completed.stdout[:500],
                "stderr": completed.stderr[:500],
                **(payload or {}),
            },
        )

    def _execute_tts(self, pending: dict[str, Any]) -> DeviceActionExecutionResult:
        text = str(pending.get("target", ""))
        if not text:
            return self._blocked_real_action(pending, "say_tts", text, "empty_tts_text")
        if self.mode == "dry-run":
            return DeviceActionExecutionResult(
                action_id=str(pending.get("action_id", "")),
                action_type="say_tts",
                target=text,
                mode=self.mode,
                status="prepared",
                side_effect_executed=False,
                reason="dry_run_no_side_effect",
                resolved_target=text[:50],
                payload={"text_length": len(text)},
            )
        if not self.tts_command:
            return self._blocked_real_action(pending, "say_tts", text, "missing_tts_command")
        return self._run_command(
            pending,
            action_type="say_tts",
            target=text,
            command=(*_command_argv(self.tts_command), text),
            resolved_target=text[:50],
            payload={"text_length": len(text)},
        )

    def _blocked_real_action(
        self,
        pending: dict[str, Any],
        action_type: str,
        target: str,
        reason: str,
    ) -> DeviceActionExecutionResult:
        return DeviceActionExecutionResult(
            action_id=str(pending.get("action_id", "")),
            action_type=action_type,
            target=target,
            mode=self.mode,
            status="blocked",
            side_effect_executed=False,
            reason=reason,
            resolved_target=target,
        )


def _media_payload(store: Any | None, item_id: str) -> dict[str, Any]:
    if store is None or not item_id:
        return {}
    return dict(store.load_inventory("media").get(item_id, {}))


def _call_contact_target(pending: dict[str, Any], store: Any | None) -> tuple[str, str]:
    contact_id = ""
    for key in pending.get("evidence_keys", ()):
        key_text = str(key)
        if key_text.startswith("contacts."):
            contact_id = key_text.split(".", 1)[1]
            break
    if not contact_id:
        contact_id = _contact_id_from_target(str(pending.get("target", "")))
    if store is None or not contact_id:
        return contact_id, ""
    payload = dict(store.load_inventory("contact").get(contact_id, {}))
    return contact_id, str(payload.get("number", ""))


def _contact_id_from_target(target: str) -> str:
    text = target.lower()
    text = re.sub(r"\b(i|can|call|calling|confirmed)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _media_item_id_from_target(target: str) -> str:
    text = target.lower()
    text = re.sub(r"\b(confirmed|playing|play|start|starting)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [word for word in text.split() if word not in {"media", "song"}]
    return " ".join(words)


def _command_argv(command: str) -> tuple[str, ...]:
    return tuple(part for part in shlex.split(command) if part)
