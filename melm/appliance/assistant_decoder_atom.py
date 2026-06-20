"""Atom-aware answer generation through intent-specific templates.

The ``AtomTemplateBackend`` reads UOL atoms from ``decision.uol_act``
and renders intent-specific template strings by filling slot values
extracted from atom roles (theme, agent, attribute, location, etc.) plus
the predicate lemma as ``{verb}``.

This is the first step toward replacing the per-intent handler functions
in ``assistant_synthesis._answer()`` with a generic atom-based renderer.
"""

import re
from typing import Any


class AtomTemplateBackend:
    """Generate answer text from UOL atoms using intent-specific templates.

    Templates use ``{role_name}`` placeholders. Slot values are extracted
    from the first atom's roles.
    """

    _DEFAULT_TEMPLATES: dict[str, str] = {
        "weather": "The weather {theme} is expected to be pleasant.",
        "meal_suggestion": "How about a meal suggestion for something to eat?",
        "assistant_identity": "I am a local assistant running on your device.",
        "assistant_status": "I am running and available.",
        "health_advice": "Please consult a medical professional for health concerns.",
        "social_contact": "I can help you contact {theme} when you're ready.",
        "personal_memory": "I remember that from our earlier conversation.",
        "autobiographical_memory": "Here is what I recall from our conversations.",
        "story": "Let me tell you a story about {theme}.",
        "media_playback": "I can play media for you.",
        "common_sense_safety": "That may not be safe to do in public.",
        "social_greeting": "Hello! How can I help you today?",
    }

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        self._templates = templates or {}

    def generate(
        self,
        intent: str,
        uol_act: dict | Any | None,
        evidence: dict[str, Any] | None = None,
        *,
        extra_templates: dict[str, str] | None = None,
    ) -> str | None:
        """Generate answer text from UOL atoms, or *None* if not possible."""
        if uol_act is None:
            return None
        if not self._has_atoms(uol_act):
            return None
        template = self._resolve_template(intent, extra_templates)
        if template is None:
            return None
        roles = self._extract_roles(uol_act)
        return self._render(template, roles)

    @staticmethod
    def _has_atoms(uol_act: dict | Any) -> bool:
        if not isinstance(uol_act, dict):
            return False
        content = uol_act.get("content", [])
        return bool(content) and isinstance(content, (list, tuple))

    def _resolve_template(
        self, intent: str, extra: dict[str, str] | None,
    ) -> str | None:
        templates = extra or self._templates
        if intent in templates:
            return templates[intent]
        if intent in self._DEFAULT_TEMPLATES:
            return self._DEFAULT_TEMPLATES[intent]
        return None

    def _extract_roles(self, uol_act: dict | Any) -> dict[str, str]:
        roles: dict[str, str] = {}
        if not isinstance(uol_act, dict):
            return roles
        content = uol_act.get("content", [])
        if not content:
            return roles
        atom = content[0] if isinstance(content, (list, tuple)) else {}
        if not isinstance(atom, dict):
            return roles
        for entry in atom.get("roles", []):
            if isinstance(entry, dict):
                r = entry.get("role", "")
                v = entry.get("value", "")
                if r and isinstance(v, str) and r not in roles:
                    roles[r] = v
        pred = atom.get("predicate", {}) or {}
        if isinstance(pred, dict):
            roles["verb"] = str(pred.get("lemma", "") or pred.get("id", ""))
        return roles

    def _render(self, template: str, roles: dict[str, str]) -> str:
        result = template
        for key, value in roles.items():
            result = result.replace("{" + key + "}", value)
        result = re.sub(r"\{[^}]+\}", "", result)
        result = re.sub(r" +", " ", result).strip()
        return result
