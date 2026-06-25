"""Atom-aware answer generation through intent-specific templates.

The ``AtomTemplateBackend`` reads UOL atoms from ``decision.uol_act``
and renders intent-specific template strings by filling slot values
extracted from atom roles (theme, agent, attribute, location, etc.) plus
the predicate lemma as ``{verb}``.

V0.4+ enhancement: atoms connected by AtomLinks (causes/caused_by/enables/
prevents) are resolved so causal templates can use ``{cause}``, ``{effect}``,
``{cause_verb}``, ``{effect_verb}``, and other role placeholders drawn from
the linked atoms' predicates and role assignments.

This is the first step toward replacing the per-intent handler functions
in ``assistant_synthesis._answer()`` with a generic atom-based renderer.

The ``AtomDecoderBackend`` wraps ``AtomTemplateBackend`` as a
``DecoderBackend``-protocol-compliant backend registered in the
``ConstrainedDecoder`` registry so that ``_decode_verified()`` can
dispatch through the standard decoder chain.
"""

import re
from typing import Any

_ATOM_TEMPLATES_CACHE: dict[str, str] | None = None


def _load_atom_templates_cached() -> dict[str, str]:
    global _ATOM_TEMPLATES_CACHE
    if _ATOM_TEMPLATES_CACHE is None:
        try:
            from melm.contracts.validation import load_atom_templates
            _ATOM_TEMPLATES_CACHE = load_atom_templates()
        except Exception:
            _ATOM_TEMPLATES_CACHE = {}
    return _ATOM_TEMPLATES_CACHE


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
        "causal_explanation": "{effect} happens because {cause}.",
        "causal_prediction": "If {cause}, then {effect} may happen.",
    }

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        self._templates = templates or {}
        if not self._templates:
            contract = _load_atom_templates_cached()
            if contract:
                self._templates = dict(contract)

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
        contract = _load_atom_templates_cached()
        if intent in contract:
            return contract[intent]
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
        atoms = content if isinstance(content, (list, tuple)) else [content]
        for idx, atom in enumerate(atoms[:5]):
            if not isinstance(atom, dict):
                continue
            for entry in atom.get("roles", []):
                if isinstance(entry, dict):
                    r = entry.get("role", "")
                    v = entry.get("value", "")
                    if r and isinstance(v, str):
                        if r not in roles:
                            roles[r] = v
                        else:
                            suffix = 1
                            while f"{r}_{suffix}" in roles:
                                suffix += 1
                            roles[f"{r}_{suffix}"] = v
            if idx == 0:
                pred = atom.get("predicate", {}) or {}
                if isinstance(pred, dict):
                    roles["verb"] = str(pred.get("lemma", "") or pred.get("id", ""))
        # AtomLink-aware: resolve causal links between atoms
        links = self._extract_causal_links(uol_act, atoms)
        roles.update(links)
        return roles

    def _extract_causal_links(
        self, uol_act: dict | Any, atoms: list[dict],
    ) -> dict[str, str]:
        """Extract cause/effect roles from AtomLinks between atoms.

        Returns a dict with keys like ``{cause}``, ``{effect}``, etc.
        so causal templates (e.g. ``causal_explanation``) can render
        ``{effect} happens because {cause}``.
        """
        result: dict[str, str] = {}
        atom_by_pred: dict[str, dict] = {}
        for atom in atoms:
            pid = (atom.get("predicate") or {}).get("id", "").lower()
            if pid:
                atom_by_pred[pid] = atom

        for atom in atoms:
            links = atom.get("links") or {}
            causes = links.get("causes") or []
            caused_by = links.get("caused_by") or []
            enables = links.get("enables") or []
            prevents = links.get("prevents") or []

            if not (causes or caused_by or enables or prevents):
                continue

            # If this atom causes something: it's the cause
            for target_pid in causes:
                target = atom_by_pred.get(target_pid) or {}
                result["cause"] = self._atom_text(atom)
                result["cause_verb"] = (atom.get("predicate") or {}).get("lemma", target_pid)
                result["effect"] = self._atom_text(target)
                result["effect_verb"] = (target.get("predicate") or {}).get("lemma", target_pid)
                for role_entry in target.get("roles", []):
                    rn = role_entry.get("role", "") if isinstance(role_entry, dict) else ""
                    rv = role_entry.get("value", "") if isinstance(role_entry, dict) else ""
                    if rn and rv:
                        result.setdefault(f"effect_{rn}", rv)

            # If this atom is caused_by something: that something is the cause
            for source_pid in caused_by:
                source = atom_by_pred.get(source_pid) or {}
                result["effect"] = self._atom_text(atom)
                result["effect_verb"] = (atom.get("predicate") or {}).get("lemma", "")
                result["cause"] = self._atom_text(source)
                result["cause_verb"] = (source.get("predicate") or {}).get("lemma", source_pid)
                for role_entry in atom.get("roles", []):
                    rn = role_entry.get("role", "") if isinstance(role_entry, dict) else ""
                    rv = role_entry.get("value", "") if isinstance(role_entry, dict) else ""
                    if rn and rv:
                        result.setdefault(f"effect_{rn}", rv)

            for _ in enables:
                result["relation"] = "enables"
            for _ in prevents:
                result["relation"] = "prevents"

        # If we found causal links but missing a role value, fill from fallback
        if "cause" not in result:
            for atom in atoms:
                links = atom.get("links") or {}
                if links.get("causes") or links.get("caused_by") or links.get("enables") or links.get("prevents"):
                    result.setdefault("cause", self._atom_text(atom))
                    break

        return result

    @staticmethod
    def _atom_text(atom: dict) -> str:
        """Return the most descriptive text for an atom."""
        pred = atom.get("predicate") or {}
        lemma = str(pred.get("lemma", "") or pred.get("id", "") or "")
        roles_list = atom.get("roles", [])
        theme_vals = [str(r.get("value", "")) for r in roles_list if isinstance(r, dict) and r.get("role") in ("theme", "patient", "object")]
        valid_themes = [v for v in theme_vals if v and v != lemma]
        if valid_themes:
            return " ".join(valid_themes)
        return lemma

    def _render(self, template: str, roles: dict[str, str]) -> str:
        result = template
        for key, value in roles.items():
            result = result.replace("{" + key + "}", value)
        result = re.sub(r"\{[^}]+\}", "", result)
        result = re.sub(r" +", " ", result).strip()
        return result


class AtomDecoderBackend:
    """``DecoderBackend``-protocol wrapper for ``AtomTemplateBackend``.

    Reads ``grammar.uol_act`` and ``grammar.intent`` to generate
    atom-aware template-filled answers. Returns empty string when
    no UOL act is available (signals caller to try next backend).
    """

    name = "atom"

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        self._backend = AtomTemplateBackend(templates=templates)

    def decode(self, plan: Any, grammar: Any) -> str:
        uol_act = getattr(grammar, "uol_act", None)
        intent = getattr(grammar, "intent", "")
        if uol_act is None or not intent:
            return ""
        result = self._backend.generate(intent, uol_act)
        return result or ""


def build_atom_backend(kwargs: dict[str, Any] | None = None) -> AtomDecoderBackend:
    """Factory function that creates an AtomDecoderBackend.

    Used by the ConstrainedDecoder registry for lazy construction.
    Accepts optional ``kwargs`` with a ``templates`` key.
    """
    kwargs = kwargs or {}
    return AtomDecoderBackend(templates=kwargs.get("templates"))
