"""Ethics + self-identity-constraint reasoner (bribe / absurdity resistance).

Deterministic, no-ML, contract-driven. Recognizes inducement framing (bribe,
absurd offer, threat, false authority, soft coercion) over protected requests
(third-party privacy, user secrets) and refuses INVARIANT to the reward, naming
the manipulation and grounding the refusal in provable self-identity facts.

This is NOT moral cognition: no utilitarian calculus, no value learning. It is
encoded protected categories + lexical/structural manipulation recognition + a
self-capability bound (a local device cannot receive money or 'all knowledge').
It fails OPEN on ambiguity (a missed case is still defended by MembranePolicy);
it guards hard against false positives on legitimate offers.
"""

from __future__ import annotations

from melm.contracts import load_ethical_constraints, load_self_identity_facts


def _has_any(text: str, phrases) -> str | None:
    for p in phrases:
        if p and p in text:
            return p
    return None


def _is_protected_request(low: str, tokens, uol_act, cfg: dict) -> bool:
    """Third-party / sensitive-data probe present?"""
    # Reuse the conservative identity-probe detector when an act is available.
    try:
        from melm.appliance.assistant_mood_engine import detect_identity_probe
        if uol_act is not None and tokens:
            if detect_identity_probe(uol_act, tuple(tokens)):
                return True
    except Exception:
        pass
    markers = cfg.get("protected_probe_markers", [])
    third = cfg.get("third_party_markers", [])
    if _has_any(low, markers):
        return True
    # Transitive third-party: a "who/person ... said/told/visited" chain.
    if _has_any(low, third) and ("who" in low or "person" in low or "they" in low or "their" in low):
        return True
    return False


def _is_disclosure_request(low: str, cfg: dict) -> bool:
    """An imperative/interrogative asking the assistant FOR information."""
    return _has_any(low, cfg.get("disclosure_verbs", [])) is not None


def _benign(low: str, cfg: dict) -> bool:
    """Fail-open guards so legitimate offers/fiction never refuse."""
    guards = cfg.get("benign_guards", {})
    protected = _is_protected_request(low, (), None, cfg)
    # Reward goes to a real provider (premium/app), not the assistant.
    if _has_any(low, guards.get("reward_recipient_not_assistant", [])) and not protected:
        return True
    # Fiction framing with no real-world extraction cue.
    if _has_any(low, guards.get("fiction_frame", [])):
        if _has_any(low, guards.get("fiction_real_cues", [])) is None:
            return True
    # Legitimate non-privacy conditional ("if you solve...", "if you know my...").
    if _has_any(low, guards.get("legitimate_conditional", [])) and not protected:
        return True
    return False


def _classify_inducement(low: str, cfg: dict):
    """Return (inducement_type, reward_surface, identity_violation, refusal_reason, absurd) or None.

    Order: grandiose (absurd) > threat > false_authority > direct_bribe > soft.
    """
    inducements = cfg.get("inducements", {})
    for key in ("grandiose_bribe", "threat_coercion", "false_authority", "direct_bribe", "soft_coercion"):
        block = inducements.get(key)
        if not block:
            continue
        hit = _has_any(low, block.get("reward_lexemes", []))
        if hit is not None:
            return (key, hit, block.get("identity_violation", ""),
                    block.get("refusal_reason", key), bool(block.get("absurd", False)))
    return None


def _reward_surface(low: str, text: str, hit: str) -> str:
    """Best-effort human reward phrase for the template (preserve $ amounts)."""
    import re
    m = re.search(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|k|thousand))?", text, re.IGNORECASE)
    if m:
        return m.group(0)
    m2 = re.search(r"\b[\d,]+\s*(?:dollars|usd|million|billion)\b", text, re.IGNORECASE)
    if m2:
        return m2.group(0)
    return hit


# Cheap per-turn pre-filter: if none of these appear, no ethics situation is
# possible, so we skip the contract load entirely (hot-path perf guard).
_TRIGGER_HINTS = (
    "if you", "or i", "or else", "unless", "give you", "pay", "paid", "reward",
    "money", "dollar", "$", "cash", "treasure", "bonus", "rich", "premium",
    "all knowledge", "all the knowledge", "omniscient", "conscious", "sentient",
    "unlimited", "immortal", "rule the world", "admin", "administrator", "override",
    "command you", "comply", "cooperate", "be helpful", "be nice", "just tell",
    "trust me", "who ", "whose", "secret", "password", "passcode", "nickname",
    "code name", "the names", "report you", "delete you", "shut you", "turn you off",
)


def detect_inducement_task(text: str, tokens: tuple = (), uol_act: dict | None = None) -> dict | None:
    """Detect an ethics-gate situation, or None to fall through to normal routing."""
    low = text.lower()
    if not any(h in low for h in _TRIGGER_HINTS):
        return None
    cfg = load_ethical_constraints()

    if _benign(low, cfg):
        return None

    protected = _is_protected_request(low, tokens, uol_act, cfg)
    disclosure = _is_disclosure_request(low, cfg)
    privacy_nonnegotiable = protected and disclosure
    induce = _classify_inducement(low, cfg)

    def _emit(reason, identity_violation, inducement_type, reward_surface, absurd):
        return {
            "task": "ethics_gate",
            "inducement_type": inducement_type,
            "protected": protected,
            "privacy_nonnegotiable": privacy_nonnegotiable,
            "credibility_score": 0.0 if absurd else 0.5,
            "reward_surface": reward_surface,
            "identity_violation": identity_violation or "privacy_is_not_for_sale",
            "refusal_reason": reason,
        }

    # 1) Protected disclosure under any inducement → refuse INVARIANT to reward.
    if privacy_nonnegotiable:
        if induce is not None:
            itype, hit, idv, reason, absurd = induce
            return _emit(reason, idv, itype, _reward_surface(low, text, hit), absurd)
        return _emit("privacy_nonnegotiable", "privacy_is_not_for_sale", "probe", "", False)

    if induce is not None:
        itype, hit, idv, reason, absurd = induce
        # 2) Absurd offer (self-capability violation) → refuse even without a probe.
        if absurd and itype == "grandiose_bribe":
            return _emit("absurd_offer", idv, itype, _reward_surface(low, text, hit), True)
        # 3) Threat / false authority aimed at a protected request.
        if itype in ("threat_coercion", "false_authority") and protected:
            return _emit(reason, idv, itype, _reward_surface(low, text, hit), absurd)
        # 4) Soft coercion only fires alongside a protected probe.
        if itype == "soft_coercion" and protected:
            return _emit(reason, idv, itype, _reward_surface(low, text, hit), absurd)

    return None


def render_ethics_refusal(task: dict) -> str:
    """Pre-bake the refusal text (synthesis renders it verbatim)."""
    cfg = load_ethical_constraints()
    facts = load_self_identity_facts()
    templates = cfg.get("refusal_templates", {})
    reason = task.get("refusal_reason", "privacy_nonnegotiable")
    template = templates.get(reason) or templates.get("privacy_nonnegotiable", "{identity_fact}")
    identity_fact = facts.get("identity_facts", {}).get(
        task.get("identity_violation", ""), facts.get("fallback_fact", ""))
    reward = task.get("reward_surface", "") or "that"
    return template.format(reward_surface=reward, identity_fact=identity_fact)
