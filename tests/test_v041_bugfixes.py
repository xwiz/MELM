"""P0 bugfix tests for v0.4.1 — mood, negation, session memory, greeting idioms."""

import json

# ── PH1-A: Mood expression routing ─────────────────────────────────────────

def test_mood_negative_routes_to_assistant_behavior():
    """'I'm feeling sad' should route to assistant_behavior, not open_domain."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("I'm feeling sad")
    assert decision.intent == "assistant_behavior", (
        f"Expected assistant_behavior, got {decision.intent} (reason={decision.reason})"
    )

def test_mood_positive_routes_to_assistant_behavior():
    """'I feel happy' should route to assistant_behavior."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("I feel happy")
    assert decision.intent == "assistant_behavior", (
        f"Expected assistant_behavior, got {decision.intent} (reason={decision.reason})"
    )

def test_mood_copular_negative_routes_correctly():
    """'I am sad' should route to assistant_behavior."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("I am sad")
    assert decision.intent == "assistant_behavior", (
        f"Expected assistant_behavior, got {decision.intent} (reason={decision.reason})"
    )

def test_non_mood_statement_not_affected():
    """'I need help with math' should NOT route to assistant_behavior via mood gate."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("I need help with math")
    assert decision.reason != "emotional_expression", (
        f"Expected non-emotional routing, got reason={decision.reason}"
    )

def test_mood_affect_carried_on_decision():
    """Mood-routed decisions should carry utterance_affect with correct valence."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("I'm feeling sad")
    assert decision.utterance_affect is not None
    assert decision.utterance_affect.valence < 0


# ── PH1-B: Negated action-verb facts ──────────────────────────────────────

def test_knowledge_classify_filters_user_referent():
    """"I did not break the vase" should be filtered as personal (subject=user)."""
    from melm.appliance.assistant_knowledge import classify_knowledge
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "break", "lemma": "break"},
            "roles": [
                {"role": "agent", "value": "user"},
                {"role": "theme", "value": "vase"},
            ],
            "context": {"polarity": "negative", "negation_scope": True},
        }],
    }
    assert classify_knowledge(uol, "I did not break the vase") is None, (
        "Personal negated actions should be filtered from knowledge classification"
    )

def test_extract_proposition_uses_pred_id():
    """Non-copular verbs not in world_relations should use pred_id, not fall back to 'is_a'."""
    from melm.appliance.assistant_knowledge import extract_proposition
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "push", "lemma": "push"},
            "roles": [
                {"role": "agent", "value": "boy"},
                {"role": "theme", "value": "cart"},
            ],
            "context": {"polarity": "positive"},
        }],
    }
    prop = extract_proposition(uol)
    assert prop is not None
    assert prop["relation"] == "push", (
        f"Expected 'push', got '{prop['relation']}'"
    )

def test_extract_proposition_non_copular_third_person():
    """Third-person negated action verb facts should extract correctly."""
    from melm.appliance.assistant_knowledge import extract_proposition
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "push", "lemma": "push"},
            "roles": [
                {"role": "agent", "value": "boy"},
                {"role": "theme", "value": "cart"},
            ],
            "context": {"polarity": "negative", "negation_scope": True},
        }],
    }
    prop = extract_proposition(uol)
    assert prop is not None
    assert prop["subject"] == "boy"
    assert prop["relation"] == "push"
    assert prop["object"] == "cart"

def test_knowledge_third_person_negated_does_not_filter_user():
    """Third-person negated claims should pass through knowledge classifier."""
    from melm.appliance.assistant_knowledge import classify_knowledge
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "break", "lemma": "break"},
            "roles": [
                {"role": "agent", "value": "boy"},
                {"role": "theme", "value": "vase"},
            ],
            "context": {"polarity": "negative", "negation_scope": True},
        }],
    }
    result = classify_knowledge(uol, "The boy did not break the vase")
    assert result == "negated_fact", f"Expected negated_fact, got {result}"


# ── PH1-C: Session memory / "remind" routing ──────────────────────────────

def test_remind_me_routes_to_personal_memory():
    """"Remind me to buy milk" should route to personal_memory."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("Remind me to buy milk")
    assert decision.intent == "personal_memory", (
        f"Expected personal_memory, got {decision.intent} (reason={decision.reason})"
    )

def test_remind_us_routes_to_personal_memory():
    """"Remind us to call mom" should route to personal_memory."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("Remind us to call mom")
    assert decision.intent == "personal_memory", (
        f"Expected personal_memory, got {decision.intent} (reason={decision.reason})"
    )

def test_remind_is_not_routed_as_open_domain():
    """'Remind' utterances should not fall through to open_domain."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("Remind me to buy milk")
    assert decision.reason != "understood_open_domain", (
        f"Should not be open_domain, got reason={decision.reason}"
    )

def test_remind_me_that_routes_to_personal_memory():
    """"Remind me that I have a meeting" should route to personal_memory."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("Remind me that I have a meeting")
    assert decision.intent == "personal_memory", (
        f"Expected personal_memory, got {decision.intent} (reason={decision.reason})"
    )


# ── PH1-D: Greeting idiom routing ─────────────────────────────────────────

def test_wassup_routes_to_social_greeting():
    """"wassup?" should route to social_greeting."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("wassup?")
    assert decision.intent == "social_greeting", (
        f"Expected social_greeting, got {decision.intent} (reason={decision.reason})"
    )

def test_whats_up_routes_to_social_greeting():
    """"what's up?" should route to social_greeting."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("what's up?")
    assert decision.intent == "social_greeting", (
        f"Expected social_greeting, got {decision.intent} (reason={decision.reason})"
    )

def test_sup_routes_to_social_greeting():
    """"sup" should route to social_greeting."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("sup")
    assert decision.intent == "social_greeting", (
        f"Expected social_greeting, got {decision.intent} (reason={decision.reason})"
    )

def test_howdy_routes_to_social_greeting():
    """"howdy" should route to social_greeting (ensures existing greetings still work)."""
    from melm.appliance import OnDeviceAssistantRouter, LocalAssistantProfile
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    decision = router.handle("howdy")
    assert decision.intent == "social_greeting", (
        f"Expected social_greeting, got {decision.intent} (reason={decision.reason})"
    )
