"""Tests for AtomTemplateBackend — atom-aware answer generation."""

from melm.appliance.assistant_decoder_atom import AtomTemplateBackend


def test_atom_template_backend_weather():
    backend = AtomTemplateBackend()
    uol = {
        "act": "question",
        "content": [{
            "kind": "state",
            "predicate": {"id": "weather.check", "lemma": "weather"},
            "roles": [{"role": "theme", "value": "today"}],
            "context": {"polarity": "positive"},
        }],
    }
    result = backend.generate("weather", uol, {})
    assert result is not None
    assert "weather" in result.lower()


def test_atom_template_backend_unknown_intent():
    assert AtomTemplateBackend().generate("no_such_intent", {}, {}) is None


def test_atom_template_backend_empty_atoms():
    assert AtomTemplateBackend().generate("weather", {"act": "x", "content": []}, {}) is None


def test_atom_template_backend_identity():
    backend = AtomTemplateBackend()
    uol = {
        "act": "question",
        "content": [{
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [{"role": "theme", "value": "you"}, {"role": "attribute", "value": "who"}],
            "context": {"polarity": "positive"},
        }],
    }
    result = backend.generate("assistant_identity", uol, {})
    assert result is not None
    assert "assistant" in result.lower()


def test_atom_template_backend_custom_templates():
    backend = AtomTemplateBackend()
    custom = {"test_intent": "custom: {theme}"}
    uol = {
        "act": "question",
        "content": [{
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [{"role": "theme", "value": "hello"}],
            "context": {},
        }],
    }
    assert backend.generate("test_intent", uol, extra_templates=custom) == "custom: hello"


def test_atom_template_backend_renders_verb():
    backend = AtomTemplateBackend(templates={"test": "verb is {verb}"})
    uol = {
        "act": "claim",
        "content": [{
            "predicate": {"id": "run", "lemma": "run"},
            "roles": [],
            "context": {},
        }],
    }
    result = backend.generate("test", uol)
    assert result == "verb is run"


def test_atom_template_backend_multi_atom_roles():
    """_extract_roles collects roles from up to 5 atoms with suffix dedup."""
    backend = AtomTemplateBackend(templates={"multi": "{theme} and {theme_1} via {verb}"})
    uol = {
        "act": "request",
        "content": [
            {
                "kind": "event",
                "predicate": {"id": "eat", "lemma": "eat"},
                "roles": [{"role": "theme", "value": "pasta"}, {"role": "location", "value": "kitchen"}],
            },
            {
                "kind": "event",
                "predicate": {"id": "cook", "lemma": "cook"},
                "roles": [{"role": "theme", "value": "tomatoes"}, {"role": "agent", "value": "chef"}],
            },
        ],
    }
    result = backend.generate("multi", uol)
    assert result is not None
    assert "pasta" in result
    assert "tomatoes" in result
    assert "eat" in result or "cook" in result
    assert result == "pasta and tomatoes via eat"


def test_atom_template_backend_multi_atom_bounded():
    """Only the first 5 atoms are processed for role extraction."""
    backend = AtomTemplateBackend()
    atoms = []
    for i in range(7):
        atoms.append({
            "kind": "event",
            "predicate": {"id": f"verb_{i}", "lemma": f"verb_{i}"},
            "roles": [{"role": "theme", "value": f"value_{i}"}],
        })
    uol = {"act": "request", "content": atoms}
    # With multi-atom, theme appears 5 times: theme, theme_1, theme_2, theme_3, theme_4
    # The 6th and 7th are skipped
    roles = backend._extract_roles(uol)
    assert "theme" in roles
    assert "theme_1" in roles
    assert "theme_2" in roles
    assert "theme_3" in roles
    assert "theme_4" in roles
    assert "theme_5" not in roles
    assert roles.get("verb") == "verb_0"


def test_atom_template_backend_loads_from_contract():
    """AtomTemplateBackend loads templates from atom_templates.v1.json contract."""
    backend = AtomTemplateBackend()
    assert "weather" in backend._templates or any(
        k in backend._templates for k in ("gibberish", "complaint_response")
    )
    result = backend.generate("gibberish", {"act": "x", "content": [{"predicate": {"id": "x"}, "roles": []}]})
    if result is not None:
        assert "understand" in result.lower() or "rephrase" in result.lower()


def test_atom_template_backend_contract_fallback_chain():
    """Template resolution: extra_templates > self._templates > contract > _DEFAULT_TEMPLATES."""
    backend = AtomTemplateBackend()
    # The contract should have 'gibberish' and 'weather' is in DEFAULT_TEMPLATES
    # Make sure the fallback chain works for a valid intent
    uol = {
        "act": "question",
        "content": [{
            "kind": "state",
            "predicate": {"id": "weather.check", "lemma": "weather"},
            "roles": [{"role": "theme", "value": "today"}],
        }],
    }
    # Should resolve from either self._templates (contract) or _DEFAULT_TEMPLATES
    result = backend.generate("weather", uol, {})
    assert result is not None
    assert "weather" in result.lower()


def test_atom_template_backend_gibberish_resolution():
    """Gibberish template resolves from contract or defaults."""
    backend = AtomTemplateBackend()
    uol = {
        "act": "statement",
        "content": [{
            "kind": "state",
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [],
        }],
    }
    result = backend.generate("gibberish", uol, {})
    assert result is not None
    assert any(word in result.lower() for word in ("understand", "rephrase", "did not"))


def test_atom_template_backend_complaint_resolution():
    """Complaint_response template resolves from contract or defaults."""
    backend = AtomTemplateBackend()
    uol = {
        "act": "statement",
        "content": [{
            "kind": "state",
            "predicate": {"id": "be", "lemma": "be"},
            "roles": [],
        }],
    }
    result = backend.generate("complaint_response", uol, {})
    assert result is not None
    assert any(word in result.lower() for word in ("hear", "try again", "specifically"))
