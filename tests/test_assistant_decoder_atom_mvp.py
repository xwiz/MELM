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
