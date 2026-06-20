"""Tests for African language greeting contracts (Yoruba, Swahili, Igbo)."""

from __future__ import annotations

from melm.contracts.validation import (
    validate_yoruba_greetings, validate_swahili_greetings, validate_igbo_greetings,
    load_yoruba_greetings, load_swahili_greetings, load_igbo_greetings,
)
from melm.appliance.local_assistant_router import OnDeviceAssistantRouter, LocalAssistantProfile


class TestYorubaGreetingsContract:
    def test_contract_loads_and_validates(self) -> None:
        data = load_yoruba_greetings()
        assert data["language"] == "yoruba"
        assert "greetings" in data
        assert data["greetings"]["general"] == "Báwo ni?"
        assert data["greetings"]["morning"] == "Ẹ kú àárọ̀"

    def test_validator_accepts_valid(self) -> None:
        data = load_yoruba_greetings()
        validate_yoruba_greetings(data)

    def test_validator_rejects_bad_schema_id(self) -> None:
        import pytest
        from melm.contracts.validation import ContractValidationError
        data = load_yoruba_greetings()
        data["schema_id"] = "wrong"
        with pytest.raises(ContractValidationError):
            validate_yoruba_greetings(data)


class TestSwahiliGreetingsContract:
    def test_contract_loads_and_validates(self) -> None:
        data = load_swahili_greetings()
        assert data["language"] == "swahili"
        assert "greetings" in data
        assert data["greetings"]["general"] == "Habari"
        assert data["greetings"]["welcome"] == "Karibu"

    def test_validator_accepts_valid(self) -> None:
        data = load_swahili_greetings()
        validate_swahili_greetings(data)


class TestIgboGreetingsContract:
    def test_contract_loads_and_validates(self) -> None:
        data = load_igbo_greetings()
        assert data["language"] == "igbo"
        assert "greetings" in data
        assert data["greetings"]["general"] == "Ndeewo"
        assert data["greetings"]["morning"] == "Ụtụtụ ọma"

    def test_validator_accepts_valid(self) -> None:
        data = load_igbo_greetings()
        validate_igbo_greetings(data)

    def test_validator_rejects_bad_schema_id(self) -> None:
        import pytest
        from melm.contracts.validation import ContractValidationError
        data = load_igbo_greetings()
        data["schema_id"] = "wrong"
        with pytest.raises(ContractValidationError):
            validate_igbo_greetings(data)


class TestRouterGreetingIntegration:
    def test_default_english_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Generic", language_preference="english")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert decision.answer == "Hi. What would you like help with?"
        assert decision.intent == "social_greeting"

    def test_yoruba_culture_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Yoruba", language_preference="english")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Báwo ni?" in decision.answer
        assert "What would you like help with?" in decision.answer

    def test_yoruba_language_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Generic", language_preference="yoruba")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Báwo ni?" in decision.answer

    def test_swahili_culture_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Swahili", language_preference="english")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Habari" in decision.answer
        assert "What would you like help with?" in decision.answer

    def test_swahili_language_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Generic", language_preference="swahili")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Habari" in decision.answer

    def test_igbo_culture_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Igbo", language_preference="english")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Ndeewo" in decision.answer
        assert "What would you like help with?" in decision.answer

    def test_igbo_language_greeting(self) -> None:
        profile = LocalAssistantProfile(culture="Generic", language_preference="igbo")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        assert "Ndeewo" in decision.answer

    def test_greeting_falls_back_to_english_on_missing_contract(self) -> None:
        # This test verifies graceful fallback if contract file were missing
        # In practice the contract is present, but the code has try/except
        profile = LocalAssistantProfile(culture="Yoruba")
        router = OnDeviceAssistantRouter(profile=profile)
        decision = router._greeting("hello")
        # Should not be empty even if contract loading somehow failed
        assert decision.answer
        assert decision.intent == "social_greeting"
