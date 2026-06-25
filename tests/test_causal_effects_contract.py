from melm.contracts import load_causal_effects


def test_causal_effects_loads():
    data = load_causal_effects()
    assert "rain" in data["rules"]
    assert "eat" in data["rules"]


def test_causal_effect_has_effects_and_confidence():
    data = load_causal_effects()
    rain = data["rules"]["rain"]
    assert "effects" in rain
    assert "confidence" in rain
    assert "physical" in rain["effects"]
