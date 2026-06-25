import json
from pathlib import Path
from scripts.extract_causal_effects import extract, _convert_verb


def test_convert_verb_basic():
    data = {
        "patient_states": {"physical": ["clean", "restored"]},
        "patient_types": ["object"],
    }
    rule = _convert_verb("clean", data)
    assert rule is not None
    assert "physical" in rule["effects"]
    assert "clean" in rule["effects"]["physical"]


def test_convert_verb_no_states_returns_none():
    data = {"patient_states": {}, "patient_types": []}
    assert _convert_verb("glorp", data) is None


def test_convert_verb_unchanged_filtered():
    data = {"patient_states": {"physical": ["unchanged"]}, "patient_types": ["object"]}
    assert _convert_verb("stagnate", data) is None


def test_extract_produces_valid_contract(tmp_path):
    nameless = tmp_path / "nameless.json"
    nameless.write_text(json.dumps({
        "abandon": {
            "patient_states": {
                "emotional": ["abandoned", "ignored"],
                "mental": ["forgotten"],
            },
            "patient_types": ["person"],
        },
        "clean": {
            "patient_states": {"physical": ["clean", "restored"]},
            "patient_types": ["object"],
        },
    }))
    output = tmp_path / "causal_effects.json"
    count = extract(nameless_path=nameless, output_path=output)
    assert count >= 2
    with open(output) as f:
        data = json.load(f)
    assert data["schema_id"] == "melm.causal_effects.v1"
    assert "abandon" in data["rules"]
    assert "clean" in data["rules"]
