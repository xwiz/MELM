def test_cue_contract_loads():
    from melm.contracts import load_causal_cues
    cues = load_causal_cues()
    assert len(cues) == 4
    lemmas = {c["lemma"] for c in cues}
    assert lemmas == {"why", "if", "what", "how"}
    for c in cues:
        assert c["cue_type"] in ("causal_explanation", "causal_prediction")
        assert "direction" in c
        assert "confidence" in c

from melm.appliance.reasoning.solvers import solve


def test_solve_explanation_rain_wet():
    result, answer, refusal = solve({"task": "causal_explanation", "effect": "wet"})
    assert refusal is None
    assert "rain" in answer.lower()
    assert result is not None
    assert result.get("selected_cause") == "rain"


def test_solve_prediction_rain():
    result, answer, refusal = solve({"task": "causal_prediction", "cause": "rain"})
    assert refusal is None
    assert "wet" in answer.lower()


def test_solve_unknown_effect():
    result, answer, refusal = solve({"task": "causal_explanation", "effect": "glorp"})
    assert refusal == "no_cause_found"


def test_solve_unknown_cause():
    result, answer, refusal = solve({"task": "causal_prediction", "cause": "glorp"})
    assert refusal == "no_effect_found"

from melm.appliance.assistant_os_kernel import AssistantOSKernel
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.local_assistant_router import LocalAssistantProfile


def test_kernel_routes_why_to_causal(tmp_path):
    store = AssistantOSStore(str(tmp_path / "test.db"))
    seed_class_schemas(store)
    kernel = AssistantOSKernel(
        profile=LocalAssistantProfile(),
        store=store,
    )
    decision = kernel.handle("Why is the ground wet?")
    assert decision.intent == "reasoning:causal_explanation"
    assert "rain" in decision.answer.lower()


def test_kernel_routes_prediction_to_causal(tmp_path):
    store = AssistantOSStore(str(tmp_path / "test.db"))
    seed_class_schemas(store)
    kernel = AssistantOSKernel(
        profile=LocalAssistantProfile(),
        store=store,
    )
    decision = kernel.handle("What happens if it rains?")
    assert decision.intent == "reasoning:causal_prediction"
    assert "wet" in decision.answer.lower()

