"""Interpret Phase 1 validation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhaseFinding:
    area: str
    status: str
    finding: str
    next_step: str


def interpret_phase1(payload: dict[str, Any]) -> list[PhaseFinding]:
    """Create concise findings from a Phase 1 report payload."""

    findings = [
        _interpret_memory(payload),
    ]
    if payload.get("authored_dialogue"):
        findings.append(_interpret_authored_dialogue(payload))
    if payload.get("sample_transcript"):
        findings.append(_interpret_sample_transcript(payload))
    findings.extend(
        [
            _interpret_tokenizer_lm(payload),
            _interpret_morphology_boundary(payload),
            _interpret_grounding(payload),
        ]
    )
    if payload.get("state_resolution"):
        findings.append(_interpret_state_resolution(payload))
    return findings


def _interpret_memory(payload: dict[str, Any]) -> PhaseFinding:
    memory = payload["memory_report"]
    strict = payload.get("memory_strict_report")
    gain = memory["absolute_gain"]
    strict_text = ""
    if strict:
        strict_text = f" Strict recall@1 gain is {strict['absolute_gain']:.0%}."
    by_category = memory.get("by_category") or {}
    ablations = payload.get("memory_ablation_report") or {}
    ablation_text = ""
    if ablations:
        temporal = ablations.get("entity_action_temporal", {}).get("absolute_gain", 0.0)
        causal = ablations.get("entity_action_causal", {}).get("absolute_gain", 0.0)
        full = ablations.get("event_memory", {}).get("absolute_gain", gain)
        ablation_text = (
            f" Ablation: temporal-only gain {temporal:.0%}, "
            f"causal-only gain {causal:.0%}, combined gain {full:.0%}."
        )
    abstention = payload.get("memory_abstention") or {}
    calibrated_runs = abstention.get("calibrated") or {}
    calibrated = (
        calibrated_runs.get("selected")
        or calibrated_runs.get("score_with_evidence_veto")
        or calibrated_runs.get("top_score")
        or {}
    )
    event_abstention = calibrated.get("evaluation_report") or abstention.get("event_memory_best") or {}
    abstention_text = ""
    calibration_risk = False
    if event_abstention:
        threshold = calibrated.get("threshold", event_abstention.get("threshold", 0.0))
        positive_recall = event_abstention.get("positive_recall", 0.0)
        negative_abstention = event_abstention.get("negative_abstention", 0.0)
        abstention_text = (
            f" Held-out calibrated abstention at threshold {threshold:.2f} has "
            f"{event_abstention['accuracy']:.0%} accuracy, "
            f"{positive_recall:.0%} positive recall, and "
            f"{negative_abstention:.0%} negative abstention."
        )
        calibration_risk = negative_abstention < 0.80 or positive_recall < 0.75
    temporal_gains = [
        category.get("absolute_gain", 0.0)
        for name, category in by_category.items()
        if name.startswith("temporal_")
    ]
    temporal_gain = sum(temporal_gains) / len(temporal_gains) if temporal_gains else 0.0
    passed = gain >= 0.15
    status = "pass" if passed else "needs_work"
    if passed and calibration_risk:
        status = "pass_with_calibration_risk"
    return PhaseFinding(
        area="event_memory",
        status=status,
        finding=(
            f"Structured event memory beats RAG by {gain:.0%} absolute recall@k. "
            f"The gain is concentrated in temporal-neighbor cases (average {temporal_gain:.0%})."
            f"{strict_text}"
            f"{ablation_text}"
            f"{abstention_text}"
        ),
        next_step=(
            "Tune score calibration and add no-answer dialogue cases before treating this as robust."
            if calibration_risk
            else "Add harder entity-conflict cases and then validate on non-synthetic dialogue before treating this as robust."
            if passed
            else "Inspect retrieval scoring and add temporal/entity metadata only where it improves recall."
        ),
    )


def _interpret_tokenizer_lm(payload: dict[str, Any]) -> PhaseFinding:
    reports = payload["tokenizer_lm_reports"]
    best = min(reports, key=lambda item: item["nll_per_token"])
    morph = next(item for item in reports if item["tokenizer"] == "heuristic_morpheme")
    decision = payload.get("tokenizer_decision")
    status = "pass" if best["tokenizer"] == "heuristic_morpheme" else "no_downstream_win_yet"
    decision_text = ""
    next_step = (
        "Run the tokenizer comparison on BabyLM-style data with trained BPE/Unigram tokenizers."
        if status != "pass"
        else "Keep morphology in the next tokenizer ablation and verify on a larger corpus."
    )
    if decision:
        status = (
            "primary_candidate"
            if decision["decision"] == "primary_candidate"
            else "auxiliary_only"
            if decision["decision"] == "auxiliary_only"
            else "reject_for_now"
        )
        decision_text = (
            f" Decision: {decision['decision']} with LM NLL gain "
            f"{decision['lm_nll_gain']:.3f} and boundary-F1 gain "
            f"{decision['boundary_f1_gain']:.0%}."
        )
        next_step = decision["recommendation"]
    stability = payload.get("tokenizer_stability")
    stability_text = ""
    if stability:
        stability_text = (
            f" Cross-fold stability shows morphology win rate "
            f"{stability['morph_win_rate']:.0%}, average LM gain "
            f"{stability['average_lm_nll_gain']:.3f}, and stable-primary="
            f"{stability['stable_primary_candidate']}."
        )
        if not stability["stable_primary_candidate"] and status == "primary_candidate":
            status = "unstable_primary_candidate"
            next_step = (
                "Keep morphology as a candidate, but require a larger corpus and stable folds before promoting it."
            )
    return PhaseFinding(
        area="tokenizer_lm",
        status=status,
        finding=(
            f"The best held-out unigram-LM probe is {best['tokenizer']} "
            f"(NLL/token {best['nll_per_token']:.3f}); heuristic morphology is "
            f"{morph['nll_per_token']:.3f}."
            f"{decision_text}"
            f"{stability_text}"
        ),
        next_step=next_step,
    )


def _interpret_authored_dialogue(payload: dict[str, Any]) -> PhaseFinding:
    dialogue = payload["authored_dialogue"]
    memory = dialogue["memory_report"]
    abstention = dialogue["abstention_report"]
    memory_passed = dialogue["memory_gate"]["passed"]
    abstention_passed = dialogue["abstention_gate"]["passed"]
    passed = memory_passed and abstention_passed
    return PhaseFinding(
        area="authored_dialogue",
        status="probe_pass" if passed else "needs_work",
        finding=(
            f"On the hand-authored dialogue smoke test, event memory reaches "
            f"{memory['event_memory_recall_at_k']:.0%} recall@k versus "
            f"{memory['rag_recall_at_k']:.0%} for RAG. The transferred evidence selector gets "
            f"{abstention['positive_recall']:.0%} positive recall and "
            f"{abstention['negative_abstention']:.0%} negative abstention."
        ),
        next_step=(
            "Replace this smoke fixture with transcript-derived cases and keep the same gates."
            if passed
            else "Inspect authored dialogue misses before trusting synthetic memory gains."
        ),
    )


def _interpret_sample_transcript(payload: dict[str, Any]) -> PhaseFinding:
    sample = payload["sample_transcript"]
    memory = sample["memory_report"]
    abstention = sample["abstention_report"]
    state = sample.get("state_resolution_report")
    memory_passed = sample["memory_gate"]["passed"]
    abstention_passed = sample["abstention_gate"]["passed"]
    state_passed = (
        True
        if not state
        else state["accuracy"] >= 0.95 and state["false_positive_rate"] <= 0.05
    )
    passed = memory_passed and abstention_passed and state_passed
    state_text = ""
    if state:
        state_text = (
            f" State resolution covers {sample['state_cases']} cases at "
            f"{state['accuracy']:.0%} accuracy with "
            f"{state['false_positive_rate']:.0%} false positives."
        )
    return PhaseFinding(
        area="sample_transcript",
        status="smoke_pass" if passed else "needs_work",
        finding=(
            f"The annotated transcript compiler runs end to end: "
            f"{sample['turns']} turns become {sample['events']} events, "
            f"with event memory recall {memory['event_memory_recall_at_k']:.0%} "
            f"versus RAG {memory['rag_recall_at_k']:.0%}. Evidence admission reaches "
            f"{abstention['positive_recall']:.0%} positive recall and "
            f"{abstention['negative_abstention']:.0%} negative abstention."
            f"{state_text}"
        ),
        next_step=(
            "Use this compiler on real transcript snippets and track source-turn provenance for every event."
            if passed
            else "Fix annotation compilation or retrieval misses before adding larger transcript data."
        ),
    )


def _interpret_morphology_boundary(payload: dict[str, Any]) -> PhaseFinding:
    reports = payload["morphology_boundary_reports"]
    morph = next(item for item in reports if item["tokenizer"] == "heuristic_morpheme")
    return PhaseFinding(
        area="morphology_boundary",
        status="probe_pass" if morph["f1"] >= 0.80 else "needs_work",
        finding=(
            f"Heuristic morphology reaches {morph['f1']:.0%} boundary F1 on the tiny gold fixture."
        ),
        next_step=(
            "Replace the tiny hand fixture with MorphoLex/CELEX-derived examples; this result is alignment-only."
        ),
    )


def _interpret_grounding(payload: dict[str, Any]) -> PhaseFinding:
    grounding = payload["grounding"]
    accuracy = grounding["accuracy"]
    return PhaseFinding(
        area="state_grounding",
        status="probe_pass" if accuracy == 1.0 else "needs_work",
        finding=f"State grounding gets {accuracy:.0%} accuracy on {grounding['cases']} seed cases.",
        next_step="Expand precondition, contradiction, figurative-language, and context-blindness cases.",
    )


def _interpret_state_resolution(payload: dict[str, Any]) -> PhaseFinding:
    report = payload["state_resolution"]
    accuracy = report["accuracy"]
    false_positive_rate = report["false_positive_rate"]
    return PhaseFinding(
        area="state_resolution",
        status="probe_pass" if accuracy >= 0.95 and false_positive_rate <= 0.05 else "needs_work",
        finding=(
            f"Explicit event-state tracking gets {accuracy:.0%} accuracy on "
            f"{report['cases']} synthetic object-location cases, with "
            f"{false_positive_rate:.0%} false positives on unknown objects."
        ),
        next_step=(
            "Port the same state annotations into transcript-derived dialogue fixtures."
            if accuracy >= 0.95 and false_positive_rate <= 0.05
            else "Inspect state-update extraction before relying on persistent memory answers."
        ),
    )
