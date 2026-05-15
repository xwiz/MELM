"""Generate a Phase 1 validation report."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.benchmarks import (
    authored_child_dialogue_fixture,
    generate_synthetic_evidence_benchmark,
    generate_synthetic_episodic_benchmark,
    load_annotated_transcript_benchmark,
    morphology_fixture,
    state_grounding_fixture,
    synthetic_state_resolution_fixture,
)
from melm.data import load_train_validation
from melm.evaluation import abstention_gate, interpret_phase1, memory_gate
from melm.grounding import check_transition
from melm.memory import (
    EventMemory,
    best_abstention_report,
    calibrate_abstention_threshold,
    evaluate_abstention,
    evaluate_memory,
    evaluate_memory_variants,
    evaluate_state_resolution,
    split_evidence_cases,
    sweep_abstention_thresholds,
)
from melm.tokenization import (
    build_default_tokenizers,
    compare_tokenizers,
    compare_unigram_lms,
    decide_tokenizer_strategy,
    evaluate_boundary_f1,
    evaluate_tokenizer_lm_stability,
    split_for_lm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="MELM_whitepaper.md", help="Text file or directory for tokenizer metrics.")
    parser.add_argument("--manifest", help="Corpus manifest with train/validation/test splits.")
    parser.add_argument("--stories", type=int, default=25, help="Synthetic episodic story count.")
    parser.add_argument("--k", type=int, default=2, help="Recall@k for memory retrieval.")
    parser.add_argument("--out-dir", default="reports", help="Directory for report artifacts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts, train_texts, validation_texts, text_source = load_train_validation(
        path=args.text,
        manifest_path=args.manifest,
    )
    tokenizers = build_default_tokenizers(texts, train_texts=train_texts)
    tokenizer_reports = compare_tokenizers(tokenizers, texts)
    tokenizer_lm_reports = compare_unigram_lms(tokenizers, train_texts, validation_texts)
    morphology_reports = [
        evaluate_boundary_f1(tokenizer, morphology_fixture())
        for tokenizer in tokenizers
    ]
    tokenizer_decision = decide_tokenizer_strategy(
        tokenizer_lm_reports,
        morphology_reports,
    )
    tokenizer_stability = evaluate_tokenizer_lm_stability(texts)

    events, cases = generate_synthetic_episodic_benchmark(stories=args.stories, distractors_per_story=1)
    memory = EventMemory(events)
    memory_report = evaluate_memory(memory, cases, k=args.k)
    strict_memory_report = evaluate_memory(memory, cases, k=1)
    memory_ablation_report = evaluate_memory_variants(memory, cases, k=args.k)
    memory_gate_report = memory_gate(
        memory_report.event_memory_recall_at_k,
        memory_report.rag_recall_at_k,
    )
    evidence_events, evidence_cases = generate_synthetic_evidence_benchmark(
        stories=args.stories,
        distractors_per_story=1,
    )
    evidence_memory = EventMemory(evidence_events)
    rag_abstention_reports = sweep_abstention_thresholds(
        evidence_memory,
        evidence_cases,
        k=args.k,
        retriever="rag",
    )
    event_abstention_reports = sweep_abstention_thresholds(
        evidence_memory,
        evidence_cases,
        k=args.k,
        retriever="event_memory",
    )
    hybrid_abstention_reports = sweep_abstention_thresholds(
        evidence_memory,
        evidence_cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )
    rag_abstention_best = best_abstention_report(rag_abstention_reports)
    event_abstention_best = best_abstention_report(event_abstention_reports)
    hybrid_abstention_best = best_abstention_report(hybrid_abstention_reports)
    calibration_cases, evaluation_cases = split_evidence_cases(evidence_cases)
    calibrated_top_score = calibrate_abstention_threshold(
        evidence_memory,
        calibration_cases,
        evaluation_cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="top_score",
    )
    calibrated_hybrid = calibrate_abstention_threshold(
        evidence_memory,
        calibration_cases,
        evaluation_cases,
        k=args.k,
        retriever="event_memory",
        confidence_method="score_with_evidence_veto",
    )
    selected_calibrated_abstention = _select_calibrated_abstention(
        [calibrated_top_score, calibrated_hybrid]
    )
    abstention_gate_report = abstention_gate(
        selected_calibrated_abstention.evaluation_report.negative_abstention,
        positive_recall=selected_calibrated_abstention.evaluation_report.positive_recall,
    )
    dialogue_events, dialogue_recall_cases, dialogue_evidence_cases = authored_child_dialogue_fixture()
    dialogue_memory = EventMemory(dialogue_events)
    dialogue_memory_report = evaluate_memory(dialogue_memory, dialogue_recall_cases, k=args.k)
    dialogue_abstention_report = evaluate_abstention(
        dialogue_memory,
        dialogue_evidence_cases,
        k=args.k,
        threshold=selected_calibrated_abstention.threshold,
        retriever="event_memory",
        confidence_method=selected_calibrated_abstention.confidence_method,
    )
    dialogue_memory_gate = memory_gate(
        dialogue_memory_report.event_memory_recall_at_k,
        dialogue_memory_report.rag_recall_at_k,
    )
    dialogue_abstention_gate = abstention_gate(
        dialogue_abstention_report.negative_abstention,
        positive_recall=dialogue_abstention_report.positive_recall,
    )
    sample_transcript = _maybe_run_sample_transcript(
        selected_calibrated_abstention.threshold,
        selected_calibrated_abstention.confidence_method,
        args.k,
    )

    grounding_cases = state_grounding_fixture()
    grounding_correct = 0
    grounding_results = []
    for case in grounding_cases:
        result = check_transition(case.current, case.required, case.effects)
        correct = result.valid == case.should_be_valid
        grounding_correct += int(correct)
        grounding_results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "expected_valid": case.should_be_valid,
                "actual_valid": result.valid,
                "correct": correct,
                "missing": {key: sorted(value) for key, value in result.missing.items()},
                "conflicts": result.conflicts,
            }
        )

    state_resolution_events, state_resolution_cases = synthetic_state_resolution_fixture(
        stories=args.stories,
        distractors_per_story=1,
    )
    state_resolution_report = evaluate_state_resolution(
        state_resolution_events,
        state_resolution_cases,
    )

    payload = {
        "text_source": text_source,
        "tokenizer_reports": [_to_jsonable(report) for report in tokenizer_reports],
        "tokenizer_lm_reports": [_to_jsonable(report) for report in tokenizer_lm_reports],
        "morphology_boundary_reports": [_to_jsonable(report) for report in morphology_reports],
        "tokenizer_decision": _to_jsonable(tokenizer_decision),
        "tokenizer_stability": _to_jsonable(tokenizer_stability),
        "memory_report": _to_jsonable(memory_report),
        "memory_strict_report": _to_jsonable(strict_memory_report),
        "memory_ablation_report": _to_jsonable(memory_ablation_report),
        "memory_abstention": {
            "rag": _to_jsonable(rag_abstention_reports),
            "event_memory": _to_jsonable(event_abstention_reports),
            "event_memory_hybrid": _to_jsonable(hybrid_abstention_reports),
            "rag_best": _to_jsonable(rag_abstention_best),
            "event_memory_best": _to_jsonable(event_abstention_best),
            "event_memory_hybrid_best": _to_jsonable(hybrid_abstention_best),
            "calibrated": {
                "top_score": _to_jsonable(calibrated_top_score),
                "score_with_evidence_veto": _to_jsonable(calibrated_hybrid),
                "selected": _to_jsonable(selected_calibrated_abstention),
            },
            "gate": _to_jsonable(abstention_gate_report),
        },
        "authored_dialogue": {
            "events": len(dialogue_events),
            "recall_cases": len(dialogue_recall_cases),
            "evidence_cases": len(dialogue_evidence_cases),
            "memory_report": _to_jsonable(dialogue_memory_report),
            "abstention_report": _to_jsonable(dialogue_abstention_report),
            "memory_gate": _to_jsonable(dialogue_memory_gate),
            "abstention_gate": _to_jsonable(dialogue_abstention_gate),
        },
        "sample_transcript": _to_jsonable(sample_transcript) if sample_transcript else None,
        "memory_gate": _to_jsonable(memory_gate_report),
        "grounding": {
            "cases": len(grounding_cases),
            "accuracy": grounding_correct / len(grounding_cases) if grounding_cases else 0.0,
            "results": grounding_results,
        },
        "state_resolution": _to_jsonable(state_resolution_report),
    }
    findings = interpret_phase1(payload)
    payload["interpretation"] = [_to_jsonable(finding) for finding in findings]

    json_path = out_dir / "phase1_report.json"
    md_path = out_dir / "phase1_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MELM Phase 1 Validation Report",
        "",
        f"Text source: `{payload['text_source']}`",
        "",
        "## Interpretation",
        "",
        "| Area | Status | Finding | Next Step |",
        "|---|---|---|---|",
    ]
    for finding in payload["interpretation"]:
        lines.append(
            f"| {finding['area']} | {finding['status']} | "
            f"{finding['finding']} | {finding['next_step']} |"
        )

    lines.extend(
        [
            "",
            "## Tokenizer Metrics",
            "",
            "| Tokenizer | Tokens/Word | Unique | Fallback |",
            "|---|---:|---:|---:|",
        ]
    )
    for report in payload["tokenizer_reports"]:
        lines.append(
            f"| {report['tokenizer']} | {report['tokens_per_word']:.3f} | "
            f"{report['unique_tokens']} | {report['fallback_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Tokenizer LM Probe",
            "",
            "| Tokenizer | NLL/Token | Perplexity | Vocab | Validation Tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for report in payload["tokenizer_lm_reports"]:
        lines.append(
            f"| {report['tokenizer']} | {report['nll_per_token']:.3f} | "
            f"{report['perplexity']:.2f} | {report['vocabulary']} | {report['validation_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Morphology Boundary Probe",
            "",
            "| Tokenizer | Precision | Recall | F1 | Exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for report in payload["morphology_boundary_reports"]:
        lines.append(
            f"| {report['tokenizer']} | {report['precision']:.2%} | "
            f"{report['recall']:.2%} | {report['f1']:.2%} | {report['exact_match']:.2%} |"
        )

    tokenizer_decision = payload["tokenizer_decision"]
    lines.extend(
        [
            "",
            "## Tokenizer Decision",
            "",
            f"- Decision: {tokenizer_decision['decision']}",
            f"- Gate passed: {tokenizer_decision['gate_passed']}",
            f"- Best LM baseline: {tokenizer_decision['best_lm_tokenizer']} "
            f"(NLL/token {tokenizer_decision['best_baseline_nll_per_token']:.3f})",
            f"- Morphology LM: {tokenizer_decision['morph_tokenizer']} "
            f"(NLL/token {tokenizer_decision['morph_nll_per_token']:.3f})",
            f"- LM NLL gain: {tokenizer_decision['lm_nll_gain']:.3f}",
            f"- Best boundary baseline: {tokenizer_decision['best_boundary_tokenizer']} "
            f"(F1 {tokenizer_decision['best_baseline_boundary_f1']:.2%})",
            f"- Morphology boundary F1: {tokenizer_decision['morph_boundary_f1']:.2%}",
            f"- Boundary F1 gain: {tokenizer_decision['boundary_f1_gain']:.2%}",
            f"- Recommendation: {tokenizer_decision['recommendation']}",
        ]
    )

    tokenizer_stability = payload.get("tokenizer_stability")
    if tokenizer_stability:
        lines.extend(
            [
                "",
                "## Tokenizer Stability",
                "",
                f"- Documents: {tokenizer_stability['documents']}",
                f"- Folds: {tokenizer_stability['folds']}",
                f"- Morphology win rate: {tokenizer_stability['morph_win_rate']:.2%}",
                f"- Stable primary candidate: {tokenizer_stability['stable_primary_candidate']}",
                f"- Best average baseline: {tokenizer_stability['best_baseline_tokenizer']} "
                f"(NLL/token {tokenizer_stability['best_baseline_average_nll_per_token']:.3f})",
                f"- Morphology average NLL/token: {tokenizer_stability['morph_average_nll_per_token']:.3f}",
                f"- Average LM NLL gain: {tokenizer_stability['average_lm_nll_gain']:.3f}",
                "",
                "| Tokenizer | Fold Wins | Average NLL/Token |",
                "|---|---:|---:|",
            ]
        )
        for tokenizer, nll in tokenizer_stability["average_nll_per_token"].items():
            lines.append(
                f"| {tokenizer} | {tokenizer_stability['winner_counts'].get(tokenizer, 0)} | {nll:.3f} |"
            )

    memory = payload["memory_report"]
    strict_memory = payload["memory_strict_report"]
    gate = payload["memory_gate"]
    lines.extend(
        [
            "",
            "## Event Memory",
            "",
            f"- Cases: {memory['cases']}",
            f"- RAG recall: {memory['rag_recall_at_k']:.2%}",
            f"- Event memory recall: {memory['event_memory_recall_at_k']:.2%}",
            f"- Absolute gain: {memory['absolute_gain']:.2%}",
            f"- RAG MRR: {memory['rag_mrr_at_k']:.2%}",
            f"- Event memory MRR: {memory['event_memory_mrr_at_k']:.2%}",
            f"- MRR gain: {memory['mrr_gain']:.2%}",
            f"- Strict RAG recall@1: {strict_memory['rag_recall_at_k']:.2%}",
            f"- Strict event memory recall@1: {strict_memory['event_memory_recall_at_k']:.2%}",
            f"- Strict absolute gain@1: {strict_memory['absolute_gain']:.2%}",
            f"- Gate passed: {gate['passed']}",
            "",
            "### By Category",
            "",
            "| Category | Cases | RAG | Event Memory | Gain | MRR Gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, report in memory["by_category"].items():
        lines.append(
            f"| {category} | {report['cases']} | {report['rag_recall_at_k']:.2%} | "
            f"{report['event_memory_recall_at_k']:.2%} | {report['absolute_gain']:.2%} | "
            f"{report['mrr_gain']:.2%} |"
        )

    lines.extend(
        [
            "",
            "### Component Ablation",
            "",
            "| Variant | Recall | Gain vs RAG | MRR | MRR Gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, report in payload["memory_ablation_report"].items():
        lines.append(
            f"| {name} | {report['event_memory_recall_at_k']:.2%} | "
            f"{report['absolute_gain']:.2%} | {report['event_memory_mrr_at_k']:.2%} | "
            f"{report['mrr_gain']:.2%} |"
        )

    abstention = payload["memory_abstention"]
    lines.extend(
        [
            "",
            "### Evidence Abstention",
            "",
            "| Retriever | Threshold | Accuracy | Precision | Positive Recall | Negative Abstention | False Positive Rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("rag", "event_memory", "event_memory_hybrid"):
        for report in abstention[name]:
            lines.append(
                f"| {name} | {report['threshold']:.2f} | {report['accuracy']:.2%} | "
                f"{report['precision']:.2%} | {report['positive_recall']:.2%} | "
                f"{report['negative_abstention']:.2%} | {report['false_positive_rate']:.2%} |"
            )
    lines.extend(
        [
            "",
            "Best abstention thresholds:",
            "",
            f"- RAG: threshold {abstention['rag_best']['threshold']:.2f}, "
            f"accuracy {abstention['rag_best']['accuracy']:.2%}, "
            f"negative abstention {abstention['rag_best']['negative_abstention']:.2%}.",
            f"- Event memory: threshold {abstention['event_memory_best']['threshold']:.2f}, "
            f"accuracy {abstention['event_memory_best']['accuracy']:.2%}, "
            f"negative abstention {abstention['event_memory_best']['negative_abstention']:.2%}.",
            f"- Event memory hybrid: threshold {abstention['event_memory_hybrid_best']['threshold']:.2f}, "
            f"accuracy {abstention['event_memory_hybrid_best']['accuracy']:.2%}, "
            f"negative abstention {abstention['event_memory_hybrid_best']['negative_abstention']:.2%}.",
            "",
            "Held-out calibrated thresholds:",
            "",
            _calibrated_abstention_line("Top score", abstention["calibrated"]["top_score"]),
            _calibrated_abstention_line(
                "Score + evidence veto",
                abstention["calibrated"]["score_with_evidence_veto"],
            ),
            _calibrated_abstention_line(
                "Selected",
                abstention["calibrated"]["selected"],
            ),
            f"- Abstention gate passed: {abstention['gate']['passed']} "
            f"(combined metric {abstention['gate']['metric']:.2f}).",
        ]
    )

    grounding = payload["grounding"]
    lines.extend(
        [
            "",
            "## Authored Dialogue Smoke",
            "",
        ]
    )
    dialogue = payload.get("authored_dialogue")
    if dialogue:
        dialogue_memory = dialogue["memory_report"]
        dialogue_abstention = dialogue["abstention_report"]
        lines.extend(
            [
                f"- Events: {dialogue['events']}",
                f"- Recall cases: {dialogue['recall_cases']}",
                f"- Evidence cases: {dialogue['evidence_cases']}",
                f"- RAG recall: {dialogue_memory['rag_recall_at_k']:.2%}",
                f"- Event memory recall: {dialogue_memory['event_memory_recall_at_k']:.2%}",
                f"- Absolute gain: {dialogue_memory['absolute_gain']:.2%}",
                f"- Event memory MRR: {dialogue_memory['event_memory_mrr_at_k']:.2%}",
                f"- Memory gate passed: {dialogue['memory_gate']['passed']}",
                f"- Evidence threshold: {dialogue_abstention['threshold']:.2f}",
                f"- Evidence method: {dialogue_abstention['confidence_method']}",
                f"- Evidence accuracy: {dialogue_abstention['accuracy']:.2%}",
                f"- Evidence precision: {dialogue_abstention['precision']:.2%}",
                f"- Evidence positive recall: {dialogue_abstention['positive_recall']:.2%}",
                f"- Evidence negative abstention: {dialogue_abstention['negative_abstention']:.2%}",
                f"- Abstention gate passed: {dialogue['abstention_gate']['passed']}",
                "",
                "### Authored Dialogue By Category",
                "",
                "| Category | Cases | RAG | Event Memory | Gain |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for category, report in dialogue_memory["by_category"].items():
            lines.append(
                f"| {category} | {report['cases']} | {report['rag_recall_at_k']:.2%} | "
                f"{report['event_memory_recall_at_k']:.2%} | {report['absolute_gain']:.2%} |"
            )

    sample_transcript = payload.get("sample_transcript")
    if sample_transcript:
        transcript_memory = sample_transcript["memory_report"]
        transcript_abstention = sample_transcript["abstention_report"]
        transcript_state = sample_transcript.get("state_resolution_report")
        lines.extend(
            [
                "",
                "## Annotated Transcript Smoke",
                "",
                f"- Annotation source: `{sample_transcript['annotation_path']}`",
                f"- Turns: {sample_transcript['turns']}",
                f"- Events: {sample_transcript['events']}",
                f"- Recall cases: {sample_transcript['recall_cases']}",
                f"- Evidence cases: {sample_transcript['evidence_cases']}",
                f"- State cases: {sample_transcript['state_cases']}",
                f"- RAG recall: {transcript_memory['rag_recall_at_k']:.2%}",
                f"- Event memory recall: {transcript_memory['event_memory_recall_at_k']:.2%}",
                f"- Absolute gain: {transcript_memory['absolute_gain']:.2%}",
                f"- Evidence accuracy: {transcript_abstention['accuracy']:.2%}",
                f"- Evidence precision: {transcript_abstention['precision']:.2%}",
                f"- Evidence positive recall: {transcript_abstention['positive_recall']:.2%}",
                f"- Evidence negative abstention: {transcript_abstention['negative_abstention']:.2%}",
                f"- Memory gate passed: {sample_transcript['memory_gate']['passed']}",
                f"- Abstention gate passed: {sample_transcript['abstention_gate']['passed']}",
            ]
        )
        if transcript_state:
            lines.extend(
                [
                    f"- State resolution accuracy: {transcript_state['accuracy']:.2%}",
                    f"- State resolution false positive rate: {transcript_state['false_positive_rate']:.2%}",
                ]
            )

    lines.extend(
        [
            "",
            "## State Grounding",
            "",
            f"- Cases: {grounding['cases']}",
            f"- Accuracy: {grounding['accuracy']:.2%}",
        ]
    )
    state_resolution = payload.get("state_resolution")
    if state_resolution:
        lines.extend(
            [
                "",
                "## State Resolution",
                "",
                f"- Cases: {state_resolution['cases']}",
                f"- Accuracy: {state_resolution['accuracy']:.2%}",
                f"- Answer rate: {state_resolution['answer_rate']:.2%}",
                f"- False positive rate: {state_resolution['false_positive_rate']:.2%}",
                "",
                "| Category | Cases | Accuracy | Answer Rate | False Positive Rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for category, report in state_resolution["by_category"].items():
            lines.append(
                f"| {category} | {report['cases']} | {report['accuracy']:.2%} | "
                f"{report['answer_rate']:.2%} | {report['false_positive_rate']:.2%} |"
            )
    lines.append("")
    return "\n".join(lines)


def _calibrated_abstention_line(label: str, run: dict[str, Any]) -> str:
    report = run["evaluation_report"]
    return (
        f"- {label}: threshold {run['threshold']:.2f}, "
        f"calibration cases {run['calibration_cases']}, evaluation cases {run['evaluation_cases']}, "
        f"eval accuracy {report['accuracy']:.2%}, eval precision {report['precision']:.2%}, "
        f"eval positive recall {report['positive_recall']:.2%}, "
        f"eval negative abstention {report['negative_abstention']:.2%}."
    )


def _select_calibrated_abstention(runs):
    def key(run):
        report = run.evaluation_report
        gate = abstention_gate(
            report.negative_abstention,
            positive_recall=report.positive_recall,
        )
        return (
            int(gate.passed),
            report.accuracy,
            report.precision,
            report.positive_recall,
            report.negative_abstention,
        )

    return max(runs, key=key)


def _maybe_run_sample_transcript(threshold: float, confidence_method: str, k: int) -> dict[str, Any] | None:
    annotation_path = Path("benchmarks/sample_transcript_annotations.jsonl")
    if not annotation_path.exists():
        return None

    benchmark = load_annotated_transcript_benchmark(annotation_path)
    memory = EventMemory(benchmark.events)
    memory_report = evaluate_memory(memory, benchmark.recall_cases, k=k)
    abstention_report = evaluate_abstention(
        memory,
        benchmark.evidence_cases,
        k=k,
        threshold=threshold,
        retriever="event_memory",
        confidence_method=confidence_method,
    )
    memory_gate_report = memory_gate(
        memory_report.event_memory_recall_at_k,
        memory_report.rag_recall_at_k,
    )
    abstention_gate_report = abstention_gate(
        abstention_report.negative_abstention,
        positive_recall=abstention_report.positive_recall,
    )
    state_resolution_report = (
        evaluate_state_resolution(benchmark.events, benchmark.state_cases)
        if benchmark.state_cases
        else None
    )
    return {
        "annotation_path": str(annotation_path),
        "turns": len(benchmark.turns),
        "events": len(benchmark.events),
        "recall_cases": len(benchmark.recall_cases),
        "evidence_cases": len(benchmark.evidence_cases),
        "state_cases": len(benchmark.state_cases),
        "memory_report": memory_report,
        "abstention_report": abstention_report,
        "state_resolution_report": state_resolution_report,
        "memory_gate": memory_gate_report,
        "abstention_gate": abstention_gate_report,
    }


if __name__ == "__main__":
    main()
