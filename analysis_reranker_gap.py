"""Analyze reranker score distributions — bridge path vs fallback path routing.

Measures the gap: for each bridge-served intent, what rerank_score does the
correct frame achieve when routed through the fallback path (linker + reranker
without the bridge pre-check)? Would a rerank_score threshold gate be viable?
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from melm.appliance import local_assistant_router as router_module
from melm.appliance.assistant_frame_linker import FrameLinker
from melm.appliance.assistant_frame_ranker import E3CandidateReranker
from melm.appliance.functional_grammar import parse_functional_relations

# Re-export for script use
from melm.appliance.local_assistant_router import (
    _FRAME_LINKER_MIGRATED_INTENTS,
    _is_question_like,
    _is_request_like,
    _tokenize,
    _IN_MEMORY_LEXICON,
    _semantic_family_terms,
)


@dataclass
class TestCase:
    utterance: str
    expected_intent: str
    bridge_should_route: bool  # Would the actual bridge accept this?
    note: str = ""


@dataclass
class RunResult:
    utterance: str
    expected_intent: str
    bridge_should_route: bool
    top_linker_intent: str
    top_linker_frame_id: str
    top_linker_score: float
    top_linker_threshold: float
    top_reranker_intent: str
    top_reranker_frame_id: str
    top_reranker_rule_score: float
    top_reranker_score: float
    correct_in_linker: bool
    correct_in_reranker: bool
    linker_candidates: list[dict[str, Any]] = field(default_factory=list)
    reranker_candidates: list[dict[str, Any]] = field(default_factory=list)


# ── Test utterances per intent ──────────────────────────────────────────────
# Each intent has:
#   TRUE POSITIVES: utterances the bridge correctly routes YES
#   FALSE POSITIVES: utterances that are NOT the intent but might look close
#   FALSE NEGATIVES: valid utterances the bridge might miss

TEST_CASES: dict[str, list[TestCase]] = {
    "story": [
        # True positives (bridge routes YES)
        TestCase("tell me a story", "story", True, "basic story request"),
        TestCase("tell me a story about rain", "story", True, "story with topic"),
        TestCase("read me a story", "story", True, "read variant"),
        TestCase("make up a story", "story", True, "make variant"),
        TestCase("give me a story", "story", True, "give variant"),
        TestCase("tell me a story about a brave dog", "story", True, "story with specific topic"),
        # Bridge rejects (FALSE for bridge → NOT a story)
        TestCase("same people tell stories", "story", False, "declarative, no request"),
        TestCase("the story was long", "story", False, "past declarative, no request"),
        TestCase("I like stories", "story", False, "preference, no request"),
        TestCase("story time is over", "story", False, "no request structure"),
        TestCase("he tells stories", "story", False, "third-person declarative"),
    ],
    "weather": [
        # True positives
        TestCase("what is the weather like today", "weather", True, "basic weather query"),
        TestCase("how is the weather", "weather", True, "how variant"),
        TestCase("will it rain today", "weather", True, "rain question"),
        TestCase("what is the forecast", "weather", True, "forecast query"),
        TestCase("is it going to rain", "weather", True, "going to rain"),
        TestCase("what is the weather in Lagos", "weather", True, "weather + location"),
        # Concept questions — bridge explicitly rejects these
        TestCase("what is weather", "weather", False, "concept question — bridge blocks"),
        TestCase("how does weather work", "weather", False, "concept question — bridge blocks"),
        TestCase("explain weather systems", "weather", False, "explanation — bridge blocks"),
        TestCase("what does weather mean", "weather", False, "definition — bridge blocks"),
        # Bridge rejects (no weather terms)
        TestCase("is it going to be hot", "weather", False, "no explicit weather term"),
    ],
    "common_sense_safety": [
        # True positives
        TestCase("is it safe to go outside without a shirt", "common_sense_safety", True, "safety question"),
        TestCase("can I go to the park without clothes", "common_sense_safety", True, "park + clothes"),
        TestCase("should I wear a shirt to the park", "common_sense_safety", True, "wear question"),
        TestCase("is it okay to walk outside without clothes", "common_sense_safety", True, "walk + clothes"),
        # Bridge rejects
        TestCase("the park is nice", "common_sense_safety", False, "no safety frame"),
        TestCase("I like my shirt", "common_sense_safety", False, "no safety question"),
        TestCase("go to the park", "common_sense_safety", False, "no clothing/undress terms"),
        TestCase("class is canceled", "common_sense_safety", False, "class isn't a safety context"),
    ],
    "health_advice": [
        # True positives
        TestCase("I feel sick", "health_advice", True, "feeling sick"),
        TestCase("I have a headache", "health_advice", True, "condition report"),
        TestCase("what should I do for a fever", "health_advice", True, "advice question"),
        TestCase("how can I sleep better", "health_advice", True, "sleep advice"),
        TestCase("I need help with anxiety", "health_advice", True, "mental health"),
        TestCase("my head hurts", "health_advice", True, "pain report"),
        # Bridge rejects
        TestCase("the hospital is far", "health_advice", False, "no personal health context"),
        TestCase("sick people need rest", "health_advice", False, "general statement, not personal"),
        TestCase("define fever", "health_advice", False, "definition, not personal advice"),
    ],
    "media_playback": [
        # True positives
        TestCase("play some music", "media_playback", True, "basic play music"),
        TestCase("play calm piano", "media_playback", True, "play specific media"),
        TestCase("play rain sounds", "media_playback", True, "play sounds"),
        TestCase("start the music", "media_playback", True, "start variant"),
        TestCase("play something with sounds", "media_playback", True, "special-cased"),
        # Bridge rejects
        TestCase("the music is nice", "media_playback", False, "no play/start action"),
        TestCase("I like music", "media_playback", False, "preference, no action"),
        TestCase("play is my favorite", "media_playback", False, "play as noun, not action"),
        TestCase("start the car", "media_playback", False, "not media"),
    ],
    "meal_suggestion": [
        # True positives
        TestCase("suggest something to eat", "meal_suggestion", True, "direct suggestion"),
        TestCase("recommend a meal", "meal_suggestion", True, "recommend variant"),
        TestCase("what should I eat for dinner", "meal_suggestion", True, "what should I eat"),
        TestCase("what can I eat for breakfast", "meal_suggestion", True, "what can I eat"),
        TestCase("what should I cook", "meal_suggestion", True, "cook variant"),
        # Bridge rejects
        TestCase("you cook well", "meal_suggestion", False, "not a user choice frame"),
        TestCase("she eats pasta", "meal_suggestion", False, "third-person, no frame"),
        TestCase("cook the rice", "meal_suggestion", False, "imperative without user choice"),
        TestCase("I eat rice", "meal_suggestion", False, "declarative, no question/request"),
    ],
    "social_contact": [
        # True positives
        TestCase("call my mom", "social_contact", True, "call contact"),
        TestCase("phone my mom", "social_contact", True, "phone variant"),
        TestCase("call mom", "social_contact", True, "call without possessive"),
        TestCase("ring my mom", "social_contact", True, "ring variant"),
        TestCase("reach my mom", "social_contact", True, "reach variant"),
        # Bridge rejects
        TestCase("call the meeting", "social_contact", False, "no person target"),
        TestCase("phone battery is dead", "social_contact", False, "phone as object"),
        TestCase("my phone number", "social_contact", False, "phone as noun"),
        TestCase("the phone is ringing", "social_contact", False, "not a contact action"),
        TestCase("call the doctor", "social_contact", False, "doctor not in trusted contacts"),
    ],
    "autobiographical_memory": [
        # True positives
        TestCase("what did we talk about last time", "autobiographical_memory", True, "shared context"),
        TestCase("summarize our conversation", "autobiographical_memory", True, "summarize session"),
        TestCase("recap what happened in our last session", "autobiographical_memory", True, "recap session"),
        TestCase("what did I ask you earlier", "autobiographical_memory", True, "self query"),
        TestCase("tell me about our last conversation", "autobiographical_memory", True, "tell about"),
        # Bridge rejects
        TestCase("summarize the book", "autobiographical_memory", False, "no user context"),
        TestCase("what happened in the movie", "autobiographical_memory", False, "no we/our"),
        TestCase("recap the game", "autobiographical_memory", False, "not autobiographical"),
    ],
    "personal_memory": [
        # True positives
        TestCase("remember when I went to school", "personal_memory", True, "memory recall"),
        TestCase("do you remember my favorite color", "personal_memory", True, "remember attribute"),
        TestCase("what is my school schedule", "personal_memory", True, "routine query"),
        TestCase("what time do I go to school", "personal_memory", True, "routine time"),
        TestCase("forget what I said", "personal_memory", True, "forget variant"),
        # Bridge rejects
        TestCase("I remember that place", "personal_memory", False, "no first-person memory context? actually might pass"),
        TestCase("remember the alamo", "personal_memory", False, "generic, not personal"),
    ],
}


def load_lexicon_copy() -> dict[str, frozenset[str]]:
    """Return a copy of the module-level IN_MEMORY_LEXICON."""
    return dict(router_module._IN_MEMORY_LEXICON)


def build_frame_templates() -> dict[str, Any]:
    from melm.contracts import CONTRACT_ROOT, validate_frame_templates
    import json
    path = CONTRACT_ROOT / "frame_templates.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_frame_templates(payload)
    return payload["templates"]


def run_test(test: TestCase, linker: FrameLinker, reranker: E3CandidateReranker,
             templates: dict[str, Any]) -> RunResult:
    tokens = _tokenize(test.utterance)
    text = test.utterance

    is_q = _is_question_like(text, tokens)
    is_r = _is_request_like(tokens)

    lexicon = load_lexicon_copy()

    # Run linker
    candidates = linker.score(tokens, lexicon, is_question_like=is_q, is_request_like=is_r)

    linker_top_intent = candidates[0].intent if candidates else "none"
    linker_top_frame_id = candidates[0].frame_id if candidates else "none"
    linker_top_score = candidates[0].score if candidates else 0.0
    linker_top_threshold = candidates[0].threshold if candidates else 0.0
    correct_in_linker = bool(candidates and candidates[0].intent == test.expected_intent)

    # Run reranker (no UOL token_roles = no-UOL fallback)
    reranked = reranker.rerank(candidates, tokens, lexicon,
                               is_question_like=is_q, is_request_like=is_r,
                               token_roles=None)

    reranker_top_intent = reranked[0].intent if reranked else "none"
    reranker_top_frame_id = reranked[0].frame_id if reranked else "none"
    reranker_top_rule_score = reranked[0].rule_score if reranked else 0.0
    reranker_top_score = reranked[0].rerank_score if reranked else 0.0
    correct_in_reranker = bool(reranked and reranked[0].intent == test.expected_intent)

    # Also attempt UOL-based reranking
    parse = parse_functional_relations(tokens, question_mark="?" in text)
    uol_reranked = []
    if parse is not None and parse.token_roles:
        uol_reranked = reranker.rerank(candidates, tokens, lexicon,
                                       is_question_like=is_q, is_request_like=is_r,
                                       token_roles=parse.token_roles)

    # Capture all candidates for analysis
    linker_info = []
    for c in candidates:
        linker_info.append({
            "frame_id": c.frame_id,
            "intent": c.intent,
            "score": c.score,
            "threshold": c.threshold,
            "components": dict(c.score_components),
        })

    reranker_info = []
    for s in reranked:
        reranker_info.append({
            "frame_id": s.frame_id,
            "intent": s.intent,
            "rule_score": s.rule_score,
            "rerank_score": s.rerank_score,
            "explanation": s.rerank_explanation,
            "threshold": s.threshold,
        })

    result = RunResult(
        utterance=test.utterance,
        expected_intent=test.expected_intent,
        bridge_should_route=test.bridge_should_route,
        top_linker_intent=linker_top_intent,
        top_linker_frame_id=linker_top_frame_id,
        top_linker_score=linker_top_score,
        top_linker_threshold=linker_top_threshold,
        top_reranker_intent=reranker_top_intent,
        top_reranker_frame_id=reranker_top_frame_id,
        top_reranker_rule_score=reranker_top_rule_score,
        top_reranker_score=reranker_top_score,
        correct_in_linker=correct_in_linker,
        correct_in_reranker=correct_in_reranker,
        linker_candidates=linker_info,
        reranker_candidates=reranker_info,
    )

    return result


def print_separator(char="=", width=100):
    print(char * width)


def summarize_results(all_results: dict[str, list[RunResult]]):
    print_separator()
    print("INTENT-LEVEL SUMMARY")
    print_separator()

    for intent, results in all_results.items():
        true_positives = [r for r in results if r.bridge_should_route]
        false_positives = [r for r in results if not r.bridge_should_route]

        tp_linker_correct = sum(1 for r in true_positives if r.correct_in_linker)
        tp_reranker_correct = sum(1 for r in true_positives if r.correct_in_reranker)
        fp_linker_wrong = sum(1 for r in false_positives if not r.correct_in_linker)
        fp_reranker_wrong = sum(1 for r in false_positives if not r.correct_in_reranker)

        print(f"\n{intent}:")
        print(f"  TRUE POSITIVES (bridge says YES): {len(true_positives)} utterances")
        print(f"    Linker top1 correct: {tp_linker_correct}/{len(true_positives) or 1}")
        print(f"    Reranker top1 correct: {tp_reranker_correct}/{len(true_positives) or 1}")
        if true_positives:
            avg_rerank = sum(r.top_reranker_score for r in true_positives) / len(true_positives)
            min_rerank = min(r.top_reranker_score for r in true_positives)
            max_rerank = max(r.top_reranker_score for r in true_positives)
            print(f"    Rerank score range: [{min_rerank:.4f}, {max_rerank:.4f}] avg={avg_rerank:.4f}")

        print(f"  BRIDGE REJECTS (bridge says NO): {len(false_positives)} utterances")
        print(f"    Linker top1 correct (not intent): {fp_linker_wrong}/{len(false_positives) or 1}")
        print(f"    Reranker top1 correct (not intent): {fp_reranker_wrong}/{len(false_positives) or 1}")

        # For bridge rejects, what does the linker/reranker produce?
        if false_positives:
            avg_fp_rerank = sum(r.top_reranker_score for r in false_positives) / len(false_positives)
            print(f"    Avg rerank score (top candidate): {avg_fp_rerank:.4f}")
            # Check if any bridge reject still has the intent frame in candidates
            has_intent_frame = sum(
                1 for r in false_positives
                if any(c["intent"] == intent for c in r.linker_candidates)
            )
            print(f"    Has {intent} frame in candidates: {has_intent_frame}/{len(false_positives)}")

    print()
    print_separator()
    print("RERANKER INTENT-CHANGE ANALYSIS")
    print_separator()

    total_switches = 0
    total_cases = 0
    for intent, results in all_results.items():
        for r in results:
            total_cases += 1
            if r.top_linker_intent != r.top_reranker_intent:
                total_switches += 1

    print(f"\nReranker changed top intent: {total_switches}/{total_cases} cases ({100*total_switches/total_cases:.1f}%)")

    # Per-intent: how often does reranker change the top intent?
    for intent, results in all_results.items():
        switches = sum(1 for r in results if r.top_linker_intent != r.top_reranker_intent)
        print(f"  {intent}: {switches}/{len(results)} ({100*switches/len(results):.1f}%)")

    # Threshold gate analysis
    print()
    print_separator()
    print("THRESHOLD GATE VIABILITY ANALYSIS")
    print_separator()
    print("\nWould a rerank_score >= threshold gate catch FPs without blocking TPs?")
    print()

    for intent, results in all_results.items():
        true_positives = [r for r in results if r.bridge_should_route]
        false_positives = [r for r in results if not r.bridge_should_route]

        if not true_positives:
            continue

        # Find the minimum rerank_score among correct TPs
        tp_scores = [r.top_reranker_score for r in true_positives if r.correct_in_reranker]
        if not tp_scores:
            tp_scores = [r.top_reranker_score for r in true_positives]

        fp_scores = [r.top_reranker_score for r in false_positives]

        min_tp = min(tp_scores) if tp_scores else 0
        max_fp = max(fp_scores) if fp_scores else 0

        # Explore thresholds
        print(f"  {intent}:")
        print(f"    TP rerank scores: {[f'{s:.4f}' for s in sorted(tp_scores)]}")
        print(f"    FP rerank scores: {[f'{s:.4f}' for s in sorted(fp_scores)]}")
        print(f"    Min TP={min_tp:.4f}, Max FP={max_fp:.4f}")

        # Find optimal threshold
        import itertools
        all_unique = sorted(set(tp_scores + fp_scores))
        best_threshold = 0.0
        best_tp_kept = 0
        best_fp_blocked = 0
        best_f1 = 0.0

        for threshold in [0.0] + all_unique:
            tp_kept = sum(1 for s in tp_scores if s >= threshold)
            fp_blocked = sum(1 for s in fp_scores if s < threshold)
            # "F1" for this gate
            precision = tp_kept / (tp_kept + (len(fp_scores) - fp_blocked)) if (tp_kept + len(fp_scores) - fp_blocked) > 0 else 0
            recall = tp_kept / len(tp_scores) if tp_scores else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_tp_kept = tp_kept
                best_fp_blocked = fp_blocked

        print(f"    Best gate: rerank_score >= {best_threshold:.4f}")
        print(f"      TP kept: {best_tp_kept}/{len(tp_scores)} ({100*best_tp_kept/len(tp_scores):.0f}%)")
        print(f"      FP blocked: {best_fp_blocked}/{len(fp_scores)} ({100*best_fp_blocked/len(fp_scores):.0f}%)")
        print(f"      Gate F1: {best_f1:.3f}")

    # Candidate depth analysis
    print()
    print_separator()
    print("CANDIDATE DEPTH ANALYSIS")
    print_separator()
    print("\nHow many candidates does the linker produce per utterance?")

    for intent, results in all_results.items():
        avg_linker = sum(len(r.linker_candidates) for r in results) / len(results)
        avg_reranker = sum(len(r.reranker_candidates) for r in results) / len(results)
        max_linker = max(len(r.linker_candidates) for r in results)
        max_reranker = max(len(r.reranker_candidates) for r in results)
        print(f"  {intent}: avg candidates linker={avg_linker:.1f} reranker={avg_reranker:.1f} max={max_linker}/{max_reranker}")


def main():
    print_separator()
    print("RERANKER SCORE DISTRIBUTION ANALYSIS")
    print("Bridge path vs Fallback path routing gap")
    print_separator()

    print("\nLoading modules...")
    linker = FrameLinker()
    reranker = E3CandidateReranker()
    templates = build_frame_templates()

    templates_list = sorted(templates.keys())
    print(f"Loaded {len(templates_list)} templates: {', '.join(templates_list)}")

    lexicon = load_lexicon_copy()
    print(f"Lexicon size: {len(lexicon)} tokens")

    migrated = sorted(_FRAME_LINKER_MIGRATED_INTENTS)
    print(f"Migrated intents: {', '.join(migrated)}")

    print()
    print_separator()
    print("DETAILED RESULTS PER UTTERANCE")
    print_separator()

    all_results: dict[str, list[RunResult]] = {}

    for intent, cases in TEST_CASES.items():
        print(f"\n--- {intent} ({len(cases)} utterances) ---")
        intent_results: list[RunResult] = []
        for test in cases:
            result = run_test(test, linker, reranker, templates)
            intent_results.append(result)

            bridge_status = "BRIDGE_YES" if result.bridge_should_route else "bridge_no"
            correct_linker = "Y" if result.correct_in_linker else "N"
            correct_reranker = "Y" if result.correct_in_reranker else "N"

            print(f"  {bridge_status:12s} L={correct_linker} R={correct_reranker} "
                  f"linker={result.top_linker_intent:25s}({result.top_linker_score:.3f}) "
                  f"reranker={result.top_reranker_intent:25s}({result.top_reranker_score:.4f}) "
                  f"| {test.utterance}")

            # Show all reranker candidates
            if len(result.reranker_candidates) > 1:
                for s in result.reranker_candidates[1:]:
                    expected_mark = " <--" if s["intent"] == test.expected_intent else ""
                    print(f"    |-- {s['intent']:25s} frame={s['frame_id']:30s} "
                          f"rule={s['rule_score']:.3f} rerank={s['rerank_score']:.4f}"
                          f"{expected_mark}")

        all_results[intent] = intent_results

    print()
    summarize_results(all_results)

    # Overall summary
    print()
    print_separator()
    print("BOTTOM LINE")
    print_separator()

    all_tp = sum(len([r for r in results if r.bridge_should_route]) for results in all_results.values())
    all_fp = sum(len([r for r in results if not r.bridge_should_route]) for results in all_results.values())
    all_tp_linker_correct = sum(
        sum(1 for r in results if r.bridge_should_route and r.correct_in_linker)
        for results in all_results.values()
    )
    all_tp_reranker_correct = sum(
        sum(1 for r in results if r.bridge_should_route and r.correct_in_reranker)
        for results in all_results.values()
    )

    print(f"\n  Bridge-accepted (TP) utterances: {all_tp}")
    print(f"    Linker top-1 correct: {all_tp_linker_correct}/{all_tp} ({100*all_tp_linker_correct/all_tp:.1f}%)")
    print(f"    Reranker top-1 correct: {all_tp_reranker_correct}/{all_tp} ({100*all_tp_reranker_correct/all_tp:.1f}%)")
    print()
    print("  Interpretation:")
    print("    If reranker score >= threshold were the ONLY gate (no bridge pre-check),")
    print("    the % above shows how many bridge-accepted utterances would still route correctly.")


if __name__ == "__main__":
    main()
