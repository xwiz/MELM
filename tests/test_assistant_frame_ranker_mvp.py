"""Tests for the E3 frame reranker (M5) using the sealed minimal-pair set."""

import json
import unittest
from pathlib import Path
from typing import Any

from melm.appliance.assistant_frame_linker import FrameCandidate, FrameLinker
from melm.appliance.assistant_frame_ranker import E3CandidateReranker, ScoredCandidate
from melm.contracts import load_frame_minimal_pairs


def _lexicon_from_pairs(pairs_payload: dict[str, Any]) -> dict[str, frozenset[str]]:
    return {
        token: frozenset(classes)
        for token, classes in pairs_payload["lexicon"].items()
    }


def _run_pair(
    linker: FrameLinker,
    reranker: E3CandidateReranker,
    pair: dict[str, Any],
    lexicon: dict[str, frozenset[str]],
) -> dict[str, Any]:
    tokens = tuple(pair["tokens"])
    candidates = linker.score(
        tokens, lexicon,
        is_question_like=pair["is_question_like"],
        is_request_like=pair["is_request_like"],
    )
    reranked = reranker.rerank(
        candidates, tokens, lexicon,
        is_question_like=pair["is_question_like"],
        is_request_like=pair["is_request_like"],
    )

    top3_ids = [s.frame_id for s in reranked[:3]]
    top1_id = reranked[0].frame_id if reranked else None

    return {
        "id": pair["id"],
        "expected_top": pair["expected_top_frame"],
        "expected_top3": list(pair["expected_top3"]),
        "top1": top1_id,
        "top3": top3_ids,
        "n_candidates": len(reranked),
        "reranked": reranked,
    }


class E3RerankerLoadContractTests(unittest.TestCase):
    """E3 reranker loads frame templates at init."""

    def test_loads_templates(self) -> None:
        reranker = E3CandidateReranker()
        self.assertGreaterEqual(len(reranker._templates), 9)

    def test_minimal_pairs_contract_loads(self) -> None:
        payload = load_frame_minimal_pairs()
        self.assertIn("pairs", payload)
        self.assertIn("lexicon", payload)
        self.assertGreaterEqual(len(payload["pairs"]), 20)


class E3RerankerContentTokensTests(unittest.TestCase):
    """_content_tokens filters stopwords correctly."""

    def test_filters_stopwords(self) -> None:
        from melm.appliance.assistant_frame_ranker import _content_tokens
        tokens = ("tell", "me", "a", "story", "about", "rain")
        result = _content_tokens(tokens)
        self.assertEqual(result, ["tell", "story", "rain"])

    def test_all_stopwords_returns_empty(self) -> None:
        from melm.appliance.assistant_frame_ranker import _content_tokens
        result = _content_tokens(("a", "the", "is", "it"))
        self.assertEqual(result, [])

    def test_no_stopwords_preserves_all(self) -> None:
        from melm.appliance.assistant_frame_ranker import _content_tokens
        result = _content_tokens(("rain", "snow", "story"))
        self.assertEqual(result, ["rain", "snow", "story"])


class E3RerankerMinimalPairsTests(unittest.TestCase):
    """Rerank all minimal pairs; measure top-3 recall and precision."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs_payload = load_frame_minimal_pairs()
        cls.lexicon = _lexicon_from_pairs(cls.pairs_payload)
        cls.linker = FrameLinker()
        cls.reranker = E3CandidateReranker()

    def test_all_pairs_have_expected_classes_in_lexicon(self) -> None:
        """Verify every content-bearing token in every pair has a lexicon entry."""
        from melm.appliance.assistant_frame_ranker import _STOP_TOKENS
        missing: list[str] = []
        for pair in self.pairs_payload["pairs"]:
            for token in pair["tokens"]:
                if token in _STOP_TOKENS:
                    continue
                if token not in self.lexicon:
                    missing.append(f"{pair['id']}:{token}")
        self.assertEqual(missing, [], f"content tokens missing from lexicon: {missing}")

    def test_top1_recall_meets_threshold(self) -> None:
        """expected_top_frame must be reranked #1."""
        results: list[dict[str, Any]] = []
        for pair in self.pairs_payload["pairs"]:
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            results.append(r)

        applicable = [r for r in results if r["expected_top"] is not None]
        correct = sum(1 for r in applicable if r["top1"] == r["expected_top"])
        recall = correct / len(applicable) if applicable else 1.0

        failures = [
            f"{r['id']}: expected top1={r['expected_top']}, got {r['top1']}"
            for r in applicable if r["top1"] != r["expected_top"]
        ]

        self.assertGreaterEqual(
            recall, 0.85,
            f"top-1 recall {recall:.3f} ({correct}/{len(applicable)}) below 0.85\n"
            + "\n".join(failures),
        )

    def test_top3_recall_meets_threshold(self) -> None:
        """expected_top_frame must appear in top-3 after reranking."""
        results: list[dict[str, Any]] = []
        for pair in self.pairs_payload["pairs"]:
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            results.append(r)

        applicable = [r for r in results if r["expected_top"] is not None]
        correct = sum(1 for r in applicable if r["expected_top"] in r["top3"])
        recall = correct / len(applicable) if applicable else 1.0

        failures = [
            f"{r['id']}: expected {r['expected_top']} in top-3 {r['top3']}"
            for r in applicable if r["expected_top"] not in r["top3"]
        ]

        self.assertGreaterEqual(
            recall, 0.95,
            f"top-3 recall {recall:.3f} ({correct}/{len(applicable)}) below 0.95\n"
            + "\n".join(failures),
        )

    def test_precision_top1_meets_threshold(self) -> None:
        """Precision@1: of all applicable cases, fraction where top-1 is correct.

        Measures precision against two subsets:
        - All cases: expected_top_frame ≠ null (overall).
        - Precision-target cases: precision_target=true (targeted precision metric).

        Target ≥95% overall, ≥95% on precision-target cases toward M5 exit
        criterion of ≥98%.
        """
        results: list[dict[str, Any]] = []
        for pair in self.pairs_payload["pairs"]:
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            results.append(r)

        applicable = [r for r in results if r["expected_top"] is not None]
        correct = sum(1 for r in applicable if r["top1"] == r["expected_top"])
        precision = correct / len(applicable) if applicable else 1.0

        failures = [
            f"{r['id']}: expected {r['expected_top']}, got {r['top1']} — "
            f"scores: {[(s.frame_id, s.rerank_score) for s in r['reranked'][:3]]}"
            for r in applicable if r["top1"] != r["expected_top"]
        ]

        self.assertGreaterEqual(
            precision, 0.95,
            f"precision@1 (all) {precision:.3f} ({correct}/{len(applicable)}) below 0.95\n"
            + "\n".join(failures),
        )

        # Precision-target subset: pairs where precision_target=true
        pair_map = {p["id"]: p for p in self.pairs_payload["pairs"]}
        precision_target_results = [
            r for r in results
            if pair_map.get(r["id"], {}).get("precision_target", False)
            and r["expected_top"] is not None
        ]
        pt_correct = sum(1 for r in precision_target_results if r["top1"] == r["expected_top"])
        pt_precision = pt_correct / len(precision_target_results) if precision_target_results else 1.0

        pt_failures = [
            f"{r['id']}: expected {r['expected_top']}, got {r['top1']}"
            for r in precision_target_results if r["top1"] != r["expected_top"]
        ]

        self.assertGreaterEqual(
            pt_precision, 0.95,
            f"precision@1 (precision-target) {pt_precision:.3f} "
            f"({pt_correct}/{len(precision_target_results)}) below 0.95\n"
            + "\n".join(pt_failures),
        )

    def test_precision_no_false_local_safety(self) -> None:
        """Zero false-local safety cases: no safety frame ranks ABOVE the expected
        top frame.  Safety frames may appear as lower-ranked secondary candidates
        (e.g. health_advice for "remember when I was sick" is valid), but must
        never displace the correct primary intent."""
        safety_intents = {"common_sense_safety", "health_advice"}
        for pair in self.pairs_payload["pairs"]:
            expected = pair["expected_top_frame"]
            if expected is None:
                continue
            if expected in safety_intents:
                continue
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            for i, sc in enumerate(r["reranked"]):
                if sc.frame_id in safety_intents and sc.frame_id != expected:
                    # Safety frame appeared — it must not be higher than the expected frame.
                    expected_rank = next(
                        (j for j, s in enumerate(r["reranked"]) if s.frame_id == expected),
                        None,
                    )
                    if expected_rank is None or i < expected_rank:
                        self.fail(
                            f"{pair['id']}: safety frame {sc.frame_id} "
                            f"(rank={i}, score={sc.rerank_score}) ranked above "
                            f"expected {expected} (rank={expected_rank}) for "
                            f"'{pair['utterance']}'"
                        )

    def test_rerank_improves_over_rule_score_on_tiebreak_cases(self) -> None:
        """On tiebreak cases (discriminator=tiebreak_requires_reranker), the
        reranker must place expected_top_frame as #1."""
        tiebreak_pairs = [
            p for p in self.pairs_payload["pairs"]
            if p.get("discriminator") == "tiebreak_requires_reranker"
        ]
        for pair in tiebreak_pairs:
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            self.assertEqual(
                r["top1"], pair["expected_top_frame"],
                f"{pair['id']}: reranker failed tiebreak — "
                f"expected {pair['expected_top_frame']}, got {r['top1']}. "
                f"Scores: {[(s.frame_id, s.rerank_score, s.rerank_explanation) for s in r['reranked'][:3]]}",
            )

    def test_null_expected_top_returns_no_candidates(self) -> None:
        """When expected_top_frame is null (exclude should fire), no candidates pass threshold."""
        exclude_pairs = [
            p for p in self.pairs_payload["pairs"]
            if p["expected_top_frame"] is None
        ]
        for pair in exclude_pairs:
            r = _run_pair(self.linker, self.reranker, pair, self.lexicon)
            self.assertEqual(
                r["n_candidates"], 0,
                f"{pair['id']}: expected no candidates (exclude), got {r['n_candidates']}: "
                + str([s.frame_id for s in r["reranked"]]),
            )

    def test_scored_candidate_fields_populated(self) -> None:
        """Every ScoredCandidate has all required fields."""
        sample_pair = self.pairs_payload["pairs"][0]
        r = _run_pair(self.linker, self.reranker, sample_pair, self.lexicon)
        for sc in r["reranked"]:
            self.assertIsInstance(sc.frame_id, str)
            self.assertIsInstance(sc.intent, str)
            self.assertIsInstance(sc.rule_score, float)
            self.assertIsInstance(sc.rerank_score, float)
            self.assertIsInstance(sc.rerank_explanation, str)
            self.assertIsInstance(sc.score_components, dict)
            self.assertGreaterEqual(sc.rerank_score, 0.0)

    def test_reranked_list_sorted_by_score_desc(self) -> None:
        """Reranked candidates are sorted by rerank_score descending."""
        sample_pair = self.pairs_payload["pairs"][0]
        r = _run_pair(self.linker, self.reranker, sample_pair, self.lexicon)
        for i in range(len(r["reranked"]) - 1):
            self.assertGreaterEqual(
                r["reranked"][i].rerank_score,
                r["reranked"][i + 1].rerank_score,
            )


class E3RerankerActionAlignmentTests(unittest.TestCase):
    """_action_alignment_score correctly identifies action token matches."""

    def setUp(self) -> None:
        self.templates = {
            "story": {
                "activation": {"action_tokens": ["tell", "read", "make", "give"]},
            },
            "weather": {
                "activation": {"action_tokens": []},
            },
        }

    def test_content_token_matches_action_tokens(self) -> None:
        from melm.appliance.assistant_frame_ranker import _action_alignment_score, _content_tokens
        content = _content_tokens(("tell", "me", "a", "story"))
        score = _action_alignment_score("story", self.templates, content)
        self.assertAlmostEqual(score, 1.0)

    def test_no_action_tokens_returns_zero(self) -> None:
        from melm.appliance.assistant_frame_ranker import _action_alignment_score, _content_tokens
        content = _content_tokens(("rain", "today"))
        score = _action_alignment_score("weather", self.templates, content)
        self.assertAlmostEqual(score, 0.0)

    def test_no_match_returns_zero(self) -> None:
        from melm.appliance.assistant_frame_ranker import _action_alignment_score, _content_tokens
        content = _content_tokens(("rain", "today"))
        score = _action_alignment_score("story", self.templates, content)
        self.assertAlmostEqual(score, 0.0)


class E3RerankerRoleCoverageTests(unittest.TestCase):
    """_role_coverage_score measures fraction of content tokens serving a frame."""

    def setUp(self) -> None:
        self.templates = {
            "story": {
                "activation": {
                    "required_classes": ["narrative_content"],
                    "optional_classes": [],
                    "exclude_classes": ["abstract_concept"],
                    "action_tokens": ["tell", "read"],
                },
            },
        }
        self.lexicon = {
            "tell": frozenset({"narrative_content"}),
            "story": frozenset({"narrative_content"}),
            "rain": frozenset({"weather_phenomenon"}),
            "define": frozenset({"definition_request"}),
        }

    def test_all_content_tokens_serve_frame(self) -> None:
        from melm.appliance.assistant_frame_ranker import _role_coverage_score
        score = _role_coverage_score(
            "story", ("tell", "a", "story"), self.lexicon, self.templates,
        )
        # content tokens: ["tell", "story"] → both have narrative_content → 2/2=1.0
        self.assertAlmostEqual(score, 1.0)

    def test_half_content_tokens_serve_frame(self) -> None:
        from melm.appliance.assistant_frame_ranker import _role_coverage_score
        score = _role_coverage_score(
            "story", ("tell", "a", "story", "rain"), self.lexicon, self.templates,
        )
        # content tokens: ["tell", "story", "rain"] → 2/3 have narrative_content → ~0.667
        self.assertAlmostEqual(score, 2.0 / 3.0, places=4)

    def test_exclude_classes_not_counted_as_coverage(self) -> None:
        """Exclude classes should not contribute to coverage score."""
        from melm.appliance.assistant_frame_ranker import _role_coverage_score
        score = _role_coverage_score(
            "story", ("tell", "a", "story", "define"), self.lexicon, self.templates,
        )
        # "define" → definition_request, which is NOT in required/optional/action_tokens
        # content: ["tell", "story", "define"] → "define" does NOT contribute
        # expected: 2/3 ≈ 0.667 (tell + story contribute)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=4)


class E3RerankerSpecificityTests(unittest.TestCase):
    """_class_specificity_score differs per frame (is a discriminator)."""

    def setUp(self) -> None:
        from melm.appliance.assistant_frame_ranker import _load_frame_templates, _build_semantic_hierarchy, _class_specificity_score, _content_tokens  # noqa: E501
        self.templates = _load_frame_templates()
        self.hierarchy = _build_semantic_hierarchy()
        self._score = _class_specificity_score
        self._content = _content_tokens
        self.lexicon = {
            "rain": frozenset({"weather_phenomenon"}),
            "today": frozenset({"temporal_descriptor"}),
            "tell": frozenset({"narrative_content"}),
            "story": frozenset({"narrative_content"}),
            "call": frozenset({"contact_action", "communication_action"}),
            "mom": frozenset({"social_relation"}),
        }

    def test_different_frames_get_different_scores(self) -> None:
        content = self._content(("tell", "a", "story", "about", "rain"))
        scores = {}
        for fid in ("story", "weather", "social_contact"):
            scores[fid] = self._score(fid, content, self.lexicon, self.hierarchy, self.templates)
        self.assertNotEqual(
            scores["story"], scores["weather"],
            f"story and weather get same specificity: {scores}",
        )

    def test_no_relevant_classes_returns_zero(self) -> None:
        content = self._content(("rain", "today"))
        score = self._score("story", content, self.lexicon, self.hierarchy, self.templates)
        # "rain" and "today" have no narrative_content → 0
        self.assertAlmostEqual(score, 0.0)


class E3RerankerPredicateActionAlignmentTests(unittest.TestCase):
    """_predicate_action_alignment_score uses main_predicate UOL role."""

    def setUp(self) -> None:
        self.templates = {
            "story": {
                "activation": {"action_tokens": ["tell", "read", "make", "give"]},
            },
            "weather": {
                "activation": {"action_tokens": []},
            },
            "meal_suggestion": {
                "activation": {"action_tokens": ["eat", "cook", "have", "suggest"]},
            },
        }

    def _make_roles(self, entries: list[dict[str, str]]) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"index": i, "token": e["token"], "lemma": e["lemma"],
             "role": e["role"], "meaning": e.get("meaning", ""), "weight": 1.0}
            for i, e in enumerate(entries)
        )

    def test_main_predicate_lemma_matches_action_tokens(self) -> None:
        from melm.appliance.assistant_frame_ranker import _predicate_action_alignment_score
        roles = self._make_roles([
            {"token": "tell", "lemma": "tell", "role": "main_predicate", "meaning": "tell:communication"},
            {"token": "story", "lemma": "story", "role": "semantic_object", "meaning": "narrative_content"},
        ])
        score = _predicate_action_alignment_score("story", self.templates, roles)
        self.assertAlmostEqual(score, 1.0)

    def test_main_predicate_lemma_no_match(self) -> None:
        from melm.appliance.assistant_frame_ranker import _predicate_action_alignment_score
        roles = self._make_roles([
            {"token": "rain", "lemma": "rain", "role": "main_predicate", "meaning": "rain:weather"},
            {"token": "today", "lemma": "today", "role": "content_nominal", "meaning": "temporal_descriptor"},
        ])
        score = _predicate_action_alignment_score("story", self.templates, roles)
        self.assertAlmostEqual(score, 0.0)

    def test_secondary_predicate_matches_at_0_5(self) -> None:
        """When a secondary_predicate_candidate (not main) matches, returns 0.5."""
        from melm.appliance.assistant_frame_ranker import _predicate_action_alignment_score
        roles = self._make_roles([
            {"token": "I", "lemma": "i", "role": "grammatical_subject", "meaning": "user"},
            {"token": "want", "lemma": "want", "role": "main_predicate", "meaning": "want:desire"},
            {"token": "to", "lemma": "to", "role": "relation_marker", "meaning": "to"},
            {"token": "eat", "lemma": "eat", "role": "secondary_predicate_candidate", "meaning": "eat:consumption"},
            {"token": "pasta", "lemma": "pasta", "role": "semantic_object", "meaning": "food_item"},
        ])
        score = _predicate_action_alignment_score("meal_suggestion", self.templates, roles)
        self.assertAlmostEqual(score, 0.5)

    def test_no_action_tokens_returns_zero(self) -> None:
        from melm.appliance.assistant_frame_ranker import _predicate_action_alignment_score
        roles = self._make_roles([
            {"token": "rain", "lemma": "rain", "role": "main_predicate", "meaning": "rain:weather"},
        ])
        score = _predicate_action_alignment_score("weather", self.templates, roles)
        self.assertAlmostEqual(score, 0.0)

    def test_no_matching_role_returns_zero(self) -> None:
        """When token_roles exist but no predicate role is helpful, return 0.0."""
        from melm.appliance.assistant_frame_ranker import _predicate_action_alignment_score
        roles = self._make_roles([
            {"token": "the", "lemma": "the", "role": "determiner", "meaning": "nominal_scope"},
            {"token": "weather", "lemma": "weather", "role": "content_nominal", "meaning": "weather_phenomenon"},
        ])
        score = _predicate_action_alignment_score("story", self.templates, roles)
        self.assertAlmostEqual(score, 0.0)


class E3RerankerObjectAlignmentTests(unittest.TestCase):
    """_object_alignment_score uses semantic_object UOL role."""

    def setUp(self) -> None:
        self.templates = {
            "story": {
                "activation": {
                    "required_classes": ["narrative_content"],
                    "optional_classes": [],
                },
            },
            "weather": {
                "activation": {
                    "required_classes": ["weather_phenomenon"],
                    "optional_classes": ["temporal_descriptor"],
                },
            },
            "meal_suggestion": {
                "activation": {
                    "required_classes": ["food_item"],
                    "optional_classes": [],
                },
            },
        }
        self.lexicon = {
            "pasta": frozenset({"food_item"}),
            "rain": frozenset({"weather_phenomenon"}),
            "story": frozenset({"narrative_content"}),
        }

    def _make_roles(self, entries: list[dict[str, str]]) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"index": i, "token": e["token"], "lemma": e["lemma"],
             "role": e["role"], "meaning": e.get("meaning", ""), "weight": 1.0}
            for i, e in enumerate(entries)
        )

    def test_object_class_matches_frame_classes(self) -> None:
        from melm.appliance.assistant_frame_ranker import _object_alignment_score
        roles = self._make_roles([
            {"token": "eat", "lemma": "eat", "role": "main_predicate", "meaning": "eat:consumption"},
            {"token": "pasta", "lemma": "pasta", "role": "semantic_object", "meaning": "food_item"},
        ])
        score = _object_alignment_score("meal_suggestion", self.templates, roles, self.lexicon)
        self.assertAlmostEqual(score, 1.0)

    def test_object_class_no_match_for_frame(self) -> None:
        from melm.appliance.assistant_frame_ranker import _object_alignment_score
        roles = self._make_roles([
            {"token": "tell", "lemma": "tell", "role": "main_predicate", "meaning": "tell:communication"},
            {"token": "story", "lemma": "story", "role": "semantic_object", "meaning": "narrative_content"},
        ])
        # story object does NOT match weather (needs weather_phenomenon)
        score = _object_alignment_score("weather", self.templates, roles, self.lexicon)
        self.assertAlmostEqual(score, 0.0)

    def test_no_semantic_object_returns_zero(self) -> None:
        from melm.appliance.assistant_frame_ranker import _object_alignment_score
        roles = self._make_roles([
            {"token": "tell", "lemma": "tell", "role": "main_predicate", "meaning": "tell:communication"},
        ])
        score = _object_alignment_score("story", self.templates, roles, self.lexicon)
        self.assertAlmostEqual(score, 0.0)

    def test_no_frame_classes_returns_zero(self) -> None:
        from melm.appliance.assistant_frame_ranker import _object_alignment_score
        tmpl_no_classes = {"no_class_frame": {"activation": {"required_classes": [], "optional_classes": []}}}
        roles = self._make_roles([
            {"token": "eat", "lemma": "eat", "role": "main_predicate", "meaning": "eat:consumption"},
            {"token": "pasta", "lemma": "pasta", "role": "semantic_object", "meaning": "food_item"},
        ])
        score = _object_alignment_score("no_class_frame", tmpl_no_classes, roles, self.lexicon)
        self.assertAlmostEqual(score, 0.0)


class E3RerankerUOLIntegrationTests(unittest.TestCase):
    """rerank() with token_roles produces correct outputs."""

    def setUp(self) -> None:
        self.linker = FrameLinker()
        self.reranker = E3CandidateReranker()
        self.lexicon = {
            "rain": frozenset({"weather_phenomenon"}),
            "today": frozenset({"temporal_descriptor"}),
            "tell": frozenset({"communication_action", "narrative_content"}),
            "story": frozenset({"narrative_content"}),
            "eat": frozenset({"consumption_action", "food_item"}),
            "pasta": frozenset({"food_item"}),
        }

    def _make_roles(self, entries: list[dict[str, str]]) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"index": i, "token": e["token"], "lemma": e["lemma"],
             "role": e["role"], "meaning": e.get("meaning", ""), "weight": 1.0}
            for i, e in enumerate(entries)
        )

    def test_uol_rerank_produces_valid_candidates(self) -> None:
        """rerank() with token_roles returns ScoredCandidate list."""
        candidates = self.linker.score(
            ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        roles = self._make_roles([
            {"token": "tell", "lemma": "tell", "role": "main_predicate", "meaning": "tell:communication"},
            {"token": "a", "lemma": "a", "role": "determiner", "meaning": "nominal_scope"},
            {"token": "story", "lemma": "story", "role": "semantic_object", "meaning": "narrative_content"},
        ])
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
            token_roles=roles,
        )
        self.assertGreater(len(reranked), 0)
        for sc in reranked:
            self.assertIsInstance(sc, ScoredCandidate)

    def test_uol_explanation_includes_predicate(self) -> None:
        """With token_roles, the top candidate's explanation uses 'predicate='."""
        candidates = self.linker.score(
            ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        roles = self._make_roles([
            {"token": "tell", "lemma": "tell", "role": "main_predicate", "meaning": "tell:communication"},
            {"token": "a", "lemma": "a", "role": "determiner", "meaning": "nominal_scope"},
            {"token": "story", "lemma": "story", "role": "semantic_object", "meaning": "narrative_content"},
        ])
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
            token_roles=roles,
        )
        self.assertGreater(len(reranked), 0)
        top = reranked[0]
        self.assertIn("predicate=", top.rerank_explanation,
                      f"top candidate '{top.frame_id}' lacks 'predicate=': {top.rerank_explanation}")

    def test_uol_object_alignment_appears_when_match(self) -> None:
        """When semantic_object matches frame classes, 'object=' appears."""
        candidates = self.linker.score(
            ("eat", "pasta"), self.lexicon,
            is_question_like=False, is_request_like=False,
        )
        roles = self._make_roles([
            {"token": "eat", "lemma": "eat", "role": "main_predicate", "meaning": "eat:consumption"},
            {"token": "pasta", "lemma": "pasta", "role": "semantic_object", "meaning": "food_item"},
        ])
        reranked = self.reranker.rerank(
            candidates, ("eat", "pasta"), self.lexicon,
            is_question_like=False, is_request_like=False,
            token_roles=roles,
        )
        has_object = any("object=" in sc.rerank_explanation for sc in reranked)
        self.assertTrue(has_object, "no explanation entry contains 'object='")

    def test_uol_rerank_allows_empty_token_roles(self) -> None:
        """Empty token_roles tuple should not crash."""
        candidates = self.linker.score(
            ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
            token_roles=(),
        )
        self.assertGreater(len(reranked), 0)


class E3RerankerIntegrationTests(unittest.TestCase):
    """End-to-end reranker integration with FrameLinker."""

    def setUp(self) -> None:
        self.linker = FrameLinker()
        self.reranker = E3CandidateReranker()
        self.lexicon = {
            "rain": frozenset({"weather_phenomenon"}),
            "today": frozenset({"temporal_descriptor"}),
            "tell": frozenset({"communication_action", "narrative_content"}),
            "story": frozenset({"narrative_content"}),
        }

    def test_reranker_accepts_empty_candidates(self) -> None:
        result = self.reranker.rerank([], ("foo", "bar"), self.lexicon)
        self.assertEqual(result, [])

    def test_reranker_preserves_single_candidate(self) -> None:
        candidates = self.linker.score(
            ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        self.assertGreaterEqual(len(reranked), 1)
        self.assertEqual(reranked[0].frame_id, "story")

    def test_reranker_explanation_is_readable(self) -> None:
        candidates = self.linker.score(
            ("tell", "a", "story", "rain"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story", "rain"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        self.assertGreater(len(reranked), 0)
        for sc in reranked:
            self.assertIn("rule=", sc.rerank_explanation)

    def test_score_property_returns_rerank_score(self) -> None:
        """ScoredCandidate.score must return rerank_score so the router
        uses the learned reranker output for routing decisions."""
        candidates = self.linker.score(
            ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        reranked = self.reranker.rerank(
            candidates, ("tell", "a", "story"), self.lexicon,
            is_question_like=False, is_request_like=True,
        )
        for sc in reranked:
            self.assertEqual(sc.score, sc.rerank_score)

    def test_rerank_score_can_rescue_below_threshold_rule_score(self) -> None:
        """When rerank_score exceeds threshold but rule_score does not,
        ScoredCandidate.score must reflect rerank_score so routing accepts it."""
        from melm.appliance.assistant_frame_ranker import ScoredCandidate
        candidate = ScoredCandidate(
            frame_id="test_frame",
            intent="test_intent",
            rule_score=0.30,
            rerank_score=0.85,
            rerank_explanation="boosted by predicate alignment",
            threshold=0.40,
        )
        self.assertGreaterEqual(candidate.score, candidate.threshold)


if __name__ == "__main__":
    unittest.main()
