"""Minimal morpheme/root/meaning inference MVP.

The goal is not to model English perfectly. This module tests whether a tiny
weighted morpheme/root graph can infer useful meaning features for held-out
words and route utterances into answer/action/update modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z']+")
QUESTION_STARTS = {"what", "where", "who", "why", "how", "when", "does", "do", "is"}
COMMAND_STARTS = {"please", "kindly"}
NOUNY_SUFFIXES = ("ness",)
INFLECTIONAL_SUFFIXES = ("ing", "ed", "s")
FEATURE_LABELS = {
    "ability": "can/able",
    "abstract_quality": "abstract quality",
    "activity": "activity",
    "aid": "help/aid",
    "after": "after",
    "arrival": "arrival",
    "benefit": "benefit",
    "before": "before",
    "building": "building",
    "change_location": "change location",
    "clarity": "clarity",
    "comprehension": "understanding",
    "concern": "concern",
    "creation": "creation",
    "desire": "desire",
    "error": "error",
    "excess": "excess",
    "fluid": "fluid",
    "giving": "giving",
    "greeting": "greeting",
    "hardness": "stone/hardness",
    "home": "home",
    "hope": "hope",
    "incorrectness": "incorrectness",
    "joy": "joy",
    "knowledge_gain": "knowledge gain",
    "knowledge_transfer": "knowledge transfer",
    "learning": "learning",
    "light": "light",
    "manner": "manner",
    "mineral": "stone/mineral",
    "motion": "motion",
    "negation": "not/negation",
    "person": "person",
    "play": "play",
    "positive_valence": "positive feeling",
    "protection": "protection",
    "reading": "reading",
    "repeat": "again/repeat",
    "safety": "safety",
    "teaching": "teaching",
    "text": "text",
    "visibility": "visibility",
    "water": "water",
    "writing": "writing",
}


@dataclass(frozen=True)
class MeaningComponent:
    component_id: str
    form: str
    kind: str
    gloss: str
    features: dict[str, float]
    confidence: float
    ipa: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class LexemeEntry:
    word: str
    components: tuple[str, ...]
    gloss: str
    features: dict[str, float]
    confidence: float
    notes: str = ""


@dataclass(frozen=True)
class WordCase:
    word: str
    expected_components: tuple[str, ...]
    expected_features: dict[str, float]
    forbidden_features: dict[str, float]
    expected_gloss_contains: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class UtteranceCase:
    utterance: str
    expected_intent: str
    target_word: str
    expected_features: dict[str, float]
    expected_response_kind: str
    category: str


@dataclass(frozen=True)
class MeaningCorpus:
    components: dict[str, MeaningComponent]
    lexemes: dict[str, LexemeEntry]
    word_cases: tuple[WordCase, ...]
    utterance_cases: tuple[UtteranceCase, ...]


@dataclass(frozen=True)
class MeaningInference:
    word: str
    component_ids: tuple[str, ...]
    features: dict[str, float]
    gloss: str
    confidence: float
    unknown_spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class UtteranceInference:
    utterance: str
    intent: str
    response_kind: str
    target_word: str
    meaning: MeaningInference | None


@dataclass(frozen=True)
class CaseResult:
    item: str
    category: str
    expected: str
    predicted: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeaningMvpReport:
    word_cases: int
    word_accuracy: float
    utterance_cases: int
    utterance_accuracy: float
    overall_accuracy: float
    word_results: tuple[CaseResult, ...]
    utterance_results: tuple[CaseResult, ...]


class MeaningInferencer:
    """Tiny weighted component graph for word and utterance inference."""

    def __init__(self, corpus: MeaningCorpus) -> None:
        self.corpus = corpus
        self._components_by_form = {
            component.form: component
            for component in corpus.components.values()
        }
        self._lexemes_by_word = dict(corpus.lexemes)
        self._segment_forms = sorted(
            {
                component.form
                for component in corpus.components.values()
                if component.kind == "root"
            },
            key=lambda item: (-len(item), item),
        )

    def infer_word(self, word: str) -> MeaningInference:
        normalized = _normalize_word(word)
        if normalized in self._lexemes_by_word:
            entry = self._lexemes_by_word[normalized]
            component_ids = self._expand_component_ids(entry.components)
            features = dict(entry.features)
            return MeaningInference(
                word=normalized,
                component_ids=component_ids,
                features=_round_features(features),
                gloss=entry.gloss,
                confidence=entry.confidence,
            )

        component_ids, unknown_spans = self._decompose(normalized)
        features = self._features_for_components(component_ids)
        confidence = _mean(
            self.corpus.components[component_id].confidence
            for component_id in component_ids
            if component_id in self.corpus.components
        )
        return MeaningInference(
            word=normalized,
            component_ids=tuple(component_ids),
            features=_round_features(features),
            gloss=_gloss_from_features(features),
            confidence=round(confidence, 3),
            unknown_spans=tuple(unknown_spans),
        )

    def infer_utterance(self, utterance: str) -> UtteranceInference:
        tokens = [_normalize_word(token) for token in TOKEN_RE.findall(utterance.lower())]
        if not tokens:
            return UtteranceInference(
                utterance=utterance,
                intent="clarification_needed",
                response_kind="ask_clarification",
                target_word="",
                meaning=None,
            )

        intent = self._classify_intent(utterance, tokens)
        target_word = self._target_word(tokens, intent)
        meaning = self.infer_word(target_word) if target_word else None
        response_kind = self._response_kind(intent, meaning)
        return UtteranceInference(
            utterance=utterance,
            intent=intent,
            response_kind=response_kind,
            target_word=target_word,
            meaning=meaning,
        )

    def _classify_intent(self, utterance: str, tokens: list[str]) -> str:
        if tokens[:1] == ["by"] and "mean" in tokens:
            return "clarification"
        if "means" in tokens or ("mean" in tokens and tokens[:1] != ["what"]):
            return "knowledge_transfer"
        if utterance.strip().endswith("?") or tokens[0] in QUESTION_STARTS:
            if self._looks_like_bad_command(tokens):
                return "clarification_needed"
            return "question"
        if tokens[0] in COMMAND_STARTS:
            return "command"
        return "knowledge_transfer"

    def _target_word(self, tokens: list[str], intent: str) -> str:
        if intent == "command":
            return tokens[1] if len(tokens) > 1 else ""
        if intent == "question":
            if "does" in tokens and "mean" in tokens:
                index = tokens.index("does")
                return tokens[index + 1] if index + 1 < len(tokens) else ""
            return _first_content_token(tokens)
        if intent == "knowledge_transfer":
            if len(tokens) >= 2 and tokens[0] in {"a", "an", "the"}:
                return tokens[1]
            if "means" in tokens:
                return tokens[max(0, tokens.index("means") - 1)]
        if intent == "clarification":
            return tokens[1] if len(tokens) > 1 else ""
        if intent == "clarification_needed":
            return _first_unknownish_token(tokens, self)
        return ""

    def _response_kind(self, intent: str, meaning: MeaningInference | None) -> str:
        if intent == "command":
            if meaning is None or _is_nominalized(meaning.word):
                return "ask_clarification"
            return "action_frame"
        if intent == "question":
            return "meaning_answer"
        if intent == "knowledge_transfer":
            return "store_candidate"
        if intent == "clarification":
            return "clarification_update"
        return "ask_clarification"

    def _looks_like_bad_command(self, tokens: list[str]) -> bool:
        if tokens and tokens[0] in {"could", "can", "would"} and "you" in tokens:
            target = _first_unknownish_token(tokens, self)
            return bool(target and _is_nominalized(target))
        return False

    def _decompose(self, word: str) -> tuple[list[str], list[str]]:
        component_ids: list[str] = []
        unknown_spans: list[str] = []
        remaining = word
        trailing_suffix_ids: list[str] = []

        for prefix in _components_of_kind(self.corpus, "prefix"):
            after_prefix = remaining[len(prefix.form):]
            if (
                remaining.startswith(prefix.form)
                and len(remaining) > len(prefix.form) + 2
                and self._has_plausible_base(after_prefix)
            ):
                component_ids.append(prefix.component_id)
                remaining = after_prefix
                break

        remaining, trailing_suffix_ids = self._strip_known_suffixes(remaining)
        for suffix in INFLECTIONAL_SUFFIXES:
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
                remaining = remaining[: -len(suffix)]
                break

        if remaining in self._lexemes_by_word:
            component_ids.extend(self._expand_component_ids(self._lexemes_by_word[remaining].components))
        else:
            component_ids.extend(self._segment_remainder(remaining, unknown_spans))

        for suffix_id in reversed(trailing_suffix_ids):
            if suffix_id not in component_ids:
                component_ids.append(suffix_id)

        return component_ids, unknown_spans

    def _strip_known_suffixes(self, text: str) -> tuple[str, list[str]]:
        suffix_ids: list[str] = []
        remaining = text
        changed = True
        while changed:
            changed = False
            for suffix in _components_of_kind(self.corpus, "suffix"):
                base = self._suffix_base(remaining, suffix.form)
                if base is None:
                    continue
                suffix_ids.append(suffix.component_id)
                remaining = base
                changed = True
                break
        return remaining, suffix_ids

    def _has_known_base(self, text: str) -> bool:
        if text in self._lexemes_by_word:
            return True
        return any(
            component.kind == "root" and text == component.form
            for component in self.corpus.components.values()
        )

    def _has_plausible_base(self, text: str) -> bool:
        if self._has_known_base(text):
            return True
        return any(
            component.kind == "root" and text.startswith(component.form)
            for component in self.corpus.components.values()
        )

    def _suffix_base(self, word: str, suffix: str) -> str | None:
        if not word.endswith(suffix) or len(word) <= len(suffix) + 2:
            return None
        base = word[: -len(suffix)]
        if self._has_known_base(base):
            return base
        if self._has_known_base(base + "e"):
            return base + "e"
        return None

    def _segment_remainder(self, text: str, unknown_spans: list[str]) -> list[str]:
        selected: list[str] = []
        cursor = 0
        while cursor < len(text):
            match = None
            for form in self._segment_forms:
                if text.startswith(form, cursor):
                    match = form
                    break
            if match is None:
                cursor += 1
                continue
            component = self._components_by_form[match]
            if component.component_id not in selected:
                selected.append(component.component_id)
            cursor += len(match)

        covered = "".join(self.corpus.components[item].form for item in selected if item in self.corpus.components)
        if not selected or (covered and len(covered) < max(2, len(text) // 2)):
            unknown_spans.append(text)
        return selected

    def _expand_component_ids(self, component_ids: tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for component_id in component_ids:
            if component_id in self.corpus.components:
                expanded.append(component_id)
        return tuple(expanded)

    def _features_for_components(self, component_ids: list[str]) -> dict[str, float]:
        features: dict[str, float] = {}
        has_negation = False
        for component_id in component_ids:
            component = self.corpus.components.get(component_id)
            if component is None:
                continue
            if component.kind == "prefix" and component.form == "un":
                has_negation = True
            for feature, value in component.features.items():
                features[feature] = features.get(feature, 0.0) + (value * component.confidence)

        if has_negation:
            for feature in ("positive_valence", "desire", "willingness", "benefit"):
                if feature in features:
                    features[feature] *= -0.7
        if {"root:wil", "root:cuma"} <= set(component_ids):
            features["greeting"] = max(features.get("greeting", 0.0), 0.75)
        return features


def load_meaning_corpus(path: str | Path) -> MeaningCorpus:
    components: dict[str, MeaningComponent] = {}
    lexemes: dict[str, LexemeEntry] = {}
    word_cases: list[WordCase] = []
    utterance_cases: list[UtteranceCase] = []

    for record in _read_jsonl(Path(path)):
        schema = record.get("schema")
        if schema == "melm.meaning_component.v1":
            component = MeaningComponent(
                component_id=str(record["component_id"]),
                form=str(record["form"]),
                kind=str(record["kind"]),
                gloss=str(record["gloss"]),
                features=_float_map(record.get("features") or {}),
                confidence=float(record.get("confidence", 1.0)),
                ipa=str(record["ipa"]) if record.get("ipa") else None,
                notes=str(record.get("notes") or ""),
            )
            components[component.component_id] = component
        elif schema == "melm.lexeme.v1":
            entry = LexemeEntry(
                word=_normalize_word(str(record["word"])),
                components=tuple(str(item) for item in record.get("components") or ()),
                gloss=str(record["gloss"]),
                features=_float_map(record.get("features") or {}),
                confidence=float(record.get("confidence", 1.0)),
                notes=str(record.get("notes") or ""),
            )
            lexemes[entry.word] = entry
        elif schema == "melm.word_case.v1":
            word_cases.append(
                WordCase(
                    word=_normalize_word(str(record["word"])),
                    expected_components=tuple(str(item) for item in record.get("expected_components") or ()),
                    expected_features=_float_map(record.get("expected_features") or {}),
                    forbidden_features=_float_map(record.get("forbidden_features") or {}),
                    expected_gloss_contains=tuple(str(item) for item in record.get("expected_gloss_contains") or ()),
                    category=str(record.get("category") or "overall"),
                )
            )
        elif schema == "melm.utterance_case.v1":
            utterance_cases.append(
                UtteranceCase(
                    utterance=str(record["utterance"]),
                    expected_intent=str(record["expected_intent"]),
                    target_word=_normalize_word(str(record["target_word"])),
                    expected_features=_float_map(record.get("expected_features") or {}),
                    expected_response_kind=str(record["expected_response_kind"]),
                    category=str(record.get("category") or "overall"),
                )
            )
        else:
            raise ValueError(f"Unsupported meaning MVP schema: {schema!r}")

    return MeaningCorpus(
        components=components,
        lexemes=lexemes,
        word_cases=tuple(word_cases),
        utterance_cases=tuple(utterance_cases),
    )


def evaluate_meaning_mvp(corpus: MeaningCorpus) -> MeaningMvpReport:
    inferencer = MeaningInferencer(corpus)
    word_results = tuple(_evaluate_word_case(inferencer, case) for case in corpus.word_cases)
    utterance_results = tuple(
        _evaluate_utterance_case(inferencer, case)
        for case in corpus.utterance_cases
    )
    word_accuracy = _accuracy(word_results)
    utterance_accuracy = _accuracy(utterance_results)
    total_cases = len(word_results) + len(utterance_results)
    total_passed = sum(result.passed for result in word_results + utterance_results)
    return MeaningMvpReport(
        word_cases=len(word_results),
        word_accuracy=word_accuracy,
        utterance_cases=len(utterance_results),
        utterance_accuracy=utterance_accuracy,
        overall_accuracy=total_passed / total_cases if total_cases else 0.0,
        word_results=word_results,
        utterance_results=utterance_results,
    )


def _evaluate_word_case(inferencer: MeaningInferencer, case: WordCase) -> CaseResult:
    inference = inferencer.infer_word(case.word)
    expected_components = set(case.expected_components)
    component_hit = expected_components <= set(inference.component_ids)
    feature_hit = _features_meet_thresholds(inference.features, case.expected_features)
    forbidden_hit = _features_below_caps(inference.features, case.forbidden_features)
    gloss_hit = all(item in inference.gloss for item in case.expected_gloss_contains)
    passed = component_hit and feature_hit and forbidden_hit and gloss_hit
    return CaseResult(
        item=case.word,
        category=case.category,
        expected=", ".join(case.expected_components),
        predicted=", ".join(inference.component_ids),
        passed=passed,
        details={
            "features": inference.features,
            "gloss": inference.gloss,
            "confidence": inference.confidence,
            "component_hit": component_hit,
            "feature_hit": feature_hit,
            "forbidden_hit": forbidden_hit,
            "gloss_hit": gloss_hit,
        },
    )


def _evaluate_utterance_case(inferencer: MeaningInferencer, case: UtteranceCase) -> CaseResult:
    inference = inferencer.infer_utterance(case.utterance)
    meaning = inference.meaning
    intent_hit = inference.intent == case.expected_intent
    target_hit = inference.target_word == case.target_word
    response_hit = inference.response_kind == case.expected_response_kind
    feature_hit = bool(meaning) and _features_meet_thresholds(meaning.features, case.expected_features)
    passed = intent_hit and target_hit and response_hit and feature_hit
    return CaseResult(
        item=case.utterance,
        category=case.category,
        expected=f"{case.expected_intent}/{case.expected_response_kind}/{case.target_word}",
        predicted=f"{inference.intent}/{inference.response_kind}/{inference.target_word}",
        passed=passed,
        details={
            "features": meaning.features if meaning else {},
            "components": meaning.component_ids if meaning else (),
            "intent_hit": intent_hit,
            "target_hit": target_hit,
            "response_hit": response_hit,
            "feature_hit": feature_hit,
        },
    )


def _features_meet_thresholds(features: dict[str, float], expected: dict[str, float]) -> bool:
    return all(features.get(feature, 0.0) >= threshold for feature, threshold in expected.items())


def _features_below_caps(features: dict[str, float], forbidden: dict[str, float]) -> bool:
    return all(features.get(feature, 0.0) <= cap for feature, cap in forbidden.items())


def _gloss_from_features(features: dict[str, float]) -> str:
    if not features:
        return "unknown meaning"
    top = sorted(features.items(), key=lambda item: (-abs(item[1]), item[0]))[:4]
    labels = [
        ("not " + _feature_label(feature)) if value < 0 else _feature_label(feature)
        for feature, value in top
    ]
    return "inferred meaning: " + ", ".join(labels)


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _components_of_kind(corpus: MeaningCorpus, kind: str) -> list[MeaningComponent]:
    return [
        component
        for component in corpus.components.values()
        if component.kind == kind
    ]


def _first_content_token(tokens: list[str]) -> str:
    for token in tokens:
        if token not in {"what", "does", "do", "mean", "means", "the", "a", "an"}:
            return token
    return ""


def _first_unknownish_token(tokens: list[str], inferencer: MeaningInferencer) -> str:
    for token in tokens:
        if token in {"could", "can", "would", "you", "please", "the", "a", "an"}:
            continue
        if token in inferencer.corpus.lexemes or inferencer.infer_word(token).component_ids:
            return token
    return ""


def _is_nominalized(word: str) -> bool:
    return word.endswith(NOUNY_SUFFIXES)


def _normalize_word(word: str) -> str:
    normalized = re.sub(r"[^a-z']", "", word.lower())
    if normalized.endswith("'s") and len(normalized) > 3:
        normalized = normalized[:-2]
    return normalized


def _round_features(features: dict[str, float]) -> dict[str, float]:
    return {
        feature: round(value, 3)
        for feature, value in sorted(features.items())
        if abs(value) >= 0.001
    }


def _float_map(value: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _accuracy(results: tuple[CaseResult, ...]) -> float:
    return sum(result.passed for result in results) / len(results) if results else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
            records.append(record)
    return records
