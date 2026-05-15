"""Public long-term memory benchmark adapters.

The first adapter targets LoCoMo because it is small enough to run in a
dependency-light test loop and exposes gold evidence dialog ids. The evaluation
here is retrieval-grounded: systems must retrieve the sessions containing the
gold evidence, not generate final natural-language answers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Iterable
import json


LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
TOKEN_RE = re.compile(r"[a-z0-9']+")
SESSION_KEY_RE = re.compile(r"session_(\d+)$")
DIALOG_ID_RE = re.compile(r"D(\d+):(\d+)")
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "single_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}
TEMPORAL_TERMS = {
    "after",
    "before",
    "first",
    "last",
    "latest",
    "recent",
    "recently",
    "when",
    "earlier",
    "later",
}


@dataclass(frozen=True)
class PublicMemoryDocument:
    doc_id: str
    sample_id: str
    session_id: str
    session_number: int
    time_index: int
    date_time: str
    raw_text: str
    observation: str
    session_summary: str
    event_summary: str
    speakers: tuple[str, ...]

    def text_for(self, mode: str) -> str:
        parts = [self.date_time, self.raw_text]
        if "observation" in mode:
            parts.append(self.observation)
        if "summary" in mode:
            parts.append(self.session_summary)
        if "event_summary" in mode:
            parts.append(self.event_summary)
        return "\n".join(part for part in parts if part)

    @property
    def entities(self) -> set[str]:
        entities = set(self.speakers)
        entities.add(self.session_id.lower())
        entities.update(_capitalized_terms(self.raw_text))
        entities.update(_capitalized_terms(self.observation))
        entities.update(_capitalized_terms(self.session_summary))
        return {entity.lower() for entity in entities if entity}


@dataclass(frozen=True)
class PublicMemoryQuestion:
    question_id: str
    sample_id: str
    question: str
    answer: str
    has_gold_answer: bool
    category_id: int
    category: str
    evidence_dialog_ids: tuple[str, ...]
    evidence_session_ids: tuple[str, ...]


@dataclass(frozen=True)
class PublicMemoryBenchmark:
    name: str
    source_path: str
    documents_by_sample: dict[str, list[PublicMemoryDocument]]
    questions: list[PublicMemoryQuestion]

    @property
    def documents(self) -> list[PublicMemoryDocument]:
        docs: list[PublicMemoryDocument] = []
        for sample_docs in self.documents_by_sample.values():
            docs.extend(sample_docs)
        return docs


@dataclass(frozen=True)
class PublicMemoryPrediction:
    question_id: str
    sample_id: str
    category: str
    evidence_count: int
    retrieved_doc_ids: tuple[str, ...]
    recall: float
    hit: bool
    full_evidence: bool


@dataclass(frozen=True)
class PublicArchitectureReport:
    architecture: str
    description: str
    questions: int
    mean_recall: float
    hit_at_k: float
    full_evidence_at_k: float
    predictions: list[PublicMemoryPrediction]
    by_category: dict[str, "PublicArchitectureReport"] | None = None


@dataclass(frozen=True)
class PublicMemoryComparisonReport:
    benchmark: str
    source_path: str
    k: int
    documents: int
    questions: int
    architectures: dict[str, PublicArchitectureReport]
    advantage_vs_mem0_arch: float
    advantage_vs_zep_arch: float
    advantage_vs_memgpt_arch: float
    gate_passed: bool


@dataclass(frozen=True)
class PublicContextBudgetPrediction:
    question_id: str
    sample_id: str
    category: str
    architecture: str
    token_budget: int
    packed_doc_ids: tuple[str, ...]
    packed_tokens: int
    evidence_recall: float
    answer_token_recall: float | None
    answer_supported: bool | None


@dataclass(frozen=True)
class PublicContextBudgetReport:
    architecture: str
    token_budget: int
    questions: int
    answer_questions: int
    evidence_recall: float
    answer_support_rate: float
    mean_answer_token_recall: float
    predictions: list[PublicContextBudgetPrediction]


@dataclass(frozen=True)
class RetrievalCandidate:
    doc_id: str
    score: float


def load_locomo_public_memory_benchmark(path: str | Path) -> PublicMemoryBenchmark:
    """Load LoCoMo into session-level public memory retrieval records."""

    source_path = Path(path)
    samples = json.loads(source_path.read_text(encoding="utf-8"))
    documents_by_sample: dict[str, list[PublicMemoryDocument]] = {}
    questions: list[PublicMemoryQuestion] = []

    for sample in samples:
        sample_id = str(sample["sample_id"])
        conversation = sample["conversation"]
        observation = sample.get("observation", {})
        session_summary = sample.get("session_summary", {})
        event_summary = sample.get("event_summary", {})
        session_keys = sorted(
            (key for key in conversation if SESSION_KEY_RE.fullmatch(key)),
            key=lambda key: int(SESSION_KEY_RE.fullmatch(key).group(1)),  # type: ignore[union-attr]
        )
        documents: list[PublicMemoryDocument] = []
        speakers = tuple(
            str(conversation.get(key, "")).lower()
            for key in ("speaker_a", "speaker_b")
            if conversation.get(key)
        )
        for time_index, session_key in enumerate(session_keys, start=1):
            session_number = int(SESSION_KEY_RE.fullmatch(session_key).group(1))  # type: ignore[union-attr]
            raw_text = "\n".join(_turn_text(turn) for turn in conversation[session_key])
            documents.append(
                PublicMemoryDocument(
                    doc_id=f"{sample_id}::S{session_number}",
                    sample_id=sample_id,
                    session_id=f"S{session_number}",
                    session_number=session_number,
                    time_index=time_index,
                    date_time=str(conversation.get(f"{session_key}_date_time", "")),
                    raw_text=raw_text,
                    observation=str(observation.get(f"{session_key}_observation", "")),
                    session_summary=str(session_summary.get(f"{session_key}_summary", "")),
                    event_summary=str(event_summary.get(f"events_{session_key}", "")),
                    speakers=speakers,
                )
            )
        documents_by_sample[sample_id] = documents

        for index, qa in enumerate(sample.get("qa", []), start=1):
            evidence_dialog_ids = tuple(str(item) for item in qa.get("evidence", ()) if item)
            evidence_sessions = tuple(
                dict.fromkeys(
                    _dialog_to_session_id(dialog_id)
                    for dialog_id in evidence_dialog_ids
                    if _dialog_to_session_id(dialog_id)
                )
            )
            category_id = int(qa.get("category", 0))
            questions.append(
                PublicMemoryQuestion(
                    question_id=f"{sample_id}::q{index}",
                    sample_id=sample_id,
                    question=str(qa["question"]),
                    answer=str(qa.get("answer") or ""),
                    has_gold_answer=bool(qa.get("answer")),
                    category_id=category_id,
                    category=CATEGORY_NAMES.get(category_id, f"category_{category_id}"),
                    evidence_dialog_ids=evidence_dialog_ids,
                    evidence_session_ids=tuple(
                        f"{sample_id}::{session_id}" for session_id in evidence_sessions
                    ),
                )
            )

    return PublicMemoryBenchmark(
        name="locomo_session_evidence_retrieval",
        source_path=str(source_path),
        documents_by_sample=documents_by_sample,
        questions=questions,
    )


ARCHITECTURE_CONTEXT_MODES = {
    "vector_rag": "raw",
    "mem0_additive_arch": "raw",
    "memgpt_tiered_arch": "summary_raw",
    "zep_temporal_graph_arch": "observation",
    "melm_memory_os": "melm_adaptive",
}


def evaluate_public_memory_architectures(
    benchmark: PublicMemoryBenchmark,
    *,
    k: int = 5,
    max_questions: int | None = None,
    include_event_summaries: bool = True,
) -> PublicMemoryComparisonReport:
    """Evaluate local memory architecture implementations on public evidence retrieval."""

    questions = [
        question
        for question in benchmark.questions
        if question.evidence_session_ids
    ]
    if max_questions is not None:
        questions = questions[:max_questions]

    melm_description = (
        "MELM Memory OS: raw turns plus extracted observations, session summaries, "
        "event summaries, temporal query routing, and question-guided raw snippets"
        if include_event_summaries
        else "MELM Memory OS: raw turns plus extracted observations, session summaries, temporal query routing, and question-guided raw snippets"
    )
    retriever_factories = {
        "vector_rag": (
            "bag-of-words cosine over raw session text",
            lambda docs: LexicalSessionRetriever(docs, text_mode="raw", scoring="cosine"),
        ),
        "mem0_additive_arch": (
            "ADD-only memory proxy: raw session memories scored by BM25 plus entity boosts",
            lambda docs: LexicalSessionRetriever(docs, text_mode="raw", scoring="bm25", entity_boost=True),
        ),
        "memgpt_tiered_arch": (
            "tiered-memory proxy: raw session text plus session summaries",
            lambda docs: LexicalSessionRetriever(docs, text_mode="summary", scoring="bm25", entity_boost=True),
        ),
        "zep_temporal_graph_arch": (
            "temporal-graph proxy: extracted observations plus graph/session neighbor expansion",
            lambda docs: LexicalSessionRetriever(
                docs,
                text_mode="observation",
                scoring="bm25",
                entity_boost=True,
                temporal_expansion=True,
            ),
        ),
        "melm_memory_os": (
            melm_description,
            lambda docs: LexicalSessionRetriever(
                docs,
                text_mode="observation_summary_event_summary" if include_event_summaries else "observation_summary",
                scoring="bm25",
                entity_boost=True,
                temporal_expansion=True,
                multi_index=False,
            ),
        ),
    }

    reports: dict[str, PublicArchitectureReport] = {}
    for architecture, (description, factory) in retriever_factories.items():
        predictions: list[PublicMemoryPrediction] = []
        retrievers = {
            sample_id: factory(documents)
            for sample_id, documents in benchmark.documents_by_sample.items()
        }
        for question in questions:
            retriever = retrievers[question.sample_id]
            retrieved = retriever.retrieve(question.question, k=k)
            predictions.append(_prediction(question, retrieved))
        reports[architecture] = _summarize_public_predictions(
            architecture,
            description,
            predictions,
        )

    melm_recall = reports["melm_memory_os"].mean_recall
    return PublicMemoryComparisonReport(
        benchmark=benchmark.name,
        source_path=benchmark.source_path,
        k=k,
        documents=len(benchmark.documents),
        questions=len(questions),
        architectures=reports,
        advantage_vs_mem0_arch=melm_recall - reports["mem0_additive_arch"].mean_recall,
        advantage_vs_zep_arch=melm_recall - reports["zep_temporal_graph_arch"].mean_recall,
        advantage_vs_memgpt_arch=melm_recall - reports["memgpt_tiered_arch"].mean_recall,
        gate_passed=(
            melm_recall > reports["mem0_additive_arch"].mean_recall
            and melm_recall > reports["zep_temporal_graph_arch"].mean_recall
            and melm_recall > reports["memgpt_tiered_arch"].mean_recall
        ),
    )


def evaluate_public_context_budget(
    benchmark: PublicMemoryBenchmark,
    comparison: PublicMemoryComparisonReport,
    *,
    token_budget: int = 1200,
) -> dict[str, PublicContextBudgetReport]:
    """Evaluate how much evidence/answer content fits after retrieval under a token budget."""

    questions = {
        question.question_id: question
        for question in benchmark.questions
        if question.evidence_session_ids
    }
    documents = {document.doc_id: document for document in benchmark.documents}
    reports: dict[str, PublicContextBudgetReport] = {}
    for architecture, architecture_report in comparison.architectures.items():
        context_mode = ARCHITECTURE_CONTEXT_MODES.get(architecture, "raw")
        predictions: list[PublicContextBudgetPrediction] = []
        for retrieval_prediction in architecture_report.predictions:
            question = questions[retrieval_prediction.question_id]
            packed_doc_ids, packed_text, packed_tokens = _pack_context(
                retrieval_prediction.retrieved_doc_ids,
                documents,
                mode=context_mode,
                query=question.question,
                token_budget=token_budget,
            )
            expected = set(question.evidence_session_ids)
            evidence_recall = (
                len(expected & set(packed_doc_ids)) / len(expected)
                if expected
                else 1.0
            )
            answer_token_recall: float | None = None
            answer_supported: bool | None = None
            if question.has_gold_answer:
                answer_token_recall = _answer_token_recall(question.answer, packed_text)
                answer_supported = (
                    _normalized_text(question.answer) in _normalized_text(packed_text)
                    or answer_token_recall >= 0.80
                )
            predictions.append(
                PublicContextBudgetPrediction(
                    question_id=question.question_id,
                    sample_id=question.sample_id,
                    category=question.category,
                    architecture=architecture,
                    token_budget=token_budget,
                    packed_doc_ids=packed_doc_ids,
                    packed_tokens=packed_tokens,
                    evidence_recall=evidence_recall,
                    answer_token_recall=answer_token_recall,
                    answer_supported=answer_supported,
                )
            )
        reports[architecture] = _summarize_context_budget(
            architecture,
            token_budget,
            predictions,
        )
    return reports


class LexicalSessionRetriever:
    """Dependency-light retrieval engine for public benchmark comparison."""

    def __init__(
        self,
        documents: list[PublicMemoryDocument],
        *,
        text_mode: str,
        scoring: str,
        entity_boost: bool = False,
        temporal_expansion: bool = False,
        multi_index: bool = False,
    ) -> None:
        self.documents = tuple(documents)
        self.text_mode = text_mode
        self.scoring = scoring
        self.entity_boost = entity_boost
        self.temporal_expansion = temporal_expansion
        self.multi_index = multi_index
        self._doc_by_id = {document.doc_id: document for document in self.documents}
        self._texts = {document.doc_id: document.text_for(text_mode) for document in self.documents}
        self._vectors = {doc_id: _token_counts(text) for doc_id, text in self._texts.items()}
        self._bm25 = BM25Index(self._texts)
        self._raw_bm25 = BM25Index({document.doc_id: document.raw_text for document in self.documents})
        self._observation_bm25 = BM25Index({document.doc_id: document.observation for document in self.documents})
        self._summary_bm25 = BM25Index({document.doc_id: document.session_summary for document in self.documents})

    def retrieve(self, query: str, *, k: int) -> tuple[str, ...]:
        if self.multi_index:
            scored = self._multi_index_scores(query)
        elif self.scoring == "bm25":
            scored = self._bm25_scores(query)
        elif self.scoring == "cosine":
            scored = self._cosine_scores(query)
        else:
            raise ValueError(f"unsupported scoring mode: {self.scoring!r}")
        if self.entity_boost:
            scored = self._with_entity_boost(query, scored)
        ranked = sorted(scored, key=lambda item: (item.score, item.doc_id), reverse=True)
        if self.temporal_expansion and _has_temporal_query(query):
            ranked = self._expand_temporal_neighbors(ranked, query=query)
        return tuple(candidate.doc_id for candidate in ranked[:k])

    def _bm25_scores(self, query: str) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(doc_id, score)
            for doc_id, score in self._bm25.score(query).items()
        ]

    def _cosine_scores(self, query: str) -> list[RetrievalCandidate]:
        query_vec = _token_counts(query)
        return [
            RetrievalCandidate(doc_id, _cosine(query_vec, doc_vec))
            for doc_id, doc_vec in self._vectors.items()
        ]

    def _multi_index_scores(self, query: str) -> list[RetrievalCandidate]:
        raw = self._raw_bm25.score(query)
        observation = self._observation_bm25.score(query)
        summary = self._summary_bm25.score(query)
        scores: dict[str, float] = {}
        for doc_id in self._texts:
            scores[doc_id] = (
                0.52 * raw.get(doc_id, 0.0)
                + 0.28 * observation.get(doc_id, 0.0)
                + 0.28 * summary.get(doc_id, 0.0)
            )
        return [RetrievalCandidate(doc_id, score) for doc_id, score in scores.items()]

    def _with_entity_boost(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        query_entities = _query_entities(query)
        query_terms = set(_tokens(query))
        boosted: list[RetrievalCandidate] = []
        for candidate in candidates:
            document = self._doc_by_id[candidate.doc_id]
            entity_overlap = len(document.entities & query_entities)
            speaker_overlap = len(set(document.speakers) & query_terms)
            boosted.append(
                RetrievalCandidate(
                    candidate.doc_id,
                    candidate.score + entity_overlap * 0.35 + speaker_overlap * 0.20,
                )
            )
        return boosted

    def _expand_temporal_neighbors(
        self,
        ranked: list[RetrievalCandidate],
        *,
        query: str,
    ) -> list[RetrievalCandidate]:
        by_id = {candidate.doc_id: candidate for candidate in ranked}
        query_lower = query.lower()
        window = 1
        if "first" in query_lower or "last" in query_lower or "latest" in query_lower:
            window = 2
        for candidate in ranked[:3]:
            document = self._doc_by_id[candidate.doc_id]
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                neighbor_id = f"{document.sample_id}::S{document.session_number + delta}"
                if neighbor_id not in self._doc_by_id:
                    continue
                neighbor_score = candidate.score * (0.92 if abs(delta) == 1 else 0.82)
                if neighbor_id not in by_id or neighbor_score > by_id[neighbor_id].score:
                    by_id[neighbor_id] = RetrievalCandidate(neighbor_id, neighbor_score)
        return sorted(by_id.values(), key=lambda item: (item.score, item.doc_id), reverse=True)


class BM25Index:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        self.term_freqs = {doc_id: _token_counts(text) for doc_id, text in documents.items()}
        self.doc_freqs: Counter[str] = Counter()
        for counts in self.term_freqs.values():
            self.doc_freqs.update(counts.keys())
        self.doc_lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.term_freqs.items()}
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

    def score(self, query: str) -> dict[str, float]:
        query_terms = _tokens(query)
        scores = {doc_id: 0.0 for doc_id in self.documents}
        total_docs = max(1, len(self.documents))
        for term in query_terms:
            doc_freq = self.doc_freqs.get(term, 0)
            if not doc_freq:
                continue
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for doc_id, counts in self.term_freqs.items():
                freq = counts.get(term, 0)
                if not freq:
                    continue
                doc_length = self.doc_lengths[doc_id]
                denominator = freq + 1.2 * (
                    1 - 0.75 + 0.75 * doc_length / max(1.0, self.avg_doc_length)
                )
                scores[doc_id] += idf * (freq * 2.2) / denominator
        return scores


def _prediction(question: PublicMemoryQuestion, retrieved_doc_ids: tuple[str, ...]) -> PublicMemoryPrediction:
    expected = set(question.evidence_session_ids)
    observed = set(retrieved_doc_ids)
    matched = expected & observed
    recall = len(matched) / len(expected) if expected else 1.0
    return PublicMemoryPrediction(
        question_id=question.question_id,
        sample_id=question.sample_id,
        category=question.category,
        evidence_count=len(expected),
        retrieved_doc_ids=retrieved_doc_ids,
        recall=recall,
        hit=bool(matched),
        full_evidence=expected.issubset(observed),
    )


def _summarize_public_predictions(
    architecture: str,
    description: str,
    predictions: list[PublicMemoryPrediction],
) -> PublicArchitectureReport:
    by_category_predictions: dict[str, list[PublicMemoryPrediction]] = defaultdict(list)
    for prediction in predictions:
        by_category_predictions[prediction.category].append(prediction)
    report = _report_without_categories(architecture, description, predictions)
    return PublicArchitectureReport(
        architecture=report.architecture,
        description=report.description,
        questions=report.questions,
        mean_recall=report.mean_recall,
        hit_at_k=report.hit_at_k,
        full_evidence_at_k=report.full_evidence_at_k,
        predictions=predictions,
        by_category={
            category: _report_without_categories(architecture, description, category_predictions)
            for category, category_predictions in sorted(by_category_predictions.items())
        },
    )


def _report_without_categories(
    architecture: str,
    description: str,
    predictions: list[PublicMemoryPrediction],
) -> PublicArchitectureReport:
    count = len(predictions)
    if not count:
        return PublicArchitectureReport(architecture, description, 0, 0.0, 0.0, 0.0, [], None)
    return PublicArchitectureReport(
        architecture=architecture,
        description=description,
        questions=count,
        mean_recall=sum(prediction.recall for prediction in predictions) / count,
        hit_at_k=sum(1 for prediction in predictions if prediction.hit) / count,
        full_evidence_at_k=sum(1 for prediction in predictions if prediction.full_evidence) / count,
        predictions=predictions,
        by_category=None,
    )


def _turn_text(turn: dict) -> str:
    parts = [str(turn.get("speaker", "")), ":", str(turn.get("text", ""))]
    if turn.get("blip_caption"):
        parts.extend([" Image:", str(turn["blip_caption"])])
    return " ".join(part for part in parts if part)


def _dialog_to_session_id(dialog_id: str) -> str:
    match = DIALOG_ID_RE.fullmatch(dialog_id.replace("(", "").replace(")", ""))
    return f"S{match.group(1)}" if match else ""


def _tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in TOKEN_RE.findall(text.lower())]


def _token_counts(text: str) -> Counter[str]:
    return Counter(_tokens(text))


def _normalize_token(token: str) -> str:
    if token.endswith("'s") and len(token) > 3:
        return token[:-2]
    return token


def _capitalized_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        if match.group(0).lower() not in {"The", "This", "That", "There"}
    }


def _query_entities(query: str) -> set[str]:
    return _capitalized_terms(query) | {
        token
        for token in _tokens(query)
        if token.startswith("d") and token[1:].isdigit()
    }


def _has_temporal_query(query: str) -> bool:
    terms = set(_tokens(query))
    return bool(terms & TEMPORAL_TERMS)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _pack_context(
    doc_ids: Iterable[str],
    documents: dict[str, PublicMemoryDocument],
    *,
    mode: str,
    query: str,
    token_budget: int,
) -> tuple[tuple[str, ...], str, int]:
    packed_ids: list[str] = []
    packed_parts: list[str] = []
    used = 0
    for doc_id in doc_ids:
        document = documents.get(doc_id)
        if document is None:
            continue
        text = _context_text(document, mode, query=query)
        tokens = _tokens(text)
        if not tokens:
            continue
        if used + len(tokens) > token_budget and packed_ids:
            continue
        if used + len(tokens) > token_budget:
            tokens = tokens[:token_budget]
            text = " ".join(tokens)
        used += len(tokens)
        packed_ids.append(doc_id)
        packed_parts.append(text)
        if used >= token_budget:
            break
    return tuple(packed_ids), "\n".join(packed_parts), used


def _answer_token_recall(answer: str, context: str) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_terms = set(_tokens(context))
    return sum(1 for token in answer_tokens if token in context_terms) / len(answer_tokens)


def _normalized_text(text: str) -> str:
    return " ".join(_tokens(text))


def _summarize_context_budget(
    architecture: str,
    token_budget: int,
    predictions: list[PublicContextBudgetPrediction],
) -> PublicContextBudgetReport:
    answer_predictions = [
        prediction
        for prediction in predictions
        if prediction.answer_supported is not None
    ]
    return PublicContextBudgetReport(
        architecture=architecture,
        token_budget=token_budget,
        questions=len(predictions),
        answer_questions=len(answer_predictions),
        evidence_recall=(
            sum(prediction.evidence_recall for prediction in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        answer_support_rate=(
            sum(1 for prediction in answer_predictions if prediction.answer_supported)
            / len(answer_predictions)
            if answer_predictions
            else 0.0
        ),
        mean_answer_token_recall=(
            sum(prediction.answer_token_recall or 0.0 for prediction in answer_predictions)
            / len(answer_predictions)
            if answer_predictions
            else 0.0
        ),
        predictions=predictions,
    )


def _context_text(document: PublicMemoryDocument, mode: str, *, query: str = "") -> str:
    if mode == "raw":
        return "\n".join(part for part in (document.date_time, document.raw_text) if part)
    if mode == "summary":
        return "\n".join(part for part in (document.date_time, document.session_summary) if part)
    if mode == "summary_raw":
        return "\n".join(
            part
            for part in (document.date_time, document.session_summary, document.raw_text)
            if part
        )
    if mode == "observation":
        return "\n".join(part for part in (document.date_time, document.observation) if part)
    if mode == "observation_summary_event_summary":
        return "\n".join(
            part
            for part in (
                document.date_time,
                document.observation,
                document.session_summary,
                document.event_summary,
            )
            if part
        )
    if mode == "melm_adaptive":
        compact = "\n".join(
            part
            for part in (
                document.date_time,
                document.observation,
                document.session_summary,
                document.event_summary,
            )
            if part
        )
        snippets = _question_guided_snippets(document.raw_text, query=query, max_tokens=160)
        return "\n".join(part for part in (compact, snippets) if part)
    if mode == "observation_summary":
        return "\n".join(
            part
            for part in (document.date_time, document.observation, document.session_summary)
            if part
        )
    return document.text_for(mode)


def _question_guided_snippets(raw_text: str, *, query: str, max_tokens: int) -> str:
    query_terms = set(_tokens(query))
    if not query_terms:
        return ""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(lines):
        line_terms = set(_tokens(line))
        overlap = len(query_terms & line_terms)
        speaker_bonus = 0.25 if any(term in line_terms for term in query_terms if term[:1].isupper()) else 0.0
        if overlap:
            scored.append((overlap + speaker_bonus, -index, line))
    if not scored:
        return ""
    selected: list[str] = []
    used = 0
    for _, _, line in sorted(scored, reverse=True):
        line_tokens = _tokens(line)
        if used + len(line_tokens) > max_tokens and selected:
            continue
        if used + len(line_tokens) > max_tokens:
            line = " ".join(line_tokens[: max_tokens - used])
            line_tokens = _tokens(line)
        selected.append(line)
        used += len(line_tokens)
        if used >= max_tokens:
            break
    return "\n".join(selected)
