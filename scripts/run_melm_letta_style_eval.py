"""Run MELM Appliance on a Letta Evals-style JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance import MelmAppliance


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/letta_eval/locomo_letta_dataset.jsonl")
    parser.add_argument("--memory", default="artifacts/letta_eval/locomo_memory.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--out-json", default="reports/melm_letta_style_eval.json")
    parser.add_argument("--out-md", default="reports/melm_letta_style_eval.md")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.dataset))
    full_appliance = MelmAppliance.load_jsonl(args.memory)
    appliances_by_sample: dict[str, MelmAppliance] = {}
    predictions = []
    for record in records:
        question = record.get("metadata", {}).get("question") or record["input"]
        sample_id = str(record.get("metadata", {}).get("sample_id", ""))
        if sample_id not in appliances_by_sample:
            sample_records = [
                memory
                for memory in full_appliance.records
                if memory.metadata.get("sample_id") == sample_id
            ]
            appliances_by_sample[sample_id] = MelmAppliance(sample_records or full_appliance.records)
        answer = appliances_by_sample[sample_id].answer(
            question,
            k=args.k,
            token_budget=args.token_budget,
        )
        strict_score = _contains_score(answer.answer, record["ground_truth"])
        token_recall = _answer_token_recall(answer.answer, record["ground_truth"])
        answer_supported = 1.0 if strict_score or token_recall >= 0.80 else 0.0
        expected_evidence = [
            str(item)
            for item in record.get("metadata", {}).get("evidence_session_ids", [])
        ]
        citation_recall = _evidence_recall(answer.citations, expected_evidence)
        retrieval_recall = _evidence_recall(answer.retrieved_ids, expected_evidence)
        predictions.append(
            {
                "id": record.get("id"),
                "question_id": record.get("metadata", {}).get("question_id"),
                "category": record.get("metadata", {}).get("category"),
                "input": record["input"],
                "ground_truth": record["ground_truth"],
                "answer": answer.answer,
                "contains_score": strict_score,
                "answer_token_recall": token_recall,
                "answer_supported": answer_supported,
                "citation_evidence_recall": citation_recall,
                "retrieval_evidence_recall": retrieval_recall,
                "confidence": answer.confidence,
                "citations": list(answer.citations),
                "retrieved_ids": list(answer.retrieved_ids),
                "evidence_session_ids": expected_evidence,
                "context_tokens": answer.context_tokens,
            }
        )
    payload = {
        "target": "melm_appliance_local",
        "dataset": args.dataset,
        "memory": args.memory,
        "k": args.k,
        "token_budget": args.token_budget,
        "samples": len(predictions),
        "contains_accuracy": (
            sum(item["contains_score"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "answer_support_rate": (
            sum(item["answer_supported"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "mean_answer_token_recall": (
            sum(item["answer_token_recall"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "mean_citation_evidence_recall": (
            sum(item["citation_evidence_recall"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "mean_retrieval_evidence_recall": (
            sum(item["retrieval_evidence_recall"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "by_category": _by_category(predictions),
        "predictions": predictions,
    }
    _write_outputs(payload, Path(args.out_json), Path(args.out_md))

    print("MELM Letta-style eval")
    print(f"- samples={payload['samples']}")
    print(f"- contains_accuracy={payload['contains_accuracy']:.2%}")
    print(f"- answer_support_rate={payload['answer_support_rate']:.2%}")
    print(f"- mean_answer_token_recall={payload['mean_answer_token_recall']:.2%}")
    print(f"- mean_citation_evidence_recall={payload['mean_citation_evidence_recall']:.2%}")
    print(f"- out_json={args.out_json}")
    print(f"- out_md={args.out_md}")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
    return records


def _contains_score(answer: str, ground_truth: str) -> float:
    normalized_answer = _norm(answer)
    normalized_truth = _norm(ground_truth)
    if not normalized_truth:
        return 0.0
    if normalized_truth in normalized_answer:
        return 1.0
    truth_tokens = normalized_truth.split()
    if not truth_tokens:
        return 0.0
    answer_terms = set(normalized_answer.split())
    return 1.0 if all(token in answer_terms for token in truth_tokens) else 0.0


def _answer_token_recall(answer: str, ground_truth: str) -> float:
    answer_terms = set(_norm(answer).split())
    truth_terms = _content_terms(ground_truth)
    if not truth_terms:
        truth_terms = _norm(ground_truth).split()
    if not truth_terms:
        return 0.0
    return len(set(truth_terms) & answer_terms) / len(set(truth_terms))


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in _norm(text).split()
        if token not in STOPWORDS or any(ch.isdigit() for ch in token)
    ]


def _evidence_recall(actual_ids: object, expected_ids: list[str]) -> float:
    expected = {str(item) for item in expected_ids if str(item)}
    if not expected:
        return 0.0
    actual = {str(item) for item in actual_ids or []}
    return len(actual & expected) / len(expected)


def _norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _by_category(predictions: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for prediction in predictions:
        buckets.setdefault(str(prediction.get("category", "unknown")), []).append(prediction)
    return {
        category: {
            "samples": len(items),
            "contains_accuracy": sum(item["contains_score"] for item in items) / len(items),
            "answer_support_rate": sum(item["answer_supported"] for item in items) / len(items),
            "mean_answer_token_recall": sum(item["answer_token_recall"] for item in items) / len(items),
            "mean_citation_evidence_recall": sum(item["citation_evidence_recall"] for item in items) / len(items),
            "mean_retrieval_evidence_recall": sum(item["retrieval_evidence_recall"] for item in items) / len(items),
        }
        for category, items in sorted(buckets.items())
    }


def _write_outputs(payload: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict) -> str:
    lines = [
        "# MELM Letta-Style Evaluation",
        "",
        f"Target: `{payload['target']}`",
        f"Dataset: `{payload['dataset']}`",
        f"Memory: `{payload['memory']}`",
        f"Samples: `{payload['samples']}`",
        f"Contains accuracy: `{payload['contains_accuracy']:.2%}`",
        f"Answer support rate: `{payload['answer_support_rate']:.2%}`",
        f"Mean answer-token recall: `{payload['mean_answer_token_recall']:.2%}`",
        f"Mean citation evidence recall: `{payload['mean_citation_evidence_recall']:.2%}`",
        f"Mean retrieval evidence recall: `{payload['mean_retrieval_evidence_recall']:.2%}`",
        "",
        "## By Category",
        "",
        "| Category | Samples | Contains accuracy | Answer support | Token recall | Citation evidence recall | Retrieval evidence recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in payload["by_category"].items():
        lines.append(
            "| "
            f"{category} | "
            f"{row['samples']} | "
            f"{row['contains_accuracy']:.2%} | "
            f"{row['answer_support_rate']:.2%} | "
            f"{row['mean_answer_token_recall']:.2%} | "
            f"{row['mean_citation_evidence_recall']:.2%} | "
            f"{row['mean_retrieval_evidence_recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            "This local run consumes the same JSONL dataset exported for Letta Evals. "
            "It is not an official Letta comparison; it gives MELM's score on the "
            "shared eval pack before running a real Letta target.",
            "",
            "Interpretation note: contains accuracy grades the final extractive answer. "
            "Evidence recall grades whether the memory layer surfaced the gold sessions. "
            "A large gap between the two means the retrieval appliance is ahead of the "
            "current no-LLM answer composer.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
