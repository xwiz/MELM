"""Symbolic state tracking for BabyLM-style entity tracking cases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from melm.memory.schema import Event

from .multiple_choice import MultipleChoiceCase


BOX_CONTENTS_RE = re.compile(
    r"Box (?P<box>\d+) contains (?P<contents>.*?)(?=, Box \d+ contains|$)"
)
MOVE_RE = re.compile(
    r"Move (?P<items>.+?) from Box (?P<src>\d+) to Box (?P<dst>\d+)$"
)
REMOVE_RE = re.compile(r"Remove (?P<items>.+?) from Box (?P<src>\d+)$")
PUT_RE = re.compile(r"Put (?P<items>.+?) into Box (?P<dst>\d+)$")
QUERY_RE = re.compile(r"Box (?P<box>\d+) contains\s*$")


@dataclass(frozen=True)
class BoxStatePrediction:
    case_id: str
    category: str
    label_index: int
    predicted_index: int | None
    correct: bool
    predicted_contents: tuple[str, ...] | None


@dataclass(frozen=True)
class BoxStateTrackingReport:
    cases: int
    correct: int
    accuracy: float
    abstentions: int
    by_category: dict[str, "BoxStateTrackingReport"]
    predictions: list[BoxStatePrediction]


class BoxStateTracker:
    """Resolve the box contents implied by entity-tracking prompts."""

    def __init__(self, boxes: dict[int, set[str]] | None = None) -> None:
        self.boxes: dict[int, set[str]] = {
            int(box): set(contents)
            for box, contents in (boxes or {}).items()
        }

    @classmethod
    def from_prompt(cls, prompt: str) -> tuple["BoxStateTracker", int | None]:
        events, query_box = entity_tracking_events_from_prompt(prompt)
        tracker = cls.from_events(events)
        return tracker, query_box

    @classmethod
    def from_events(cls, events: list[Event] | tuple[Event, ...]) -> "BoxStateTracker":
        tracker = cls()
        for event in sorted(events, key=lambda item: (item.time_index, item.event_id)):
            operation = event.metadata.get("operation", "")
            if operation == "initial_contents":
                box = int(event.metadata["box"])
                tracker.boxes[box] = set(_items_from_metadata(event))
            elif operation == "move":
                items = _items_from_metadata(event)
                src = int(event.metadata["source_box"])
                dst = int(event.metadata["target_box"])
                tracker.boxes.setdefault(src, set()).difference_update(items)
                tracker.boxes.setdefault(dst, set()).update(items)
            elif operation == "remove":
                items = _items_from_metadata(event)
                src = int(event.metadata["source_box"])
                tracker.boxes.setdefault(src, set()).difference_update(items)
            elif operation == "put":
                items = _items_from_metadata(event)
                dst = int(event.metadata["target_box"])
                tracker.boxes.setdefault(dst, set()).update(items)
        return tracker

    def apply(self, operation: str) -> bool:
        move = MOVE_RE.match(operation)
        if move:
            items = _parse_items(move.group("items"))
            src = int(move.group("src"))
            dst = int(move.group("dst"))
            self.boxes.setdefault(src, set()).difference_update(items)
            self.boxes.setdefault(dst, set()).update(items)
            return True

        remove = REMOVE_RE.match(operation)
        if remove:
            items = _parse_items(remove.group("items"))
            src = int(remove.group("src"))
            self.boxes.setdefault(src, set()).difference_update(items)
            return True

        put = PUT_RE.match(operation)
        if put:
            items = _parse_items(put.group("items"))
            dst = int(put.group("dst"))
            self.boxes.setdefault(dst, set()).update(items)
            return True

        return False

    def contents(self, box: int) -> tuple[str, ...]:
        return tuple(sorted(self.boxes.get(box, set())))


def predict_entity_tracking_option(case: MultipleChoiceCase) -> int | None:
    """Predict the answer option using explicit box-state resolution."""

    tracker, query_box = BoxStateTracker.from_prompt(case.prompt)
    if query_box is None:
        return None
    expected_contents = tracker.contents(query_box)
    for index, option in enumerate(case.options):
        if _parse_option(option) == expected_contents:
            return index
    return None


def evaluate_entity_tracking_symbolic(cases: list[MultipleChoiceCase]) -> BoxStateTrackingReport:
    """Evaluate BabyLM entity tracking with deterministic state algebra."""

    predictions: list[BoxStatePrediction] = []
    for case in cases:
        tracker, query_box = BoxStateTracker.from_prompt(case.prompt)
        predicted_contents = tracker.contents(query_box) if query_box is not None else None
        predicted_index = None
        if predicted_contents is not None:
            for index, option in enumerate(case.options):
                if _parse_option(option) == predicted_contents:
                    predicted_index = index
                    break
        predictions.append(
            BoxStatePrediction(
                case_id=case.case_id,
                category=case.category,
                label_index=case.label_index,
                predicted_index=predicted_index,
                correct=predicted_index == case.label_index,
                predicted_contents=predicted_contents,
            )
        )
    return _summarize_predictions(predictions)


def entity_tracking_events_from_prompt(
    prompt: str,
    *,
    case_id: str = "case",
) -> tuple[list[Event], int | None]:
    """Parse a BabyLM entity-tracking prompt into explicit state events."""

    parts = [part.strip() for part in prompt.split(".") if part.strip()]
    if not parts:
        return [], None

    event_specs: list[dict[str, object]] = []
    for box, items in sorted(_parse_initial_boxes(parts[0]).items()):
        event_specs.append(
            {
                "source_span": f"Box {box} contains {_format_items(items)}.",
                "action_or_state": "contains",
                "objects": tuple(items),
                "location": f"Box {box}",
                "metadata": {
                    "operation": "initial_contents",
                    "box": str(box),
                    "items": _items_metadata(items),
                },
            }
        )

    for operation_text in parts[1:-1]:
        spec = _operation_event_spec(operation_text)
        if spec is not None:
            event_specs.append(spec)

    query = QUERY_RE.search(parts[-1])
    query_box = int(query.group("box")) if query else None
    events: list[Event] = []
    for index, spec in enumerate(event_specs):
        event_id = f"{case_id}_e{index:03d}"
        events.append(
            Event(
                event_id=event_id,
                source_span=str(spec["source_span"]),
                time_index=index,
                action_or_state=str(spec["action_or_state"]),
                objects=tuple(spec["objects"]),
                location=spec.get("location"),
                previous_event_id=f"{case_id}_e{index - 1:03d}" if index > 0 else None,
                next_event_id=(
                    f"{case_id}_e{index + 1:03d}"
                    if index + 1 < len(event_specs)
                    else None
                ),
                metadata=dict(spec["metadata"]),
            )
        )
    return events, query_box


def _summarize_predictions(predictions: list[BoxStatePrediction]) -> BoxStateTrackingReport:
    buckets: dict[str, list[BoxStatePrediction]] = defaultdict(list)
    for prediction in predictions:
        buckets[prediction.category].append(prediction)

    correct = sum(1 for prediction in predictions if prediction.correct)
    abstentions = sum(1 for prediction in predictions if prediction.predicted_index is None)
    return BoxStateTrackingReport(
        cases=len(predictions),
        correct=correct,
        accuracy=correct / len(predictions) if predictions else 0.0,
        abstentions=abstentions,
        by_category={
            category: _summarize_bucket(bucket)
            for category, bucket in sorted(buckets.items())
        },
        predictions=predictions,
    )


def _summarize_bucket(predictions: list[BoxStatePrediction]) -> BoxStateTrackingReport:
    correct = sum(1 for prediction in predictions if prediction.correct)
    abstentions = sum(1 for prediction in predictions if prediction.predicted_index is None)
    return BoxStateTrackingReport(
        cases=len(predictions),
        correct=correct,
        accuracy=correct / len(predictions) if predictions else 0.0,
        abstentions=abstentions,
        by_category={},
        predictions=[],
    )


def _parse_initial_boxes(text: str) -> dict[int, set[str]]:
    boxes: dict[int, set[str]] = {}
    for match in BOX_CONTENTS_RE.finditer(text):
        boxes[int(match.group("box"))] = set(_parse_items(match.group("contents")))
    return boxes


def _operation_event_spec(operation: str) -> dict[str, object] | None:
    move = MOVE_RE.match(operation)
    if move:
        items = _parse_items(move.group("items"))
        src = move.group("src")
        dst = move.group("dst")
        return {
            "source_span": operation + ".",
            "action_or_state": "move",
            "objects": items,
            "location": f"Box {dst}",
            "metadata": {
                "operation": "move",
                "source_box": src,
                "target_box": dst,
                "items": _items_metadata(items),
            },
        }

    remove = REMOVE_RE.match(operation)
    if remove:
        items = _parse_items(remove.group("items"))
        src = remove.group("src")
        return {
            "source_span": operation + ".",
            "action_or_state": "remove",
            "objects": items,
            "location": f"Box {src}",
            "metadata": {
                "operation": "remove",
                "source_box": src,
                "items": _items_metadata(items),
            },
        }

    put = PUT_RE.match(operation)
    if put:
        items = _parse_items(put.group("items"))
        dst = put.group("dst")
        return {
            "source_span": operation + ".",
            "action_or_state": "put",
            "objects": items,
            "location": f"Box {dst}",
            "metadata": {
                "operation": "put",
                "target_box": dst,
                "items": _items_metadata(items),
            },
        }
    return None


def _parse_option(text: str) -> tuple[str, ...]:
    return _parse_items(text.rstrip("."))


def _parse_items(text: str) -> tuple[str, ...]:
    normalized = text.lower().strip().rstrip(".")
    if normalized == "nothing":
        return ()
    normalized = re.sub(r"\bthe\b", "", normalized)
    return tuple(
        sorted(
            item.strip(" ,")
            for item in normalized.split(" and ")
            if item.strip(" ,")
        )
    )


def _items_metadata(items: tuple[str, ...] | set[str]) -> str:
    return "\t".join(sorted(items))


def _items_from_metadata(event: Event) -> tuple[str, ...]:
    value = event.metadata.get("items", "")
    return tuple(item for item in value.split("\t") if item)


def _format_items(items: tuple[str, ...] | set[str]) -> str:
    sorted_items = sorted(items)
    if not sorted_items:
        return "nothing"
    return "the " + " and the ".join(sorted_items)
