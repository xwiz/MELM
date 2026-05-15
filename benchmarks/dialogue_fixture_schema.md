# Dialogue Benchmark Fixture Schema

Dialogue benchmarks use three JSONL files.

## Events

Each line is a `melm.event.v1` object:

- `event_id`: stable unique ID.
- `source_span`: source text for the event.
- `time_index`: integer ordering key.
- `actors`: list of normalized actors.
- `action_or_state`: normalized action/state label.
- `objects`: list of salient objects/places/entities.
- `location`: optional location string.
- `previous_event_id` / `next_event_id`: optional temporal links.
- `causal_links`: optional list of supporting prior event IDs.
- `metadata`: optional string map for source references.

## Recall Cases

Each line is a `melm.episodic_case.v1` object:

- `query`: question to ask.
- `expected_event_id`: event that must appear in retrieved evidence.
- `category`: evaluation bucket such as `direct`, `temporal_after`, `temporal_before`, `causal_source`, `entity_conflict`, or `witness`.

## Evidence Cases

Each line is a `melm.evidence_case.v1` object:

- `query`: question to ask.
- `expected_event_id`: event that must appear for answerable cases, or `null` for no-answer cases.
- `category`: positive or negative evaluation bucket.
- `story_id`: optional grouping key used for calibration/evaluation splits.

Use `null` rather than an empty string for unanswerable evidence cases.

## Commands

```powershell
python scripts\validate_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
python scripts\run_authored_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
```

## Annotated Transcript Source

For transcript-derived work, use a single annotation JSONL file containing:

- `melm.transcript_turn.v1` records for source turns;
- `melm.event.v1` records for extracted event/state annotations;
- `melm.episodic_case.v1` records for positive recall probes;
- `melm.evidence_case.v1` records for answerability/no-answer probes.

Every event in an annotated transcript should include `metadata.source_turn_id`
pointing to a `melm.transcript_turn.v1` record. This makes audit and error
analysis possible when an event-memory result looks suspicious.

Validation checks that temporal and causal links point to existing event IDs,
that linked events have plausible time order, and that transcript-sourced events
reference a valid source turn.

Compile an annotated transcript into the three benchmark fixture files with:

```powershell
python scripts\build_dialogue_benchmark_from_transcript.py --annotations benchmarks\sample_transcript_annotations.jsonl
python scripts\analyze_dialogue_benchmark.py --events benchmarks\sample_transcript_events.jsonl --recall-cases benchmarks\sample_transcript_recall_cases.jsonl --evidence-cases benchmarks\sample_transcript_evidence_cases.jsonl
```
