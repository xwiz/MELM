# Support/Refunds External Blind Handoff

Use this handoff when giving the next support/refunds batch to an external
author or reviewer. The goal is to create a batch that can be scored once
without tuning MELM on it first.

## Files

- Preregistration: `benchmarks/support_refunds_external_blind_preregistration.json`
- Authoring protocol: `docs/support_refunds_dataset_authoring.md`
- Seed example only: `benchmarks/support_refunds_authored.jsonl`
- Freeze command: `python scripts\freeze_support_refund_dataset.py --dataset <path>`
- Scoring command: `python scripts\run_authored_support_refund_benchmark.py --dataset <path> --freeze-manifest <manifest>`

## External Author Instructions

Create a JSONL file with `melm.support_refunds.dataset.v1` metadata followed by
turns, fact events, guard cases, and memory cases. The metadata record must use:

```json
{
  "schema": "melm.support_refunds.dataset.v1",
  "dataset_id": "melm_support_refunds_external_blind_v0_1",
  "vertical": "support_refunds",
  "authoring_mode": "external_blind_batch",
  "authoring_protocol": "docs/support_refunds_dataset_authoring.md",
  "split": "external_blind",
  "requires_external_blind_batch": false,
  "external_blind_batch": true,
  "annotator_count": 2,
  "overlap_labeled_percent": 20,
  "adjudication_record_path": "benchmarks/support_refunds_external_blind_adjudication.jsonl",
  "known_limitations": []
}
```

Do not run MELM, vector RAG, temporal/entity RAG, or any rule-engine failure
analysis while writing the batch. Use the internal seed only to understand the
record shapes, not to copy scenarios.

## Freeze And Score

After the JSONL is complete, freeze it before scoring:

```powershell
python scripts\validate_support_refund_dataset.py --dataset benchmarks\support_refunds_external_blind.jsonl
python scripts\freeze_support_refund_dataset.py --dataset benchmarks\support_refunds_external_blind.jsonl
```

Then run the benchmark against the frozen hash:

```powershell
python scripts\run_authored_support_refund_benchmark.py `
  --dataset benchmarks\support_refunds_external_blind.jsonl `
  --freeze-manifest reports\support_refunds_external_blind_freeze_manifest.json `
  --out-json reports\melm_external_blind_support_refund_benchmark.json `
  --out-md reports\melm_external_blind_support_refund_benchmark.md
```

If the JSONL changes after freezing, the scoring command fails until a new
manifest is intentionally written. First-pass results should be reported even if
MELM fails the blind gates.
