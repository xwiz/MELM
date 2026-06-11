# Support/Refunds External Blind Freeze Manifest

Dataset: `benchmarks\support_refunds_authored.jsonl`
Dataset SHA-256: `c2af00f6e61c7e428788113192971b99da1b33283e033393389f877ec79c6103`
Preregistration: `melm_support_refunds_external_blind_v0_1_prereg`
Preregistration SHA-256: `b594214bd9fb246a759302a86f484d3ae19c6784d70e49aef03cf8334ec44a7a`
Generated UTC: `2026-05-25T16:29:43+00:00`

- Schema validation passed: `True`
- Preregistration passed: `False`
- Frozen before scoring: `True`
- Turns/fact events/facts: `22` / `56` / `56`
- Guard cases: `13`
- Memory cases: `40`

## Guard Categories

| Category | Count |
|---|---:|
| `approval_required` | 1 |
| `duplicate_refund` | 1 |
| `fraud_flag` | 1 |
| `identity_missing_or_false` | 1 |
| `malformed_action` | 1 |
| `missing_order` | 1 |
| `not_delivered` | 1 |
| `outside_return_window` | 1 |
| `stale_approval` | 1 |
| `stale_state_trap` | 1 |
| `valid_high_value` | 1 |
| `valid_low_value` | 2 |

## Memory Categories

| Category | Count |
|---|---:|
| `approval_recall` | 5 |
| `contradiction_resolution` | 2 |
| `current_state` | 16 |
| `policy_recall` | 3 |
| `risk_recall` | 2 |
| `stale_state_update` | 2 |
| `unknown_order` | 10 |

## Validation

- metadata.dataset_id 'melm_support_refunds_authored_v0_1' does not match preregistration dataset_id 'melm_support_refunds_external_blind_v0_1'
- metadata.external_blind_batch must be true
- metadata.requires_external_blind_batch must be false
- turns=22 is below preregistered minimum 40
- fact_events=56 is below preregistered minimum 80
- guard_cases=13 is below preregistered minimum 24
- memory_cases=40 is below preregistered minimum 60
- metadata.annotator_count=0 is below preregistered minimum 2
- metadata.overlap_labeled_percent=0 is below preregistered minimum 20
- metadata.adjudication_record_path is required
- guard category 'approval_required' count 1 is below preregistered minimum 2
- guard category 'duplicate_refund' count 1 is below preregistered minimum 2
- guard category 'fraud_flag' count 1 is below preregistered minimum 2
- guard category 'identity_missing_or_false' count 1 is below preregistered minimum 2
- guard category 'malformed_action' count 1 is below preregistered minimum 2
- guard category 'missing_order' count 1 is below preregistered minimum 2
- guard category 'not_delivered' count 1 is below preregistered minimum 2
- guard category 'outside_return_window' count 1 is below preregistered minimum 2
- guard category 'stale_approval' count 1 is below preregistered minimum 2
- guard category 'stale_state_trap' count 1 is below preregistered minimum 2
- guard category 'valid_high_value' count 1 is below preregistered minimum 2
- memory category 'approval_recall' count 5 is below preregistered minimum 6
- memory category 'contradiction_resolution' count 2 is below preregistered minimum 4
- memory category 'current_state' count 16 is below preregistered minimum 20
- memory category 'risk_recall' count 2 is below preregistered minimum 4
- memory category 'stale_state_update' count 2 is below preregistered minimum 4
