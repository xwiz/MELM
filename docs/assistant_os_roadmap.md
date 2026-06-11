# Local Assistant OS — Roadmap

Authority: Part IV §18 of `docs/local_assistant_os_mvp_plan_v2.md`.
Completed work moves to git history, not to an ever-growing list.

## Implementation milestones

| Phase | Scope and deliverable | Exit gate |
|-------|----------------------|-----------|
| **M0 — Recover truth** (days) | Fix D1-D5, preserve current behavior, commit the tree, add CI, label internal fixtures honestly | `pytest`, `pi-smoke`, and `shortcut-audit` green in CI; v1 superseded; changing a debug label does not fail a behavioral gate |
| **M1 — Contract kernel** (week 1) | Contract registry; schemas for UOL, semantic classes, frames, route decisions, evidence, answer plans, stance, and model manifests; adapters around current rules/templates; integrated perf harness | Current regression suite passes through validators; initial 60-case normative UOL set passes; incompatible versions fail closed; dev + Pi benchmark JSON produced |
| **M2 — Meaning substrate** (weeks 2-3) | Factored lexicon, semantic-class registry, frame registry, capability manifest, atlas/learning ledgers; migrate media, weather, and story vocabulary from code to seeded data | 100% route agreement on existing regression cases for migrated families; deleting a seed row changes behavior predictably; no user/atlas write can enable a capability or action |
| **M3 — Learning vertical slice** (weeks 3-5) | Detect → define → quarantine → test → promote → reuse → correct for nouns, modifiers, and one verb class; restart persistence | Sealed ≥60-word dictionary set: ≥80% correct next-turn use and retention; zero reserved-namespace promotions; zero capability grants; correction/rollback trace queryable end to end |
| **M4 — Bounded generation** (parallel after M1; weeks 2-5) | Benchmark small decoder candidates; wire `AnswerPlan`, constrained decoding, verifier, and template fallback; fine-tune only after zero-shot decision artifact | On Pi: report tok/s/TTFT/RSS; 0 unsafe applied outputs; 100% fallback on verifier failure; model accepted on ≥70% of eligible rendering cases and retains ≥95% of required constraints; go/no-go recorded before training spend |
| **M5 — Learned frame linking** (weeks 5-7) | E3 reranks rule-generated candidates using UOL + semantic support; K0 remains gate owner | On sealed set: top-3 frame recall ≥95%, accepted-route precision ≥98%, zero false-local safety cases; no regression on supported minimal pairs |
| **M6 — Learned UOL families** (weeks 6-9) | E2 enters shadow mode, then takes ownership only of qualified construction families such as particles and coordination | Per promoted family on ≥50 sealed examples: slot/role F1 beats rule owner, no safety regression, p95 parse latency within budget; ownership flip and rollback are data-only |
| **M7 — v0.2 integration** (weeks 9-10) | Default-on qualified experts, fallbacks, blind/user-derived evaluation, Pi package and hardware report | All hard safety invariants green; dictionary and external NLU bars reported; p95 TTFT <1.5 s and RSS <1.2 GB; if E4 ships, >30 tok/s, otherwise local-generation claims are explicitly reduced |

## Critical path

**M0 → M1 → M2 → M3 → M5 → M6 → M7**

**Parallel falsifier:** M4 starts after M1 (generation feasibility measured
without waiting for learned NLU). A no-go removes E4 from v0.2; it does not
block the lexicon, atlas, UOL, or routing thesis.

## Milestone rules

- A hard safety gate is binary: privacy leak, unconfirmed action, unsupported
  applied factual claim, incompatible contract, or capability escalation must
  be zero.
- Quality targets report denominator, confidence interval, dataset hash, and
  whether the set is authored, sealed, user-derived, or external.
- No phase exits on component metrics alone; its end-to-end path and fallback
  must also pass.
- A model improvement may increase accepted coverage; it may not weaken the
  verifier, membrane, capability manifest, or action gate.
- Each milestone produces one reviewable artifact directory and one rollback
  point. New packaging commands remain frozen until M3 is green.

## Completed phases

Completed phases are recorded in git history and removed from this file.
No completed-slices list is maintained (v1's pattern is discontinued).
