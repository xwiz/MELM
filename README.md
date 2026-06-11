# MELM

## Authoritative Direction

The current product MVP is **MELM Local Assistant OS v0.1**:

```text
membrane policy
  + homeostatic state
  + autobiographical memory
  + user/self model
  + opportunity planner
  + local inventories
  + budgeted evidence runtime
  + local/tool/action/cloud triage
```

Use [docs/local_assistant_os_mvp_plan.md](docs/local_assistant_os_mvp_plan.md) as
the authoritative architecture and build plan. Root-level whitepaper,
validation, and research-review files are evidence sources only. They must not
be used to steer the current MVP if they imply a model-first, tokenizer-first,
or generic-chatbot direction.

## Root Document Roles

- [MELM_whitepaper.md](MELM_whitepaper.md): supporting research thesis and
  validation ladder. It is not the product build plan and cannot override the
  local assistant OS architecture.
- [MELM_validation_implementation_plan.md](MELM_validation_implementation_plan.md):
  supporting validation track for tokenizer, memory, model, and dialogue
  evidence. Promote only the parts that strengthen the assistant OS kernel.
- [MELM_implementation_research_review_2026.md](MELM_implementation_research_review_2026.md):
  historical de-risking review. Treat its model-first sections as background,
  not as current MVP direction.
- [docs/archive/](docs/archive/): non-authoritative historical drafts.

## Architecture Guardrail

MELM follows the `db-claw` / SemanticSQL lesson: typed frames own accepted
behavior. For chat, the frame is not allowed to be a shape inferred from the
current utterance alone. A candidate UOL relation becomes an accepted ChatFrame
only after it binds to the assistant's lived context: past chat sessions,
referenced objects, user/self memory, local inventories, current action state,
and the assistant world atlas that records semantic/relational strengths across
those experiences.

Phrase or vocabulary tables are allowed only as secondary meaning hints for
noun, verb, analogy, idiom, or domain interpretation. They must not create the
route, action, object, frame, or atlas edge by themselves. New chat capability
should be implemented as reusable UOL decomposition plus experience-grounded
ChatFrame transitions, then backed by tests and trace evidence.

The foundational order is:

```text
tokens
-> weighted functional roles and ranked predicates
-> subject/action/object/complement/recipient/modifier relations
-> UOL projection
-> experience lookup: past sessions + referenced objects + user/self memory
   + local inventories + action state + world-atlas relational strengths
-> ChatFrame construction from UOL plus experience-grounded atlas evidence
-> capability and policy match
-> local/tool/action/cloud/clarify/reject route
```

The assistant world atlas is the growth surface for MELM. It should accumulate
relations such as `user -> likes -> bedtime stories`, `weather_cache -> resolved
-> morning planning`, `story_gap -> caused -> inventory_build`, or
`trusted_contact -> requires -> explicit local setup` from event ledgers,
digests, inventories, and user facts. Static grammar can propose that "help me
grow in my career" is a goal/advice relation, but only atlas and memory evidence
can decide whether MELM has a local career-coaching frame, should ask a clarifier,
or should hand off to cloud.

`cloud_handoff` is a capability result, not evidence that the sentence was
unintelligible. Understood open-domain requests use
`weighted_functional_relation` and `understood_open_domain`; only fragments with
no defensible predicate structure remain `unknown_intent`. Browser/CLI debug
exposes candidate predicates, relation edges, parse coverage, and semantic
unknowns.

Concrete code smell: the primary intent classifier and post-route UOL slot
helpers must not call secondary phrase/marker-table helpers or
`_secondary_meaning_*`. Marker matching may explain/debug hints after token-role
composition or resolve a requested inventory item, but accepted behavior must
come from UOL/ChatFrame evidence.
In the runnable MVP this boundary is explicit: handled local, cached-tool,
device-action, and private-boundary routes must be owned by
`AssistantFrameRegistry` and expose `frame_registry=melm.assistant_frame_registry.v1`,
a non-empty `frame_id`, `source_policy=primary_uol_chatframe_only`, and
`secondary_hint_policy=debug_only_never_primary_route` in parse/debug/audit
evidence. A good answer without that frame ownership is not enough evidence.
Bare fragments are not accepted behavior: `your name` by itself must not become
an assistant-identity answer, while `what is your name?`, `tell me your name`,
or `your name?` may route only through identity token roles.
Assistant self-awareness routes have the same rule: identity questions must map
through `melm.identity_uol_composition.v1`, and status/ledger/next-step
questions must map through `melm.self_status_uol_composition.v1`. A duplicate
kernel phrase list for `who are you`, `what have you done`, `show your ledger`,
or similar surface text is drift. In the current MVP secondary hint tables are
concept-token hints only; request-shaped phrases belong in UOL/ChatFrame
composition tests, not in secondary lexical tables.
Identity support must be an identity-frame relation (`who/what/how` operator +
copula/modal + assistant deixis + allowed scope/capability roles), so variants
such as `who exactly are you on this device` work without allowing task frames
such as `who are you calling` to become self-model shortcuts.

The same guard applies outside self-awareness: bare domain words such as
`story`, `bedtime`, `doctor`, `medicine`, `dinner`, `naked`, `call`, or
`family` must not route by themselves. They need a compatible question, request,
object, target, or safety frame before they become accepted local behavior.
Concrete names and titles are stricter: a named trusted contact, song, story
model, or other inventory object must come from `UserModel`/local inventory data
for the current profile/session, never from a global hardcoded name list inside
the classifier. Relationship/category words such as `mom`, `caregiver`, or
`song` may act as grammar roles, but exact personal targets must be memory
evidence.
Generic capability or status words have the same limit: `what can you do to
improve my health` is a health-advice frame, not self-identity; `can you
explain cloud computing`, `can you explain story structure`, and `can you
explain weather systems` are unsupported/open-domain asks, not runtime-status,
story, or weather-cache routes. Private-cloud export asks such as `send my
favorite color to the cloud` must also surface through the Basic NLP -> UOL ->
ChatFrame debug map as a memory-policy route, not as a hidden pre-parse
shortcut. In debug output this must look like a private memory export frame
(`user / send / facts.favorite_color -> external_cloud_model`) with
`request_private_memory_cloud_boundary`, not a generic `recall / user_profile`
shortcut.
Weather, meal, and autobiographical memory have concrete anti-shortcut tests:
`what is weather` and `how does weather work` remain open-domain concept asks,
while `what is the weather` is a cache-capable observation frame; `can you cook
dinner` is not a meal recommendation, while `what can I cook for dinner` is a
user-choice meal frame; `what was the last thing I asked you` uses the shared
autobiographical UOL/ChatFrame scope, while statements such as `I dropped the
last thing yesterday` stay unsupported. The persistent kernel must call the
shared autobiographical frame composer, not an exact recall-phrase list.

Secondary hints must not hide uncertainty. A structurally understood turn may
still contain semantically unresolved content words, which remain visible in
`semantic_unknown_tokens` and can trigger opted-in research. For example,
`what is a story` is an open-domain copular question with a real UOL relation,
not a local story request; `story` remains a semantic research topic and the
route remains `cloud_handoff`.

Memory facts must keep ownership in their keys and policy metadata. For example,
`my child's school` is `facts.child_school` with `child_local` scope; it must not
collapse into generic `facts.school` or `profile.age` routing evidence.
Explicitly shareable memory is a separate policy case: a fact may cross cloud
only when its stored policy says `consent=true`, `local_only=false`, and
`cloud_eligible=true`. Baselines labelled vocabulary-only must remain
secondary-lexical baselines and must not call the UOL/ChatFrame classifier.

## Current Runnable Evidence

```powershell
python scripts\local_assistant_os_cli.py eval --json
python scripts\local_assistant_os_cli.py run-lifecycle --reset --json
python scripts\local_assistant_os_cli.py run-lifecycle-suite --json
python scripts\local_assistant_os_cli.py run-household-week --reset --json
python scripts\local_assistant_os_cli.py run-open-traces --reset --json
python scripts\local_assistant_os_cli.py run-transcript-replay --reset --json
python scripts\local_assistant_os_cli.py export-transcript-replay --db artifacts\local_assistant_os\assistant_v01.sqlite --out artifacts\local_assistant_os\event_ledger_transcript_replay.jsonl --json
python scripts\local_assistant_os_cli.py calibrate-event-ledger --db artifacts\local_assistant_os\assistant_v01.sqlite --work-dir artifacts\local_assistant_os\event_ledger_calibration --controls-json config\safe_lifecycle_controls.example.json --min-total-turns 4 --min-local-resolution-rate 0.5 --json
python scripts\local_assistant_os_cli.py v01-evidence-pack --db artifacts\local_assistant_os\assistant_v01.sqlite --work-dir artifacts\local_assistant_os\v01_evidence_pack --auto-lifecycle --json
python scripts\local_assistant_os_cli.py import-transcript-replay --input path\raw_chat.jsonl --out artifacts\local_assistant_os\imported_transcript_replay.jsonl --controls-json config\safe_lifecycle_controls.example.json --replace "Maya=<person_1>" --json
python scripts\local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks\sample_local_assistant_raw_transcript.jsonl --controls-json config\safe_lifecycle_controls.example.json --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts\local_assistant_os\sample_transcript_calibration.json --reset --json
python scripts\local_assistant_os_cli.py candidate-session-audit --db artifacts\local_assistant_os\assistant_v01.sqlite --session all --capture-surface cli_chat --json
python scripts\local_assistant_os_cli.py write-source-attestation --event-ledger-db artifacts\local_assistant_os\assistant_v01.sqlite --event-ledger-session all --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts\local_assistant_os\source_attestation.json --json
python scripts\local_assistant_os_cli.py write-host-app-attestation --host-app-config-json config\host_actions.json --capture-surface target_device_cli --media-app-configured --call-app-configured --not-demo-recorder --real-app-commands-acknowledged --human-reviewed --out artifacts\local_assistant_os\host_app_attestation.json --json
python scripts\local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db artifacts\local_assistant_os\assistant_v01.sqlite --event-ledger-session all --event-source-kind redacted_user_session --source-attestation-json artifacts\local_assistant_os\source_attestation.json --transcript-calibration-report-json artifacts\local_assistant_os\sample_transcript_calibration.json --inventory-soak-report-json artifacts\local_assistant_os\live_inventory_soak_matrix.json --host-app-config-json config\host_actions.json --host-app-attestation-json artifacts\local_assistant_os\host_app_attestation.json --run-host-app-probe --out artifacts\local_assistant_os\v01_blocker_evidence.json --json
python scripts\local_assistant_os_cli.py v01-blocker-rehearsal --reset --json
python scripts\local_assistant_os_cli.py v01-progress --json
python scripts\local_assistant_os_cli.py autoimmune-smoke --reset --json
python scripts\local_assistant_os_cli.py synthesis-variant-smoke --reset --json
python scripts\local_assistant_os_cli.py synthesis-stress-smoke --reset --json
python scripts\local_assistant_os_cli.py setup-integration-smoke --reset --json
python scripts\local_assistant_os_cli.py host-action-smoke --reset --json
python scripts\local_assistant_os_cli.py host-app-probe --reset --json
python scripts\local_assistant_os_cli.py capability-probe --reset --json
python scripts\local_assistant_os_cli.py shortcut-audit --json
python scripts\local_assistant_os_cli.py v01-audit --json
python scripts\local_assistant_os_cli.py v01-acceptance --reset --json
python scripts\local_assistant_os_cli.py parse-debug --utterance "wow you don't know who you are?" --json
python scripts\local_assistant_os_cli.py ask --utterance "Can you explain quasar algebra to my zorbulator?" --improvement-opt-in --json
python scripts\local_assistant_os_cli.py improvement-queue --json
python scripts\local_assistant_os_cli.py dataset-audit --reset --json
python scripts\local_assistant_os_cli.py resource-report --reset --json
python scripts\local_assistant_os_cli.py pi-smoke --reset --json
python scripts\local_assistant_os_cli.py pi-bundle --reset --zip --json
python scripts\local_assistant_os_cli.py verify-bundle --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --json
python scripts\local_assistant_os_cli.py bootstrap-runtime --reset --json
python scripts\local_assistant_os_cli.py api-smoke --reset --json
python scripts\local_assistant_os_cli.py api-session-smoke --reset --json
python scripts\local_assistant_os_cli.py ui-smoke --reset --json
python scripts\local_assistant_os_cli.py launcher-smoke --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --reset --json
python scripts\local_assistant_os_cli.py first-run-smoke --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --json
python scripts\local_assistant_os_cli.py archive-smoke --archive artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle.zip --reset --json
python scripts\local_assistant_os_cli.py chat --turn "Tell me a story." --turn "What is the weather today?" --json
python scripts\local_assistant_os_cli.py target-report --reset --json
python scripts\local_assistant_os_cli.py refresh-weather --offline-json benchmarks\sample_open_meteo_forecast.json --json
python scripts\local_assistant_os_cli.py schedule-refreshes --offline-samples --json
python scripts\local_assistant_os_cli.py run-jobs --json
python scripts\local_assistant_os_cli.py inventory-soak --offline-samples --source both --cycles 2 --story-limit 3 --min-story-models 12 --json
python scripts\local_assistant_os_cli.py inventory-soak-matrix --reset --json
python scripts\local_assistant_os_cli.py inventory-soak-matrix --live --reset --out artifacts\local_assistant_os\live_inventory_soak_matrix.json --json
python scripts\local_assistant_os_cli.py inventory-diversity-smoke --reset --json
python scripts\local_assistant_os_cli.py inventory-retry-smoke --reset --json
python scripts\local_assistant_os_cli.py inventory-failure-smoke --reset --json
python scripts\local_assistant_os_cli.py import-media --cold-start --manifest benchmarks\local_media_manifest.json --limit 2 --json
python scripts\local_assistant_os_cli.py ask --utterance "Play calm piano." --json
python scripts\local_assistant_os_cli.py ask --utterance "Yes, play calm piano." --action-mode dry-run --json
python scripts\local_assistant_os_cli.py action-smoke --reset --json
python scripts\local_assistant_os_cli.py memory-replay --query story --json
python scripts\local_assistant_os_cli.py memory-replay --sessions 3 --events-per-session 1 --json
python scripts\local_assistant_os_cli.py memory-digest --sessions 20 --events-per-session 3 --json
```

Current verified signals:

- `105/105` deterministic assistant eval cases pass across `12` profiles.
- `0` privacy leaks, `0` unsafe local actions, `0` wrong local answers, and `0`
  fake latest-news local answers in the current eval.
- The current eval records `97` bounded synthesis traces with `0` low-quality
  applied traces and empty warning counts.
- The 17-step cold lifecycle logs `17` membrane decisions and `17`
  homeostatic snapshots, builds story/weather inventory, confirms one action,
  and preserves offline limits.
- The 3-scenario / 34-turn lifecycle suite covers child cold start, adult
  media/routine/household setup, and elder sparse offline/contact behavior with
  `0` privacy leaks, unconfirmed actions, fake latest-news answers, low-quality
  applied synthesis, or dangling memory links.
- The 37-turn household-week trace covers setup, weather, story inventory,
  media/contact actions, cancellation/replay, private-cloud rejection, offline
  latest-news refusal, consent revocation, and detailed six-session memory
  digest recall with `11/11` architecture checks and `0` privacy/action/fake-news,
  dangling-link, or low-quality synthesis failures.
- `run-open-traces` adds a messier 2-scenario / 29-turn gate through the real
  kernel/store/scheduler/weather/action/debug path. Current run passes with
  `0.655` local/device resolution, weather miss-to-hit, story cloud-to-local,
  media/contact setup-to-action transitions, `0` privacy/action/fake-news
  failures, and priority signals showing self-observation pressure.
- `run-transcript-replay` adds an authored 25-user-turn JSONL transcript replay
  gate through the same real kernel/store/debug path without per-turn expected
  answers, routes, or response text. Current run passes with `0.68`
  local/device resolution, `profile_update` for simple age/location facts,
  weather/story/media/contact setup-to-ready transitions, private-cloud
  blocking, long-horizon digest recall, all Basic NLP -> UOL -> ChatFrame maps,
  memory-digest quality over the floor, and a transcript-level baseline win:
  current kernel `17/25` local/device vs best static baseline `7/25`, `+0.40`
  local-resolution gain, `7` fewer cloud handoffs, and `3` fewer
  clarifications over the same user turns.
- `import-transcript-replay` converts raw local chat JSONL into the same replay
  format by keeping only user rows, redacting email/phone/URL/long-number tokens
  plus optional manual replacements, and dropping static expected answer/route
  fields. Imported fixtures are calibration material for broader lifecycles; the
  authored replay remains the strict baseline-win architecture gate unless an
  imported fixture explicitly sets `required_baseline_win`.
- `export-transcript-replay` converts the local SQLite event ledger into a
  replay fixture with user utterances, session/day labels, capture provenance,
  and safe lifecycle controls only. Stored answers, routes, reasons, and
  assistant responses are not exported as expectations, so replay must
  rediscover behavior through the kernel. Capture provenance is part of the
  evidence contract: scripted CLI, scripted API/UI smokes, interactive CLI, and
  served browser UI turns must remain distinguishable in reports.
- `calibrate-event-ledger` runs event-ledger export, replay, and aggregate
  threshold scoring in one command for real local browser/CLI sessions. Add
  `--auto-lifecycle` when the replay should let the assistant schedule refreshes,
  run safe offline jobs, and build its memory digest from runtime state rather
  than from per-turn controls.
- `v01-evidence-pack` is the preferred single-command package for a real local
  session DB: it writes event-ledger export, calibration, blocker evidence,
  progress, and source-note artifacts. Development sessions stay development
  evidence with `candidate_blockers_satisfied=0`; reviewed redacted sessions
  require source attestation before any blocker row becomes candidate evidence.
  `candidate-session-audit` preflights an existing DB/session before writing an
  attestation, using the same session selector as export and calibration. It
  also emits a projection-only blocker view that reuses `v01-blocker-evidence`
  row logic to show which rows would be eligible after source attestation and
  which still need digest, live-inventory, or configured-app artifacts. Pass
  optional `--transcript-calibration-report-json`,
  `--inventory-soak-report-json`, or host-app config/attestation flags to fold
  those artifacts into the projection without promoting them to evidence.
  A fully scripted CLI/API/UI smoke ledger cannot become candidate user
  evidence even if attestation flags are supplied; source attestation now
  requires every packaged turn to have matching capture provenance from
  imported redacted transcripts, interactive CLI, served browser UI
  (`browser_ui` plus the served page capture token), or target-device capture.
- `calibrate-transcript-replay` composes import plus replay over one or more raw
  chat JSONL files and aggregates redaction counts, stripped static fields,
  routes, intents, complexity, safety totals, debug-map coverage, and baseline
  required/strict-pass counts. `--controls-json` may supply only safe non-answer
  lifecycle controls such as `run_reflection`, `schedule_refreshes`,
  `execute_jobs`, `network_available`, `execute_opportunities`, and aggregate
  calibration thresholds; route, intent, reason, and answer expectations are
  rejected so imported logs cannot become static fixtures. It now has explicit
  aggregate gate thresholds for total turns, local-resolution rate,
  route/intent diversity, persisted synthesis trace count, priority-signal
  samples, memory-digest quality, strict baseline wins, redaction, static-field
  dropping, debug maps, critical safety cleanliness, and primary UOL/ChatFrame
  routing evidence that does not promote secondary phrase hints into primary
  routes. Its generated
  `next_candidate_commands` are audit-first and session-scoped:
  `candidate-session-audit` over the replay DB/session, then
  `write-source-attestation --event-ledger-session all`, then
  `v01-evidence-pack` or `v01-blocker-evidence`. This keeps imported transcript
  evidence on the same anti-static shortcut/provenance path as live browser/CLI
  sessions instead of treating calibration as candidate proof by itself.
  Pass `--out <calibration-report.json>` to write
  the strict report consumed by `v01-blocker-evidence
  --transcript-calibration-report-json`; without that report the digest/route
  blocker remains missing rather than inferred from loose event-ledger fields.
  That report must also bind back to the current attested replay/event SQLite
  DB path and SHA-256, and preserve imported-redacted transcript capture
  provenance; a standalone strict-looking JSON report is not blocker evidence.
  Imported user turns now carry `imported_redacted_transcript` capture
  provenance into the replayed SQLite event ledger and calibration aggregate, so
  user-derived traces can be distinguished from authored fixtures and scripted
  CLI/browser sessions.
  The portable bundle self-check runs a lightweight thresholded gate against the
  fake raw transcript fixture; real user-derived completion runs should add a
  safe controls file plus flags such as
  `--min-synthesis-traces`,
  `--require-priority-signals`, `--require-memory-digest-quality`, and
  `--require-strict-baseline-win`.
- `v01-blocker-evidence` packages the remaining six blocker rows in one honest
  report. Development sessions can show useful evidence, but only
  `--event-source-kind redacted_user_session` or `target_device_user_session`
  plus a valid session-scoped `--source-attestation-json` can count as
  candidate user-derived evidence. The attestation must match the current
  event-ledger DB hash and
  declare redaction, absence of static expectations/answers/routes/reasons,
  capture provenance for the ledger turns, human review, and non-scripted or
  imported-redacted capture provenance. Scripted CLI smokes remain development
  evidence even when they pass replay/calibration thresholds. The synthesis and
  planner rows also require positive `--min-synthesis-traces` and
  `--min-priority-signal-samples` floors; zero-threshold runs are smoke
  evidence only and must not become candidate blocker evidence. The digest row
  requires `candidate_digest_route_calibration_passed`, which means strict
  digest/baseline gates plus calibration-report binding to the same attested
  event-ledger DB. Host-app evidence has the same anti-shortcut boundary:
  `write-host-app-attestation` must bind `config/host_actions.json` by SHA-256,
  assert real media/call app commands, and pass the recorder/demo detector before
  `configured_target_device_apps` can become candidate evidence. The command
  never claims `architecture_complete`. In this report, `passed=true` means the
  report assembled without read/runtime errors; it does not mean the blockers are
  satisfied. Use `report_valid`, `candidate_evidence_complete`,
  `candidate_blockers_satisfied`, and `remaining_blocker_count` for that
  distinction.
- `v01-blocker-rehearsal --reset --json` is the development-only honesty
  harness for that path. It runs a real scripted `chat` ledger, exports and
  replays it through `v01-blocker-evidence`, writes a development source note,
  then feeds the report into `v01-progress`. The expected result is useful
  `development_evidence_present` for lifecycle/synthesis/planner rows,
  `candidate_blockers_satisfied=0`, and digest/live-inventory/target-app rows
  still unclaimed.
- `v01-progress --json` combines the lightweight `v01-audit` boundary with a
  `v01-blocker-evidence` report, or an explicit `--blocker-evidence-json`, so
  operators can see core readiness, candidate blocker count, missing blockers,
  and next commands without promoting fixture/development evidence to
  completion.
- `shortcut-audit --json` is the runnable anti-shortcut guard used by
  `v01-audit`: it probes identity/weather/meal/autobiographical behavior and
  scans primary router/kernel source boundaries so phrase tables remain
  secondary debug evidence rather than accepted routes. The source scan covers
  the primary classifier, post-route slot helpers, the concept-token secondary
  hint table, identity composition, self-status composition, and the shared
  autobiographical-memory composer; a phrase-list shortcut in any of those
  blocks is drift.
- `autoimmune-smoke` runs a 26-turn / 34-check privacy/action/cache boundary
  suite through the real SQLite/kernel path: private-cloud blocks, generic
  cloud handoff without private evidence, explicitly shareable public-profile
  cloud allowance with policy-indexed dashboard accounting,
  mixed public-profile plus private household cloud blocking without partial
  leakage, household/shared-device memory cloud blocking,
  household and personal consent revocation with reload, child-owned memory
  setup/revocation without generic school fallback, stale weather cache
  exclusion, invented confirmation-target blocking across contact and media
  actions, cancel/replay protection, parent/child private-cloud blocking,
  child-location private-cloud blocking, and prior-conversation export blocking.
- `synthesis-variant-smoke` runs 10 bounded story/advice/tool/memory turns
  through the real SQLite/kernel path: story wording variants including
  "Tell me a tale" without the word `story`, general health variants including
  "healthy" without the exact `health` token, urgent health safety, cached
  weather, local meal choice, recent-session summary, and long-horizon digest
  recall. It requires clean synthesis quality, citations, complete ledgers, and
  primary `slot_role_relation` / UOL ChatFrame evidence with phrase hints kept
  out of the primary route.
- `synthesis-stress-smoke` extends that proof to a 24-turn / 3-session trace
  with identity, runtime status, five story variants, four general-health asks,
  urgent health, weather, meal, school-clothing, profile-memory, last-question,
  recent-session, and long-horizon digest recall. Current evidence passes
  `14/14` checks with `22` local answers, `2` cached-tool answers, clean cited
  synthesis on every turn, min quality `0.799`, and primary UOL/ChatFrame
  routing only.
- `host-action-smoke` runs real command-mode media/contact actions through
  harmless local recorder commands, proving the executor passes an existing
  local media file and resolved trusted-contact target without shell execution.
- `host-app-probe` reports whether target-device media/call commands are
  configured and, when commands are supplied through args or environment,
  executes them through the same typed confirmation gate. The default
  browser/CLI acceptance path passes only as an explicit unconfigured/skipped
  report; target-device app acceptance should add `--require-configured`.
  Portable installs can first run `write-host-actions-demo-config --out
  config/host_actions.local_recorder.json --overwrite --json`, then
  `host-app-probe --config-json config/host_actions.local_recorder.json
  --require-configured --json` to prove configured action execution without
  opening a real app. This recorder path is development evidence only. For
  appliance validation, copy
  `config/host_actions.example.json` to `config/host_actions.json`, fill
  `media_player_command`, `call_command`, and optional `media_dir`, then run
  `host-app-probe --config-json config/host_actions.json --require-configured
  --json`, `write-host-app-attestation --host-app-config-json
  config/host_actions.json --capture-surface target_device_cli
  --media-app-configured --call-app-configured --not-demo-recorder
  --real-app-commands-acknowledged --human-reviewed --out
  artifacts/local_assistant_os/host_app_attestation.json --json`, and then
  `v01-acceptance --host-app-config-json config/host_actions.json
  --require-host-app-configured --json`.
- `api-session-smoke` and `serve` can use the same configured typed action gate
  with `--action-mode real --host-app-config-json config/host_actions.json`;
  dry-run remains the default browser/CLI acceptance mode.
- `capability-probe` runs an 18-case realistic surface probe through the real
  SQLite/kernel path: original eight assistant asks resolve local/device,
  open-domain/code/latest/unknown examples hand off to cloud, private-cloud
  examples are blocked, action confirmations stay dry-run, and every case
  reports Basic NLP -> UOL -> ChatFrame mapping, complexity score, unknown
  tokens, route, reason, primary routing basis, and secondary debug hints.
- `dataset-audit` validates the seed/source fixture set, SHA-256 hashes, story
  metadata, local media manifest, 7-day weather sample, Gutenberg/Archive source
  candidates, 29-turn open trace fixture, 25-turn transcript replay fixture, and
  SQLite bootstrap into the seed profile. The seed facts must carry explicit
  `local_only`, `cloud_eligible`, and `scope` policy metadata before any runtime
  smoke can hide a bad dataset. Official story rows now use `narrative_frame`
  instead of a persisted answer `template`; legacy `template` rows are readable
  only as compatibility input.
- The resource report stays stdlib-only with SQLite and no required network,
  vector database, or ML framework.
- `pi-smoke` is the single compact v0.1 readiness gate: it verifies required
  datasets, the full `dataset-audit`, seeded SQLite memory, local story synthesis behind membrane and
  homeostasis, the 17-step lifecycle, typed media/contact actions with resolved
  local targets, the `synthesis-variant-smoke` and `synthesis-stress-smoke`
  bounded local synthesis gates, the `setup-integration-smoke`
  routine/household/trusted-contact gap-to-setup proof,
  the 29-turn open trace debug-parser gate, the 25-turn transcript
  replay gate with baseline comparison, the offline both-source inventory soak,
  the multi-niche `inventory-diversity-smoke` source-query gate, the
  `inventory-retry-smoke` transient-source retry gate, the
  `inventory-failure-smoke` negative source gate, clean ledgers, and no required
  network/vector DB/ML framework.
- `pi-bundle` creates a portable browser/CLI bundle with the runnable CLI,
  local Python package, seed/source/media fixtures, router anti-static
  regression file, runbook, manifest, generic Unix launchers, Windows
  PowerShell/`.cmd` launchers, optional zip archive, and a self-check that runs
  `dataset-audit`, `pi-smoke`, `autoimmune-smoke`,
  `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
  `capability-probe`, `shortcut-audit`, `v01-audit`, `v01-progress`, `api-smoke`,
  `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `launcher-smoke`, `run-open-traces`,
  `run-transcript-replay`, and `calibrate-transcript-replay` from inside the
  copied bundle root.
- `verify-bundle` verifies the copied bundle manifest before target-device
  proof: every listed file must exist with matching byte count and SHA-256
  hash, required portable commands including `dataset-audit`, `autoimmune-smoke`,
  `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
  `capability-probe`, `shortcut-audit`, `v01-audit`, `v01-progress`,
  `v01-evidence-pack`, `launcher-smoke`, `first-run-smoke`, and
  `run-open-traces`/`run-transcript-replay`/
  `calibrate-transcript-replay` must be present, and the self-check must have
  run with `shortcut_audit_passed=true`.
- The portable bundle includes `bin/first_run.sh`, `bin/start_app.sh`,
  `bin/health_check.sh`, Windows `.ps1`/`.cmd` wrappers, the older
  Raspberry/Linux launchers, and a `systemd/melm-local-assistant.service.example`
  user-service template; the verifier requires those launcher files.
- `bootstrap-runtime` creates the usable local runtime database, imports initial
  media metadata, verifies local story/weather/school-safety chat turns, writes
  clean membrane/homeostasis/event ledgers, and prints the next `ask`, `serve`,
  and `dashboard` commands.
- `api-smoke` starts the stdlib localhost API, verifies `/health`, posts one
  non-mutating identity-challenge parse to `/parse-debug`, posts one story ask
  to `/ask`, confirms membrane/homeostasis/event persistence, verifies
  `GET /dashboard` plus non-static `GET /event-transcript-replay`, and shuts
  the server down.
- `api-session-smoke` runs an 11-turn local API session over assistant
  identity, story, weather, school-safety, media confirmation, health, profile
  memory, meal, and trusted contact confirmation with no cloud/fetch path and
  clean action gates; it then proves `POST /calibrate-event-ledger` can replay
  that live event ledger without exporting stored answers, routes, or reasons.
  Its turns are explicitly labeled `scripted_api_smoke`, so they remain
  development evidence and cannot satisfy user-derived blocker rows by source
  attestation alone.
- `ui-smoke` loads the dependency-free browser chat shell served from `/`,
  verifies it is wired to `/health`, `/ask`, `/event-transcript-replay`, and
  `/calibrate-event-ledger`, posts local identity/status, story, and
  action-confirmation turns, verifies the Basic NLP -> UOL -> ChatFrame debug
  frame and parse-only endpoint, and confirms the same SQLite
  event/membrane/homeostasis/calibration path is used.
- `launcher-smoke` starts the app through the platform start launcher, verifies
  localhost `/health` through the platform health launcher, checks the browser
  shell plus parse endpoint, and shuts the server down.
- `first-run-smoke` executes the packaged first-run launcher after bundle
  creation, then verifies the nested bundle integrity check, dataset audit,
  target report, bootstrap runtime, UI smoke, and launcher smoke all pass and
  leave a usable runtime DB.
- `archive-smoke` treats the zip as the product: it rejects unsafe archive
  paths, extracts into a fresh work directory, finds the portable bundle root,
  runs `verify-bundle`, and executes `first-run-smoke` from the extracted copy.
- Status answers now cite persisted `self_status.self_observation` trends from
  SQLite `self_state`, including local-resolution rate, cache readiness,
  queued/completed job health, importer quality, synthesis warnings, action
  health, and next observed needs. A bounded `runtime_health_history` also
  records local-resolution deltas, cache-gap persistence, and readiness
  transitions so planner priorities can use more than the latest snapshot.
- `chat` runs a cross-platform terminal session through the same kernel/store
  path; scripted `--turn` sessions are regression-tested.
- `parse-debug`, `GET/POST /parse-debug`, and every `/ask` response expose a
  basic NLP -> UOL -> ChatFrame mapping with explicit stage outputs; the
  browser auto-opens the debug frame for suspicious, unknown, or high-unknown
  token turns. The frame now includes token roles, compositional parse details,
  primary domain evidence, secondary domain hints, secondary meaning hints, unknown-token lists, UOL slot sources,
  ChatFrame capabilities, primary routing basis, and secondary debug hints so
  misroutes can be debugged without guessing. Identity challenges map to
  `self_model` through `who/what` + copula + second-person/possessive UOL
  composition, not phrase-level routing; non-identity MVP turns such as
  weather, story, media, meal, and trusted-contact requests now also expose
  `slot_role_relation` decomposition, with phrases kept as secondary hints;
  household/routine/child/session recall map to `household_memory`,
  `routine_memory`, `facts.child_school`, and `conversation_events` instead of
  a vague generic profile object. Secondary lexical evidence is token-sequence
  bounded: `weather` cannot create an `eat` hint, `yesterday` cannot confirm an
  action, `play` cannot be recovered from `replay`/`display`, and bare verbs
  such as `play` or nouns such as `phone` cannot route media/contact actions
  without a compatible object or action frame.
- Every persisted turn now receives a separate response-integrity assessment:
  `understanding_score`, `response_integrity_score`, `overall_score`, score
  components, flags, and research topics. Understanding uses lexical coverage,
  UOL parse strength, primary composition, intent resolution, and route
  agreement; response integrity uses route confidence, grounded synthesis,
  route discipline, and membrane/privacy integrity. A safe cloud handoff can
  therefore score as an honest response while still exposing a language gap.
  Browser sessions have stable local session IDs and an explicit
  `Improve from this session` opt-in. Only opted-in low-confidence turns enter
  `improvement_candidates`; `improvement-queue` exposes the quarantined queue.
  Candidates cannot mutate the live router and are not cloud-exportable.
  Redaction, external-model research, held-out evaluation, and explicit
  promotion remain mandatory later stages rather than hidden online learning.
- `target-report` collects Python/SQLite/platform/memory/disk facts and runs
  `dataset-audit`, `pi-smoke`, `autoimmune-smoke`, `synthesis-variant-smoke`,
  `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`,
  `host-app-probe`, `capability-probe`, `v01-audit`, `api-smoke`,
  `api-session-smoke`, `ui-smoke`, and `bootstrap-runtime`, plus the 29-turn
  `run-open-traces` parser gate, the 25-turn `run-transcript-replay` gate, and
  the raw-chat `calibrate-transcript-replay` calibration gate;
  `--require-raspberry-pi` is now optional appliance validation, not the v0.1
  browser/CLI completion gate.
- `v01-acceptance --reset --json` is the one-command browser/CLI
  release-candidate matrix. It runs `target-report`, a real scripted `chat`
  session, and `v01-audit`, then reports requirements for datasets/bootstrap,
  Pi smoke plus inventory matrix, terminal CLI, API/browser UI, synthesis and
  transcript gates, setup/action gates, direct `shortcut-audit --json`
  anti-static UOL/ChatFrame discipline, and the explicit full-architecture
  blocker boundary.
- `v01-audit --json` reports the current completion boundary: the browser/CLI
  core can be evidenced by the existing gates and the UOL/ChatFrame
  anti-static-shortcut regression guard, while full architecture completion
  remains blocked on user-derived synthesis/lifecycle traces, longer live
  inventory soak, planner calibration, digest/route threshold calibration, and
  configured target-device app commands.
- `refresh-weather` replays the bundled Open-Meteo-shaped fixture into local
  weather cache rows, with `--live` available for real Open-Meteo HTTP.
- `schedule-refreshes` queues Pi-budgeted story metadata and weather refresh
  jobs; `run-jobs` turns imported metadata into cited local story answers and
  weather refresh jobs into cached weather answers.
- Completed story metadata refreshes are preserved as cycles, so dashboard
  trends can show inventory quality and import health across repeated runs.
- `inventory-soak` now runs repeated refresh cycles as a pass/fail readiness
  gate. The compact offline gate requires both Project Gutenberg CSV and
  Internet Archive metadata coverage, story inventory growth, metadata-quality
  scores above floor, failure-mode observability fields, clean safety flags, and
  no network use. `pi-smoke` includes that gate so inventory refresh evidence
  cannot drift into an optional side demo.
- `inventory-soak-matrix` runs three cold-start inventory profiles
  (`both_extended`, `internet_archive_query`, and `gutenberg_replay`) for at
  least nine total refresh cycles. It verifies both source families, zero failed
  import cycles, cold-start story inventory growth, quality-floor compliance,
  failure observability, no offline network use, and a future story ask that
  routes locally from imported inventory with primary UOL/ChatFrame evidence.
  `pi-smoke`, `target-report`, `pi-bundle`, and `verify-bundle` surface this
  gate so it cannot drift into a standalone demo.
  For remaining-blocker evidence, `v01-blocker-evidence` now treats the live
  inventory row as candidate evidence only when the report is
  `melm.inventory_soak_matrix.v1`, `mode=live_metadata`, `network_used=true`,
  all strict matrix checks pass, every run has live fetch counters, every run DB
  artifact exists under the reported matrix directory with a matching SHA-256,
  story inventory rows and future local story events are verified inside those
  DBs, and each run reports primary UOL/ChatFrame story evidence. A report-shaped
  JSON file without those artifacts remains missing evidence. Current live
  evidence artifact:
  `artifacts/local_assistant_os/live_inventory_matrix_probe_escalated.json`
  passes 3 cold-start profiles / 9 live cycles with both source families,
  0 failed import cycles, 10 story rows added, verified run DB hashes, and
  local future story events; the saved
  `artifacts/local_assistant_os/live_inventory_blocker_evidence.json` maps this
  to one candidate blocker while keeping architecture completion false.
- `inventory-diversity-smoke` runs folktale, bedtime, and adventure source-query
  niches through the same scheduler/job/importer path, verifies each query
  reaches the executed import job, and then proves each resulting DB routes a
  story request locally from inventory. `pi-smoke` includes this as a readiness
  sub-gate.
- `inventory-retry-smoke` starts transiently failing localhost Gutenberg and
  Internet Archive-shaped sources, requires both importers to retry, writes
  observable fetch-attempt health into the dashboard path, and proves a cold
  story ask becomes local only after the imported inventory is reloaded.
- `inventory-failure-smoke` runs malformed Internet Archive JSON, byte-budget
  exhaustion, and empty-source cases through the same scheduler/job/importer
  path. It requires observable failed/completed job state, zero fabricated story
  inventory, and a future story ask that stays `cloud_handoff /
  missing_story_model`; `pi-smoke`, `target-report`, and `pi-bundle` surface
  this as release evidence.
- `memory-replay` and chat-native recall query linked autobiographical event
  memory locally by text, intent, route, session, or bounded recent-session
  windows; recent-session chat summaries now extract capability transitions,
  open local gaps, action state, and boundary controls from cited `events.*`
  evidence; conversation export to cloud is blocked as private event memory.
- `memory-digest` compacts long-horizon local event/session memory into a cited
  `memory_digest.*` inventory row for multi-day recall without raw transcript
  stuffing; the row now includes remembered threads, session summaries,
  capability transitions, active limits, open loops, and an inspectable quality
  score for debuggable local self-memory.
- True cold-start CLI asks no longer preload the seed inventory; empty media,
  routine, and household gaps surface local setup opportunities instead of
  pretending capability exists.
- `import-media` imports local media manifest or directory metadata into SQLite
  with `local_device` provenance; `build_media_index` now uses that inventory
  path instead of hardcoded media names.
- Confirmed media/contact actions now pass through a typed local executor:
  dry-run records the prepared target without side effects, and real mode blocks
  unless an explicit command is configured.
- `action-smoke` exercises media and trusted-contact actions through the same
  confirmation gate; dry-run prepares both actions, while regression tests prove
  real mode can execute configured commands and resolve the contact target from
  local inventory.
- Routine, household, and trusted-contact setup opportunities now persist local
  setup requests; they do not create facts or contacts until the user supplies
  them explicitly. `setup-integration-smoke` proves that cold gaps stay empty,
  setup requests require user-supplied values, later explicit routine/household
  facts are scoped local, and trusted-contact actions still pass through the
  confirmation gate.

## Drift Rule

Do not grow this repo as a generic chatbot or a model-first demo. New work should
answer one of these questions:

- Does it improve the local membrane, memory, self-model, or action gate?
- Does it reduce avoidable cloud dependence without creating privacy/safety
  regressions?
- Does it make local evidence, inventories, or synthesis more useful on a
  Raspberry-Pi-class budget?
- Does it produce falsifying evidence that changes the OS plan?
