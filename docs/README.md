# MELM Docs

This directory holds project planning, architecture, and validation documents
for the MELM build.

## Authoritative documents

- `assistant_os_spec.md`: **authoritative architecture specification.**
  Documents the target typed-expert-cascade architecture, the skill/knowledge
  boundary, the knowledge-first (auto-research) design pattern, foundational
  rules, and the anti-regression checklist.
- `local_assistant_os_mvp_plan_v2.md`: **authoritative execution plan.**
  Milestones M0-M7, gates, timelines, contracts, and build order.

## Supporting documents

- `../README.md`: root landing guide and drift rule.
- `../MELM_whitepaper.md`: revised validation-first whitepaper; supporting
  thesis only, not current product direction.
- `../MELM_validation_implementation_plan.md`: supporting six-month validation
  track; it does not supersede the Local Assistant OS plan.
- `../MELM_implementation_research_review_2026.md`: historical de-risking
  review; model-first recommendations are background unless promoted by
  measured OS-plan evidence.
- `archive/`: non-authoritative historical drafts. The old root-level Word
  whitepaper was moved here because it predated the Local Assistant OS
  alignment and could mislead product direction.
- `local_assistant_os_mvp_plan.md`: superseded v0.1 implementation context for
  the Local Assistant OS kernel, including membrane policy, homeostatic state,
  autobiographical memory, lifecycle probes, inventory planning, dashboarding,
  a 3-scenario / 34-turn lifecycle suite, a 37-turn household-week trace,
  multi-profile eval gates, the compact `pi-smoke` readiness gate, the
  `dataset-audit` seed/source/bootstrap gate,
  portable `pi-bundle` browser/CLI package gate, the `verify-bundle` manifest
  integrity gate, generic Unix and Windows launchers, optional Raspberry/Linux
  launchers, the `bootstrap-runtime` usable runtime database gate, the
  `api-smoke` local API gate, the `api-session-smoke` working-MVP API session
  gate, the `ui-smoke` dependency-free browser chat gate, the cross-platform
  `chat` CLI path, the `launcher-smoke` platform launcher proof, the
  `first-run-smoke` packaged first-run launcher proof, the `archive-smoke`
  extracted zip handoff proof, the `capability-probe` can/cannot-handle
  surface gate, the
  `run-open-traces` 2-scenario/29-turn open trace gate, the
  `run-transcript-replay` 25-turn non-static transcript replay gate, the `autoimmune-smoke`
  privacy/action/cache boundary gate, the `synthesis-variant-smoke`
  bounded story/advice/tool/memory synthesis gate, the `synthesis-stress-smoke`
  multi-session bounded synthesis stress gate, the `setup-integration-smoke`
  routine/household/trusted-contact gap-to-user-setup gate, the
  `host-action-smoke` harmless real command-mode action gate, the
  `host-app-probe` target media/call app configuration gate, the
  `shortcut-audit` runnable UOL/ChatFrame anti-shortcut source/behavior guard,
  `v01-audit` completion-boundary audit with that guard, the `v01-acceptance` browser/CLI
  release-candidate requirement matrix,
  `inventory-soak-matrix` cold-start multi-source inventory gate, the
  `inventory-diversity-smoke` multi-niche source-query inventory gate, the
  `inventory-retry-smoke` transient-source retry inventory gate, the
  `inventory-failure-smoke` negative source-failure inventory gate, the
  `target-report` host evidence artifact, chat-native event recall,
  long-horizon memory digests, cited bounded local synthesis, and
  Basic NLP -> UOL -> ChatFrame debug parsing through CLI, `/ask`, and
  `GET/POST /parse-debug`, plus local evidence endpoints
  `GET /dashboard`, `GET /event-transcript-replay`, and
  `POST /calibrate-event-ledger` for turning served browser UI or other
  provenance-labeled local sessions into non-static replay calibration. The
  scripted API/UI smoke commands exercise the same path but stay development
  evidence. The debug frame includes token roles,
  compositional parse details, primary domain evidence, secondary domain hints,
  domain hints, secondary meaning hints,
  unknown-token lists, UOL slot sources, ChatFrame capabilities, primary
  routing basis, secondary debug hints, `frame_registry`, `frame_id`, and
  `secondary_hint_policy=debug_only_never_primary_route`. Per the `db-claw` / SemanticSQL
  rule, phrase tables may only provide additive secondary meaning evidence; UOL
  slots and ChatFrame gates own the accepted route. Non-identity MVP routes now
  expose `slot_role_relation` decomposition too, so story/weather/media/contact
  behavior is debugged through the same frame path rather than phrase hits.
  Accepted local/cached/device/private-boundary routes must also be owned by
  `melm.assistant_frame_registry.v1` with `source_policy=primary_uol_chatframe_only`;
  `shortcut-audit --json` checks this with
  `primary_routes_owned_by_frame_registry=true`.
  Persisted chat turns also receive a multi-factor response-integrity record.
  It separates language understanding from response integrity so an honest
  abstention does not masquerade as comprehension. The browser exposes the
  scores in the debug frame and offers session-level improvement opt-in;
  `improvement-queue --json` lists only consented, low-confidence candidates.
  This queue is quarantined: it cannot alter the live frame registry/router and
  cannot be exported to a cloud research model before a later redaction and
  promotion gate.
  Transcript replay and calibration now require `primary_uol_chatframe_not_secondary_phrase_route`:
  local answers, cached tools, and device actions must cite `token_role_relation`
  or `slot_role_relation` primary evidence, while secondary phrase/lexical hints
  remain debug-only.
  Self-awareness has two explicit primary compositions:
  `melm.identity_uol_composition.v1` for identity/name/capability asks and
  `melm.self_status_uol_composition.v1` for status, ledger, local/cloud, and
  next-step asks. Reintroducing a kernel phrase list for those routes is drift.
  Owned memories must keep
  scoped evidence keys such as `facts.child_school` instead of collapsing into
  generic age/school shortcuts. Secondary lexical evidence is token-sequence
  bounded, so substring hits such as `eat` inside `weather`, `yes` inside
  `yesterday`, `play` inside `replay`/`display`, or noun-only `phone` contact
  routing are regressions. Bare domain words such as `story`, `bedtime`,
  `doctor`, `medicine`, `dinner`, `naked`, `call`, or `family` must also stay
  inert until a compatible question, request, object, target, or safety frame is
  present. Weather concepts such as `what is weather` must not become cache
  requests, meal prompts such as `can you cook dinner` must not become local meal
  advice, and autobiographical recall must use the shared UOL/ChatFrame scope
  composer rather than an exact recall-phrase list. The primary intent
  classifier and post-route UOL slot helpers must not call phrase/marker-table
  helpers or `_secondary_meaning_*`; marker matching is secondary/debug evidence
  or requested-inventory resolution only.
  Bare identity fragments such as `your name` stay unsupported unless a
  question or request frame supplies the missing relation.
  Unsupported turns must keep secondary hint words in `unknown_tokens` unless
  a primary UOL/ChatFrame composition accepts them; for example, `what is a
  story` can expose a secondary `story` hint while still remaining an unknown
  cloud-handoff case. Private-memory cloud exports must parse as boundary
  frames such as `user / send / facts.favorite_color -> external_cloud_model`
  with `request_private_memory_cloud_boundary`, never as generic
  `recall / user_profile` shortcuts.
  Baselines labelled vocabulary-only must remain secondary-lexical baselines,
  not thin wrappers around the UOL/ChatFrame classifier. Explicitly shareable
  memory may cross cloud only when the stored fact policy is
  `consent=true`, `local_only=false`, and `cloud_eligible=true`; ordinary
  private, child, household, routine, and conversation memories stay local or
  blocked.
  `import-transcript-replay` adds the redacted raw-chat import path for broader
  calibration without turning imported logs into static answer fixtures.
  `export-transcript-replay` and `GET /event-transcript-replay` convert local
  SQLite event-ledger user turns into non-static replay evidence while omitting
  stored answers, routes, reasons, and assistant responses. The export includes
  capture provenance so scripted CLI, scripted API/UI smokes, interactive CLI,
  and served browser UI turns do not collapse into identical evidence.
  `calibrate-event-ledger` and `POST /calibrate-event-ledger` wrap event-ledger
  export, replay, and aggregate threshold scoring for real local browser/CLI
  sessions. `--auto-lifecycle` can be used on replay/calibration commands to
  let the assistant schedule refreshes, run safe offline jobs, and build memory
  digests from its own runtime state instead of authored per-turn controls.
  `v01-evidence-pack` is the concise packaging command for that evidence path:
  it writes the event-ledger export, calibration report, blocker evidence,
  progress report, and source-note artifacts from one local session DB.
  `calibrate-transcript-replay` runs import plus replay across raw chat JSONL
  files and aggregates route, complexity, safety, redaction, static-field-drop,
  and debug-map evidence with explicit aggregate thresholds/checks. Strict
  digest/route blocker evidence must be written with
  `--out <calibration-report.json>` and then passed to `v01-blocker-evidence
  --transcript-calibration-report-json`; loose event-ledger fields do not clear
  that blocker. The report must also list and hash-match the same attested
  replay/event SQLite DB and preserve imported-redacted transcript capture
  provenance. Imported user turns preserve `imported_redacted_transcript`
  capture provenance through the generated replay JSONL, replayed SQLite event
  ledger, and calibration aggregate. Its generated `next_candidate_commands`
  now start with `candidate-session-audit` over the replay DB/session and keep
  attestation/evidence commands session-scoped with `--event-ledger-session all`,
  so imported calibration output cannot bypass the same provenance preflight as
  browser/CLI event-ledger sessions. A separate `--controls-json` file may add only non-answer
  lifecycle controls such as reflection, refresh scheduling, job execution,
  network availability, and aggregate thresholds; route/intent/reason/answer
  expectations are rejected.
  The shipped template is `config/safe_lifecycle_controls.example.json`.
  `candidate-session-audit` preflights a DB/session before attestation, using
  the same session selector across event export, calibration, and provenance
  validation. Its blocker projection is explicitly planning-only: it reuses
  `v01-blocker-evidence` row logic to show what source attestation would unlock
  while still requiring written attestation and artifact-bound blocker evidence.
  Optional transcript-calibration, inventory-soak, and host-app artifacts can be
  supplied to the audit projection so one report shows the remaining evidence
  gaps without minting candidate rows.
  `write-source-attestation` writes a hash-bound source attestation for real
  local event-ledger sessions. The attestation records `event_ledger_session`
  so mixed DBs can attest a clean imported/browser/CLI session without
  pretending unrelated scripted sessions are part of the package.
  `v01-blocker-evidence` packages the remaining
  six blocker rows and separates development-authored evidence from candidate
  user-derived evidence; candidate user-derived rows require
  `--source-attestation-json` with matching DB hash and event capture provenance.
  Fully scripted CLI/API/UI smoke ledgers are not candidate user evidence even
  with attestation flags; valid source attestation requires imported redacted
  transcript, interactive CLI, served browser UI (`browser_ui` with the served
  page capture token), or
  target-device provenance to cover every packaged turn, not merely one
  non-scripted turn mixed into a scripted smoke ledger.
  Its top-level `passed=true` means the report assembled cleanly, not that all
  blocker evidence is complete; use `report_valid`,
  `candidate_evidence_complete`, `candidate_blockers_satisfied`, and
  `remaining_blocker_count` to interpret readiness.
  Synthesis/planner blocker rows additionally require positive
  `--min-synthesis-traces` and `--min-priority-signal-samples` floors; a zero
  threshold is accepted only for lightweight smoke reporting, not candidate
  evidence. Digest/route candidate evidence additionally requires
  `candidate_digest_route_calibration_passed`, which combines strict
  digest/baseline gates with calibration-report binding to the current attested
  DB. Candidate live-inventory evidence similarly requires an
  `inventory-soak-matrix --live` report bound to generated artifacts:
  schema `melm.inventory_soak_matrix.v1`, `mode=live_metadata`, live fetch
  counters in every run, strict matrix checks, matching per-run DB SHA-256
  hashes, story inventory rows, future local story events, and primary
  UOL/ChatFrame story evidence. Candidate target-app rows require
  `--host-app-attestation-json` bound to a non-recorder `config/host_actions.json`.
  Recorder/demo configs remain development rehearsals and never claim architecture completion.
  `v01-blocker-rehearsal --reset --json` proves the event-ledger to
  blocker-evidence to progress chain with real development ledger events, but it
  must keep candidate blocker count at zero and leave digest/live-inventory/
  target-app rows unclaimed.
  `shortcut-audit --json` is the direct regression artifact for static shortcut
  drift: it checks live identity/weather/meal/autobiographical behavior and
  source boundaries around the primary classifier, post-route slot helpers,
  secondary hint table, identity composition, self-status composition, the
  shared autobiographical-memory composer, and kernel autobiographical recall
  gate. The portable
  bundle self-check runs it, and `verify-bundle --json` requires
  `portable_shortcut_audit_command` plus `shortcut_audit_passed=true`.
  `v01-progress --json` summarizes the current audit plus blocker evidence into
  one non-mutating progress report for target operators.
  The
  target-report includes dataset audit, autoimmune smoke, synthesis-variant
  smoke, synthesis-stress smoke, setup-integration smoke, host-action smoke,
  host-app probe, capability probe, v01 audit, UI/API smokes,
  bootstrap runtime, open-trace debug-parser
  path, transcript replay path, and the nested `pi-smoke`
  inventory-soak-matrix/inventory-diversity/inventory-retry/inventory-failure
  gates; the
  portable bundle also exposes launcher and post-build first-run proofs.
- `../benchmarks/local_assistant_os_seed.json`: initial local assistant OS seed
  dataset for profile facts and small local inventories.
- `../benchmarks/public_domain_story_metadata.json`: metadata-only public-domain
  story inventory fixture used by the first local inventory builder.
- `../benchmarks/sample_gutenberg_catalog.csv` and
  `../benchmarks/sample_internet_archive_search.json`: replayable source
  responses for metadata-only public-domain story import tests.
- `../benchmarks/local_assistant_open_traces.json`: transcript-like assistant
  fixture for the open-ended kernel/store/scheduler/weather/action/debug gate.
- `../benchmarks/local_assistant_transcript_replay.jsonl`: authored user-turn
  transcript replay fixture with no per-turn expected answer/route text; it
  drives the real kernel/store/debug path and scores complexity, route
  diversity, memory-digest quality, and transcript-level baseline deltas over
  the same user turns.
- `../benchmarks/local_media_manifest.json`: tiny local media manifest fixture
  for the `import-media` CLI and `build_media_index` opportunity path; real
  devices can replace it with a scanned local media directory. `action-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`, and
  `pi-smoke` use the same media/contact action gate to verify dry-run or
  configured command-mode execution with resolved local targets. Portable
  target installs can first generate `config/host_actions.local_recorder.json`
  with `write-host-actions-demo-config` and run `host-app-probe --config-json`
  against it as a safe configured-gate rehearsal. For real appliance validation,
  copy `../config/host_actions.example.json` to `config/host_actions.json`, pass
  it with `host-app-probe --config-json`, write `write-host-app-attestation`
  against the same config hash, and then use `v01-blocker-evidence` or
  `v01-acceptance --host-app-config-json`; those files configure app command
  argv prefixes only and do not participate in chat routing. `api-session-smoke`
  and `serve` can exercise the same commands with `--action-mode real`; dry-run
  remains the portable default.
- `grounded_child_chat_mvp.md`: bounded child-room architecture proof and
  route/budget sub-gate evidence for the assistant OS plan.
- `grounded_child_chat_mvp_direction.md`: retained supporting direction memo;
  superseded by `local_assistant_os_mvp_plan_v2.md` for the authoritative
  execution plan.
- `roadmap.md`: current six-month validation status.
- `corpus_selection_2026.md`: BabyLM corpus choice and tokenizer/training results.
- `tokenizer_strategy_2026.md`: corrected hybrid morphology-plus-Unigram tokenizer strategy and current evidence.
- `babylm_evaluation_2026.md`: official BabyLM 2026 evaluation adapter and fast-BLiMP checkpoint results.
- `babylm_reproduction.md`: local commands for corpus, tokenizer, checkpoint, and eval reproduction.
- `../reports/entity_tracking_symbolic.md`: symbolic state-tracking baseline for BabyLM entity tracking.
- `../reports/babylm_2026_small_model_stage_plan.md`: generated run card for the next checkpointed tokenizer/model stage.
- `../reports/babylm_2026_small_model_stage_gate.md`: completed 23.3M-parameter tokenizer stage decision.
- `../reports/babylm_2026_state_assisted_entity_tracking.md`: first state-first/LM-fallback integration result.
- `../reports/memory_integration_gate.md`: decision gate for moving to the persistent dialogue demo.
- `../reports/persistent_dialogue_demo.md`: first evidence-gated persistent dialogue demo scaffold.
- `../reports/persistent_dialogue_session_demo.md`: reloadable JSONL-backed persistent dialogue session demo.
- `../reports/transcript_session_demo.md`: transcript-derived persistent session smoke report.
- `../reports/morpheme_meaning_mvp.md`: first constructed morpheme/root/meaning validation report.
- `sound_symbolism_deferred.md`: rationale for deferring active sound-symbolism claims.
- `../reports/melm_vs_rag_resource_efficiency.md`: current resource-efficiency comparison against plain RAG.
- `../reports/babylm_2026_stage_execution_readiness.md`: local GPU/disk preflight, same-shape checkpoint smoke, and resumable-run status.
- `abstention_strategy.md`: current evidence-admission/abstention stance and open risk.
- `support_refunds_external_blind_handoff.md`: preregistered handoff and freeze workflow for the external blind support/refunds batch.
- `validation_report_template.md`: final report skeleton.
- `../reports/phase1_report.md`: current generated Phase 1 benchmark snapshot.
