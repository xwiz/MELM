# MELM Roadmap

## Current Product MVP Direction

The active product direction is the Local Assistant OS kernel and v0.2 path,
documented in `docs/local_assistant_os_mvp_plan_v2.md`. The v1 plan remains
historical v0.1 implementation context.

This does not replace the validation track below. It gives the validation work a
product-shaped target:

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

Avoid drift: tokenizer, small-model, memory, and dialogue experiments should be
promoted only when they strengthen that assistant OS substrate.

Current implementation progress:

- SQLite-backed assistant OS store added for events, user facts, self state,
  opportunities, inventories, membrane decisions, homeostatic snapshots, and
  pending actions.
- Initial seed dataset added at `benchmarks/local_assistant_os_seed.json`.
- Runnable CLI/API added at `scripts/local_assistant_os_cli.py`.
- Resource-budgeted SQLite job queue added for inventory work.
- Metadata-only story inventory builder added over
  `benchmarks/public_domain_story_metadata.json`.
- `dataset-audit` CLI command added as the seed/source/bootstrap gate: it
  validates required fixture files with SHA-256 hashes, story/media/weather
  source coverage, Gutenberg/Internet Archive story candidates, the 29-turn open
  trace fixture, the 25-turn transcript replay fixture, and SQLite bootstrap
  into the seed profile.
- `resource-report` CLI command added; current development-machine evidence is
  stdlib-only SQLite runtime, no required network/vector DB/ML framework, and
  sub-second 17-step lifecycle execution.
- `pi-smoke` CLI command added as the compact v0.1 readiness gate: it checks
  required datasets, the full dataset audit, seeded SQLite memory, local story synthesis, the 17-step
  lifecycle, typed media/contact actions with resolved local targets, the
  10-turn `synthesis-variant-smoke` and 24-turn `synthesis-stress-smoke`
  bounded synthesis gates, the `setup-integration-smoke`
  routine/household/trusted-contact setup gate, complete
  ledgers, the 29-turn open trace debug-parser gate, the 25-turn transcript
  replay gate, the offline both-source inventory soak, the multi-niche
  `inventory-diversity-smoke` source-query gate, the `inventory-retry-smoke`
  transient-source retry gate, the `inventory-failure-smoke` negative source
  gate, clean safety flags, and stdlib Python + SQLite with no required
  network/vector DB/ML framework.
- `pi-bundle` CLI command added as the portable browser/CLI package gate: it
  copies the runnable CLI, local Python package, required fixtures, plan docs,
  runbook, generic Unix launchers, Windows PowerShell/`.cmd` launchers, and
  Raspberry/Linux launchers; writes a SHA-256 manifest and optional zip archive;
  and runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`,
  `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
  `capability-probe`, `v01-audit`, `api-smoke`, `api-session-smoke`,
  `ui-smoke`, `bootstrap-runtime`, `launcher-smoke`, `run-open-traces`, and
  `run-transcript-replay`, and `calibrate-transcript-replay` from inside the
  copied bundle root as a self-check.
- `capability-probe` CLI command added as the can/cannot-handle surface gate:
  it runs 18 realistic assistant asks through the real SQLite/kernel path,
  reports local/device/cloud/blocked buckets, complexity, unknown tokens,
  primary routing basis, secondary debug hints, dry-run action confirmations,
  and unsupported examples. It is included in `target-report` and `pi-bundle`.
- `v01-audit` CLI command added as the completion-boundary report: it confirms
  the browser/CLI v0.1 core evidence path and UOL/ChatFrame anti-static
  shortcut guard are defined while keeping `architecture_complete=false` until
  user-derived synthesis/lifecycle traces, longer live inventory soak,
  planner/digest threshold calibration, and configured target-app probes are
  evidenced.
- `v01-acceptance` CLI command added as the release-candidate evidence matrix:
  it runs `target-report`, scripted `chat`, and `v01-audit`, then reports
  requirement rows for datasets/bootstrap, readiness smoke plus inventory
  matrix, terminal CLI, API/browser UI, transcript/synthesis, setup/action,
  anti-static UOL/ChatFrame discipline, optional bundle status, and the explicit
  full-architecture blocker boundary.
- `host-app-probe` CLI command added as the target-app integration gate: by
  default it reports unconfigured media/call app commands without pretending to
  execute them; with supplied commands or `MELM_MEDIA_PLAYER_COMMAND` /
  `MELM_CALL_COMMAND`, or `--config-json config/host_actions.json`, it
  executes through the same typed confirmation gate. `write-host-actions-demo-config`
  now creates a local recorder config so each target can rehearse the configured
  gate before swapping in real media/call app commands. The portable bundle
  carries `config/host_actions.example.json` as the real target install template.
  `target-report`, `pi-bundle`, and `verify-bundle` surface this probe, while
  target-device acceptance can require it with `--require-configured` or
  `v01-acceptance --host-app-config-json config/host_actions.json
  --require-host-app-configured --json`.
- `verify-bundle` CLI command added as the bundle integrity gate: it verifies
  manifest-listed files, byte counts, SHA-256 hashes, required portable
  commands, stdlib-only declarations, and non-skipped self-check status before
  the copied bundle is used as portable browser/CLI proof.
- Cross-platform launcher files are now generated into the bundle and required
  by `verify-bundle`: generic Unix `first_run/start_app/health_check` scripts,
  Windows PowerShell/`.cmd` wrappers, Raspberry/Linux launchers, and
  `systemd/melm-local-assistant.service.example`.
- `launcher-smoke` CLI command added as the packaged launcher proof: it starts
  the app through the platform start launcher, verifies localhost `/health`
  through the platform health launcher, checks the browser shell and parse
  endpoint, and shuts the process down cleanly.
- `first-run-smoke` CLI command added as the post-build first-run launcher
  proof: it executes the generated platform first-run script from a completed
  portable bundle, parses the emitted JSON reports, and verifies bundle
  integrity, dataset audit, target report, bootstrap runtime, UI smoke, and
  launcher smoke all pass from the copied bundle root.
- `archive-smoke` CLI command added as the zip handoff proof: it rejects unsafe
  archive entries, extracts the bundle archive into a fresh work directory,
  finds exactly one bundle root, runs `verify-bundle`, and executes
  `first-run-smoke` from the extracted copy.
- `bootstrap-runtime` CLI command added as the first usable-runtime gate: it
  creates the actual assistant DB, imports initial media metadata, verifies
  local story/weather/school-safety turns, checks clean ledgers, and reports
  next `ask`, `serve`, and `dashboard` commands.
- `api-smoke` CLI command added as the local app/API gate: it starts the stdlib
  localhost API, verifies `/health`, posts one non-mutating parser request to
  `/parse-debug`, posts one story ask to `/ask`, checks
  membrane/homeostasis/event persistence, verifies `/dashboard` plus non-static
  `/event-transcript-replay`, and shuts the server down.
- `api-session-smoke` CLI command added as the working-MVP API session gate: it
  runs `11` localhost `/ask` turns covering assistant identity, story, weather,
  school safety, media confirmation, health, profile memory, meal, and
  trusted-contact confirmation with no cloud/fetch routes, two confirmed
  dry-run actions, clean persisted ledgers, non-static API export, and live
  `POST /calibrate-event-ledger` replay calibration.
- `ui-smoke` CLI command added as the browser-chat gate: it loads the
  dependency-free local UI served at `/`, verifies `/health`, `/ask`,
  `/event-transcript-replay`, and `/calibrate-event-ledger` wiring, verifies the
  Basic NLP -> UOL -> ChatFrame debug frame plus parse-only endpoint, posts
  identity/status/story/action turns, checks operator export/calibration
  controls, and confirms the same SQLite event ledger path.
- `chat` CLI command added for cross-platform terminal sessions; scripted
  `--turn` sessions share the same kernel/store path and are regression-tested.
- `target-report` CLI command added as the target-hardware evidence artifact:
  it records Python, SQLite, platform, disk, memory, and Raspberry Pi detection
  facts, then runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`,
  `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
  `capability-probe`, `v01-audit`, `api-smoke`,
  `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `run-open-traces`, `run-transcript-replay`, and
  `calibrate-transcript-replay`;
  `--require-raspberry-pi` is optional appliance validation rather than the
  browser/CLI v0.1 completion gate.
- `dashboard` CLI command added for route/evidence/membrane/homeostasis/job
  summaries over the persisted SQLite ledger.
- `eval` CLI command added for a deterministic 105-case, 12-profile assistant
  suite covering child, adult professional, elder-care, travel/offline, and
  accessibility scenarios with `0` privacy leaks, `0` wrong-local answers, and
  `0` unsafe local actions in the current run. The expanded suite also covers
  chat-native autobiographical recall and caught/fixed a `history` versus
  `story` substring misroute.
- `run-lifecycle-suite` CLI command added for a deterministic 3-scenario /
  34-turn architecture proof across child cold-start story/weather/action
  behavior, adult media/routine/household setup, and elder sparse
  offline/contact behavior. The current suite exercises all current
  opportunity classes and reports `0` privacy leaks, unconfirmed actions, fake
  latest-news local answers, low-quality applied synthesis, or dangling memory
  links.
- `run-household-week` CLI command added for a deterministic 37-turn longer-life
  proof across household/routine/contact setup, weather cache fill/reuse, story
  cloud-to-local conversion, media manifest import, media/contact confirmation,
  cancellation/replay blocking, private-cloud rejection, offline latest-news
  refusal, consent revocation, and detailed six-session memory digest recall.
  The current trace reports `11/11` architecture checks and `0` privacy/action,
  fake-news, dangling-link, or low-quality synthesis failures.
- `run-open-traces` CLI command added for a transcript-like 2-scenario /
  29-turn gate through the real kernel, SQLite store, scheduler, Open-Meteo
  fixture, inventory jobs, action gate, self-observation history, and debug
  parser. The current run passes with `0.655` local/device resolution, required
  miss-to-hit/setup-to-action transitions, priority signals for weather/story
  pressure, and `0` privacy/action/fake-news failures.
- `run-transcript-replay` CLI command added for an authored 25-user-turn JSONL
  transcript replay gate through the same real kernel, SQLite store, scheduler,
  jobs, action gate, memory digest, and Basic NLP -> UOL -> ChatFrame debug
  parser. The fixture explicitly carries no per-turn expected answer/route text;
  the current run passes with `0.68` local/device resolution, simple
  age/location `profile_update` facts stored locally, required route/reason
  coverage, complexity/unknown-token scoring, clean safety flags, and memory
  digest quality passing.
- `import-transcript-replay` CLI command added for redacted raw-chat calibration:
  it keeps user rows, skips assistant/system rows, strips static expected
  answer/route fields, redacts email/phone/URL/long-number tokens plus optional
  manual replacements, accepts a separate `--controls-json` overlay for safe
  non-answer lifecycle controls, and emits the same replay schema for debug-map
  scoring. The controls overlay can set reflection/scheduling/job/network knobs
  and aggregate thresholds, but route/intent/reason/answer expectations are
  rejected. Authored replay remains the strict baseline-win gate.
- `export-transcript-replay` CLI command and `GET /event-transcript-replay`
  endpoint added for local event-ledger capture: they export persisted
  browser/CLI user turns into non-static replay evidence while omitting stored
  answers, routes, reasons, and assistant rows. This creates a first-class path
  from real local sessions to non-static replay evidence.
- `calibrate-event-ledger` CLI command and `POST /calibrate-event-ledger`
  endpoint added as the local-session calibration path: they export the SQLite
  event ledger, import/redact into replay JSONL, rerun the real kernel/debug
  replay, and apply aggregate thresholds without copying stored routes or
  answers.
- `calibrate-transcript-replay` CLI command added as the aggregate raw-chat
  calibration loop: it imports one or more raw JSONL transcripts, replays each
  through the real kernel/store/debug path, and summarizes redaction counts,
  stripped static fields, routes, intents, complexity, safety totals, debug-map
  coverage, and baseline required/strict-pass counts. It now exposes aggregate
  checks/thresholds for total turns, local-resolution rate, route/intent
  diversity, persisted synthesis trace count, priority-signal samples,
  memory-digest quality, strict static-baseline wins, redaction, static-field
  removal, debug-map coverage, and critical safety cleanliness. `pi-bundle` and
  `target-report` run a lightweight thresholded gate against a fake raw
  transcript fixture; real user-derived completion runs should add
  `--controls-json config/safe_lifecycle_controls.example.json`, `--min-synthesis-traces`,
  `--require-priority-signals`,
  `--require-memory-digest-quality`, and `--require-strict-baseline-win`.
- `autoimmune-smoke` CLI command added for a compact 26-turn / 34-check
  boundary suite: private-cloud rejection, generic cloud handoff without
  private evidence, explicitly shareable public-profile cloud allowance with
  policy-indexed dashboard accounting,
  mixed public-profile plus private household cloud blocking without partial
  leakage, household/shared-device memory cloud blocking, household and personal
  consent revocation with reload, child-owned age/school memory setup,
  child-school revocation without generic `facts.school` fallback, stale
  weather-cache exclusion, contact and media confirmation target mismatch,
  cancel/replay protection, parent/child private-cloud rejection,
  child-location private-cloud rejection, and prior-conversation export blocking.
  `pi-bundle` and `target-report` now include this smoke.
- `synthesis-variant-smoke` CLI command added for a compact 10-turn / 15-check
  bounded synthesis suite. It routes story variants, health variants, urgent
  health, cached weather, meal choice, recent-session summary, and long-horizon
  digest recall through the real SQLite/kernel path; requires cited synthesis,
  clean quality, complete ledgers, and primary `slot_role_relation`
  UOL/ChatFrame evidence; and proves phrase hints stay secondary. `pi-smoke`,
  `pi-bundle`, `target-report`, and `verify-bundle` now surface this gate.
- `synthesis-stress-smoke` CLI command added for a longer 24-turn / 3-session /
  14-check bounded synthesis trace. It keeps identity, status, story, health,
  urgent health, weather, meal, safety, profile-memory, last-question,
  recent-session, and long-horizon digest answers local/cached with citations;
  current evidence is `22` local answers, `2` cached-tool answers, min synthesis
  quality `0.799`, complete ledgers, and primary UOL/ChatFrame routing only.
  `pi-smoke`, `pi-bundle`, `target-report`, and `verify-bundle` now surface it.
- `host-action-smoke` CLI command added for harmless real command-mode action
  proof: local recorder commands receive an existing media file and resolved
  trusted-contact target through the same confirmation gate, without shell
  execution. `pi-bundle` and `target-report` now include this smoke.
- `parse-debug`, `GET/POST /parse-debug`, and `/ask` debug frames now expose an
  explicit basic NLP -> UOL -> ChatFrame mapping; the browser auto-opens the
  debug trace for suspicious, unknown, or high-unknown-token turns. Debug frames
  include token roles, compositional parse details, primary domain evidence,
  secondary domain hints, domain hints, secondary meaning hints,
  unknown-token lists, UOL slot sources, ChatFrame capabilities, primary routing
  basis, secondary debug hints, and `secondary_hint_policy=
  debug_only_never_primary_route`. The `db-claw` /
  SemanticSQL rule applies here: phrase tables are additive secondary evidence
  only, while UOL slots and ChatFrame gates own accepted behavior. Identity
  challenges map to `self_model` through token-role composition, and
  self-status/ledger/next-step turns now map through
  `melm.self_status_uol_composition.v1` in both the debug router and persistent
  kernel. Story/weather/media/contact-style turns expose `slot_role_relation`
  decomposition before route acceptance. Routine, household, child, and
  session-memory asks map to owned local evidence such as
  `routine_memory`, `household_memory`, `facts.child_school`, and
  `conversation_events` rather than generic phrase shortcuts. The primary
  classifier also guards against bare domain words such as story/bedtime,
  health, meal, safety, contact, or family-memory nouns routing without a
  compatible question, request, object, target, or safety frame. The primary
  intent classifier and post-route UOL slot helpers are now guarded against
  phrase/marker-table helpers and `_secondary_meaning_*`; marker matching stays
  secondary/debug evidence or requested-inventory resolution only. Unsupported turns keep secondary hint
  words in `unknown_tokens` unless a primary composition accepts them, and
  private-memory cloud exports parse as boundary frames such as
  `user / send / facts.favorite_color -> external_cloud_model` rather than
  generic memory-recall shortcuts.
- Bounded local synthesis trace added for membrane-approved local/cache answers;
  story and advice paths now expose citations to inventory, profile, weather,
  food, health-goal, and policy inputs, while blocked/cloud/action routes refuse
  synthesis.
- Metadata-only Project Gutenberg CSV and Internet Archive search importers
  added with replayable source-response fixtures and `import-stories` CLI
  ingestion into SQLite; live Gutenberg smoke imported two story candidates from
  the official catalog.
- `schedule-refreshes` CLI command added for Pi-budgeted refresh scheduling:
  thin story inventory and stale/missing weather now queue `import_story_metadata`
  and `refresh_weather_cache` jobs, and replayable offline import jobs convert
  the next story request to local.
- Bounded local synthesis now goes beyond the first deterministic story frames:
  story answers are assembled from admitted title/summary/topic/culture metadata,
  broad personal-memory asks summarize multiple cited local facts/preferences,
  health answers use richer local evidence, meal answers share an inventory
  scorer over saved food, preference, utterance scope, and cached weather,
  imported story payloads carry `narrative_frame` plus
  quality/local-fit/metadata-quality scores, and consent-revocation synthesis
  cites local privacy policy instead of echoing revoked values.
- Metadata importers now have initial production hardening: stdlib retry/backoff
  for live fetches, canonical-title dedupe before ranking, and a minimum
  metadata-quality floor before story candidates enter local selection.
- Importer/job observability and quality dashboards added: import results expose
  candidate, quality-reject, duplicate-reject, fetch, bounded multi-page
  cursor, byte-budget, and rate-limit signals; dashboard summaries now include
  importer health, pagination/rate-limit health, priority by kind, retryable
  queued jobs, and story metadata quality floor compliance.
- Completed story metadata refreshes are now preserved as repeatable cycles
  while queued/running work stays idempotent; dashboard import trends report
  recent cycles, imported/selected totals, metadata-quality averages/deltas,
  page/fetch totals, failures, and byte-budget exhaustion.
- `inventory-soak` CLI added for repeated resource-bounded refresh cycles; the
  compact offline readiness path requires Project Gutenberg CSV plus Internet
  Archive metadata source coverage, story inventory growth, quality-floor
  compliance, failure-mode observability, clean safety flags, and zero network
  use. `pi-smoke` now includes it as a first-class readiness check.
- `inventory-soak-matrix` CLI added as a stronger cold-start inventory gate: it
  runs both-source, Internet Archive-only, and Gutenberg-only profiles for at
  least nine total cycles, verifies both source families and zero failed cycles,
  and proves each future story ask routes locally from imported inventory with
  primary UOL/ChatFrame evidence. It is included in `pi-smoke`, summarized by
  `target-report`, stored in `pi-bundle` self-check evidence, and surfaced by
  `verify-bundle`.
- `inventory-diversity-smoke` CLI added for bounded multi-niche source-query
  growth: folktale, bedtime, and adventure queries run through the same
  scheduler/job/importer path, each query is reported from the executed import
  job, and each resulting DB must answer a story request locally from inventory.
  `pi-smoke` includes it as a readiness sub-gate.
- `inventory-retry-smoke` CLI added for transient source hardening: local
  Gutenberg and Internet Archive-shaped HTTP fixtures fail once, both importers
  must retry with observable fetch attempts, no external network is used, and a
  future story ask routes locally only after imported inventory reload.
- `inventory-failure-smoke` CLI added for deterministic source-failure
  hardening: malformed Internet Archive JSON, source byte-budget exhaustion, and
  empty source fixtures run through the same scheduler/job/importer path; the
  gate requires observable job failure/completion state, no fabricated story
  inventory, and a future story ask that remains `cloud_handoff /
  missing_story_model`. `pi-smoke`, `target-report`, and `pi-bundle` surface it
  as release evidence.
- Refresh scheduling now uses homeostatic/job pressure instead of flat
  priorities for story/weather refreshes: inventory gap, recent story cloud
  handoffs, weather misses, homeostatic averages/deltas, failed job counts, and
  expected local-resolution gain are persisted with scheduled jobs.
- Kernel reflection now pressure-scores all current opportunity kinds:
  story inventory, weather cache, profile-memory questions, and trusted-contact
  setup. Contact misses can outrank profile-memory misses when homeostatic
  uncertainty and local-capability pressure justify it.
- First future opportunity classes added: empty media asks create
  `build_media_index`, routine gaps create `ask_routine_memory`, and household
  gaps create `ask_household_memory`. Executing media-index setup now imports
  a local media manifest path into SQLite and changes a later media ask from
  clarify to a confirmation-gated local device action; `import-media` can also
  ingest a scanned local media directory.
- Confirmed media/contact actions now pass through a typed local executor:
  dry-run records the prepared target without side effects, and real mode
  refuses to run unless an explicit command is configured.
- `action-smoke` CLI added for typed action readiness: it imports local media,
  stores a trusted contact, requests and confirms media/contact actions, and
  reports structured execution results. Dry-run prepares both actions with no
  side effects; regressions prove real mode executes configured commands and
  resolves media paths plus local contact targets without using a shell.
- Routine, household, and trusted-contact setup opportunities now persist local
  `setup_request` records instead of writing invented facts; later local
  answers/actions change only after explicit user-supplied setup statements.
  `setup-integration-smoke --reset --json` proves the full cold-gap ->
  setup-request -> explicit local setup -> later local answer/action arc through
  the real kernel/store/action path while preserving UOL/ChatFrame debug maps
  and confirmation gating.
- Autobiographical memory now stores session and previous/next event links in
  the SQLite event ledger; dashboards report session counts, linked event
  counts, dangling links, and a safety flag for broken memory chains.
- `memory-replay` CLI added for bounded local replay/query over linked
  autobiographical events by text, intent, route, and session; dashboard memory
  summaries now include recent sessions with per-session intent/route counts.
- Chat-native autobiographical recall added for "what did we talk about" and
  "what was my last question" prompts; recall answers cite local `events.*`
  evidence, and cloud export of prior conversation memory is blocked.
- Bounded recent-session recall added: `memory-replay --sessions` returns
  recent sessions with an events-per-session cap, and chat summaries group
  cited `events.*` evidence by session; the bounded synthesizer now extracts
  capability transitions, open local gaps, action state, and boundary controls
  from those same cited events.
- Long-horizon local memory digest added: `memory-digest` compacts bounded
  multi-session event memory into a local-only `memory_digest.*` inventory row,
  dashboard memory summaries report digest coverage, and "what happened over
  the last few days" uses the cited digest instead of raw transcript stuffing.
  The digest now stores remembered threads, per-session summaries, capability
  transitions, active limits, and open loops so chat recall can explain what
  improved, what stayed blocked/local-only, and what still needs setup; it also
  carries an inspectable quality score so thin compactions can fail.
- Bounded local synthesis now writes quality-scored traces into SQLite:
  route discipline, citation coverage, evidence strength, answer specificity,
  source diversity, local privacy discipline, and warnings are summarized in
  dashboard/eval output; current eval has `0` low-quality applied synthesis
  traces and empty warning counts.
- Current warning-heavy synthesis paths were enriched without opening the cloud
  boundary: urgent-health escalation is symptom-specific without diagnosis,
  cached weather names the cache/refresh boundary, meal suggestions cite local
  food inventory, media/contact cancellation names the cleared pending action,
  public-clothing safety gives a proper-clothing boundary, and consent revocation
  names active local memory removal.
- Action confirmation replay and pending-action cancellation gates now run
  through the kernel, CLI, dashboard, and eval suite.
- Consent revocation, stale weather-cache exclusion, invented confirmation
  target blocking, and parent/child private-cloud blocking now run through the
  store/kernel/CLI/dashboard/eval path.
- Remaining v0.1 work: extend live inventory soak across longer cycle counts and
  live network/source retry modes; broaden beyond the current compact and
  24-turn synthesis gates into real user-derived transcript gates using the
  thresholded calibration path while keeping low-quality applied synthesis and
  warning counts at zero; run host-device app smokes with
  `host-app-probe --config-json config/host_actions.json --require-configured --json`
  or `v01-acceptance --host-app-config-json config/host_actions.json
  --require-host-app-configured --json` against actual media/call commands;
  connect routine, household, and contact setup beyond the current local proof
  to real local integrations; calibrate digest-quality scoring on
  user-derived stress traces for the richer long-horizon compactor; and keep
  broadening the consent, stale-cache, invented-action, and parent/child
  privacy gates beyond the current deterministic 34-turn, 37-turn, and authored
  29-turn suites.

## Six-Month Validation Track

| Month | Milestone | Status |
|---|---|---|
| 1 | BabyLM reproduction, tokenizer harness, event-memory prototype | Tokenizer/memory scaffold, BabyLM-local manifest adapter, tiny LM smoke, and reloadable artifact checks implemented |
| 2 | Tokenizer ablations and episodic benchmark draft | Active: tokenizer stage gate passed; small-model run card generated |
| 3 | 125M-370M baselines and event memory vs RAG | Planned |
| 4 | Best 370M integration if gates pass | Planned |
| 5 | Persistent dialogue demo | Planned |
| 6 | Validation report and release artifacts | Planned |

## Gate Summary

- Morphology must beat BPE/Unigram on a meaningful metric or become auxiliary.
- Event memory must beat ordinary RAG by at least 15% on controlled episodic recall.
- The best 370M model must beat the same-size BPE baseline or clearly win episodic tasks without language-quality regression.

## Current Phase 1 Snapshot

- Dependency-free tokenizer metrics are implemented.
- Tiny morphology boundary-F1 probe is implemented.
- Synthetic episodic memory-vs-RAG benchmark is implemented.
- State-grounding fixture is implemented.
- State-resolution benchmark and annotated transcript state cases are implemented.
- Tiny PyTorch causal-LM training smoke is implemented.
- BabyLM-style local manifest adapter is implemented; official corpus run pending local data.
- BabyLM 2026 Strict-Small is downloaded locally and profiled.
- Fast HF BPE/Unigram and capped-morphology tokenizer probes are implemented.
- Matched tiny neural ablation for HF BPE, HF Unigram, and capped morphology is implemented.
- Multi-seed tiny neural tokenizer ablation is implemented and stable on the BabyLM sample.
- 10/50/100-step progression supports moving capped morphology into a small BabyLM training stack.
- Tiny-LM tokenizers/checkpoints can be saved, indexed, reloaded, and re-evaluated from local artifacts.
- A child-level minimal-pair checkpoint smoke test is implemented; the combined checkpoint decision is currently `hold_for_quality_evidence`.
- Official BabyLM 2026 fast-BLiMP assets are downloaded locally and the full 13,400-case checkpoint ranking check is implemented.
- Corrected tokenizer candidate: `tiered_morph_unigram` now best matches the MELM thesis and wins full fast-BLiMP, while BPE still leads fast entity tracking.
- Tokenizer stage gate now advances `tiered_morph_unigram` to scaled neural ablation after the first larger local proxy preserves a 3.08% bpb gain over HF BPE.
- The next checkpointed BabyLM local stage is generated in `reports/babylm_2026_small_model_stage_plan.md`: four tokenizer arms, three seeds, 23.3M parameters per arm, and full fast-BLiMP/entity follow-up commands.
- A symbolic BabyLM entity-tracking baseline now reaches 100.00% on all 3,152 fast cases with zero abstentions, confirming that the entity gap should be attacked with explicit state/event memory rather than tokenizer-only changes.
- Local stage preflight passes on the RTX 4060 Laptop GPU, and a same-shape 23.3M-parameter tiered-hybrid checkpoint smoke reloads with zero validation delta; see `reports/babylm_2026_stage_execution_readiness.md`.
- The full checkpointed 23.3M-parameter local stage completed: tiered hybrid beats HF BPE by 2.38% bits/byte, wins 2/3 fast-BLiMP scoring views against HF baselines, and edges HF BPE on fast entity tracking by 0.48 percentage points.
- Small-model stage gate is `advance_to_event_memory_integration`; capped morphology remains the compression control.
- First state-memory integration check is implemented: entity prompts compile into event records, a state-first/LM-fallback policy lifts tiered-hybrid fast entity tracking from 40.42% to 100.00% on the regular-format BabyLM fast fixture.
- Memory integration gate is `advance_to_persistent_dialogue_demo`, combining state-assisted entity tracking with synthetic, authored-dialogue, sample-transcript, and abstention gates.
- Persistent dialogue demo scaffold is implemented over authored child-dialogue events: targeted causal-source and state-change evidence resolution lifts the demo to 100.00% evidence-gated accuracy, 100.00% positive recall, and 100.00% negative abstention without lowering the abstention threshold.
- Reloadable JSONL session persistence is implemented for the dialogue demo; the seeded authored session reloads 12 events and preserves 100.00% evidence-gated accuracy, 100.00% positive recall, and 100.00% negative abstention.
- Transcript-derived persistent session smoke passes on the sample annotation fixture after reload with 4 distractor events: 100.00% regular dialogue evidence accuracy, 100.00% paraphrased/noisy dialogue evidence accuracy, 100.00% state accuracy, and 100.00% event-memory recall@2 versus 66.67% RAG recall@2.
- Current Python event memory is evidence/context efficient but not yet CPU/RAM superior to RAG: both retrievers scan the full event list; a Pi-class win requires indexed memory or a Rust/C sidecar.
- Sound-symbolism has been removed from the active MVP gate and deferred as a separate research question; the current validation target is higher-confidence morpheme/root/usage inference.
- Expanded morpheme/root/meaning validation corpus and deterministic inference harness pass 22/22 novel-word cases and 6/6 utterance-routing cases over constructed high-confidence morpheme/root examples.
- Current report: `reports/phase1_report.md`.
