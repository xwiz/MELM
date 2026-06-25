# MELM Python Package

This package holds the Python implementation for the validation-first MELM build.
The active product-shaped MVP is `MELM Local Assistant OS v0.3` (MVP3) in
`../docs/assistant_os_architecture.md` and executed through
`../docs/superpowers/plans/2026-06-19-mvp3-implementation.md`. The next milestone
in design is causal reasoning (`../docs/superpowers/plans/2026-06-20-causal-reasoning.md`).
Package work should feed that kernel
rather than create a separate chatbot or model-first track.

Planned modules:

- `tokenization`: BPE, Unigram, morphology-aware, and byte/patch tokenizer experiments.
- `data`: BabyLM and synthetic event-dialogue data preparation.
- `memory`: external event-memory prototype and RAG baselines.
- `training`: small-model training orchestration.
- `evaluation`: BabyLM, episodic recall, and conversation evaluation adapters.

Implemented now:

- dependency-free tokenizer smoke harness;
- trainable dependency-free BPE baseline;
- optional fast HF BPE/Unigram baselines;
- capped-vocabulary tokenizer wrapper for fairer morphology ablations;
- morphology boundary-F1 probe;
- tiny held-out token-LM probe;
- tokenizer decision gate for morphology as primary vs auxiliary signal;
- deterministic corpus manifest scanner;
- BabyLM-style local corpus manifest adapter;
- deterministic event-memory and RAG retrieval baselines;
- MELM Guard procedural working-memory rule engine for support/refund action validation;
- support/refunds benchmark fixture with prompt-only, schema-only, temporal/entity RAG, and MELM runtime baselines;
- authored support/refunds JSONL corpus, validator, and benchmark runner for the next publication-readiness dataset step;
- public LoCoMo memory benchmark adapter comparing vector RAG, Mem0-style, MemGPT-style, Zep-style, and MELM Memory OS local architecture families;
- Letta Evals-style LoCoMo export pack for future official Letta-vs-MELM runs;
- local MELM SLM Appliance runtime with JSONL memory, bounded context packing, cited extractive answers, and CLI commands;
- grounded child-chat micro-MVP with UOL parsing, semantic atlas, state algebra, Memory OS, and budgeted SSM/attention boundary;
- Local Assistant OS probes and SQLite-backed v0.1 store for realistic routing,
  self-model learning, inventory building, lifecycle behavior, action
  confirmation, confirmation replay blocking, pending-action cancellation,
  invented confirmation-target blocking, consent revocation, child-owned memory
  revocation without generic school fallback, stale cache exclusion,
  parent/child private-cloud blocking, membrane decisions,
  homeostatic snapshots, session-linked autobiographical event chains, offline
  limits, ledger dashboarding, assistant eval metrics, a 107-case/12-profile
  realistic assistant eval, a 3-scenario / 34-turn lifecycle suite,
  a 2-scenario / 29-turn open trace gate, a 25-turn transcript replay gate,
  a same-turn transcript baseline comparison where the current kernel resolves
  `17/25` local/device versus best static baseline `7/25`,
  chat-native autobiographical recall, blocked conversation-memory cloud export,
  and cited bounded local synthesis traces;
- metadata-only Project Gutenberg CSV and Internet Archive search importers for
  local public-domain story inventory refreshes, plus Pi-budgeted refresh
  scheduling for thin story inventory and stale/missing weather cache; importers
  include stdlib retry/backoff for live fetches, canonical-title dedupe before
  ranking, a minimum metadata-quality floor, candidate/quality/duplicate
  rejection observability, bounded multi-page Internet Archive cursor walking,
  explicit page/rate-limit budgets, and fetch/page/byte-budget health metrics;
- importer/job quality dashboards and pressure-aware refresh scheduling:
  dashboard summaries expose importer health, pagination/rate-limit health,
  priority by kind, retryable queued work, and story metadata quality floor
  compliance, while story/weather refresh priority now uses inventory gap,
  recent cloud handoffs/cache misses, homeostatic averages/deltas, failed jobs,
  and expected local-resolution gain;
- kernel reflection pressure scoring now covers current non-refresh
  opportunities too: story inventory, weather cache, profile-memory questions,
  and trusted-contact setup carry inspectable `priority_signals`;
- first future opportunity classes are implemented for media, routine, and
  household setup: true cold-start media asks surface `build_media_index`,
  routine gaps surface `ask_routine_memory`, and household gaps surface
  `ask_household_memory`; executing media-index setup imports the local media
  manifest path into SQLite and changes future media asks to confirmation-gated
  local device actions;
- `import-media` can ingest either a JSON local media manifest or a scanned
  local media directory, preserving `local_device` provenance, media tags, file
  paths, and file-existence metadata without network, vector DB, or ML
  dependencies;
- confirmed media/contact actions now pass through `LocalDeviceActionExecutor`:
  dry-run records the prepared local target without side effects, while real
  mode blocks unless an explicit command is configured;
- `pi-smoke` runs the compact v0.1 readiness gate over required datasets,
  seeded SQLite memory, local story synthesis, the 17-step lifecycle, typed
  media/contact action preparation with resolved local targets, the
  synthesis-variant and synthesis-stress smokes for bounded
  story/advice/tool/memory variants and longer multi-session synthesis, complete
  ledgers, the open-trace and transcript-replay debug gates, transcript baseline
  checks, the offline both-source inventory soak, the multi-niche
  `inventory-diversity-smoke` source-query gate, the `inventory-retry-smoke`
  transient-source retry gate, the `inventory-failure-smoke` negative source
  gate, clean safety flags, and stdlib Python + SQLite constraints;
- `pi-bundle` builds a portable browser/CLI bundle with the runnable CLI,
  local Python modules, required fixtures, router anti-static regression file,
  runbook, SHA-256 manifest, generic Unix launchers, Windows PowerShell/`.cmd`
  launchers, optional zip archive, and a self-check that runs `dataset-audit`,
  `pi-smoke`, `autoimmune-smoke`,
  `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
  `capability-probe`, `shortcut-audit`, `v01-audit`, `v01-progress`,
  `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `launcher-smoke`, `run-open-traces`,
  `run-transcript-replay`, and `calibrate-transcript-replay` from the copied
  bundle root;
- `verify-bundle` checks the portable bundle manifest before Raspberry Pi
  target proof: all listed files must exist, byte counts and SHA-256 hashes must
  match, required portable commands including `portable_shortcut_audit_command`,
  `portable_v01_progress_command`, and `portable_v01_evidence_pack_command`
  must be present, and the self-check must have run with
  `shortcut_audit_passed=true` and `v01_progress_passed=true`;
- the portable bundle includes generic Unix launchers, Windows `.ps1`/`.cmd`
  launchers, Raspberry/Linux launchers, and a systemd user-service example;
  `verify-bundle` requires these files before portable proof;
- `bootstrap-runtime` creates the usable v0.1 runtime database, imports initial
  local media metadata, verifies story/weather/school-safety chat readiness,
  checks clean ledgers, and prints next `ask`, `serve`, and `dashboard`
  commands;
- `api-smoke` starts the stdlib localhost API, verifies `/health`, posts one
  story ask to `/ask`, checks membrane/homeostasis/event persistence, verifies
  `/dashboard` and non-static `/event-transcript-replay`, and shuts the server
  down;
- `api-session-smoke` runs a realistic 11-turn localhost API session over
  assistant identity, story, weather, school safety, media confirmation, health,
  profile memory, meal, and trusted-contact confirmation with no cloud/fetch
  routes and clean ledgers, then calibrates that live event ledger through
  `POST /calibrate-event-ledger`; those turns are labeled
  `scripted_api_smoke`, so they remain development evidence rather than
  candidate user-derived source evidence; candidate sessions must have every
  packaged turn captured through imported redacted transcript, interactive CLI,
  served browser UI with the served page capture token, or target-device
  provenance;
- `ui-smoke` verifies the dependency-free browser chat shell at `/`, checks
  `/health`, `/parse-debug`, `/ask`, `/event-transcript-replay`, and
  `/calibrate-event-ledger` wiring, posts identity/status/story and
  action-confirmation turns, verifies the Basic NLP -> UOL -> ChatFrame debug
  frame plus operator export/calibration controls, and confirms the same
  persisted assistant path is used;
- parser/debug work must preserve the `db-claw` / SemanticSQL guardrail:
  UOL slots and ChatFrame gates are primary, while phrase or vocabulary tables
  are only secondary meaning hints for noun, verb, analogy, idiom, or domain
  interpretation; owned memories must stay scoped, for example
  `facts.child_school` with `child_local` scope instead of generic
  age/school shortcuts; secondary lexical evidence must be token-sequence
  bounded so `weather` does not produce an `eat` hint, `yesterday` does not
  confirm an action, `play` is not recovered from `replay`/`display`, and bare
  `play`/`phone` cues do not route actions without a compatible object or
  action frame; bare `story`/`bedtime`, health, meal, safety, contact, or
  family-memory words also must not route without compatible ChatFrame roles;
  the built-in secondary hint table is concept-token only, so request-shaped
  strings such as identity/status/memory/contact asks must be covered by
  UOL/ChatFrame composition tests instead of secondary lexical entries;
  the primary intent classifier and post-route UOL slot helpers must not call
  phrase/marker-table helpers or `_secondary_meaning_*`;
  bare fragments such as `your name` must stay unsupported unless a question or
  request frame supplies the missing identity relation;
  non-identity MVP routes must also expose `slot_role_relation`
  decomposition and `primary_domain_evidence` in debug output, while phrase
  evidence remains secondary with `secondary_hint_policy=
  debug_only_never_primary_route`; unsupported turns must keep secondary hint
  words in `unknown_tokens` unless a primary UOL/ChatFrame composition accepts
  them; private-memory cloud exports must parse as boundary frames such as
  `user / send / facts.favorite_color -> external_cloud_model` with
  `request_private_memory_cloud_boundary`, never as generic
  `recall / user_profile` shortcuts; assistant identity must use
  `melm.identity_uol_composition.v1`, and self-status/ledger/next-step asks
  must use `melm.self_status_uol_composition.v1` in both the debug router and
  the persistent kernel rather than duplicate phrase lists;
- `shortcut-audit --json` now enforces that guard as both behavior and source
  evidence: it scans the primary classifier, post-route slot helpers,
  concept-token secondary hint table, identity composition, self-status
  composition, shared autobiographical-memory composer, and kernel recall gate;
- comparison baselines labelled vocabulary-only must stay secondary-lexical
  baselines and must not borrow the UOL/ChatFrame classifier;
- explicitly shareable memory may cross cloud only when the stored fact policy
  is `consent=true`, `local_only=false`, and `cloud_eligible=true`; ordinary
  private, child, household, routine, and conversation memories remain local or
  blocked;
- `chat` runs cross-platform terminal sessions through the same kernel/store
  path; scripted `--turn` sessions are regression-tested;
- `target-report` records Python, SQLite, platform, disk, memory, and optional
  Raspberry Pi detection facts, then runs `dataset-audit`, `pi-smoke`,
  `autoimmune-smoke`, `synthesis-variant-smoke`, `synthesis-stress-smoke`,
  `setup-integration-smoke`, `host-action-smoke`,
  `host-app-probe`, `capability-probe`, `v01-audit`, `api-smoke`,
  `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `run-open-traces`, and
  `run-transcript-replay`, and `calibrate-transcript-replay`;
  `--require-raspberry-pi` can optionally make hardware detection part of
  appliance-specific pass/fail;
- target-device media/call app commands can be supplied by args, environment,
  or a portable JSON file. `write-host-actions-demo-config --out
  config/host_actions.local_recorder.json --overwrite --json` creates a safe
  recorder config for rehearsing the configured gate with `host-app-probe
  --config-json config/host_actions.local_recorder.json --require-configured
  --json`; this remains development evidence only. For real appliance validation, copy
  `config/host_actions.example.json` to `config/host_actions.json`, fill
  `media_player_command`, `call_command`, and optional `media_dir`, then run
  `host-app-probe --config-json config/host_actions.json --require-configured
  --json`, write `write-host-app-attestation` against the same config hash, and
  then use `v01-blocker-evidence` or `v01-acceptance --host-app-config-json
  config/host_actions.json --require-host-app-configured --json`;
- `api-session-smoke` and `serve` can exercise those configured commands with
  `--action-mode real --host-app-config-json config/host_actions.json`, while
  dry-run remains the portable default;
- routine, household, and trusted-contact setup opportunities persist local
  `setup_request` records instead of writing invented facts; later memory or
  action routes change only after explicit user-supplied setup statements;
  `setup-integration-smoke` proves the full cold-gap, setup-request,
  explicit-local-setup, later-local-answer/action path through the real
  kernel/store/action gate;
- `v01-audit --json` reports the completion boundary, keeping the browser/CLI
  core evidence plus the UOL/ChatFrame anti-static-shortcut guard separate from
  remaining user-derived, live-source, calibration, and configured-target-app
  blockers;
- `shortcut-audit --json` is the direct anti-shortcut evidence command used by
  `v01-audit`; it combines live behavior probes with source-boundary checks so
  phrase/vocabulary tables stay secondary debug evidence only;
- `v01-progress --json` combines the completion-boundary audit with a
  `v01-blocker-evidence` report, or `--blocker-evidence-json`, so target
  operators can see exact progress without promoting development evidence to
  completion;
- `candidate-session-audit` reports a projection-only blocker view using the
  same row logic as `v01-blocker-evidence`; this helps plan source attestation
  and artifact collection, can include optional transcript-calibration,
  inventory-soak, and host-app artifacts, but does not create candidate evidence
  by itself;
- `v01-blocker-rehearsal --reset --json` runs the real development chat-ledger
  to blocker-evidence to progress chain, while requiring candidate blocker count
  to remain zero and digest/live-inventory/target-app rows to stay unclaimed;
- `calibrate-transcript-replay` now has opt-in blocker-clearing thresholds for
  persisted synthesis traces, planner priority-signal samples, memory-digest
  quality, and strict static-baseline wins; lightweight bundle self-checks keep
  these off, and `v01-blocker-evidence` treats zero synthesis/planner floors as
  smoke evidence only, while real user-derived completion runs should add
  `--controls-json config/safe_lifecycle_controls.example.json`,
  positive `--min-synthesis-traces`, `--require-priority-signals`,
  positive `--min-priority-signal-samples`,
  `--require-memory-digest-quality`, `--require-strict-baseline-win`, and
  `--out artifacts/local_assistant_os/user_transcript_calibration.json`, then
  feed that report to `v01-blocker-evidence
  --transcript-calibration-report-json`; the digest blocker accepts it only when
  the report is bound to the same attested replay/event SQLite DB path and
  SHA-256 and carries imported-redacted transcript capture provenance;
- `export-transcript-replay` and `GET /event-transcript-replay` capture
  persisted browser/CLI user turns from the local SQLite event ledger into
  replay evidence without exporting stored answers, routes, reasons, or
  assistant responses as expectations; the export includes capture provenance
  so scripted CLI, scripted API/UI smokes, interactive CLI, and served browser
  UI turns stay distinguishable;
- `calibrate-event-ledger` and `POST /calibrate-event-ledger` run that
  event-ledger export, replay, and aggregate threshold scoring for real local
  browser/CLI sessions;
- `calibrate-transcript-replay` preserves `imported_redacted_transcript`
  capture provenance from redacted raw-chat imports through replayed SQLite
  events and aggregate calibration reports, and its generated
  `next_candidate_commands` start with `candidate-session-audit` before
  session-scoped source attestation and evidence packaging;
- `v01-evidence-pack` packages a local session DB into event-ledger export,
  calibration, blocker evidence, progress, and source-note artifacts while
  preserving the development-vs-attested-user evidence boundary;
- `v01-acceptance --reset --json` runs the browser/CLI release-candidate matrix
  by combining `target-report`, a real scripted `chat` session, and `v01-audit`
  into requirement rows for datasets/bootstrap, readiness smoke plus inventory
  matrix, CLI, API/browser UI, transcript/synthesis, setup/action gates,
  direct `shortcut-audit --json` anti-static UOL/ChatFrame discipline, and the
  explicit blocker boundary;
- completed story metadata refreshes are preserved as trendable cycles while
  queued/running work stays idempotent; dashboard import trends report recent
  cycles, imported/selected totals, metadata-quality averages/deltas,
  page/fetch totals, failures, and byte-budget exhaustion;
- `inventory-soak` runs repeated Pi-budgeted refresh cycles from the CLI; the
  compact offline readiness path requires both Project Gutenberg CSV and
  Internet Archive metadata coverage, story inventory growth, metadata-quality
  scores above floor, failure-mode observability, clean safety flags, and zero
  network use; `pi-smoke` now includes this gate;
- `inventory-soak-matrix` runs three cold-start story inventory profiles for at
  least nine total cycles across both-source, Internet Archive-only, and
  Gutenberg-only modes; it requires cold-start inventory growth, both source
  families, clean quality/observability checks, and future local story synthesis
  from imported inventory with primary UOL/ChatFrame evidence; `pi-smoke`,
  `target-report`, `pi-bundle`, and `verify-bundle` expose the matrix as a
  required readiness surface, not a side demo;
- `inventory-diversity-smoke` runs multiple Internet Archive query niches through
  the same scheduler/job/importer path, verifies each query reaches the executed
  import job, and proves each niche DB can answer the next story request locally
  from inventory;
- `inventory-retry-smoke` starts localhost Project Gutenberg and Internet
  Archive-shaped sources that fail once, requires retry/backoff observability for
  both importers, records fetch attempts in dashboard-ready job metadata, and
  proves a future story ask routes locally only after imported inventory reload;
- `inventory-failure-smoke` runs malformed Internet Archive JSON, source
  byte-budget exhaustion, and empty source fixtures through the same
  scheduler/job/importer path, then verifies failures are observable, no fake
  story inventory is written, and the next story request remains
  `cloud_handoff / missing_story_model`;
- `memory-replay` and chat-native recall query linked autobiographical event
  memory locally by text, intent, route, session, or bounded recent-session
  windows; dashboard memory summaries expose recent session intent/route
  counts, chat summaries group cited events by session and now extract
  capability transitions, open local gaps, action state, and boundary controls
  from the same cited events; prior conversation export to cloud is blocked as
  private event memory;
- `memory-digest` builds a local-only long-horizon digest from bounded
  event/session memory and stores it as a cited `memory_digest.*` inventory row
  for multi-day recall without raw transcript stuffing; the digest includes
  remembered threads, per-session summaries, capability transitions, active
  limits, open loops, and an inspectable quality score;
- richer bounded local synthesis for approved local/cache routes: story answers
  assembled from admitted title/summary/topic/culture metadata, broad
  personal-memory summaries over multiple cited local facts/preferences,
  event-memory recall and transition-aware recent-session summaries, richer health/meal/contact-cancel
  guidance, story inventory `narrative_frame` plus quality/local-fit/metadata-quality
  fields, meal choices scored from food inventory, preferences, utterance scope,
  and cached weather, and consent-revocation traces that cite privacy policy
  instead of echoing revoked fact values;
- bounded-synthesis quality scoring persisted to SQLite and exposed through
  dashboards/eval: route discipline, citation coverage, evidence strength,
  answer specificity, source diversity, local privacy discipline, warnings, and
  low-quality applied synthesis counts; current eval/lifecycle synthesis
  dashboards report zero low-quality applied synthesis, and current eval warning
  counts are empty;
- response-integrity scoring persisted beside every turn: separate
  understanding, grounded-response, and overall scores with inspectable
  lexical/UOL/composition/routing/evidence/privacy components; stable browser
  sessions can explicitly opt in to a quarantined low-confidence improvement
  queue, while non-consented turns are scored but never queued and no candidate
  may mutate the live UOL/ChatFrame router;
- Memory OS support-state projection with indexed order/entity lookups;
- synthetic episodic recall fixture;
- JSONL import/export for episodic benchmark fixtures;
- hand-authored child-dialogue smoke benchmark;
- annotated transcript compiler for transcript-derived dialogue benchmarks;
- answerability and abstention calibration probes for event memory;
- held-out story-split calibration for abstention thresholds;
- state-grounding checks for preconditions and contradictions;
- explicit object-location state-resolution benchmark;
- child-level minimal-pair ranking smoke benchmark for saved checkpoints;
- official BabyLM 2026 fast-BLiMP adapter for saved-checkpoint ranking checks;
- tiny PyTorch causal-LM trainer for end-to-end baseline smoke tests;
- tiny-LM checkpoint saving, tokenizer artifact loading, and checkpoint re-evaluation;
- validation gate helpers;
- evidence-gated persistent dialogue demo and JSONL-backed session memory;
- morpheme/root/meaning validation inference harness for novel-word and utterance-routing tests;
- unit tests.

Run from the repo root:

```powershell
# Full test suite
python -m pytest tests/ --tb=short -q

# One-time setup — seeds database, verifies runtime
python -m melm bootstrap-runtime --reset --json

# Quick smoke gate (~7 min)
python -m melm pi-smoke --reset --json

# Full release-candidate check — all gates (~15 min)
python -m melm v01-acceptance --reset --json
```

Keep the first implementation in Python for fast ML iteration. Treat `C:\dev\nameless_vector` as a design reference until its test health is confirmed.
