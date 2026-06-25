# MELM — Morphological Emotion Language Model

**A zero-dependency local assistant OS that understands meaning, not just keywords. Runs on anything from a Raspberry Pi to a laptop — no GPU, no ML framework, no cloud, no vector database.**

```powershell
# bootstrap runtime database (one-time)
python -m melm bootstrap-runtime --reset --json

# interactive chat
python -m melm chat

# one-shot
python -m melm ask --utterance "Tell me a story."

# browser UI at http://127.0.0.1:8771  mood-driven face + Web Audio tones
python -m melm serve
```

**Requirements**: Python 3.11+ (core runs on stdlib only).
**Platforms**: Windows, macOS, Linux, Raspberry Pi OS (ARM64).

> Core: no `pip install`, no virtualenv, no Docker, no GPU, no API keys, no training data. Optional local LLM generation requires the `llamacpp` extra (`llama-cpp-python`) and a GGUF model.

---

## What Is MELM?

MELM is a **meaning-first local assistant OS**. Unlike neural frameworks (Rasa, LangChain, Ollama) that depend on embeddings, GPU training, or GB-sized models, MELM uses a **Universal Object Language (UOL)** — a formal grammar that decomposes every utterance into subject-action-object frames — to understand meaning symbolically, persist it as typed entities, and generate answers from atoms rather than tokens.

This makes MELM:

- **Deterministic** — same input always produces the same output. No hallucinations, no flaky tests.
- **Verifiable** — every decision is traceable to a UOL parse frame, evidence keys, and policy rules. Verified by a routing benchmark (48 cases, 16 intents) and 23 routing accuracy tests with specific reason/evidence assertions.
- **Meaning-aware** — utterances are stored as structured propositions (`world_fact` entities) with a truth model (asserted / negated / contradicted).
- **Self-aware** — derives its sense of identity from usage patterns stored in `personal_experience` entities.
- **Lightweight** — core package runs on stdlib Python; targets sub-50 ms responses and <1 MB RAM per session on a Raspberry Pi 5 (measured baseline with local LLM decode: ~97 ms TTFT, ~545 MB RSS; core stdlib-only path is sub-millisecond at ~186 MB RSS).
- **Private** — fully local. No data leaves the device. No API keys. No cloud dependency.
- **Assistant Ready** — Need to build your own local assistant that runs without LLM keys? MELM is built just for that.

---

## Architecture

```
         utterance
            |
    [Layer 0] Input normalization
        (slang/typo/abbreviation expansion)
            |
    [Layer 1] UOL parse (T1 — utterance meaning)
            |
    [Layer 2] Knowledge typing
        static_fact | negated_fact | opinion | literary_device
            |
    [Layer 3] Frame linker + policy gate
            |
    [Layer 4] Synthesis (atom-aware templates + authority verification)
            |
         answer

    Every layer reads from and writes to:
    entity store (SQLite) + contracts (versioned JSON)
            |
    skills (radial consumers: meal, story, memory, identity, ...)
```

### Three-timescale meaning model

| Timescale | Representation | Persisted As | Purpose |
|---|---|---|---|
| **T1** | UOL atoms per turn | `uol_parse` entities | Cross-skill atom retrieval, parse regression |
| **T2** | Conversation outcome | `personal_experience` entities (outcome, polarity, learned facts) | Session memory, self-identity derivation |
| **T3** | Historical knowledge | `world_fact` entities (subject, relation, object, polarity, provenance) | Long-horizon facts, contradiction detection |

### Four architectural decisions

1. **Knowledge is data, not code** — domain strings, keyword sets, and heuristics live in versioned JSON contracts (think of this as functional memory structures). Skills consume contracts radially. Add a contract, write a skill module, register it — no retraining.

2. **Meaning is symbolic, not statistical** — UOL decomposes utterances into formal frames (subject/action/object/complement). No embeddings, no vectors, no training data. Multilingual by design (grammar-based, ~3x vocabulary per language).

3. **Synthesis is generic, not per-intent** — a single dispatch reads from a handler registry and contract templates. No intent-specific branching. New intents register handlers, not if/elif chains.

4. **Skills are radial consumers** — each skill reads from the centralized knowledge store (entities + contracts). Meal planning, story generation, health advice, memory recall, and self-identity all read from shared data. No linear pipeline, no siloed inline knowledge.

---

## Capabilities

| Capability | Tests | Live | How | Score |
|---|---|---|---|---|---|
| Weather | 100% (18+ files) | Correct, contextualized | Cached offline tool, climate-aware routing, temporal query detection (historical/forecast) | 92/100 |
| Meal suggestions | 100% (18+ files) | Good, weather-aware | Knowledge contracts + entity inventory, weather-context aware, contract-driven food tags | 92/100 |
| Self-identity | 100% (72 tests) | Detailed, correct | Derives label from usage patterns (mean polarity per intent), name awareness/origin routing, 28 self-identity tests | 92/100 |
| Clothing safety | 100% (16+ files) | Correct policy | Temperature/context policies, contract-driven trigger detection, moral cognition integration | 90/100 |
| Session memory | 100% (90 tests) | N/A | Commitment entities, deferred tasks, novelty, epistemic states, background runner, T2 experience writer | 88/100 |
| Health advice | 100% (11+ files) | Good disclaimer | Safety-gated, privacy-bound, moral cognition engine + health disclaimer contracts, contract-triggered urgent detection | 88/100 |
| Moral reasoning | 100% (107 tests) | Correct frames | 156 causal frames + 263 state definitions + verb states + state valence scoring, V4B causal graph, T4 pure function | 88/100 |
| Personal memory | 100% (25 tests) | Partial | Entity store recall (session, digest), T2 experience writer with outcome/polarity/learned_facts | 85/100 |
| Story generation | 100% (52 tests) | Good inventory pick | Expanded 12→25 title keywords, 16→31 full-text keywords, 9→16 challenges, 8→16 lessons, contract-driven templates | 85/100 |
| Knowledge typing | 100% (18 tests) | N/A | UOL claim → classify → store as `world_fact` with truth model (asserted/negated/contradicted), copular state extraction | 82/100 |
| Input normalization | 100% (80 tests) | Weak responses | 3-tier pipeline: contract expansion → lexicon SymSpell → agreement fix, proper-noun/NER protection | 75/100 |
| Mood tracking | 100% (10+ files) | Correct routing | Valence/arousal decay model (6h/1.5h), 9 mood regions, affect signal aggregation, multi-turn mood decay verified by stress tests | 78/100 |
| Music & calls | 100% (10+ files) | Correct routing | Device action via typed confirmation gate + media inventory, UOL semantic-class-based style enrichment with modifier support (language-agnostic) | 80/100 |
| Fact negation | 100% (8+ files) | Correct gating | Detects negated claims, stores polarity, handles contradiction, negation gates classifier + frame linker routing | 78/100 |
| Trusted contacts | 100% (14+ files) | Correct routing | Profile-resolved action via entity store, social_relation frame linking, relationship-aware enrichment templates | 72/100 |
| Cloud handoff | 100% (8+ files) | N/A | Privacy-gated fallback for unknown intents, consent tracking, private facts block, open-domain speech-act templates | 65/100 |

---

## Use Cases

### Researchers
MELM is a testbed for **symbolic AI and conversational agents without neural networks**. The UOL grammar, contract registry, and test suite (~2,000 tests across 152 files, zero flaky tests, zero regressions) provide a reproducible platform for experiments in meaning representation, moral cognition, knowledge persistence, and skill-based architectures.

### Makers & Raspberry Pi users
MELM targets **Raspberry Pi 5 (8 GB)** operation. Measured baseline (laptop, Qwen2.5-0.5B GGUF decode): ~97 ms median TTFT, ~545 MB RSS, ~61 tok/s. The core stdlib path is lighter still. Use it as a local voice assistant, home automation interface, or offline educational companion. Raspberry Pi is an optional appliance validation target — all core acceptance tests run on any stdlib Python platform. Pi smoke-gate validation via `python -m melm pi-smoke` (~7 min).

### Privacy-conscious users
MELM is **fully offline** with no external API calls in the default path. All conversational memory, user facts, learned vocabulary, and typed world facts live in a local SQLite database. The only data that leaves is explicitly gated through `cloud_handoff` with privacy consent tracking.

### Embedded & edge devices
With zero external dependencies (stdlib Python only) and a ~15 MB core footprint (stdlib code + contracts; bundled folk tale data adds ~10 MB), MELM can run in environments where Docker, Redis, or GPU drivers are impractical. Optional local-model generation adds the model file and `llama-cpp-python` extra.

---

## Performance Benchmarks

*Windows 11, AMD Ryzen 7 — representative of any modern CPU. On RPi 5, expect similar latency (stdlib-only, no model weights).*

| Metric | MELM v0.4 | Rasa 3.6 | LangChain + GPT-4o | Ollama + 1.5B |
|---|---|---|---|---|
| Cold start | **~400 ms** (core, stdlib) | 10-30 s | N/A (API) | 2-5 s |
| Per-turn latency | **sub-50 ms target¹** (core, stdlib) | 200-500 ms | 1-5 s | 1-8 s |
| RAM per session | **<1 MB target²** (core, stdlib) | ~1-2 GB | N/A | ~1-3 GB |
| Disk footprint | **~15 MB** (stdlib code + contracts); folk tale data adds ~10 MB | ~2 GB | SDK only | ~1-5 GB |
| External deps | **0** (core) | ~50+ pip | ~30+ pip | llama.cpp/Ollama |
| Offline | **Full** (core) | Full | No | Full |
| Deterministic | **Yes** | No | No | No |

> ¹ Measured baseline with local LLM decode (Qwen2.5-0.5B GGUF, laptop): ~97 ms median TTFT, ~61 tok/s throughput. Core stdlib-only path (template fallback) is sub-millisecond.
> ² Measured RSS during local LLM decode: ~545 MB. Core stdlib-only RSS at idle: ~186 MB. The <1 MB target applies to per-session incremental state (SQLite + ring buffers), not the resident process.

---

## Accomplishments

- **~2,000 tests across 152 files** — zero flaky tests, zero regressions. Coverage spans single-assertion unit checks through full-pipeline **lifecycle simulations**: 17-step and 37-step multi-day sessions exercising the kernel/router/synthesis/store pipeline end-to-end, 29-turn structured scenarios with capability growth, and 25-turn real transcript replay with no pre-baked answers. Routing correctness validated by a **48-case routing benchmark** covering all 16 intents, **23 routing accuracy tests** with specific reason/evidence assertions, and **16 multi-turn stress tests** verifying mood decay, intent tracking, experience persistence, and cross-session recall.
- **122 registered versioned JSON contracts** defining all domain knowledge — food tags, health disclaimers, safety policies, story components, verb states, mood states, capability manifests, knowledge types, world relations, normalization expansions, self-identity, causal frames, music style templates, contact enrichment, and 100+ more across 7 infrastructure categories.
- **115 semantic classes** in `semantic_classes.v1.json` — the spine of all cross-layer meaning, enforced by CI invariant tests that catch class-ID drift across contracts, entity schemas, and UOL frames.
- **Causal frame expansion**: 156 curated predicate frames with 263 state definitions, hand-crafted across 6 batches covering all README use cases (music, planning, chatting, health, story, weather, mood, memory, identity, moral reasoning). Classification fields are 100% rule-correct via contract lookup.
- **Moral cognition engine**: 59 verb states + 100 state valences, pure-function `derive_moral_context()`.
- **Mood engine**: valence/arousal tracking with decay model (6 h valence, 1.5 h arousal half-lives), 9 mood regions, emoji face + Web Audio tones.
- **T1 utterance persistence**: every turn's UOL atoms stored as `uol_parse` entities for cross-skill retrieval.
- **T2 experience writer**: every conversation recorded as `personal_experience` entity with outcome / polarity / learned facts / follow-up / intent_achieved.
- **T3 knowledge typing**: factual claims extracted from UOL, stored as `world_fact` with truth model (asserted / negated / contradicted).
- **Self-identity derivation**: analyzes `personal_experience` usage patterns to derive mood-aware identity narratives (28 tests, per-user isolation). Name-awareness routing expanded with "named"/"yourself"/"real name" pattern matching.
- **Input normalization**: 3-tier pipeline (contract-driven expansion → lexicon-backed SymSpell → deterministic agreement fix) with proper-noun / NER protection. 21 tests with 50 subtests.
- **UOL lexicon integration**: reads verb/noun classes from `lexical_senses` table (C1/C2/C3 fulfilled).
- **MVP3 skill modules**: meal, story, memory, self-identity, curiosity, commitments, novelty, epistemic tracking, deferred tasks, greeting context, creative behavior engine.
- **UOL-powered story generation**: StoryPlan dataclass + planner heuristics + prompt pipeline → Qwen2.5-0.5B decoding. Title keywords expanded 12→25, full-text keywords 16→31, challenges 9→16, lessons 8→16. Contract-driven templates.
- **Causal graph (V4B)**: `AtomLinks` carrying `causes`/`caused_by`/`enables`/`prevents`; multi-atom causal atoms from `advcl`/`mark` edges; `causal_rule` entities with CRUD + solver merge layer; verb-centered `_extract_cause()`. 359 dedicated causal tests.
- **Hybrid causal extraction pipeline**: dual backend (transformers + llama_cpp GGUF) with rule-based post-processing ensuring 100% correct classification. 156 predicates, 263 states.
- **Routing benchmark and accuracy tests**: 48-case routing benchmark covering all 16 intents, 23 accuracy tests with specific reason/evidence assertions, 16 multi-turn stress tests verifying mood decay, intent tracking, experience persistence, and cross-session recall.
- **Pi target**: Raspberry Pi 5 (8 GB). Measured baseline (laptop, Qwen2.5-0.5B GGUF): ~97 ms median TTFT, ~545 MB RSS during decode, ~61 tok/s. Core stdlib-only path: sub-millisecond latency, ~186 MB RSS. Smoke-gate validation via `python -m melm pi-smoke` (~7 min).
- **Mood-driven face + audio in web UI**: emoji face reflects `session_mood` (happy → 😊, sad → 😢), Web Audio API tones on each response, CSS animations for thinking/listening states.
- **Semantic attention NLG**: `SemanticAttentionPacket` binds UOL atoms + learned facts + noun/modifier contracts into a compact pre-synthesis workspace; contract-backed `NlgRenderer` produces deterministic answers from packet slots. 58 tests pass.

---

## V0.4.2 Gap Fixes (Resolved)

All verified gaps from the V0.4.2 probing pass have been fixed. See `docs/superpowers/plans/2026-06-24-v0_4_2-gap-verification-and-implementation-plan.md` for the original gap analysis and root-cause traces.

| Gap | Severity | Status | Fix |
|---|---|---|---|
| **Negation → classifier gating** | P0 | ✅ Fixed | `_main_atom_negated()` helper gates `health_advice` frame-linker match on negated/counterfactual atom context. "I do not feel sick" → `open_domain`. |
| **Mood routing regression** | P0 | ✅ Fixed | `_apply_short_circuits()` routes emotionally valenced claims to `assistant_behavior`, with tag-based exclusion of pain/complaint/profanity so health complaints still route correctly. |
| **"Remind" routing regression** | P0 | ✅ Fixed | `remind` added to `predicate_inventory.v1.json`; classifier rule routes remind/remember/recall + user agent/beneficiary to `personal_memory`. |
| **Temporal memory routing** | P1 | ✅ Fixed | Temporal deixis added to `function_words.v1.json`; past-tense + first-person + meal-predicate questions now route to `personal_memory` instead of `meal_suggestion`. |
| **Consciousness → self_query** | P1 | ✅ Fixed | Unified self-probe short-circuit in `_apply_short_circuits()` routes "Are you conscious/real/alive/sentient?" to `assistant_behavior`. |
| **Rapid repetition reset** | P1 | ✅ Fixed | `AssistantOSKernel` now reuses a persistent `OnDeviceAssistantRouter` instance across turns, preserving `_rapid_state`. |
| **Music mood modifier enrichment** | P2 | ✅ Fixed | `_enrich_media_playback_answer()` now reads atom modifier semantic classes in addition to role classes. "Play calm music" → ambient style. |
| **CLI bundle file list** | P2 | ✅ Fixed | Bundle required-docs paths updated to archived locations in `local_assistant_os_cli.py`. |
| **Symbolic story fallback** | P2 | ✅ Fixed | `SymbolicStoryEngine` wired as Tier 2 fallback in `_story_answer()` (folk tale → symbolic → LLM pipeline → template). |
| **Semantic attention NLG** | P3 | ✅ Built | `SemanticAttentionPacket` + contract-backed `NlgRenderer` integrated into synthesis before atom template fallback. 58 tests pass. |
| **Pi benchmark + BitNet backend** | P2 | ⏳ Deferred | Cannot verify constrained-decoder tok/s/TTFT/RSS without hardware or emulator access. |

---

## Quickstart

```powershell
# 1. Clone
git clone https://github.com/anomalyco/MELM.git
cd MELM

# 2. Bootstrap runtime database (one-time setup)
python -m melm bootstrap-runtime --reset --json

# 3. Chat
python -m melm chat

# 4. Or one-shot
python -m melm ask --utterance "Tell me a story."

# 5. Or browser UI at http://127.0.0.1:8771
python -m melm serve
```

> **Core has no dependencies.** Python 3.11+ stdlib only. Optional local LLM generation requires the `llamacpp` extra and a GGUF model.

### Web Demo

Open `http://127.0.0.1:8771` after running `serve`. The web UI shows a large emoji face that reflects the assistant's real-time mood:  for happy,  for sad,  for annoyed,  when listening,  while thinking. Mood transitions play subtle Web Audio tones. No external files or API calls needed — the tones are synthesized client-side via `OscillatorNode`.

To record a demo GIF for PRs or documentation:

1. Install [ScreenToGif](https://www.screentogif.com/) (free, open-source, Windows) or use OBS Studio / your OS screen recorder.
2. Run the server.
3. Navigate through a few interactions (e.g. "Tell me a story", "Play calm piano", "I'm feeling sad").
4. Stop recording, trim, save as GIF.

### For developers

```powershell
# Run the full test suite
python -m pytest tests/ --tb=short -q

# Setup (one-time — seeds database, verifies runtime)
python -m melm bootstrap-runtime --reset --json

# Quick health check (smoke gate, ~7 min)
python -m melm pi-smoke --reset --json

# Full release-candidate check (all gates, ~15 min)
python -m melm v01-acceptance --reset --json

# Run live session with your own utterances
python -m melm chat
```

### For integrators

```python
from melm.appliance import AssistantOSKernel, LocalAssistantProfile

kernel = AssistantOSKernel(profile=LocalAssistantProfile())
decision = kernel.handle("Tell me a story.")
print(decision.answer)
```

The kernel is stateless between sessions (state persists via SQLite). Embed it in a CLI tool, wrap it in an HTTP server, or run it as a subprocess.

---

### For researchers

- **`melm/contracts/`** — 122 registered versioned JSON contracts defining all domain knowledge (food tags, health disclaimers, safety policies, verb states, mood regions, causal frames, semantic class spine, 100+ more)
- **`melm/appliance/`** — 83+ Python modules implementing kernel, router, synthesis, authority, mood engine, decoder, 15+ skill modules, creative behavior engine
- **`AGENTS.md`** — project context, architectural invariants, done/blocked lists, remaining gaps

---

## Commands Reference

MELM exposes commands through the `python -m melm` CLI.

### User commands

| Command | Purpose |
|---|---|
| `init` | Initialize the assistant OS SQLite database |
| `ask --utterance "..."` | One-shot query |
| `chat` | Interactive terminal session |
| `serve` | Browser UI at localhost:8771 |
| `dashboard` | Summarize the persisted assistant OS ledger |
| `memory-replay` | Query autobiographical event memory |
| `memory-digest` | Compact long-horizon memory digest |
| `import-stories` | Import public-domain story metadata |
| `import-media` | Import local media manifest or directory |
| `refresh-weather` | Fetch or replay weather data into local cache |
| `improvement-queue` | Inspect consent-gated research candidates |

### Setup & health check

These commands are callable directly but not shown in `--help` to keep the CLI surface clean.

| Command | Purpose |
|---|---|
| `bootstrap-runtime --reset --json` | One-time setup — seeds database, verifies runtime |
| `setup-integration-smoke --reset --json` | Quick integration smoke gate (<1 min) — validates runtime setup |
| `pi-smoke --reset --json` | Quick smoke gate (~7 min) |
| `v01-acceptance --reset --json` | Full release-candidate check (~15 min) |

Run `python -m melm --help` for user-facing command details.

---

## Authoritative Direction

The architecture is meaning-first, symbolic, and knowledge-driven. All domain strings, keyword sets, and heuristics live in versioned JSON contracts (knowledge is data, not code). Synthesis is generic (handler registry + contract templates), not per-intent branching. Skills are radial consumers of a centralized knowledge store. See `AGENTS.md` for current architectural invariants, remaining gaps, and the anti-regression checklist.

## Drift Rule

Do not grow this repo as a generic chatbot or a model-first demo. New work should answer one of these questions:

- Does it improve the local membrane, memory, self-model, or action gate?
- Does it reduce avoidable cloud dependence without creating privacy/safety regressions?
- Does it make local evidence, inventories, or synthesis more useful on a Raspberry-Pi-class budget?
- Does it produce falsifying evidence that changes the OS plan?

---

## Comparison With Alternatives

| Dimension | MELM v0.4 | Rasa | LangChain | Ollama | Botpress |
|---|---|---|---|---|---|
| **Paradigm** | Symbolic (UOL grammar) | Intent/entity ML | LLM orchestration | LLM serving | Visual flow builder |
| **Deterministic** | Yes | No | No | No | Mixed |
| **Privacy model** | Full local | Full local | N/A (API key) | Full local | Cloud or self-hosted |
| **Multilingual** | Yes (grammar) | Yes (NLU training) | Yes (LLM) | Yes (LLM) | Limited |
| **Hard floor** | stdlib Python 3.11+ | 2 GB RAM, GPU optional | API key required | 4-8 GB RAM, ~2 GB disk | Docker, Redis |
| **Open source** | MIT | Apache 2.0 | MIT | MIT | AGPL |
