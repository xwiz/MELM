# MELM — Morphological Emotion Language Model

**A zero-dependency local assistant OS that understands meaning, not just keywords. Runs on anything from a Raspberry Pi to a laptop — no GPU, no ML framework, no cloud, no vector database.**

```powershell
# bootstrap runtime database (one-time)
python -m melm bootstrap-runtime --reset --json

# interactive chat
python -m melm chat

# one-shot
python -m melm ask --utterance "Tell me a story."

# browser UI at http://127.0.0.1:8771  ✨ mood-driven face + Web Audio tones
python -m melm serve
```

**Requirements**: Python 3.11+ (stdlib only — no pip install needed).
**Platforms**: Windows, macOS, Linux, Raspberry Pi OS (ARM64).

> No `pip install`. No virtualenv. No Docker. No GPU. No API keys. No training data.

---

## What Is MELM?

MELM is a **symbolic reasoning engine** for conversational AI. Unlike neural approaches (Rasa, LangChain, Ollama) that depend on embeddings, GPU training, or GB-sized LLMs, MELM uses a **Universal Object Language (UOL)** — a formal grammar that decomposes utterances into subject-action-object frames — to understand meaning symbolically.

This makes MELM:

- **Deterministic** — same input always produces the same output. No hallucinations, no flaky tests.
- **Verifiable** — every decision is traceable to a UOL parse frame, evidence keys, and policy rules.
- **Lightweight** — ~8 MB disk, ~0.4 MB RAM per session, sub-50ms responses on a Raspberry Pi 5.
- **Private** — fully local. No data leaves the device. No API keys. No cloud dependency.

---

## What Is UOL?

UOL is originally designed as a way to represent memory 'embeddings/tokens' - knowledge, facts, experiences, events, meaning in a language agnostic way. For MELM, this is still work in progress. Not every knowledge has to have self-relevant, personal or entity implied meaning. The MELM architecture assumes that sentence construction is primarily 

a learned syntactical grammatical function of 'intent'/expression of will through self-learned causality/experience expressed through entity, event and action frames captured in experience/memory. Then the language is a function of these entity, action, and event frames for a specific language using the grammatical rules of that language to create tokens/sentences that achieve the original intent based on a causality implication of verb frames and noun frames (represented via UOL)

e.g.

Intent - 'Inquire emotional state'
Causality implication -> Query User Self State
UOL -> User -> State Input
English -> Are you good?
Revised form -> How are you?

## Architecture

```
utterance → UOL parse → frame linker → policy gate → synthesis → answer
                              ↕
                    entity store + contracts
                              ↕
                    skills (radial consumers)
```

Four architectural decisions distinguish MELM from other frameworks:

1. **Knowledge is data, not code** — domain strings, keyword sets, and heuristics live in versioned JSON contracts (54 registered). Skills consume contracts radially. Add a contract, write a skill module, register it — no retraining.

2. **Meaning is symbolic, not statistical** — UOL decomposes utterances into formal frames (subject/action/object/complement). No embeddings, no vectors, no training data. Multilingual by design (grammar-based, ~3x vocabulary per language).

3. **Synthesis is generic, not per-intent** — a single dispatch reads from a handler registry and contract templates. No intent-specific branching. New intents register handlers, not if/elif chains.

4. **Skills are radial consumers** — each skill reads from the centralized knowledge store (entities + contracts). Meal planning, story generation, health advice, and memory recall all read from shared data. No linear pipeline, no siloed inline knowledge.

---

## Capabilities

| Capability | How | Local |
|---|---|---|
| Story generation | UOL-driven planner → prompt pipeline → Qwen2.5-0.5B | Yes |
| Weather | Cached offline tool (Open-Meteo fixture) | Yes |
| Meal suggestions | Knowledge contracts + entity inventory | Yes |
| Health advice | Safety-gated, privacy-bound, moral cognition | Yes |
| Clothing safety | Temperature/context policies | Yes |
| Music & calls | Device action via typed confirmation gate | Yes |
| Trusted contacts | Profile-resolved action | Yes |
| Personal memory | Entity store recall (recent session, long-horizon digest) | Yes |
| Session memory | Commitment entities, deferred tasks, novelty, epistemic states | Yes |
| Mood tracking | Valence/arousal across sessions with decay model | Yes |
| Moral reasoning | Verb state contracts + state valence scoring | Yes |
| Cloud handoff | Privacy-gated fallback for unknown intents | Gated |

---

## Use Cases

### Researchers
MELM is a testbed for **symbolic AI and conversational agents without neural networks**. The UOL grammar, contract registry, and anti-regression test suite (1,331 tests) provide a reproducible platform for experiments in meaning representation, moral cognition, and skill-based architectures. See `docs/assistant_os_spec.md` for the authoritative architecture specification.

### Makers & Raspberry Pi users
MELM runs on a **Raspberry Pi 5 (8GB)** with **sub-50ms response times** and **<0.5 MB RAM**. Use it as a local voice assistant, home automation interface, or offline educational companion. The `pi-bundle` command creates a portable distribution with launcher scripts. Raspberry Pi is an optional appliance validation target — all core acceptance tests run on any stdlib Python platform.

### Privacy-conscious users
MELM is **fully offline** with no external API calls in the default path. All conversational memory, user facts, and learned vocabulary live in a local SQLite database. The only data that leaves is explicitly gated through `cloud_handoff` with privacy consent tracking.

### Embedded & edge devices
With zero external dependencies (stdlib Python only) and an 8 MB disk footprint, MELM can run in environments where Docker, Redis, or GPU drivers are impractical.

---

## Performance Benchmarks

*Windows 11, AMD Ryzen 7 — representative of any modern CPU. On RPi 5, expect similar latency (stdlib-only, no model weights).*

| Metric | MELM v0.10 | Rasa 3.6 | LangChain + GPT-4o | Ollama + 1.5B |
|---|---|---|---|---|
| Cold start | **387 ms** | 10-30 s | N/A (API) | 2-5 s |
| Per-turn latency | **7-34 ms** | 200-500 ms | 1-5 s | 1-8 s |
| RAM per session | **418 KB** | ~1-2 GB | N/A | ~1-3 GB |
| Disk footprint | **8 MB** | ~2 GB | SDK only | ~1-5 GB |
| External deps | **0** | ~50+ pip | ~30+ pip | llama.cpp/Ollama |
| Offline | **Full** | Full | No | Full |
| Deterministic | **Yes** | No | No | No |

---

## Accomplishments

- **1,331 deterministic tests** across 50+ test files. 0 flaky tests. 0 regressions.
- **54 registered JSON contracts** defining all domain knowledge (food tags, health disclaimers, safety policies, story components, verb states, mood regions, capability manifests, etc.).
- **9 classifiers migrated** from keyword-based to UOL frame linker pattern.
- **89 semantic classes** in `semantic_classes.v1.json` — the spine of all cross-layer meaning.
- **Moral cognition engine**: 59 verb states + 98 state valences, pure-function `derive_moral_context()`.
- **Mood engine**: valence/arousal tracking with decay model (6h valence, 1.5h arousal half-lives).
- **T2 experience writer**: every conversation recorded as `personal_experience` entity with outcome/polarity/learned facts.
- **UOL lexicon integration**: reads verb/noun classes from `lexical_senses` table, 16 tests.
- **Phase 4 skill modules**: meal, story, memory, curiosity, commitments, novelty, epistemic tracking.
- **UOL-powered story generation**: StoryPlan dataclass + planner heuristics + prompt pipeline → Qwen2.5-0.5B decoding.
- **Pi-compatible**: runs on Raspberry Pi 5 with 8 GB RAM. Portable bundle with verified self-check.
- **Mood-driven face + audio in web UI**: emoji face reflects `session_mood` (happy → 😊, sad → 😢), Web Audio API tones on each response, CSS animations for thinking/listening states.

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

> **No dependencies.** Python 3.11+ stdlib only. No `pip install`, no virtualenv.

### Web Demo (mood-driven face + audio)

Open `http://127.0.0.1:8771` after running `serve`. The web UI shows a large emoji face that reflects the assistant's real-time mood: 😊 for happy, 😢 for sad, 😤 for annoyed, 👂 when listening, 💭 while thinking. Mood transitions play subtle Web Audio tones (ascending arpeggios for positive moods, gentle descents for negative). No external files or API calls needed — the tones are synthesized client-side via `OscillatorNode`.

To record a demo GIF for PRs or documentation:

1. Install [ScreenToGif](https://www.screentogif.com/) (free, open-source, Windows) or use OBS Studio / your OS screen recorder.
2. Run the server.
3. Navigate through a few interactions (e.g. "Tell me a story", "Play calm piano", "I'm feeling sad").
4. Stop recording, trim, save as GIF.

### For developers

```powershell
# Run the full test suite
python -m pytest tests/ --tb=short -q

# Run evaluation suite
python scripts\local_assistant_os_cli.py eval --json

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

### For researchers

- **`docs/assistant_os_spec.md`** — authoritative architecture specification (target architecture, skill/knowledge boundary, anti-regression checklist)
- **`docs/local_assistant_os_mvp_plan_v2.md`** — execution plan (milestones, gates, timeline)
- **`melm/contracts/`** — 54 versioned JSON contracts defining all domain knowledge
- **`melm/appliance/`** — 40+ Python modules implementing kernel, router, synthesis, authority, mood engine, 7 skill modules
- **`docs/superpowers/specs/`** — design docs for creative behaviors, story generation, and other features

---

## Commands reference

MELM exposes commands through the `python -m melm` CLI. Major groups:

| Command | Purpose |
|---|---|
| `chat` | Interactive terminal session |
| `ask --utterance "..."` | One-shot query |
| `serve` | Browser UI at localhost:8771 |
| `bootstrap-runtime` | One-time database setup |
| `eval --json` | Core evaluation suite (107 cases) |
| `run-lifecycle --reset` | Cold lifecycle (17 steps) |
| `run-household-week` | 37-turn household scenario |
| `run-open-traces` | 29-turn open trace (parser gate) |
| `run-transcript-replay` | 25-turn transcript replay (baseline gate) |
| `pi-smoke` | Raspberry Pi readiness gate |
| `pi-bundle` | Portable distribution bundle |
| `v01-audit` | Current completion boundary |
| `autoimmune-smoke` | Privacy/action/cache boundary (26 turns) |
| `synthesis-variant-smoke` | Story/health/weather synthesis variants |
| `synthesis-stress-smoke` | 24-turn / 3-session synthesis stress |
| `capability-probe` | 18-case realistic surface probe |
| `setup-integration-smoke` | Setup-request to local recall/action gap gate |
| `shortcut-audit` | Anti-static UOL/ChatFrame regression guard |
| `inventory-soak` | Story metadata refresh cycles |
| `memory-replay` | Query autobiographical memory |
| `memory-digest` | Compact long-horizon memory |
| `parse-debug` | UOL debug frame for any utterance |

Run `python -m melm --help` for full details on any command.

---

## Authoritative Direction

Use [docs/assistant_os_spec.md](docs/assistant_os_spec.md) as the authoritative architecture specification and [docs/local_assistant_os_mvp_plan_v2.md](docs/local_assistant_os_mvp_plan_v2.md) as the execution plan. Root-level whitepaper and research-review files are evidence sources only — they must not steer the current MVP if they imply a model-first or generic-chatbot direction.

## Drift Rule

Do not grow this repo as a generic chatbot or a model-first demo. New work should answer one of these questions:

- Does it improve the local membrane, memory, self-model, or action gate?
- Does it reduce avoidable cloud dependence without creating privacy/safety regressions?
- Does it make local evidence, inventories, or synthesis more useful on a Raspberry-Pi-class budget?
- Does it produce falsifying evidence that changes the OS plan?

---

## Comparison With Alternatives

| Dimension | MELM v0.10 | Rasa | LangChain | Ollama | Botpress |
|---|---|---|---|---|---|
| **Paradigm** | Symbolic (UOL grammar) | Intent/entity ML | LLM orchestration | LLM serving | Visual flow builder |
| **Deterministic** | Yes | No | No | No | Mixed |
| **Privacy model** | Full local | Full local | N/A (API key) | Full local | Cloud or self-hosted |
| **Multilingual** | Yes (grammar) | Yes (NLU training) | Yes (LLM) | Yes (LLM) | Limited |
| **Hard floor** | stdlib Python 3.11+ | 2GB RAM, GPU optional | API key required | 4-8GB RAM, ~2GB disk | Docker, Redis |
| **Open source** | MIT | Apache 2.0 | MIT | MIT | AGPL |
