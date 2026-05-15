# A Morpheme-Event Language Model
## A Falsifiable Cognitive Runtime for Sample-Efficient Conversational AI

**May 2026, revised validation-first edition**

---

## Abstract

MELM, the Morpheme-Event Language Model, is now framed as a cognitive runtime for small language models. Its central claim is not that morphology alone can replace scale. The stronger claim is that many capabilities currently forced into one large neural model can be separated into explicit subsystems: a small language model for parsing and composition, procedural working memory for rules and live state, event memory for experience, semantic memory for stable knowledge, and evidence gates for grounded output.

The original MELM concept combined four ideas: morpheme-aware tokenization, compositional lexical embeddings, a small efficient backbone, and event-indexed episodic memory. Those pieces remain useful, but the fastest validation path is broader and more practical: first prove a rule-aware working-memory layer, then prove hierarchical event/state memory, and only then ask how small the neural model can become when those systems carry more of the cognitive load.

The central tests are simple:

1. Can a procedural working-memory layer extract, cache, and apply useful rules in real time more reliably than prompt-only agents?
2. Can explicit event/state memory beat ordinary RAG and long-context baselines on temporal recall, state updates, contradiction handling, and abstention?
3. Can a smaller model plus MELM runtime match or beat a larger model on constrained tasks at lower cost, latency, and context budget?
4. Does morphology-aware tokenization or auxiliary morphology supervision improve sample efficiency once the runtime is in place?

If the answer is yes, MELM earns the right to scale from a memory/rule runtime into a small-model architecture. If the answer is no, the project still produces a useful negative result: benchmark infrastructure, tokenizer comparisons, memory baselines, rule/runtime ablations, and a practical account of where the cognitive-runtime hypothesis fails.

---

## 1. Executive Summary

The MELM thesis is that language models waste capacity when they must internalize too many separable systems at once:

- rules, procedures, constraints, and live task state;
- events, state changes, temporal order, and causal links;
- semantic facts and domain concepts;
- words are built from recurring meaning-bearing parts;
- conversation and experience are organized into events, not just flat token streams.

The project is promising because 2024-2026 research has produced useful pieces: agent-memory systems such as MemGPT/Letta, Zep/Graphiti, A-MEM, Memory^3, and EM-LLM; rule and guardrail systems such as constrained decoding, policy enforcement, and agent firewalls; MorphPiece, MorphBPE, and tokenizer-alignment studies for morphology; BabyLM for disciplined low-data evaluation; and efficient sequence models such as Mamba-family and hybrid SSM/attention systems.

The project is risky because the evidence is not evenly strong:

- morphology-aware tokenization helps most clearly when it preserves statistical subword efficiency rather than replacing it; English-only gains are plausible but unproven;
- EM-LLM is a valuable no-finetune episodic memory method, not a ready-made trainable event processor for a custom Mamba model;
- rule-aware working memory can become brittle if rules are hand-authored, overfit to demos, or extracted without provenance and validation;
- agent-memory markets are already crowded, so MELM's advantage has to appear specifically in state updates, contradiction handling, and evidence-gated abstention;
- Mamba-3 is a strong research candidate, but checkpoint and kernel availability are implementation dependencies;
- a reversible morpheme-in/morpheme-out decoder is a larger engineering task than a first MVP needs;
- a real training program costs more than lower-bound FLOPs imply because evaluation, failed runs, dataloading, optimizer state, integration, and ablations dominate the calendar.

This whitepaper therefore frames MELM as a phased validation ladder:

1. **MELM Guard:** a procedural working-memory hypothesis covering rule extraction, rule caching, state tracking, policy/action validation, and evidence-backed allow/deny decisions.
2. **MELM Memory OS:** a hierarchical-memory hypothesis covering working state, event memory, semantic memory, temporal updates, contradiction handling, and abstention against RAG and long-context baselines.
3. **MELM SLM Appliance:** a small-model systems hypothesis: explicit memory and rules may reduce model size, token budget, and latency for constrained tasks.
4. **MELM Language R&D:** a compression hypothesis around morphology, tokenization, and small-backbone sample efficiency.
5. **MELM Integrated Model:** a later integration hypothesis in which a 125M-370M model absorbs the validated runtime signals; 800M scaling is treated as contingent, not assumed.

The fastest credible outcome is a 90-day external benchmark demo. The six-month outcome is a usable validation release. The publishable integrated MELM-MVP is a 9-12 month target.

---

## 2. The Thesis: Cognitive Runtime With Compression Guardrails

### 2.1 Procedural working memory

In this whitepaper, "working memory" does not mean only a short context buffer. It means a live, rule-aware state space:

- current facts, goals, variables, constraints, and open questions;
- extracted or hand-authored rules with provenance;
- cached inferences that can be reused without another full neural reasoning pass;
- action preconditions, postconditions, and policy gates;
- links into event memory and semantic memory when local state is insufficient.

This is the MELM equivalent of a fast "left-brain" system, described in engineering terms as procedural working memory plus production rules, state tracking, typed facts, and evidence-gated validation. Some questions are expected to be answerable from this layer alone. Others trigger retrieval from event memory, semantic memory, or a larger model.

### 2.2 What language compresses

Language is a compact description system for events, states, relations, and intentions. Morphology carries recurring pieces of meaning: negation, tense, plurality, agency, possibility, degree, and derivation. Discourse carries event structure: who did what, when, why, and what changed afterward.

Standard LLMs trained on BPE or Unigram tokens learn many of these structures implicitly. That works, but it may be inefficient in the low-data, small-model regime. MELM asks whether some of this structure should be represented explicitly.

The strongest tokenizer version of this claim is hybrid rather than
morpheme-only. Familiar high-frequency forms remain cheap whole lexical units,
ordinary subword variation uses a strong statistical tokenizer such as Unigram,
and explicit morphology activates when it preserves useful structure without
imposing excessive sequence length. This mirrors how the rest of MELM treats
memory: what is frequently or recently useful is retrieved quickly, while less
familiar material triggers deeper decomposition.

### 2.3 What this thesis does not claim

Storage compression is not semantic compression. A smaller tokenizer does not eliminate the need to model world knowledge. A morpheme inventory does not make reasoning automatic. Event memory does not replace learned language competence. Rule systems do not remove ambiguity, perception, or judgment. MELM's claim is narrower:

- procedural working memory may make agent behavior more reliable, auditable, and cheap;
- morphology may improve sample efficiency;
- event memory may improve long-horizon recall and consistency;
- small models may benefit more from these biases than frontier-scale models do;
- the benefits are meaningful only when measured against strong baselines.

### 2.4 Operational target

The phrase "five-year-old conversational competence" is useful as inspiration but too broad as a single benchmark. The validation target is therefore operationalized as:

- age-appropriate vocabulary and grammar;
- multi-turn consistency over controlled stories;
- simple false-belief and perspective-taking tasks;
- event recall across sessions;
- counterfactual reasoning over small synthetic worlds;
- uncertainty and refusal behavior when the model lacks evidence.

MELM does not claim human-like child cognition from isolated language tests.

---

## 3. Prior Art And Competing Approaches

### 3.1 Morphology-aware tokenization

Relevant work includes Morfessor, MorphPiece, MorphBPE, Tokens with Meaning, and tokenizer-alignment evaluations across many languages. The pattern is encouraging: aligning tokens with morphemes can improve interpretability and performance, especially under low-resource conditions or in morphologically rich languages.

The MELM risk is that English is comparatively analytic. A 12,000-morpheme vocabulary might be enough for the intended conversational scope, but it is not a safe assumption. The tokenizer claim is evaluated against 4k, 8k, 12k, 20k, BPE, Unigram, and byte/patch alternatives before any tokenizer becomes architectural doctrine.

### 3.2 Tokenizer-free and byte/patch competitors

The strongest alternative to "better linguistic tokens" is reducing or removing tokenization assumptions entirely, so tokenizer-free and byte/patch approaches are treated as first-class competitors.

Important baselines and design references:

- **MEGABYTE:** multiscale byte modeling.
- **SpaceByte:** byte modeling with explicit patch boundaries.
- **Byte Latent Transformer (BLT):** dynamic byte patches that scale better than fixed tokens in some settings.
- **T-FREE:** sparse tokenizer-free generative modeling.

These systems are direct competitors to the morpheme hypothesis. If byte/patch models win at similar compute and model size, the morphology claim weakens; MELM would then keep event memory as the primary contribution and treat morphology as auxiliary supervision rather than a required tokenizer.

### 3.3 Episodic memory and event representation

EM-LLM shows that event segmentation and temporal-contiguity retrieval can extend effective context without simply stuffing more text into the prompt. SEEM, Memory^3, A-MEM, and MemGPT/Letta-style systems strengthen the broader idea that long-term memory should be explicit and structured.

The MELM correction is important: EM-LLM is treated first as an external-memory baseline and an event-segmentation prior. Integrating event tokens into a trainable backbone is a research contribution, not glue work.

### 3.4 Efficient small backbones

Candidate model families:

- small Transformers for baseline clarity;
- Mamba/Mamba-2/state-spaces checkpoints where stable;
- Mamba-3 if suitable checkpoints and kernels are available;
- hybrid SSM/attention designs inspired by Hymba and Nemotron-style systems.

The first validation target is 125M-370M. An 800M model is reserved for the case where smaller ablations justify scaling.

### 3.5 Developmentally plausible pretraining

BabyLM is the right evaluation culture for MELM because it constrains data and rewards careful corpus construction. The proposed evaluation uses BabyLM 10M/100M-style tracks where possible, plus controlled synthetic event-dialogue data only after data quality filters are in place.

Synthetic data is grounded in event records. Free-generation synthetic dialogue is too likely to add noise. Teacher models are treated provider-agnostically; current model IDs and pricing are checked at generation time.

### 3.6 Cognitive architectures, rule engines, and agent runtime control

MELM's procedural working-memory layer has older roots than modern LLM agents. Cognitive architectures such as Soar and ACT-R represent active state, declarative memory, procedural rules, and compiled/chunked inferences as separate components. Production-rule systems and Rete-style matching show that a large set of rules can be precompiled into efficient runtime structures rather than reinterpreted from scratch on every step.

Modern LLM systems are converging on related ideas from a different direction:

- memory layers separate short-term context, long-term archival memory, user facts, and temporal knowledge graphs;
- constrained decoding and structured-output engines validate syntax and schemas during generation;
- agent firewalls, policy engines, and guardrail runtimes validate proposed actions before execution;
- domain-specific small models increasingly rely on external knowledge, policy, and tool state rather than memorizing everything internally.

MELM's differentiator is not "it has memory" or "it has rules." The differentiator is the combination: a procedural working-memory layer that can decide when local rules are enough, when event/state memory is required, when semantic memory is required, and when the system must abstain.

### 3.7 Reuse from nameless_vector

The previous project `C:\dev\nameless_vector` is useful as a parts donor, not as the primary codebase. It contains concepts MELM can reuse:

- semantic frame memory;
- state algebra and contradiction detection;
- temporal planning and causal relation ideas;
- grounding and query-routing patterns;
- hallucination failure-mode taxonomy and benchmark scaffolding.

However, `cargo test --no-default-features` timed out during planning, so that codebase remains an unverified reference until its build and test health are confirmed. The current MELM implementation uses Python equivalents for fast ML iteration. A Rust sidecar remains a possible later optimization if the Python event-memory prototype wins.

---

## 4. Proposed Architecture

MELM is best understood as a runtime around a small language model, not only as a new model architecture. The runtime decides whether a query can be handled by local rules and active state, whether it needs event/state memory, whether it needs semantic retrieval, whether it needs a larger model, or whether it should abstain.

```text
Surface text
  -> small parser/composer model
  -> procedural working memory
       - active facts
       - rules and cached inferences
       - action preconditions and policy gates
  -> hierarchical memory when needed
       - event/state memory
       - semantic memory
       - transcript/document evidence
  -> grounded response, action, or abstention
```

### 4.1 Layer 1: Procedural working memory

The first product-grade MELM layer is a rule-aware working-memory runtime. In the proposed design, it supports:

- typed facts and entities;
- current task/session state;
- production-style rules;
- cached derived facts;
- action preconditions and postconditions;
- contradiction checks;
- provenance for every fact and rule;
- allow/deny/warn decisions for proposed actions;
- routing decisions for when memory retrieval is required.

Example:

```text
query: "Can I approve this refund?"

working_memory:
  user_verified = true
  refund_amount = 420
  policy_limit_without_manager = 250
  manager_approval_present = false

decision:
  deny
  reason = "manager approval required for refunds above 250"
```

This layer validates structure and state. It does not pretend to know facts that are not present. If local facts and rules are insufficient, it routes to event memory, semantic memory, a tool, or abstention.

### 4.2 Layer 2: Hierarchical event/state memory

The first event memory is external. Its proposed schema stores:

```text
event_id
source_span
time_index
actors
action_or_state
objects
location
causal_links
salience
surprise_score
embedding
previous_event_id
next_event_id
provenance
```

Retrieval combines:

- vector similarity;
- entity match;
- temporal neighbors;
- frequency and recency of prior access;
- recency decay;
- causal links;
- contradiction checks;
- state transition resolution.

Access history is treated as a retrieval prior, not as truth. Frequently
or recently recalled events may be faster to surface, but grounding and
contradiction checks still decide whether the event is admissible evidence
for the current question.

The event store can begin with SQLite/Postgres plus FAISS or Qdrant. Property-graph tooling is optional and becomes useful only if graph traversal becomes a bottleneck.

### 4.3 Layer 3: Small model interface

The neural model begins as a parser, router, and response composer:

- parse user utterances into candidate intents, entities, and requested operations;
- propose retrieval plans or action plans;
- convert evidence into natural language;
- ask clarification questions;
- fall back to larger models only when the local runtime cannot decide.

The first proof uses existing small open models or the current 23.3M local checkpoint scaffolding. Training a larger custom model is a later phase.

### 4.4 Layer 4: Morphology-aware input and language compression

Tokenizer hypotheses:

- standard BPE baseline;
- standard Unigram baseline;
- tiered morphology-plus-Unigram tokenizer, where frequent forms remain whole, ordinary variation uses Unigram pieces, and explicit morphology is a constrained fallback/override;
- MorphBPE-style constrained tokenizer;
- morpheme tags as auxiliary features;
- byte/patch baseline inspired by BLT or SpaceByte.

Current implementation note: the preferred MELM tokenizer candidate is now a
tiered morphology-Unigram arm, not pure morphology. Early validation shows that
pure capped morphology is strong on compression, but the tiered hybrid better
matches the architecture thesis and currently gives the best BabyLM fast-BLiMP
checkpoint result among local tiny checkpoints. As of the May 2026 local gate,
the tiered hybrid has advanced to a checkpointed small-model stage, not to final
tokenizer status: it beats the best HF baseline by 3.27% bits/byte in the
200-step three-seed tiny progression and by 3.08% in the first larger local
proxy, but HF BPE still leads fast entity tracking by 0.76 percentage points.
A symbolic state tracker solves the same fast entity-tracking set at 100.00%,
which makes that gap an explicit event/state-memory integration target rather
than a pure tokenizer failure.

Updated checkpoint-stage result: the 23.3M-parameter local BabyLM stage has now
completed at 1,000 steps and three seeds. The tiered hybrid beats HF BPE by
2.38% bits/byte, wins two of three fast-BLiMP scoring views against HF baselines,
and edges HF BPE on fast entity tracking by 0.48 percentage points. The current
gate is therefore to integrate explicit event/state memory. Capped morphology
remains the compression control, not the production tokenizer.

First integration result: BabyLM fast entity-tracking prompts can now be
compiled into explicit event records and evaluated with a state-first/LM-fallback
policy. On the completed 23.3M checkpoint stage, this lifts the tiered hybrid
from 40.42% LM-only accuracy to 100.00% state-assisted accuracy on the regular
fast fixture. This is not a claim of general semantic parsing; it is a proof
that the MELM event/state layer can absorb the exact class of bookkeeping error
the small LM still makes.

The current memory integration gate combines that state-assisted result with the
existing synthetic, authored-dialogue, sample-transcript, and abstention checks.
It advances MELM to a persistent child-level dialogue demo with tiered
morphology-Unigram, explicit event/state memory, and evidence-gated responses.

The first demo scaffold now exists: it stores authored child-dialogue events,
answers only when evidence clears the gate, and abstains otherwise. On the
authored dialogue fixture it reaches 86.67% evidence-gated accuracy with 80.00%
positive recall and 100.00% negative abstention. The current answer surface is
evidence summarization, not a free-form neural chat model.

Post-validation questions:

- full FST coverage target;
- fixed 12,000-morpheme vocabulary;
- reversible morpheme-only generation.

### 4.5 Layer 5: Compositional lexical representation

The original Tree-LSTM morpheme composition remains an interesting post-validation design. The validation version starts simpler:

- ordinary token embeddings for baseline comparability;
- optional auxiliary morpheme-boundary and morpheme-tag losses;
- optional factorized embeddings only if tokenizer ablations show a clear morphology advantage.

### 4.6 Layer 6: Output generation and grounding

For the validation demo, output remains ordinary text. Grounding happens through evaluation and post-generation checks:

- event consistency;
- state-transition plausibility;
- temporal order;
- contradiction detection;
- age-appropriate response rubric.

Morpheme-tuple decoding and FST realization are deferred until morphology proves it improves downstream performance.

---

## 5. Data Strategy

### 5.1 Organic data

Candidate sources:

- BabyLM 10M and 100M tracks;
- CHILDES/TalkBank where licensing permits;
- Simple English Wikipedia;
- Project Gutenberg children's texts after license filtering;
- OpenAssistant only after strong child-suitability filtering.

### 5.2 Synthetic data

Synthetic data is restricted to examples grounded in event records. Each generated sample includes:

- source event record;
- target capability;
- teacher model identifier and date;
- rejection/filter reason if discarded;
- contamination check status.

Synthetic dialogue fills measured gaps rather than inflating the corpus.

### 5.3 Data quality metrics

The corpus pipeline reports:

- token count and word count;
- tokenizer compression ratio;
- OOV or fallback rate;
- morphological boundary F1 where gold or proxy labels exist;
- child-suitability filter rate;
- synthetic rejection rate;
- deduplication statistics;
- license/source manifest.

---

## 6. Evaluation Plan

### 6.1 Procedural working-memory evaluation

Compare:

- prompt-only agent;
- ordinary schema validation;
- rule engine without memory;
- MELM procedural working memory with state, rules, cached inferences, and provenance.

Tasks:

- action precondition checks;
- policy allow/deny/warn decisions;
- state update tracking;
- contradiction detection;
- rule conflict detection;
- routing decisions for local answer vs memory retrieval vs abstention.

Metrics:

- decision accuracy;
- false-allow and false-deny rates;
- citation/provenance precision;
- latency per decision;
- number of model calls avoided;
- token cost avoided;
- rule extraction precision/recall when rules are learned from documents.

Success signal: MELM Guard beats prompt-only and schema-only baselines on false-allow rate, traceability, and latency on at least one realistic workflow.

### 6.2 Episodic and state-memory evaluation

Compare:

- standard vector RAG;
- RAG plus temporal/entity metadata;
- event memory with entity/time/causal retrieval;
- temporal knowledge graph baseline where feasible;
- long-context baseline when model/context budget permits.

Tasks:

- direct event recall;
- temporal neighbor recall;
- cross-event causal question answering;
- current-state resolution after updates;
- contradiction detection;
- multi-session persistence;
- answerable vs unanswerable abstention.

Success signal: event/state memory beats standard RAG by at least 15% on controlled episodic recall and state-update tasks at matched context and retrieval budget.

### 6.3 Small-model runtime evaluation

Compare:

- larger model alone;
- smaller model alone;
- smaller model plus RAG;
- smaller model plus MELM runtime.

Metrics:

- task success;
- hallucination/false-answer rate;
- answerable abstention and unanswerable abstention;
- tool-call accuracy;
- latency;
- input/output tokens;
- memory and CPU/GPU footprint.

Success signal: small model plus MELM runtime matches or beats a materially larger baseline on a constrained workflow while using less context, lower latency, or lower cost.

### 6.4 Tokenizer and language modeling evaluation

Compare:

- BPE;
- Unigram;
- MorphBPE/MorphPiece-inspired tokenizer;
- morpheme auxiliary supervision;
- byte/patch baseline.

Metrics:

- validation loss/perplexity under matched model and token budgets;
- compression ratio;
- OOV/fallback rate;
- morphological boundary alignment;
- BabyLM evaluation score;
- generation quality spot checks.

### 6.5 Child-level conversational evaluation

MELM is evaluated on:

- age-appropriate vocabulary;
- syntactic generalization through BabyLM/BLiMP-style tasks;
- simple world knowledge through EWOK-style tasks;
- false-belief and perspective-taking probes;
- counterfactuals over small synthetic worlds;
- session-to-session event recall.

Human review may be used for conversation quality, but it is paired with reproducible automatic tasks.

### 6.6 Baselines

Required baselines:

- prompt-only agent;
- schema-only validation;
- existing policy/guardrail runtime where feasible;
- existing memory systems where feasible, including MemGPT/Letta-style and temporal-graph baselines;
- same-size BPE model with no event memory;
- same-size Unigram model;
- byte/patch baseline if feasible;
- BPE plus ordinary RAG;
- existing small open model plus external event memory;
- long-context baseline where practical.

The project succeeds only if MELM beats strong baselines on specific metrics.

---

## 7. Falsification Milestones

### Milestone 1: procedural working-memory smoke

Near-term test:

- define one narrow workflow with real rules, such as refunds, tutoring progress, claims triage, or device troubleshooting;
- convert the rules into typed facts, preconditions, postconditions, and action gates;
- compare MELM Guard against prompt-only and schema-only baselines;
- report false-allow, false-deny, latency, token cost, and traceability.

Success signal: the runtime blocks invalid actions and allows valid actions more reliably than prompt-only control, with every decision grounded in an explicit rule or fact.

### Milestone 2: event/state memory proof

Near-term test:

- build 500-1,000 multi-session queries from transcripts, workflow logs, or realistic synthetic sessions with held-out templates;
- include temporal updates, contradictions, stale facts, unknown facts, and state changes;
- compare vector RAG, temporal/entity RAG, long-context where feasible, and MELM event/state memory.

Success signal: event/state memory beats the best RAG baseline by at least 15% on state-update and temporal-reasoning tasks while maintaining a lower false-answer rate through abstention.

### Milestone 3: small-model runtime proof

Near-term test:

- pair MELM runtime with a small local model;
- compare against a larger model alone, the same small model alone, and small model plus RAG;
- measure task accuracy, hallucination rate, action validity, cost, latency, and context tokens.

Success signal: small model plus MELM runtime matches or beats a larger-model baseline on a constrained workflow at lower cost or latency.

### Milestone 4: external benchmark demo

Near-term test:

- package the runtime as a local API or CLI;
- publish a benchmark harness and replayable dataset;
- include at least one realistic demo with persistent memory, rule validation, and abstention;
- compare against RAG, long-context, and at least one open memory/runtime baseline where feasible.

Success signal: an external reviewer can run the benchmark and reproduce the advantage without hand-editing the fixtures.

### Milestone 5: language/model integration decision

The integrated-model hypothesis becomes worth testing when the runtime milestones show an independent advantage:

- morphology-aware tokenizer beats BPE/Unigram/byte baselines on at least one meaningful downstream metric, or morphology stays auxiliary;
- the best 125M-370M configuration beats the same-size BPE baseline on BabyLM average, or clearly wins runtime-integrated episodic/state tasks without degrading language quality;
- release demo, benchmark harness, validation report, and checkpoint if results are clean; or release a negative-results report with reproducible artifacts.

---

## 8. Roadmap, Team, And Budget

### 8.1 Six-month validation demo

| Time | Focus | Deliverable |
|---|---|---|
| Days 1-14 | MELM Guard smoke | One workflow with typed facts, rules, action gates, prompt-only comparison, false-allow/false-deny metrics |
| Days 15-30 | Event/state memory proof | 500-1,000 multi-session queries, RAG/temporal-RAG/long-context comparisons, abstention metrics |
| Days 31-60 | Small-model runtime proof | Small model plus MELM runtime versus larger model alone and small model plus RAG |
| Days 61-90 | External benchmark demo | Reproducible harness, replayable dataset, local API/CLI, realistic demo, external-review packet |
| Months 4-5 | Language/model integration | Tokenizer and morphology ablations, optional 125M-370M run if runtime milestones justify it |
| Month 6 | Release | Validation report, benchmark harness, demo, model/checkpoint if clean |

The fastest validation path is not to train a new model first. It is to prove that explicit rules, state, memory, and abstention make existing small models more reliable and cheaper on concrete workflows. Training becomes justified only after that runtime advantage is measurable.

### 8.2 9-12 month publishable MVP

Add:

- production-grade indexed event/state store;
- workflow-specific rule extraction from documents;
- external benchmark comparisons against memory and guardrail systems;
- serious 800M run;
- broader ablation matrix;
- tokenizer-free baseline;
- external memory vs integrated memory comparison;
- CPU inference optimization;
- paper, model card, and public release.

### 8.3 Compute budget

The lower-bound pretraining estimate remains useful:

```text
training FLOPs ~= 6 * parameters * tokens
```

For 800M parameters and 200M tokens, the lower bound is about `1e18` FLOPs. That is not the real budget. The practical budget must include failed runs, ablations, evaluation, dataloading, checkpointing, optimizer state, sequence length effects, kernel instability, and integration time.

Practical planning bands:

- prototype and ablations: 500-2,000 H100-equivalent hours;
- serious 370M-800M runs plus evals: 2,000-8,000 H100-equivalent hours;
- compute budget: $25k-$75k depending on provider, utilization, and reruns.

Total six-month project budget remains dominated by engineering time, not raw FLOPs.

### 8.4 Team

Two senior practitioners are enough for the validation phase:

- ML engineer: training, baselines, scaling, experiment tracking;
- NLP/retrieval engineer: tokenizer pipeline, event memory, evaluation, data quality.

A third contributor is useful only after the baseline harness is stable.

---

## 9. Out Of Scope For Six Months

- Claiming human-like five-year-old cognition.
- Full morpheme-only generation.
- Full reversible FST output decoder.
- Integrated event-token pretraining before external memory wins.
- 125M-800M training before procedural working memory and event/state memory win against baselines.
- Multilingual extension.
- Audio/prosody grounding.
- AMR/UCCA formal interoperability study.
- 800M public model unless smaller milestones strongly justify it.

---

## 10. Conclusion

MELM is a good research bet if it stays honest and leads with the runtime, not the model. The original intuition is strong: language exposes morphology, experience arrives as events, and small models may benefit from making both explicit. The stronger current intuition is broader: useful intelligence can be distributed across a small neural model, procedural working memory, rules, event/state memory, semantic memory, and evidence-gated output.

The fastest proof is therefore not an 800M model. It is a 90-day benchmark showing that a rule-aware working-memory layer plus event/state memory makes existing small models more reliable, more auditable, cheaper, and better at abstaining than prompt-only agents and ordinary RAG. If that works, MELM earns a six-month validation release and a 9-12 month push toward integrated small-model training. If it fails, the project still leaves behind useful tools and a clear map of where the cognitive-runtime hypothesis breaks.

---

## References

**Cognitive runtime, working memory, and agent control**

- Soar cognitive architecture manual. https://soar.eecs.umich.edu/soar_manual/02_TheSoarArchitecture/
- ACT-R cognitive architecture. http://act-r.psy.cmu.edu/
- Rete: A Fast Algorithm for the Many Pattern/Many Object Pattern Match Problem. https://doi.org/10.1016/0004-3702(82)90020-0
- XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. https://xgrammar.mlc.ai/
- LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents. https://arxiv.org/abs/2505.03574
- Gartner prediction on small task-specific AI models. https://www.gartner.com/en/newsroom/press-releases/2025-04-09-gartner-predicts-by-2027-organizations-will-use-small-task-specific-ai-models-three-times-more-than-general-purpose-large-language-models
- Gartner prediction on agentic AI project cancellations. https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027

**Agent memory and long-term state**

- MemGPT: Towards LLMs as Operating Systems. https://arxiv.org/abs/2310.08560
- Letta/MemGPT agent memory architecture. https://docs.letta.com/guides/agents/architectures/memgpt
- Zep: A Temporal Knowledge Graph Architecture for Agent Memory. https://arxiv.org/abs/2501.13956
- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. https://arxiv.org/abs/2504.19413
- LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. https://arxiv.org/abs/2410.10813
- MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks. https://arxiv.org/abs/2602.16313

**Morphology and tokenization**

- Sanchez-Gutierrez, C. H., Mailhot, H., Deacon, S. H., and Wilson, M. A. (2017). MorphoLex: A derivational morphological database for 70,000 English words. https://link.springer.com/article/10.3758/s13428-017-0981-8
- Creutz, M., and Lagus, K. (2007). Unsupervised models for morpheme segmentation and morphology learning. https://dl.acm.org/doi/10.1145/1187415.1187418
- Jabbar, H. (2023). MorphPiece: A Linguistic Tokenizer for Large Language Models. https://arxiv.org/abs/2307.07262
- MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training. https://arxiv.org/abs/2502.00894
- Evaluating Morphological Alignment of Tokenizers in 70 Languages. https://arxiv.org/abs/2507.06378
- Tokens with Meaning: A Hybrid Tokenization Approach. https://arxiv.org/abs/2508.14292

**Tokenizer-free and byte-level modeling**

- MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers. https://arxiv.org/abs/2305.07185
- SpaceByte: Towards Deleting Tokenization from Large Language Modeling. https://arxiv.org/abs/2404.14408
- Byte Latent Transformer: Patches Scale Better Than Tokens. https://arxiv.org/abs/2412.09871
- T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations. https://arxiv.org/abs/2406.19223

**Episodic memory and event representation**

- Fountas, Z. et al. (2024). Human-inspired Episodic Memory for Infinite Context LLMs. https://arxiv.org/abs/2407.09450
- SEEM: Structured Episodic Event Memory. https://arxiv.org/abs/2601.06411
- Memory^3: Language Modeling with Explicit Memory. https://arxiv.org/abs/2407.01178
- A-MEM: Agentic Memory for LLM Agents. https://arxiv.org/abs/2502.12110

**Architectures**

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/abs/2312.00752
- Mamba-3: Improved Sequence Modeling using State Space Principles. https://arxiv.org/abs/2603.15569
- Hymba: A Hybrid-head Architecture for Small Language Models. https://arxiv.org/abs/2411.13676
- Nemotron 3 Super. https://arxiv.org/abs/2604.12374

**Data quality, small models, and evaluation**

- Phi-3 Technical Report. https://arxiv.org/abs/2404.14219
- Phi-4 Technical Report. https://arxiv.org/abs/2412.08905
- Findings of the Second BabyLM Challenge. https://arxiv.org/abs/2412.05149
- BabyLM venues and shared tasks. https://aclanthology.org/venues/babylm/
- EleutherAI lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
- RULER long-context evaluation. https://arxiv.org/abs/2404.06654
- LongBench v2. https://arxiv.org/abs/2412.15204

**Project-adjacent source**

- `C:\dev\nameless_vector`: semantic frame memory, state algebra, temporal reasoning, grounding, and hallucination benchmark concepts. Treat as unverified until build/test health is confirmed.
