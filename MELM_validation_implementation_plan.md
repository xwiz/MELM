# MELM Validation Implementation Plan

Date: 2026-05-10  
Target: six-month validation demo  
Primary repo: `C:\Users\Son\cowork\MELM`  
Parts donor: `C:\dev\nameless_vector`

## 1. Objective

Build a validation-first MELM prototype that can prove or falsify the core morpheme-event hypothesis before committing to a custom 800M model.

The implementation must answer three questions:

1. Does morphology-aware tokenization or auxiliary morphology supervision improve sample efficiency over BPE, Unigram, and byte/patch baselines?
2. Does structured event memory beat ordinary RAG on controlled episodic recall at the same retrieval budget?
3. Does the best 125M-370M model plus event memory improve child-level dialogue and episodic consistency without degrading ordinary language quality?

The six-month target is a reproducible demo and validation report. The 9-12 month target is a publishable 800M MELM-MVP only if the gates pass.

## 2. Repository Structure

Create and maintain this structure in the MELM repo:

```text
docs/
  validation_report_template.md
  roadmap.md
melm/
  __init__.py
  README.md
benchmarks/
  README.md
experiments/
  README.md
vendor_notes/
  nameless_vector.md
MELM_whitepaper.md
MELM_implementation_research_review_2026.md
MELM_validation_implementation_plan.md
```

Planned future modules:

```text
melm/tokenization/
melm/data/
melm/memory/
melm/training/
melm/evaluation/
benchmarks/episodic_recall/
benchmarks/babylm_adapters/
benchmarks/state_grounding/
experiments/tokenizers/
experiments/memory/
experiments/models/
```

## 3. Workstreams

### A. Tokenization and morphology

Build a tokenizer harness that compares:

- BPE;
- Unigram;
- MorphBPE/MorphPiece-inspired constrained tokenizer;
- morpheme tags as auxiliary supervision;
- byte/patch baseline inspired by BLT or SpaceByte where feasible.

Metrics:

- token/word compression ratio;
- OOV or fallback rate;
- morphological boundary alignment F1 where labels exist;
- validation loss under matched model and data budgets;
- BabyLM score;
- generation quality spot checks.

Decision rule:

- If morphology wins on a meaningful downstream metric, use it for the 370M integration.
- If it does not, keep ordinary tokenization and use morphology only as auxiliary supervision or analysis.

### B. Event memory

Implement an external event-memory prototype before changing the model architecture.

Minimum event schema:

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
```

Retrieval variants:

- ordinary vector RAG;
- RAG plus temporal neighbors;
- event memory with entity, time, and causal retrieval;
- EM-LLM-style segmentation where feasible.

Decision rule:

- Event memory must beat ordinary RAG by at least 15% on controlled episodic recall at matched context and retrieval budget.
- If it does not, simplify to RAG plus temporal/entity metadata.

### C. Small-model training

Train matched 125M-370M models only after tokenizer and memory harnesses are reliable.

Required model arms:

- same-size BPE baseline;
- same-size Unigram baseline;
- best morphology-aware tokenizer or auxiliary morphology model;
- byte/patch baseline if feasible;
- best model plus external event memory.

Backbone choices:

- small Transformer for clarity;
- Mamba/Mamba-2/state-spaces checkpoint if stable;
- Mamba-3 only if checkpoints and kernels are available;
- hybrid SSM/attention only after a simple baseline exists.

Decision rule:

- The best 370M model must beat the same-size BPE baseline on BabyLM average, or clearly win custom episodic tasks without language-quality regression.

### D. Demo and reporting

The six-month demo should include:

- persistent event memory;
- controlled child-level conversation;
- recall of prior session events;
- contradiction and state-transition checks;
- benchmark report with positive or negative findings.

The release can be a success report or a negative-results report. Both are valid if reproducible.

## 4. Six-Month Roadmap

| Month | Focus | Inputs | Outputs | Gate |
|---|---|---|---|---|
| 1 | Baselines | BabyLM data, tokenizer libraries, memory sketch | BabyLM reproduction, tokenizer harness, first event-memory prototype | Baseline runs and retrieval prototype work |
| 2 | Ablations | BabyLM 10M/100M subsets, morphology tools | Tokenizer comparison, episodic benchmark draft, data filters | Morphology beats a baseline or becomes auxiliary |
| 3 | Small model and memory | 125M-370M configs, event benchmark | Model baselines, event memory vs RAG results | Event memory beats RAG by 15% or is simplified |
| 4 | Integration | Best tokenizer and memory results | Best 370M training run, optional auxiliary objectives | 370M beats BPE or wins episodic tasks |
| 5 | Demo | Best model, event memory, grounding checks | Persistent child-level dialogue demo | Demo passes scripted recall and consistency scenarios |
| 6 | Release | All metrics and artifacts | Validation report, benchmark harness, demo, checkpoint if clean | Honest success or negative-results release |

## 5. Validation Metrics

### Tokenizer metrics

- Compression ratio vs BPE/Unigram.
- OOV/fallback rate.
- Boundary alignment F1.
- Validation loss/perplexity.
- BabyLM average score.
- Degradation checks on ordinary generation.

### Memory metrics

- Direct event recall accuracy.
- Temporal neighbor recall accuracy.
- Cross-event causal QA accuracy.
- Contradiction detection precision/recall.
- Retrieval latency and context budget.

### Conversation metrics

- Age-appropriate vocabulary and grammar.
- Multi-turn consistency.
- False-belief and perspective-taking probes.
- Counterfactual reasoning over small worlds.
- Safe uncertainty behavior.

## 6. nameless_vector Reuse

Use `C:\dev\nameless_vector` as a design source only during the first implementation phase.

Reusable concepts:

- semantic frame memory;
- state algebra;
- temporal relation graph;
- grounding layer;
- routing and validation outcomes;
- hallucination failure-mode taxonomy.

Do not blindly port Rust code. First actions:

1. Document the concepts in `vendor_notes/nameless_vector.md`.
2. Build Python equivalents inside MELM for fast ML iteration.
3. Run `nameless_vector` tests separately before any code port.
4. Consider a Rust sidecar only after the Python event-memory prototype wins.

Known issue:

- `cargo test --no-default-features` timed out during planning, so build/test health is unverified.

## 7. Budget

Use these planning bands:

- prototype and ablations: 500-2,000 H100-equivalent hours;
- serious 370M-800M runs plus evals: 2,000-8,000 H100-equivalent hours;
- compute: $25k-$75k depending on provider and utilization;
- engineering: primary cost driver for the six-month validation phase.

Do not plan around the lower-bound FLOPs estimate as if it were wall-clock cost.

## 8. Acceptance Criteria

By the end of six months, the repo should contain:

- revised whitepaper;
- reproducible implementation plan;
- tokenizer harness;
- event-memory prototype;
- episodic recall benchmark;
- BabyLM evaluation adapter or documented reproduction path;
- model configs and training logs for attempted baselines;
- final validation report.

Current Month 1 execution status:

- tokenizer harness implemented with BPE, Unigram-like, morphology heuristic, and byte/patch baselines;
- event-memory, abstention, state-grounding, and state-resolution probes implemented;
- annotated transcript compiler implemented with source-turn provenance and state cases;
- BabyLM-style local manifest adapter implemented;
- BabyLM 2026 Strict-Small downloaded, profiled, and connected to sampled/full tokenizer probes;
- fast HF BPE/Unigram and capped morphology probes implemented;
- tiny PyTorch causal-LM training smoke implemented for manifest-to-tokenizer-to-training validation;
- first matched tiny neural tokenizer ablation completed, promoting capped morphology to longer scaled ablations;
- three-seed tiny neural rerun completed with stable capped-morphology advantage;
- 10/50/100-step progression completed;
- tiny-LM tokenizer/checkpoint artifacts saved, indexed, reloaded, and re-evaluated with zero delta against their training reports;
- child-level minimal-pair checkpoint smoke implemented; current combined checkpoint decision is `hold_for_quality_evidence`;
- tokenizer strategy corrected to the original hybrid thesis: `tiered_morph_unigram` combines a Unigram statistical layer, frequent whole-word fast path, and constrained morphology override; it now wins full fast-BLiMP but still trails BPE on fast entity tracking;
- tokenizer stage gate now says `advance_to_scaled_neural_ablation` after folding in the first larger local proxy: tiered hybrid has a 3.27% mean bits/byte gain over HF BPE at 200-step three-seed tiny scale, wins 3/3 fast-BLiMP scoring views, has only a -0.76 percentage-point entity-tracking delta versus the best baseline, and preserves a 3.08% bits/byte gain in the 2-layer proxy;
- first larger local proxy model also promotes tiered hybrid with a 3.08% bits/byte gain over HF BPE;
- a symbolic BabyLM entity-tracking state baseline reaches 100.00% on all 3,152 fast cases with zero abstentions, which reframes the BPE entity advantage as a state/event-memory integration target rather than a tokenizer-only blocker;
- local execution readiness now passes preflight on the RTX 4060 Laptop GPU, and a same-shape 23.3M-parameter tiered-hybrid checkpoint smoke reloads with zero validation delta; the multiseed runner now caches each tokenizer/seed arm for interruption-safe reruns;
- the generated checkpointed 23.3M-parameter small-model stage completed: tiered hybrid beats HF BPE by 2.38% bits/byte, wins 2/3 fast-BLiMP views against HF baselines, and beats HF BPE on fast entity tracking by 0.48 percentage points;
- small-model stage gate is now `advance_to_event_memory_integration`, while capped morphology remains the compression baseline;
- first state-memory integration check is implemented: BabyLM entity prompts compile into event records, and a state-first/LM-fallback policy lifts tiered-hybrid entity tracking from 40.42% to 100.00% on the regular-format fast fixture;
- memory integration gate is now `advance_to_persistent_dialogue_demo`, combining the state-assisted result with synthetic, authored-dialogue, sample-transcript, and abstention gates;
- first persistent dialogue demo scaffold is implemented over authored child-dialogue events; a narrow causal-source/state-change resolver lifts it to 100.00% evidence-gated accuracy, 100.00% positive recall, and 100.00% negative abstention without lowering the abstention threshold.
- reloadable JSONL session persistence is implemented for the dialogue demo; the seeded authored session reloads 12 events and preserves 100.00% evidence-gated accuracy, 100.00% positive recall, and 100.00% negative abstention.
- transcript-derived persistent session smoke passes on the sample annotation fixture after reload with 4 distractor events: 100.00% regular dialogue evidence accuracy, 100.00% paraphrased/noisy dialogue evidence accuracy, 100.00% state accuracy, and 100.00% event-memory recall@2 versus 66.67% RAG recall@2.
- current Python event memory is evidence/context efficient but not yet CPU/RAM superior to RAG because both retrievers scan the full event list; a Pi-class win requires indexed memory or a Rust/C sidecar;
- sound-symbolism has been removed from the active MVP gate and deferred as a separate research question because the thesis is not mature enough yet;
- expanded morpheme/root/meaning MVP corpus and deterministic inference harness pass 22/22 novel-word cases and 6/6 utterance-routing cases over constructed high-confidence morpheme/root examples.

The final result is acceptable if it clearly states either:

- MELM improves specific metrics under specific conditions; or
- the hypothesis failed specific tests and should be revised.
