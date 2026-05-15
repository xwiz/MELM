# MELM Roadmap

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
- Expanded morpheme/root/meaning MVP corpus and deterministic inference harness pass 22/22 novel-word cases and 6/6 utterance-routing cases over constructed high-confidence morpheme/root examples.
- Current report: `reports/phase1_report.md`.
