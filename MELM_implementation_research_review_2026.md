# MELM Implementation Research Review

Date: 2026-05-10  
Project reviewed: `MELM_whitepaper.md`, `MELM_whitepaper.docx`, `MELM_collaborators.csv`

## Executive Verdict

The MELM whitepaper has a real thesis worth pursuing: combine better lexical primitives with structured event memory, then evaluate under a BabyLM-style low-data regime. The strongest version of the project is not "replace scaling with morphemes" but "measure whether explicit linguistic and episodic inductive biases buy capability per parameter in small models."

The current plan is too confident in four places:

1. A fully custom morpheme-in/morpheme-out model is likely too much for a six-month MVP.
2. EM-LLM is not a drop-in trained event processor for a Mamba backbone; it is mainly a no-finetune wrapper/retrieval method around existing LLMs.
3. The compute math uses idealized FLOPs and understates data, evaluation, integration, failed runs, and tokenizer/backbone mismatch costs.
4. The evidence for morphology-aware tokenization is strongest in low-resource and morphologically rich languages; English-only gains are plausible but not guaranteed.

Recommended path: build MELM as a staged research system. First prove tokenization and event memory advantages independently against strong BPE, byte/patch, Qwen-family small language model (SLM), RAG, and long-context baselines. Only then train the integrated 370M-800M model.

## Repository State

This repo is currently documentation-only:

- `MELM_whitepaper.md`: main proposal.
- `MELM_whitepaper.docx`: document export.
- `MELM_collaborators.csv`: outreach/collaborator target list.

There is no implementation code, training pipeline, dataset manifest, benchmark harness, tokenizer prototype, or model config yet. So "existing code review" is effectively a project-doc feasibility review.

## What The Whitepaper Gets Right

The architecture is aligned with several real 2024-2026 trends:

- Efficient sequence models are improving. Mamba-3 was published in March 2026 and reports gains on retrieval, state tracking, and downstream language modeling while prioritizing inference efficiency.
- Hybrid SSM/attention models are now mainstream enough to treat seriously. Hymba and NVIDIA Nemotron 3 show that SSM + attention/MoE hybrids are a serious design family, not a fringe idea.
- Explicit memory is a live research frontier. EM-LLM, SEEM, Memory^3, A-MEM, MemGPT/Letta-style systems, and the 2025 agent-memory survey all point in the same direction: memory should be treated as a system primitive, not just a longer prompt.
- BabyLM is the right evaluation culture for this project. It creates a disciplined low-data regime and already includes grammar, pragmatics, world knowledge, and, increasingly, grounded evaluation.
- Synthetic data is credible when heavily controlled. Phi-4 is a strong example that data quality, curriculum, and synthetic generation can outperform naive scale in some settings.

## Claims That Need Correction Or De-risking

### 1. "This is not a scale problem" is too strong

The BabyLM 2024 findings found a strong relationship between training FLOPs and average performance. That directly weakens any claim that architecture and tokenization replace scale. A better claim is:

> Scale remains important, but better primitives and memory structures may improve capability per parameter in the sub-1B, low-data regime.

### 2. Morphology is promising, but not proven for English-only MELM

MorphBPE, MorphPiece, and MorphScore-style evaluations support morphology-aware tokenization. But the most reliable gains are in low-resource or morphologically rich languages. English is comparatively analytic, so the MVP must include BPE, Unigram, byte/patch, MorphBPE, and FST-assisted variants as first-class baselines.

Do not commit to a 12,000-morpheme fixed vocabulary until ablations show it wins. Treat 4k, 8k, 12k, 20k, and byte/patch alternatives as experimental arms.

### 3. A pure FST + Tree-LSTM front-end is probably too much for MVP

FST analyzers are useful, but English derivational morphology is messy: borrowed roots, opaque etymology, compounds, named entities, slang, child language, spelling variation, and idioms will create edge cases. A reversible morpheme decoder is even harder.

Recommended MVP simplification:

- Keep ordinary surface text generation at first.
- Add morpheme features as auxiliary inputs or tokenizer constraints.
- Use FST/Morfessor/MorphBPE for segmentation and analysis.
- Defer morpheme-tuple output decoding until tokenizer gains are proven.

### 4. EM-LLM is not plug-and-play Layer 3

EM-LLM is valuable, but the whitepaper overstates the integration ease. It is a no-finetune episodic-memory method for existing LLMs, not a ready-made trainable event processor inside a Mamba model.

Recommended MVP simplification:

- Build the event memory as an external service first.
- Use surprise-based segmentation, sentence/discourse boundaries, entity extraction, embeddings, and temporal-neighbor retrieval.
- Compare it against RAG, MemGPT/Letta-style virtual context, A-MEM-style dynamic notes, and LongRAG-like retrieval.
- Integrate event tokens into pretraining only after the external system wins.

### 5. Mamba-3 is promising, but checkpoint availability is a dependency

Mamba-3 exists as a 2026 paper, but the MVP should not depend on Mamba-3 checkpoints being available in the exact 370M-800M form needed. The older `state-spaces/mamba-790m-hf` checkpoint exists, and hybrid alternatives such as Hymba and Nemotron 3 provide design evidence, but Nemotron 3 Super is far too large to serve as a sub-1B warm start.

Recommended wording:

> Candidate backbones: Mamba-3 if suitable checkpoints/kernels are available; otherwise Mamba-2/state-spaces checkpoints, Hymba-style hybrid design, or a strong small Transformer baseline.

### 6. Compute math is materially optimistic

The FLOPs estimate `6 * parameters * tokens` is a useful lower-bound sanity check, but the "30 minutes per full training run" implication is unrealistic for a serious research loop. Real utilization, dataloading, optimizer states, evals, checkpoints, sequence length effects, Mamba kernels, failed runs, and ablations dominate.

Also, 800M parameters trained on 200M tokens is heavily undertrained unless warm-starting works extremely well. If the tokenizer/backbone changes too much, warm-starting may lose much of its value.

Use budget bands:

- Prototype and ablations: 500-2,000 H100-equivalent hours.
- Serious 370M-800M runs plus evals: 2,000-8,000 H100-equivalent hours.
- Practical compute budget: $25k-$75k, not because raw FLOPs require it, but because research iteration does.

### 7. "Five-year-old conversational competence" needs a narrower operational definition

The current benchmark target bundles vocabulary, conversation, theory of mind, counterfactuals, narrative sequencing, and episodic memory. That is a lot. The MVP should define a testable subset:

- Age-appropriate vocabulary and grammar.
- Multi-turn consistency over controlled stories.
- Simple false-belief and perspective-taking tasks.
- Event recall across sessions.
- Counterfactual reasoning over small synthetic worlds.
- Safe refusal / uncertainty behavior.

Do not claim human-like five-year-old competence from isolated language tests.

## Best Practical Architecture For MVP

### MELM-0: Baseline Lab

Before building a custom model, establish a reproducible baseline suite:

- BabyLM strict 10M and 100M tracks.
- BPE/Unigram tokenizer baseline.
- MorphBPE/MorphPiece-inspired tokenizer.
- Byte/patch baseline inspired by BLT or SpaceByte where feasible.
- External episodic memory baseline over a small open model.
- Qwen SLM baseline: use the current Qwen 0.6B/1.7B/4B-class open models as practical small-model controls for inference, distillation, and event-memory integration.
- Evaluation harness with BabyLM, BLiMP, EWOK, RULER/LongBench-style retrieval, and custom episodic tasks.

### MELM-1: Morpheme-Aware Small LM

Build a 125M-370M model first:

- Backbone: small Transformer and Mamba/SSM candidate side by side.
- SLM strategy: keep a Qwen-family SLM as the pragmatic reference model for local demos, synthetic-data filtering, retrieval-conditioned prompting, and teacher/student distillation into the MELM candidate.
- Input: surface tokens plus morpheme features, or MorphBPE tokenizer.
- Output: ordinary text tokens.
- Objective: next-token loss plus auxiliary morpheme-boundary / morpheme-tag prediction.

This keeps generation sane while testing whether morphology helps.

### MELM-2: External Event Memory

Use a simple event schema:

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

Retrieval should combine:

- entity match,
- vector similarity,
- temporal neighbors,
- recency decay,
- causal links,
- contradiction detection.

Store it in SQLite/Postgres plus FAISS/Qdrant initially. Use NetworkX or a property graph only when graph traversal becomes necessary.

### MELM-3: Integrated Model

Only after MELM-1 and MELM-2 win independently:

- Train 370M with morphology-aware tokenization and event-memory supervision.
- Add event-boundary auxiliary loss.
- Add retrieval-conditioned pretraining examples.
- Compare against BPE + RAG and BPE + long-context baselines.

### MELM-4: 800M Public Demo

Scale only if 370M results justify it. The demo should be:

- CPU-capable quantized inference,
- persistent event memory,
- controlled child-level dialogue,
- reproducible evaluation report,
- model card with limitations.

## Recommended Tool Stack

### Core ML

- PyTorch 2.x
- Hugging Face Transformers / Datasets / Tokenizers
- Accelerate, FSDP, or DeepSpeed
- Mamba / mamba-ssm kernels where stable
- Weights & Biases or MLflow for experiment tracking
- Hydra/OmegaConf for configs
- DVC or Git-LFS for data manifests and artifacts

### Tokenization And Morphology

- Hugging Face Tokenizers for BPE/Unigram baselines
- Morfessor 2.0 for unsupervised segmentation fallback
- MorphBPE-style constrained merges
- MorphoLex for English derivational analysis
- CELEX if licensing is acceptable
- foma or HFST for finite-state experiments
- spaCy/Stanza for POS, lemma, sentence, entity features
- BLT/SpaceByte/T-FREE-inspired byte or patch baselines as counterfactuals

### Memory

- EM-LLM for surprise-based event segmentation/retrieval ideas
- SEEM as a design reference for graph + episodic split
- MemGPT/Letta-style memory tiers for conversational persistence
- A-MEM-style dynamic memory note linking
- Memory^3 as research inspiration, not an MVP dependency
- SQLite/Postgres for event tables
- FAISS/Qdrant for embedding retrieval

### Data

- BabyLM 2024/2025 corpora
- CHILDES/TalkBank, subject to license and access constraints
- Simple English Wikipedia
- Project Gutenberg children's texts, license-filtered
- OpenAssistant only after age-appropriateness filtering
- Synthetic data generated from grounded event records, not unconstrained prompting

### Teacher Models

Use a provider-agnostic generation interface. As of this review:

- Qwen-family SLMs should be treated as the default local/open small-model strategy: use 0.6B/1.7B/4B-class checkpoints for cheap ablations, constrained generation, event-memory demos, and distillation targets before committing to custom MELM training runs.
- DeepSeek V4 preview/API exists, but model IDs and pricing should be checked at generation time.
- OpenAI's official model list currently points users toward GPT-5.2/GPT-5.1 family models, with GPT-5 mini/nano as cheaper options.
- Local or hosted Llama 3.3 70B remains useful for open-model comparison, but it is not a current frontier teacher.

### Evaluation

- EleutherAI `lm-evaluation-harness`
- BabyLM evaluation suite
- BLiMP
- EWOK
- LongBench v2, RULER, and/or InfiniteBench for long-context controls
- Custom episodic recall benchmark with controllable event graphs
- Human review rubric for age-appropriate conversation
- Contamination checks for synthetic tasks

## Revised Timeline

### 6-month version: credible demo, not definitive thesis

Month 1:
- Reproduce BabyLM baseline.
- Build tokenizer comparison harness.
- Build first event-memory prototype.

Month 2:
- Run tokenizer ablations on 10M/100M subsets.
- Build custom episodic benchmark.
- Start data-quality filters.

Month 3:
- Train 125M-370M baselines.
- Compare BPE vs morphology-aware vs byte/patch.
- Compare event memory vs RAG.

Month 4:
- Train best 370M configuration.
- Add auxiliary event-boundary/morpheme objectives if justified.

Month 5:
- Integrate persistent conversational demo.
- Run BabyLM + episodic + long-context evals.

Month 6:
- Release demo, benchmark report, data pipeline, and model/checkpoint if results are clean.

### 9-12 month version: publishable MELM-MVP

Add:

- serious 800M run,
- stronger ablation matrix,
- tokenizer-free baseline,
- external memory vs integrated memory comparison,
- CPU inference optimization,
- written paper/model card.

## Revised Kill Criteria

End of Month 1:
- BabyLM baseline reproduced.
- Tokenizer harness can report compression, OOV/fallback, morphological-boundary F1, and downstream loss.
- Event-memory prototype retrieves temporal neighbors and entity-linked events.

End of Month 2:
- Morphology-aware tokenizer beats BPE/Unigram on at least one meaningful downstream metric, or the project pivots to morphology as auxiliary supervision only.

End of Month 3:
- Event memory beats standard RAG by at least 15% on controlled episodic recall at matched context and retrieval budget.

End of Month 4:
- 370M best model beats same-size BPE baseline on BabyLM average or clearly wins on the project's custom episodic tasks without degrading language quality.

End of Month 6:
- Demo is honest: either "MELM improves X under Y conditions" or "morpheme/event assumptions failed these tests, and here is the evidence."

## Whitepaper Edits I Recommend

Replace:

> This is a tokenization, representation, and memory problem - not a scale problem.

With:

> This is not only a scale problem. MELM tests whether explicit morphology and event memory improve sample efficiency and long-horizon consistency in the sub-1B regime.

Replace:

> EM-LLM enters our build as integration work, not novel research.

With:

> EM-LLM provides a strong external-memory baseline and event-segmentation prior; integrating event tokens into pretraining remains a research contribution.

Replace:

> The empirical literature is unambiguous.

With:

> The literature is encouraging, especially for low-resource and morphologically rich languages; English-only gains must be proven by ablation.

Replace the compute section with a lower-bound FLOPs calculation plus a practical research-iteration budget.

Add an explicit "competing tokenizer-free approaches" section covering BLT, SpaceByte, MEGABYTE, and T-FREE.

Add a "checkpoint dependency" note for Mamba-3.

## Collaborator CSV Notes

The collaborator list is directionally useful. The best first outreach targets are:

- EM-LLM authors for event segmentation and retrieval sanity checks.
- BabyLM organizers for evaluation discipline.
- computational morphology researchers for tokenizer/FST review.
- lm-evaluation-harness maintainers or experienced users for benchmark integration.

Before outreach, verify current affiliations, contact channels, and whether public handles are still active. The CSV should be treated as a lead list, not a verified contact database.

## Source Notes Checked

- EM-LLM: https://arxiv.org/abs/2407.09450 and https://em-llm.github.io/
- SEEM: https://arxiv.org/abs/2601.06411
- Mamba-3: https://arxiv.org/abs/2603.15569
- Original Mamba: https://arxiv.org/abs/2312.00752
- Hymba: https://arxiv.org/abs/2411.13676
- NVIDIA Nemotron 3 Super: https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/ and https://arxiv.org/abs/2604.12374
- MorphPiece: https://arxiv.org/abs/2307.07262
- MorphBPE: https://arxiv.org/abs/2502.00894
- Morphological Alignment in 70 Languages: https://arxiv.org/abs/2507.06378
- Tokens with Meaning: https://arxiv.org/abs/2508.14292
- SpaceByte: https://arxiv.org/abs/2404.14408
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871 and https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/
- MEGABYTE: https://arxiv.org/abs/2305.07185
- T-FREE: https://arxiv.org/abs/2406.19223
- Phi-4: https://arxiv.org/abs/2412.08905
- Phi-3: https://arxiv.org/abs/2404.14219
- BabyLM 2024 findings: https://arxiv.org/abs/2412.05149
- BabyLM ACL venue: https://aclanthology.org/venues/babylm/
- MemGPT: https://arxiv.org/abs/2310.08560
- Memory^3: https://arxiv.org/abs/2407.01178
- A-MEM: https://arxiv.org/abs/2502.12110
- Memory in the Age of AI Agents: https://arxiv.org/abs/2512.13564
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- LongBench v2: https://arxiv.org/abs/2412.15204
- RULER: https://arxiv.org/abs/2404.06654
- NVIDIA H100 specs: https://www.nvidia.com/en-us/data-center/h100/
- Lambda pricing: https://lambda.ai/service/gpu-cloud/pricing
- OpenAI model list: https://platform.openai.com/docs/models
- DeepSeek API pricing/news: https://api-docs.deepseek.com/quick_start/pricing/ and https://api-docs.deepseek.com/news/news260424
- Qwen model family: https://qwen.moe/ and https://qwenlm.github.io/blog/qwen3/
