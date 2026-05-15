# MELM Tokenizer Strategy 2026

Date: 2026-05-11

## Correction

The intended MELM tokenizer is not pure morphology. It is a leveled hybrid:

1. frequent/stable lexical forms are represented as fast whole units;
2. a statistical Unigram layer handles ordinary subword variation;
3. explicit morphology overrides or augments the statistical layer when the
   morpheme split is recognizable and not too sequence-expensive;
4. character/byte fallback remains available for novel forms.

This better matches the original MELM intuition: familiar material should be
cheap to retrieve, while unfamiliar material should trigger deeper decomposition
rather than collapse to an opaque unknown token.

## Implemented Arms

- `capped_morpheme`: heuristic morphology with a capped vocabulary.
- `hybrid_morph_unigram`: first lexical-memory plus morphology fallback arm.
- `tiered_morph_unigram`: current preferred arm; HF Unigram is the statistical
  layer, frequent words get a fast path, and morphology is used as a constrained
  override.

The first `hybrid_morph_unigram` arm was useful diagnostically but overpaid in
sequence length. The stronger MELM candidate is now `tiered_morph_unigram`.

## Evidence So Far

BabyLM sample tokenizer LM probe, 4,096 tokenizer budget:

| Rank | Tokenizer | Bits/Byte | NLL/Token |
|---:|---|---:|---:|
| 1 | capped_morpheme | 2.222 | 5.618 |
| 2 | tiered_morph_unigram | 2.612 | 5.873 |
| 3 | hf_bpe | 2.729 | 6.224 |
| 4 | hf_unigram | 2.790 | 5.691 |
| 5 | hybrid_morph_unigram | 2.877 | 5.432 |

100-step tiny checkpoint validation:

| Rank | Tokenizer | Bits/Byte |
|---:|---|---:|
| 1 | capped_morpheme | 2.586 |
| 2 | tiered_morph_unigram | 3.027 |
| 3 | hf_bpe | 3.081 |
| 4 | hf_unigram | 3.240 |
| 5 | hybrid_morph_unigram | 3.501 |

Full BabyLM 2026 fast-BLiMP checkpoint ranking:

| Score Field | Winner | Tiered Morph-Unigram | Best Baseline |
|---|---|---:|---:|
| mean NLL/token | tiered_morph_unigram | 54.75% | 54.66% |
| bits/byte | tiered_morph_unigram | 52.66% | 50.29% |
| total NLL | tiered_morph_unigram | 55.34% | 52.99% |

BabyLM 2026 fast entity tracking, completion-conditioned total NLL:

| Rank | Tokenizer | Accuracy |
|---:|---|---:|
| 1 | hf_bpe | 41.88% |
| 2 | hybrid_morph_unigram | 41.62% |
| 3 | tiered_morph_unigram | 41.12% |
| 4 | hf_unigram | 39.50% |
| 5 | capped_morpheme | 17.61% |

Multi-seed tiny progression, 10-200 steps:

| Rank | Tokenizer | 10 steps | 50 steps | 100 steps | 200 steps |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 3.075 | 2.545 | 2.216 |
| 2 | tiered_morph_unigram | 3.763 | 3.539 | 2.989 | 2.679 |
| 3 | hf_bpe | 3.697 | 3.503 | 3.050 | 2.769 |
| 4 | hf_unigram | 4.168 | 3.884 | 3.204 | 2.851 |

Small local proxy model, 200 steps, 1MB/0.5MB byte cap, 2 layers, 128 hidden:

| Rank | Tokenizer | Bits/Byte | NLL/Token |
|---:|---|---:|---:|
| 1 | capped_morpheme | 2.070 | 5.232 |
| 2 | tiered_morph_unigram | 2.491 | 5.602 |
| 3 | hf_bpe | 2.570 | 5.861 |
| 4 | hf_unigram | 2.627 | 5.359 |

Checkpointed small-model stage, 23.3M parameters, 5MB/1MB byte cap, 1,000
steps, three seeds:

| Rank | Tokenizer | Mean Bits/Byte | Std |
|---:|---|---:|---:|
| 1 | capped_morpheme | 1.681 | 0.001 |
| 2 | tiered_morph_unigram | 1.928 | 0.008 |
| 3 | hf_bpe | 1.975 | 0.005 |
| 4 | hf_unigram | 1.996 | 0.007 |

Checkpointed fast-BLiMP and entity results:

| Metric | Winner | Tiered Hybrid | Best HF Baseline |
|---|---|---:|---:|
| Fast-BLiMP mean NLL/token | tiered_morph_unigram | 53.17% | 52.54% |
| Fast-BLiMP bits/byte | hf_unigram | 50.81% | 51.04% |
| Fast-BLiMP total NLL | tiered_morph_unigram | 53.67% | 52.60% |
| Fast entity tracking | tiered_morph_unigram | 40.42% | 39.94% |

## Decision

`tiered_morph_unigram` is now the main tokenizer candidate for the next MELM
integration step. The checkpointed small-model stage gate is
`advance_to_event_memory_integration`. It is the first arm that:

- matches the original hybrid hypothesis;
- beats HF BPE and HF Unigram on 100-step validation bits/byte;
- won the earlier tiny-checkpoint full fast-BLiMP under all tested scoring conventions;
- keeps improving relative to HF BPE as training length increases from 50 to
  200 steps;
- survives a larger local proxy model with a 3.08% bits/byte gain over HF BPE;
- wins the 23.3M-parameter multiseed stage over HF BPE by 2.38% bits/byte;
- wins two of three fast-BLiMP scoring views at the checkpointed stage;
- edges HF BPE on fast entity tracking at the checkpointed stage.

It is not yet a final tokenizer. Capped morphology is still a strong compression
baseline, and it remains the best compression arm. The next gate should connect
the tiered tokenizer to explicit event/state memory rather than keep tuning
tokenization in isolation.

The entity-tracking gap should be interpreted carefully. A symbolic box-state
tracker now solves the BabyLM fast entity-tracking set at `100.00%` accuracy
over 3,152 cases with zero abstentions. That means the task is a strong argument
for explicit state/event memory, not a standalone reason to reject the tiered
tokenizer.

The completed stage report is
`reports/babylm_2026_small_model_stage_gate.md`.

## Design Direction

The most promising novel MELM route is:

- keep Unigram/BPE as serious statistical competitors;
- use tiered morph-Unigram as the primary MELM candidate;
- add morphology as auxiliary supervision even when the tokenizer is not purely
  morphological;
- introduce an adaptive lexical cache later, where recency and frequency can
  promote forms into a faster lexical tier;
- evaluate tokenizers on downstream BabyLM tasks, not only compression.

This is consistent with current morphology-aware tokenization work, which tends
to preserve statistical subword efficiency while adding linguistic constraints,
and with byte/patch work such as BLT, which makes sequence-length and dynamic
segmentation a direct competitor rather than an afterthought.

## Research Context

This strategy is shaped by:

- MorphBPE-style evidence that morphology-aware tokenization is strongest when
  it preserves statistical tokenization efficiency instead of replacing it:
  `https://arxiv.org/abs/2502.00894`
- MorphTok/MoVoC-style work that uses morphology as pre-tokenization or hybrid
  vocabulary construction rather than a pure morpheme-only system:
  `https://arxiv.org/abs/2504.10335`, `https://arxiv.org/abs/2509.08812`
- BLT-style byte/patch models, which make dynamic segmentation and
  sequence-length control a direct competitor:
  `https://arxiv.org/abs/2412.09871`
- BabyLM 2026, which gives the right low-data evaluation culture for this
  question:
  `https://babylm.github.io/`,
  `https://github.com/babylm-org/babylm-eval`
