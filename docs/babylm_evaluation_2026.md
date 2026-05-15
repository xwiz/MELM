# BabyLM 2026 Evaluation Adapter

Date: 2026-05-11

## Local Assets

The official BabyLM 2026 evaluation code is mirrored locally for inspection at:

```text
local_data/babylm_eval_2026/
```

Source:

- `https://github.com/babylm-org/babylm-eval`
- local commit inspected: `467793f`
- fast eval dataset: `BabyLM-community/BabyLM-2026-Strict-Evals`

Only the fast evaluation subset was downloaded for local work:

```powershell
python scripts\profile_babylm_fast_eval.py
```

| Asset | Files | Cases |
|---|---:|---:|
| BLiMP fast | 67 | 13,400 |

The official strict-track README says fast evaluation is intended for checkpoint
testing, while full evaluation is for final models. That makes fast BLiMP a good
next gate for MELM tokenizer checkpoints.

## Current MELM Adapter

Implemented:

- `melm.benchmarks.load_blimp_fast_cases()`
- `scripts/profile_babylm_fast_eval.py`
- `scripts/run_tiny_lm_blimp_fast.py`

The adapter maps BabyLM BLiMP rows with `sentence_good`, `sentence_bad`, `UID`,
and `pair_id` into the same minimal-pair scorer used by local smoke fixtures.

## Current Results

The first full official fast-BLiMP run uses all 67 BLiMP paradigms for 13,400
minimal pairs. After adding the true `tiered_morph_unigram` arm, this is the
strongest tokenizer-quality result so far.

```powershell
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_blimp_fast_full.json --out-md reports\tiny_lm_blimp_fast_full.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field bits_per_byte --out-json reports\tiny_lm_blimp_fast_full_bits_per_byte.json --out-md reports\tiny_lm_blimp_fast_full_bits_per_byte.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field total_nll --out-json reports\tiny_lm_blimp_fast_full_total_nll.json --out-md reports\tiny_lm_blimp_fast_full_total_nll.md
```

| Score Field | Best Baseline | Tiered Hybrid | Capped Morphology |
|---|---:|---:|---:|
| mean NLL/token | `hf_unigram` 54.66% | 54.75% | 53.53% |
| bits/byte | `hf_bpe` 50.29% | 52.66% | 49.57% |
| total NLL | `hf_bpe` 52.99% | 55.34% | 51.79% |

Interpretation:

- Pure capped morphology remains competitive on official fast BLiMP.
- The true tiered morphology-Unigram arm beats the best BPE/Unigram baseline
  under all three tested scoring conventions.
- Fast entity tracking still favors HF BPE, so the promotion should be to
  longer ablation, not to final-tokenizer status.

The fast entity-tracking task is also now covered by an explicit state-tracking
baseline:

```powershell
python scripts\run_entity_tracking_symbolic.py
```

That symbolic baseline reaches `100.00%` accuracy on all 3,152 fast cases with
zero abstentions. This is not a language-model score, but it is important: the
remaining entity-tracking weakness is exactly the kind of event/state bookkeeping
MELM is supposed to externalize, rather than evidence that the tokenizer should
be rejected.

Tokenizer stage gate:

- Decision: `advance_to_scaled_neural_ablation`
- Candidate: `tiered_morph_unigram`
- Latest step: `200`
- Relative bits/byte gain over best HF baseline: `3.27%`
- Fast-BLiMP wins: `3/3`
- Entity accuracy delta versus best baseline: `-0.76%`
- First larger proxy relative bits/byte gain: `3.08%`
- Symbolic state-tracking baseline: `100.00%`

## Checkpointed Small-Model Stage

The generated 23.3M-parameter local stage has now completed for four tokenizer
arms, three seeds, and 1,000 steps over a 5MB/1MB BabyLM byte cap.

Multiseed validation bits/byte:

| Rank | Tokenizer | Mean Bits/Byte | Std |
|---:|---|---:|---:|
| 1 | capped_morpheme | 1.681 | 0.001 |
| 2 | tiered_morph_unigram | 1.928 | 0.008 |
| 3 | hf_bpe | 1.975 | 0.005 |
| 4 | hf_unigram | 1.996 | 0.007 |

Checkpointed downstream fast results:

| Metric | Winner | Tiered Hybrid | Best HF Baseline |
|---|---|---:|---:|
| Fast-BLiMP mean NLL/token | tiered_morph_unigram | 53.17% | 52.54% |
| Fast-BLiMP bits/byte | hf_unigram | 50.81% | 51.04% |
| Fast-BLiMP total NLL | tiered_morph_unigram | 53.67% | 52.60% |
| Fast entity tracking | tiered_morph_unigram | 40.42% | 39.94% |

Small-model stage gate:

- Decision: `advance_to_event_memory_integration`
- Relative bits/byte gain over best HF baseline: `2.38%`
- Fast-BLiMP wins vs HF baselines: `2/3`
- Entity accuracy delta versus best HF baseline: `+0.48%`
- Compression control: `capped_morpheme`

## State-Assisted Entity Tracking

The first event/state-memory integration check now evaluates a hybrid policy:
state memory gets first refusal; if it cannot resolve the case, the evaluator
falls back to the LM prediction.

```powershell
python scripts\run_state_assisted_entity_tracking.py
```

On the completed 23.3M checkpoint stage:

| Tokenizer | LM Accuracy | State-Assisted Accuracy | Lift | State Answer Rate |
|---|---:|---:|---:|---:|
| tiered_morph_unigram | 40.42% | 100.00% | 59.58% | 100.00% |
| hf_bpe | 39.94% | 100.00% | 60.06% | 100.00% |
| capped_morpheme | 38.58% | 100.00% | 61.42% | 100.00% |
| hf_unigram | 31.12% | 100.00% | 68.88% | 100.00% |

The parser compiles each prompt into explicit event records, then replays state
transitions. This is still an oracle-style regular-format benchmark, but it is
now wired as a state-first/LM-fallback integration path rather than a separate
side metric.

## Next Gate

The next useful official work is memory integration, not another tokenizer-only
probe. The meaningful promotion gate should compare:

- official fast BLiMP accuracy;
- official entity-tracking fast accuracy;
- validation bits/byte;
- custom episodic/state benchmarks;
- explicit symbolic/event-memory-assisted state baselines;
- same-size BPE/Unigram baselines.
