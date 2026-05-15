# Corpus Selection 2026

Date: 2026-05-11

## Selected Corpus

Selected:

- `BabyLM-community/BabyLM-2026-Strict-Small`
- `BabyLM-community/BabyLM-dev`
- `BabyLM-community/BabyLM-Test`

Local target:

```text
local_data/babylm_2026_strict_small/
```

Manifest:

```text
reports/babylm_2026_strict_small_manifest.json
```

Rationale:

- It is the current 2026 BabyLM strict-small training corpus.
- It matches the MELM Month 1/2 target: sub-1B, sample-efficiency, 10M-word scale.
- It is small enough for local tokenizer and tiny-training smoke tests.
- It is more directly relevant than generic children-story-only corpora, because it includes multiple domains and challenge baselines.

## Alternatives Considered

- `BabyLM-community/BabyLM-2026-Strict`: better for the 100M track, but too large for the first local ablation pass.
- `cambridge-climb/BabyLM`: useful 2023 reference corpus with 10M/100M/dev/test directories, but older than the 2026 challenge data.
- `nilq/babylm-10M`: convenient parquet mirror, but not the current 2026 challenge source.
- Kaggle BabyLM 2025 dataset: potentially useful later, but less direct for the immediate 2026 strict-small comparison.

## Local Download Summary

Downloaded with:

```powershell
python scripts\download_babylm_2026_strict_small.py
```

Profiled with:

```powershell
python scripts\profile_corpus_manifest.py --manifest reports\babylm_2026_strict_small_manifest.json --out-json reports\babylm_2026_strict_small_profile.json
```

Summary from local profile:

| Split | Docs | Bytes | Lines | Word-like Tokens |
|---|---:|---:|---:|---:|
| train | 6 | 54,399,840 | 1,104,112 | 9,956,033 |
| validation | 6 | 56,752,783 | 1,168,159 | 10,375,121 |
| test | 6 | 51,722,871 | 1,110,006 | 9,431,457 |

The full dependency-free tokenizer probe timed out at 10 minutes, so large-corpus
CLI probes now support deterministic byte caps:

```powershell
python scripts\run_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
python scripts\run_tokenizer_decision.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
```

## First Results

Sampled tokenizer decision on about 1MB train and 0.5MB validation:

- Decision: `auxiliary_only`
- Best LM baseline: `unigram_like`
- Best baseline NLL/token: `4.835`
- Heuristic morphology NLL/token: `6.302`
- LM NLL gain: `-1.466`
- Boundary F1 gain: `52.94%`

Tiny neural tokenizer smoke on about 0.5MB train and 0.25MB validation:

| Rank | Tokenizer | Bits/Byte | NLL/Token |
|---:|---|---:|---:|
| 1 | heuristic_morpheme | 3.051 | 7.780 |
| 2 | simple_bpe | 3.938 | 6.985 |
| 3 | unigram_like | 4.404 | 7.070 |

Interpretation:

- The BabyLM sample reinforces the conservative gate: morphology is still not a primary tokenizer.
- The tiny neural bits-per-byte result is interesting enough to keep morphology in ablations as auxiliary supervision.
- Full BabyLM ablations need a faster trained tokenizer backend or external `tokenizers` integration before using the entire corpus repeatedly.

After adding fast HF BPE/Unigram baselines and a capped morphology arm, the full
train/validation probe completed:

| Rank | Tokenizer | NLL/Token | Bits/Byte | Vocab |
|---:|---|---:|---:|---:|
| 1 | capped_morpheme | 5.842 | 2.312 | 8,192 |
| 2 | hf_bpe | 6.368 | 2.540 | 8,188 |
| 3 | hf_unigram | 5.728 | 2.569 | 8,288 |

Updated interpretation:

- Capped morphology has earned a proper neural tokenizer ablation.
- It is not yet proven as the final primary tokenizer, because bits-per-byte
  under a unigram probe is not the same as BabyLM downstream model quality.
- The next decisive test is matched tiny/small neural training with HF BPE,
  HF Unigram, and capped morphology.

Fast tokenizer decision:

- `promote_to_neural_ablation`
- best baseline: `hf_bpe`
- relative bits-per-byte gain: `9.00%`

The first matched tiny neural ablation on a byte-capped BabyLM sample also
favored capped morphology:

| Rank | Tokenizer | Bits/Byte | NLL/Token |
|---:|---|---:|---:|
| 1 | capped_morpheme | 3.301 | 8.418 |
| 2 | hf_bpe | 3.701 | 8.427 |
| 3 | hf_unigram | 4.170 | 8.427 |

Tiny LM decision:

- `promote_to_scaled_neural_ablation`
- best baseline: `hf_bpe`
- relative bits-per-byte gain: `10.79%`

This is still a smoke run. Parameter counts are now exactly matched by padding
all model vocabularies to 4096 entries. The next run should be longer and use
more seeds.

The three-seed rerun remained stable:

| Rank | Tokenizer | Mean Bits/Byte | Std | Runs |
|---:|---|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 0.005 | 3 |
| 2 | hf_bpe | 3.697 | 0.004 | 3 |
| 3 | hf_unigram | 4.168 | 0.007 | 3 |

Multi-seed decision:

- `promote_to_scaled_neural_ablation`
- best baseline: `hf_bpe`
- relative bits-per-byte gain: `10.85%`

The longer 50 and 100-step multi-seed runs strengthened the result:

| Tokenizer | 10 Steps | 50 Steps | 100 Steps |
|---|---:|---:|---:|
| capped_morpheme | 3.296 | 3.075 | 2.545 |
| hf_bpe | 3.697 | 3.503 | 3.050 |
| hf_unigram | 4.168 | 3.884 | 3.204 |

At 100 steps, capped morphology has a 16.54% relative bits-per-byte gain over
`hf_bpe`. This justifies moving from smoke runs into a small proper BabyLM
training stack with saved tokenizers/checkpoints and downstream evaluation.

The first saved checkpoint artifacts were then reloaded and re-evaluated on the
same 250k-byte validation cap:

| Rank | Tokenizer | Eval Bits/Byte | Reload Delta |
|---:|---|---:|---:|
| 1 | capped_morpheme | 2.586 | 0.000000 |
| 2 | hf_bpe | 3.081 | 0.000000 |
| 3 | hf_unigram | 3.240 | 0.000000 |

The zero delta confirms that tokenizer artifacts, model vocabularies, and
checkpoints can be reused without retraining. The next useful step is downstream
evaluation rather than more tiny-model bits-per-byte probes alone.

The first downstream-style smoke test was an eight-case child-level
minimal-pair ranking fixture. It did not support promoting morphology yet:

| Score Field | Best BPE/Unigram Accuracy | Capped Morphology Accuracy |
|---|---:|---:|
| mean NLL/token | 50.00% | 25.00% |
| bits/byte | 62.50% | 37.50% |
| total NLL | 25.00% | 12.50% |

This is exactly why the validation plan keeps separate gates. Capped morphology
has a meaningful early loss/compression signal, but it has not yet earned a
language-quality claim.

The combined checkpoint decision is therefore `hold_for_quality_evidence`: the
current candidate has a 16.07% bits-per-byte gain over HF BPE, but a -25.00%
minimal-pair accuracy delta on the mean-NLL ranking smoke test.

The official BabyLM 2026 fast-BLiMP adapter gives a larger downstream-style
gate. After adding the true `tiered_morph_unigram` arm, the full 13,400-case
fast-BLiMP set now favors the tiered hybrid:

| Score Field | Best Baseline | Tiered Hybrid | Capped Morphology |
|---|---:|---:|---:|
| mean NLL/token | `hf_unigram` 54.66% | 54.75% | 53.53% |
| bits/byte | `hf_bpe` 50.29% | 52.66% | 49.57% |
| total NLL | `hf_bpe` 52.99% | 55.34% | 51.79% |

This changes the current interpretation: pure morphology remains a strong
compression baseline, but the original MELM-style hybrid is now the main
candidate for longer training. Fast entity tracking still favors BPE at 41.88%
versus 41.12% for the tiered hybrid, so the tokenizer is not solved.

The next stage gate has now passed for the tiered hybrid:

- 200-step three-seed mean bits/byte: `2.679` for tiered hybrid vs `2.769`
  for HF BPE, a `3.27%` relative gain.
- Full fast-BLiMP: tiered hybrid wins `3/3` scoring views.
- Fast entity tracking: tiered hybrid trails the best baseline by `0.76`
  percentage points, inside the provisional tolerance.
- Larger local proxy: tiered hybrid reaches `2.491` bits/byte vs `2.570` for
  HF BPE, a `3.08%` relative gain.

Decision: advance `tiered_morph_unigram` to the checkpointed small-model
BabyLM stage, while retaining capped morphology as the compression reference
and HF BPE as the entity-tracking reference. The generated run card is
`reports/babylm_2026_small_model_stage_plan.md` and currently schedules a
23.3M-parameter local stage before any 125M-370M run.
