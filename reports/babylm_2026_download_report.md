# BabyLM 2026 Strict-Small Download Report

Downloaded: 2026-05-11

## Corpus

- Train repo: `BabyLM-community/BabyLM-2026-Strict-Small`
- Dev repo: `BabyLM-community/BabyLM-dev`
- Test repo: `BabyLM-community/BabyLM-Test`
- Local directory: `local_data/babylm_2026_strict_small`
- Manifest: `reports/babylm_2026_strict_small_manifest.json`
- Download metadata: `local_data/babylm_2026_strict_small/download_metadata.json`

`local_data/` is gitignored.

## Profile

| Split | Docs | Bytes | Lines | Word-like Tokens |
|---|---:|---:|---:|---:|
| train | 6 | 54,399,840 | 1,104,112 | 9,956,033 |
| validation | 6 | 56,752,783 | 1,168,159 | 10,375,121 |
| test | 6 | 51,722,871 | 1,110,006 | 9,431,457 |

## Initial Sampled Probe

Command:

```powershell
python scripts\run_tokenizer_decision.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
```

Result:

- Decision: `auxiliary_only`
- Best LM baseline: `unigram_like`
- Best baseline NLL/token: `4.835`
- Heuristic morphology NLL/token: `6.302`
- LM NLL gain: `-1.466`
- Boundary F1 gain: `52.94%`

Tiny neural smoke:

```powershell
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers simple_bpe,unigram_like,heuristic_morpheme --max-train-bytes 500000 --max-validation-bytes 250000 --steps 1 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 4 --out-json reports\babylm_2026_sample_tiny_lm_ablation.json --out-md reports\babylm_2026_sample_tiny_lm_ablation.md
```

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens |
|---:|---|---:|---:|---:|
| 1 | heuristic_morpheme | 3.051 | 7.780 | 67,958 |
| 2 | simple_bpe | 3.938 | 6.985 | 97,707 |
| 3 | unigram_like | 4.404 | 7.070 | 107,939 |

The tiny neural result is advisory only. It keeps morphology worth testing, but
does not override the tokenizer decision gate.

## Full Fast Tokenizer Probe

After adding optional Hugging Face `tokenizers` baselines and a capped morphology
arm, the full train/validation probe completed on the 2026 strict-small corpus:

```powershell
python scripts\run_fast_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --vocab-size 8192 --arms hf_bpe,hf_unigram,capped_morpheme --out-json reports\babylm_2026_fast_tokenizer_full.json --out-md reports\babylm_2026_fast_tokenizer_full.md
```

| Rank | Tokenizer | NLL/Token | Bits/Byte | Vocab | Validation Tokens |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 5.842 | 2.312 | 8,192 | 15,566,580 |
| 2 | hf_bpe | 6.368 | 2.540 | 8,188 | 15,694,331 |
| 3 | hf_unigram | 5.728 | 2.569 | 8,288 | 17,641,522 |

Updated interpretation:

- The old dependency-free tokenizer decision remains useful as a smoke gate, but
  it is not the right final arbiter for BabyLM-scale tokenization.
- Capped morphology now has a real full-corpus signal and should be promoted to
  the next neural ablation tier.
- This still does not prove morphology should be the final primary tokenizer:
  it must win under matched neural training budgets and downstream BabyLM tasks.

Fast tokenizer decision:

- Decision: `promote_to_neural_ablation`
- Best baseline: `hf_bpe`
- Absolute bits/byte gain: `0.229`
- Relative bits/byte gain: `9.00%`
- Next step: matched neural BabyLM ablations for capped morphology, HF BPE,
  and HF Unigram.

## Matched Tiny Neural Ablation

Command:

```powershell
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme --max-train-bytes 500000 --max-validation-bytes 250000 --steps 10 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_matched_tiny_lm_ablation.json --out-md reports\babylm_2026_matched_tiny_lm_ablation.md
```

| Rank | Tokenizer | Bits/Byte | NLL/Token | Parameters |
|---:|---|---:|---:|---:|
| 1 | capped_morpheme | 3.301 | 8.418 | 582,464 |
| 2 | hf_bpe | 3.701 | 8.427 | 582,464 |
| 3 | hf_unigram | 4.170 | 8.427 | 582,464 |

Tiny LM decision:

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative bits/byte gain: `10.79%`

Caveat: this is a 10-step tiny-model smoke run on a byte-capped subset. The
parameter counts are now exactly matched by padding each model vocabulary to
4096 entries, but the run is still too short and too small to count as a final
BabyLM model-quality result.

## Multi-Seed Tiny Neural Ablation

Command:

```powershell
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme --seeds 3,13,23 --max-train-bytes 500000 --max-validation-bytes 250000 --steps 10 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_multiseed_tiny_lm_ablation.json --out-md reports\babylm_2026_multiseed_tiny_lm_ablation.md
```

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 0.005 | 8.405 | 582,464 | 3 |
| 2 | hf_bpe | 3.697 | 0.004 | 8.420 | 582,464 | 3 |
| 3 | hf_unigram | 4.168 | 0.007 | 8.424 | 582,464 | 3 |

Multi-seed decision:

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative bits/byte gain: `10.85%`

Updated next step: run a longer 50-100 step ablation, then decide whether to
move the tokenizer experiment into a small BabyLM model training stack.

## Tiny LM Progression

Progression across 10, 50, and 100-step multi-seed matched-parameter tiny LM
ablations:

| Rank | Tokenizer | 10 Steps | 50 Steps | 100 Steps | Relative Improvement |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 3.075 | 2.545 | 22.78% |
| 2 | hf_bpe | 3.697 | 3.503 | 3.050 | 17.52% |
| 3 | hf_unigram | 4.168 | 3.884 | 3.204 | 23.12% |

At 100 steps, capped morphology has a 16.54% relative bits/byte gain over the
best baseline (`hf_bpe`). This is enough to proceed to a small proper BabyLM
training stack with saved tokenizers/checkpoints, while still treating the
result as preliminary until downstream BabyLM evaluations run.

## Saved Tiny LM Artifacts

The 100-step seed-13 arms were saved as reloadable local artifacts under
`artifacts/tiny_lm/`. The directory is gitignored, so reports record the local
artifact inventory and validation results.

```powershell
python scripts\summarize_tiny_lm_artifacts.py
python scripts\evaluate_tiny_lm_artifacts.py --manifest reports\babylm_2026_strict_small_manifest.json --root artifacts\tiny_lm --max-validation-bytes 250000 --out-json reports\tiny_lm_artifact_evaluation.json --out-md reports\tiny_lm_artifact_evaluation.md
```

| Rank | Tokenizer | Eval Bits/Byte | Delta vs Training Report |
|---:|---|---:|---:|
| 1 | capped_morpheme | 2.586 | 0.000000 |
| 2 | hf_bpe | 3.081 | 0.000000 |
| 3 | hf_unigram | 3.240 | 0.000000 |

Checkpoint reload reproduced the saved training reports exactly on the same
250k-byte validation cap. This confirms the local tokenizers, vocabularies, and
model checkpoints are reusable for the next evaluation layer.

## Minimal-Pair Checkpoint Smoke

A tiny child-level minimal-pair ranking fixture now checks whether each saved
checkpoint assigns a lower score to grammatical sentences than to close foils.
This is not a BabyLM replacement, but it is the first downstream-style warning
light beyond bits-per-byte.

```powershell
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --out-json reports\tiny_lm_minimal_pairs.json --out-md reports\tiny_lm_minimal_pairs.md
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --score-field bits_per_byte --out-json reports\tiny_lm_minimal_pairs_bits_per_byte.json --out-md reports\tiny_lm_minimal_pairs_bits_per_byte.md
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --score-field total_nll --out-json reports\tiny_lm_minimal_pairs_total_nll.json --out-md reports\tiny_lm_minimal_pairs_total_nll.md
```

| Score Field | Best BPE/Unigram Accuracy | Capped Morphology Accuracy |
|---|---:|---:|
| mean NLL/token | 50.00% | 25.00% |
| bits/byte | 62.50% | 37.50% |
| total NLL | 25.00% | 12.50% |

Interpretation: the capped morphology arm remains promising for compression and
early neural bits-per-byte, but this tiny checkpoint does not yet show a
language-quality advantage. Morphology should not be promoted to the primary
tokenizer on loss alone; the next stage needs real downstream BabyLM-style
evaluation and longer small-model training.

Pure-morphology checkpoint validation decision:

- Decision: `hold_for_quality_evidence`
- Candidate: `capped_morpheme`
- Best baseline: `hf_bpe`
- Relative bits/byte gain: `16.07%`
- Minimal-pair accuracy delta: `-25.00%`
- Recommendation: run stronger BabyLM-style evaluations before any
  primary-tokenizer claim.

## Official BabyLM Fast-BLiMP Adapter

The official BabyLM 2026 evaluation repository was mirrored locally for
inspection at `local_data/babylm_eval_2026` (`babylm-org/babylm-eval`, commit
`467793f`). The fast evaluation subset from
`BabyLM-community/BabyLM-2026-Strict-Evals` was downloaded into the strict-track
directory.

```powershell
python scripts\profile_babylm_fast_eval.py
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_blimp_fast_full.json --out-md reports\tiny_lm_blimp_fast_full.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field bits_per_byte --out-json reports\tiny_lm_blimp_fast_full_bits_per_byte.json --out-md reports\tiny_lm_blimp_fast_full_bits_per_byte.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field total_nll --out-json reports\tiny_lm_blimp_fast_full_total_nll.json --out-md reports\tiny_lm_blimp_fast_full_total_nll.md
```

Fast-BLiMP profile:

| Asset | Files | Cases |
|---|---:|---:|
| BLiMP fast | 67 | 13,400 |

After adding the true tiered morphology-Unigram arm, full fast BLiMP changed
materially:

| Score Field | Best Baseline | Tiered Hybrid | Capped Morphology |
|---|---:|---:|---:|
| mean NLL/token | `hf_unigram` 54.66% | 54.75% | 53.53% |
| bits/byte | `hf_bpe` 50.29% | 52.66% | 49.57% |
| total NLL | `hf_bpe` 52.99% | 55.34% | 51.79% |

Fast entity tracking still favors BPE:

| Rank | Tokenizer | Accuracy |
|---:|---|---:|
| 1 | hf_bpe | 41.88% |
| 2 | hybrid_morph_unigram | 41.62% |
| 3 | tiered_morph_unigram | 41.12% |
| 4 | hf_unigram | 39.50% |
| 5 | capped_morpheme | 17.61% |

Updated interpretation: pure morphology is a strong compression baseline, but
the actual MELM-style tiered morphology-Unigram tokenizer is now the main
candidate for longer small-model training. It should not be declared final
until it survives multi-seed longer schedules and entity/state tasks.

Tiered-hybrid checkpoint decision:

- Decision: `promote_to_small_model_training`
- Candidate: `tiered_morph_unigram`
- Best baseline: `hf_bpe`
- Relative bits/byte gain: `1.76%`
- Minimal-pair accuracy delta: `0.00%`
- Recommendation: proceed to longer BabyLM-style training and downstream
  evaluation.

## Tiered Hybrid Progression And Small Proxy

Three-seed matched-parameter progression now covers 10, 50, 100, and 200 steps
for HF BPE, HF Unigram, capped morphology, and tiered morphology-Unigram:

| Rank | Tokenizer | 10 Steps | 50 Steps | 100 Steps | 200 Steps |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 3.075 | 2.545 | 2.216 |
| 2 | tiered_morph_unigram | 3.763 | 3.539 | 2.989 | 2.679 |
| 3 | hf_bpe | 3.697 | 3.503 | 3.050 | 2.769 |
| 4 | hf_unigram | 4.168 | 3.884 | 3.204 | 2.851 |

The tiered arm trails BPE at 10 and 50 steps, then overtakes it at 100 and 200
steps. At 200 steps its relative bits/byte gain over HF BPE is `3.27%`.

Tokenizer stage gate:

- Decision: `advance_to_small_model_ablation`
- Candidate: `tiered_morph_unigram`
- Fast-BLiMP wins: `3/3`
- Entity-tracking delta versus best baseline: `-0.76%`

First larger local proxy:

```powershell
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --max-train-bytes 1000000 --max-validation-bytes 500000 --steps 200 --sequence-length 96 --embedding-dim 128 --layers 2 --heads 4 --batch-size 8 --max-vocab-size 4096 --pad-vocab-to-max-size --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_small_proxy_tokenizer_ablation.json --out-md reports\babylm_2026_small_proxy_tokenizer_ablation.md
```

| Rank | Tokenizer | Bits/Byte | NLL/Token |
|---:|---|---:|---:|
| 1 | capped_morpheme | 2.070 | 5.232 |
| 2 | tiered_morph_unigram | 2.491 | 5.602 |
| 3 | hf_bpe | 2.570 | 5.861 |
| 4 | hf_unigram | 2.627 | 5.359 |

Small-proxy decision: `promote_to_scaled_neural_ablation` for
`tiered_morph_unigram`, with a `3.08%` bits/byte gain over HF BPE.
