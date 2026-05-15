# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `100`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 2.545 | 0.039 | 6.490 | 582464 | 3 |
| 2 | tiered_morph_unigram | 2.989 | 0.033 | 6.690 | 582464 | 3 |
| 3 | hf_bpe | 3.050 | 0.028 | 6.945 | 582464 | 3 |
| 4 | hf_unigram | 3.204 | 0.031 | 6.476 | 582464 | 3 |

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative gain: `1.98%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for capped morphology, HF BPE, and HF Unigram.
