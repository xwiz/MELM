# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `50`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.075 | 0.035 | 7.842 | 582464 | 3 |
| 2 | hf_bpe | 3.503 | 0.023 | 7.978 | 582464 | 3 |
| 3 | tiered_morph_unigram | 3.539 | 0.033 | 7.919 | 582464 | 3 |
| 4 | hf_unigram | 3.884 | 0.037 | 7.851 | 582464 | 3 |

- Decision: `do_not_scale_yet`
- Best baseline: `hf_bpe`
- Relative gain: `-1.01%`
- Recommendation: Keep morphology as auxiliary supervision unless longer neural ablations recover.
