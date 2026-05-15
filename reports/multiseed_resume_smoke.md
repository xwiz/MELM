# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `13`
Steps: `1`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | hf_bpe | 3.916 | 0.000 | 6.403 | 47008 | 1 |
| 2 | tiered_morph_unigram | 4.068 | 0.000 | 6.431 | 47008 | 1 |

- Decision: `do_not_scale_yet`
- Best baseline: `hf_bpe`
- Relative gain: `-3.89%`
- Recommendation: Keep morphology as auxiliary supervision unless longer neural ablations recover.
