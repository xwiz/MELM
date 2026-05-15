# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `10`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 0.005 | 8.405 | 582464 | 3 |
| 2 | hf_bpe | 3.697 | 0.004 | 8.420 | 582464 | 3 |
| 3 | tiered_morph_unigram | 3.763 | 0.002 | 8.422 | 582464 | 3 |
| 4 | hf_unigram | 4.168 | 0.007 | 8.424 | 582464 | 3 |

- Decision: `do_not_scale_yet`
- Best baseline: `hf_bpe`
- Relative gain: `-1.78%`
- Recommendation: Keep morphology as auxiliary supervision unless longer neural ablations recover.
