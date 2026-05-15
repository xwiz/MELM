# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `200`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 2.216 | 0.011 | 5.651 | 582464 | 3 |
| 2 | tiered_morph_unigram | 2.679 | 0.011 | 5.995 | 582464 | 3 |
| 3 | hf_bpe | 2.769 | 0.008 | 6.307 | 582464 | 3 |
| 4 | hf_unigram | 2.851 | 0.006 | 5.761 | 582464 | 3 |

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative gain: `3.27%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for tiered_morph_unigram, capped morphology, HF BPE, and HF Unigram.
