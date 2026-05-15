# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `1000`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 1.681 | 0.001 | 4.208 | 23319808 | 3 |
| 2 | tiered_morph_unigram | 1.928 | 0.008 | 4.640 | 23319808 | 3 |
| 3 | hf_bpe | 1.975 | 0.005 | 4.802 | 23319808 | 3 |
| 4 | hf_unigram | 1.996 | 0.007 | 4.383 | 23319808 | 3 |

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative gain: `2.38%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for tiered_morph_unigram, capped morphology, HF BPE, and HF Unigram.
