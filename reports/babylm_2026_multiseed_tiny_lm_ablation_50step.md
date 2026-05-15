# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `50`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.075 | 0.035 | 7.842 | 582464 | 3 |
| 2 | hf_bpe | 3.503 | 0.023 | 7.978 | 582464 | 3 |
| 3 | hf_unigram | 3.884 | 0.037 | 7.851 | 582464 | 3 |

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative gain: `12.22%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for capped morphology, HF BPE, and HF Unigram.
