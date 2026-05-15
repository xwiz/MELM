# Multi-Seed Tiny LM Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Seeds: `3, 13, 23`
Steps: `10`

| Rank | Tokenizer | Mean Bits/Byte | Std | Mean NLL/Token | Mean Params | Runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 0.005 | 8.405 | 582464 | 3 |
| 2 | hf_bpe | 3.697 | 0.004 | 8.420 | 582464 | 3 |
| 3 | hf_unigram | 4.168 | 0.007 | 8.424 | 582464 | 3 |

- Decision: `promote_to_scaled_neural_ablation`
- Best baseline: `hf_bpe`
- Relative gain: `10.85%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for capped morphology, HF BPE, and HF Unigram.
