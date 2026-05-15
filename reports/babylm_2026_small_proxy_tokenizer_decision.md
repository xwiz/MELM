# Tiny LM Ablation Decision

Source report: `reports\babylm_2026_small_proxy_tokenizer_ablation.json`

- Decision: `promote_to_scaled_neural_ablation`
- Candidate: `tiered_morph_unigram`
- Best baseline: `hf_bpe`
- Candidate bits/byte: `2.491`
- Best baseline bits/byte: `2.570`
- Absolute bits/byte gain: `0.079`
- Relative gain: `3.08%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for tiered_morph_unigram, capped morphology, HF BPE, and HF Unigram.

This supports a longer neural ablation only; it is not a final BabyLM score.
