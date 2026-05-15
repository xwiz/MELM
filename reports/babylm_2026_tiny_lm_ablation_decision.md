# Tiny LM Ablation Decision

Source report: `reports\babylm_2026_matched_tiny_lm_ablation.json`

- Decision: `promote_to_scaled_neural_ablation`
- Candidate: `capped_morpheme`
- Best baseline: `hf_bpe`
- Candidate bits/byte: `3.301`
- Best baseline bits/byte: `3.701`
- Absolute bits/byte gain: `0.399`
- Relative gain: `10.79%`
- Recommendation: Run longer BabyLM neural ablations with matched compute for capped morphology, HF BPE, and HF Unigram.

This supports a longer neural ablation only; it is not a final BabyLM score.
