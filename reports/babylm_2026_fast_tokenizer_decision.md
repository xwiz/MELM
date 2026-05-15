# Fast Tokenizer Decision

Source report: `reports\babylm_2026_fast_tokenizer_full.json`

- Decision: `promote_to_neural_ablation`
- Candidate: `capped_morpheme`
- Best baseline: `hf_bpe`
- Candidate bits/byte: `2.312`
- Best baseline bits/byte: `2.540`
- Absolute bits/byte gain: `0.229`
- Relative gain: `9.00%`
- Recommendation: Run matched neural BabyLM ablations for capped morphology, HF BPE, and HF Unigram.

This promotes the tokenizer to matched neural ablation only; it is not a final model-quality claim.
