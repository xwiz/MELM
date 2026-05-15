# Tokenizer Stage Gate

Progression: `reports/babylm_2026_tiered_tiny_lm_progression.json`
Entity report: `reports/tiny_lm_entity_tracking_fast.json`
Proxy decision: `reports\babylm_2026_small_proxy_tokenizer_decision.json`

- Decision: `advance_to_scaled_neural_ablation`
- Candidate: `tiered_morph_unigram`
- Latest step: `200`
- Candidate bits/byte: `2.679`
- Best baseline: `hf_bpe`
- Best baseline bits/byte: `2.769`
- Relative bits/byte gain: `3.27%`
- Fast-BLiMP wins: `3/3`
- Entity best baseline: `hf_bpe`
- Candidate entity accuracy: `41.12%`
- Best baseline entity accuracy: `41.88%`
- Entity accuracy delta: `-0.76%`
- Proxy decision: `promote_to_scaled_neural_ablation`
- Proxy best baseline: `hf_bpe`
- Proxy relative bits/byte gain: `3.08%`
- Proxy supports scale: `True`
- Recommendation: Schedule longer matched BabyLM-style neural ablations for tiered morphology-Unigram, HF BPE, HF Unigram, and capped morphology.
