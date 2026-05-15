# BabyLM 2026 Small-Model Stage Gate

Multiseed report: `reports/babylm_2026_small_model_stage_multiseed.json`
Entity report: `reports/babylm_2026_small_model_stage_entity_tracking.json`
Symbolic entity report: `reports\babylm_2026_small_model_stage_entity_tracking_symbolic.json`

- Decision: `advance_to_event_memory_integration`
- Candidate: `tiered_morph_unigram`
- Candidate bits/byte: `1.928`
- Best HF baseline: `hf_bpe`
- Best HF baseline bits/byte: `1.975`
- Relative bits/byte gain: `2.38%`
- Compression control: `capped_morpheme`
- Compression control bits/byte: `1.681`
- Fast-BLiMP wins vs HF baselines: `2/3`
- Entity best HF baseline: `hf_bpe`
- Candidate entity accuracy: `40.42%`
- Best HF entity accuracy: `39.94%`
- Entity accuracy delta: `0.48%`
- Symbolic entity accuracy: `100.00%`
- Recommendation: Keep tiered morphology-Unigram as the primary MELM tokenizer candidate, keep capped morphology as a compression control, and integrate explicit event/state memory next.

Interpretation: this gate can promote the next integration step, but it does not make the tokenizer final.
