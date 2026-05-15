# Tiny LM Checkpoint Validation Decision

Artifact report: `reports\tiny_lm_artifact_evaluation.json`
Minimal-pair report: `reports\tiny_lm_minimal_pairs.json`

- Decision: `promote_to_small_model_training`
- Candidate: `tiered_morph_unigram`
- Best baseline: `hf_bpe`
- Candidate bits/byte: `3.027`
- Best baseline bits/byte: `3.081`
- Relative bits/byte gain: `1.76%`
- Candidate minimal-pair accuracy: `50.00%`
- Best baseline minimal-pair accuracy: `50.00%`
- Minimal-pair accuracy delta: `0.00%`
- Recommendation: Proceed to longer BabyLM-style training and downstream evaluation.
