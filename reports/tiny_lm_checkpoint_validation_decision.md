# Tiny LM Checkpoint Validation Decision

Artifact report: `reports\tiny_lm_artifact_evaluation.json`
Minimal-pair report: `reports\tiny_lm_minimal_pairs.json`

- Decision: `hold_for_quality_evidence`
- Candidate: `capped_morpheme`
- Best baseline: `hf_bpe`
- Candidate bits/byte: `2.586`
- Best baseline bits/byte: `3.081`
- Relative bits/byte gain: `16.07%`
- Candidate minimal-pair accuracy: `25.00%`
- Best baseline minimal-pair accuracy: `50.00%`
- Minimal-pair accuracy delta: `-25.00%`
- Recommendation: Capped morphology has a loss/compression signal, but downstream smoke results lag baselines; run stronger BabyLM-style evaluations before primary-tokenizer claims.
