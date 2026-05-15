# MELM Validation Suite

Source report: `reports\phase1_report.json`

Overall passed: `True`
Hard failures: `0`

| Check | Severity | Status | Metric | Threshold | Detail |
|---|---|---:|---:|---:|---|
| synthetic_event_memory_gain | hard | PASS | 0.470 | 0.150 | Proceed with structured event memory |
| held_out_abstention_calibration | hard | PASS | 1.000 | 1.000 | score_with_evidence_veto at threshold 1.25; positive recall 75.00%, negative abstention 85.33% |
| tokenizer_decision_consistency | hard | PASS | -1.151 | 0.000 | decision=auxiliary_only; LM gain -1.151; boundary gain 57.14% |
| tokenizer_stability | advisory | FAIL | -0.879 | 0.000 | morphology win rate 0.00%; best baseline unigram_like |
| state_grounding_seed_accuracy | hard | PASS | 1.000 | 0.950 | 4 seed cases |
| state_resolution | hard | PASS | 1.000 | 0.950 | 100 object-location cases; false positive rate 0.00% |
| authored_dialogue_memory_gain | hard | PASS | 0.200 | 0.150 | Proceed with structured event memory |
| authored_dialogue_abstention | hard | PASS | 1.067 | 1.000 | positive recall 80.00%; negative abstention 100.00% |
| sample_transcript_memory_gain | hard | PASS | 0.333 | 0.150 | Proceed with structured event memory |
| sample_transcript_abstention | hard | PASS | 1.250 | 1.000 | positive recall 100.00%; negative abstention 100.00% |
| sample_transcript_state_resolution | hard | PASS | 1.000 | 0.950 | 4 annotated state cases; false positive rate 0.00% |
