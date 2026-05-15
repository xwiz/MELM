# MELM Phase 1 Validation Report

Text source: `MELM_whitepaper.md`

## Interpretation

| Area | Status | Finding | Next Step |
|---|---|---|---|
| event_memory | pass | Structured event memory beats RAG by 47% absolute recall@k. The gain is concentrated in temporal-neighbor cases (average 100%). Strict recall@1 gain is 50%. Ablation: temporal-only gain 25%, causal-only gain 22%, combined gain 47%. Held-out calibrated abstention at threshold 1.25 has 79% accuracy, 75% positive recall, and 85% negative abstention. | Add harder entity-conflict cases and then validate on non-synthetic dialogue before treating this as robust. |
| authored_dialogue | probe_pass | On the hand-authored dialogue smoke test, event memory reaches 100% recall@k versus 80% for RAG. The transferred evidence selector gets 80% positive recall and 100% negative abstention. | Replace this smoke fixture with transcript-derived cases and keep the same gates. |
| sample_transcript | smoke_pass | The annotated transcript compiler runs end to end: 8 turns become 7 events, with event memory recall 100% versus RAG 67%. Evidence admission reaches 100% positive recall and 100% negative abstention. State resolution covers 4 cases at 100% accuracy with 0% false positives. | Use this compiler on real transcript snippets and track source-turn provenance for every event. |
| tokenizer_lm | auxiliary_only | The best held-out unigram-LM probe is unigram_like (NLL/token 4.810); heuristic morphology is 5.961. Decision: auxiliary_only with LM NLL gain -1.151 and boundary-F1 gain 57%. Cross-fold stability shows morphology win rate 0%, average LM gain -0.879, and stable-primary=False. | Use morphology for auxiliary supervision or analysis, not as the primary tokenizer yet. |
| morphology_boundary | probe_pass | Heuristic morphology reaches 100% boundary F1 on the tiny gold fixture. | Replace the tiny hand fixture with MorphoLex/CELEX-derived examples; this result is alignment-only. |
| state_grounding | probe_pass | State grounding gets 100% accuracy on 4 seed cases. | Expand precondition, contradiction, figurative-language, and context-blindness cases. |
| state_resolution | probe_pass | Explicit event-state tracking gets 100% accuracy on 100 synthetic object-location cases, with 0% false positives on unknown objects. | Port the same state annotations into transcript-derived dialogue fixtures. |

## Tokenizer Metrics

| Tokenizer | Tokens/Word | Unique | Fallback |
|---|---:|---:|---:|
| whitespace | 1.000 | 1119 | 0.00% |
| byte_patch | 1.191 | 3165 | 0.00% |
| heuristic_morpheme | 1.232 | 901 | 0.00% |
| unigram_like | 1.536 | 625 | 32.67% |
| simple_bpe | 1.701 | 498 | 0.00% |

## Tokenizer LM Probe

| Tokenizer | NLL/Token | Perplexity | Vocab | Validation Tokens |
|---|---:|---:|---:|---:|
| unigram_like | 4.810 | 122.71 | 625 | 2303 |
| simple_bpe | 5.396 | 220.59 | 498 | 2198 |
| heuristic_morpheme | 5.961 | 387.93 | 901 | 1487 |
| whitespace | 6.517 | 676.37 | 1119 | 1241 |
| byte_patch | 9.411 | 12224.26 | 3131 | 1269 |

## Morphology Boundary Probe

| Tokenizer | Precision | Recall | F1 | Exact |
|---|---:|---:|---:|---:|
| whitespace | 0.00% | 0.00% | 0.00% | 0.00% |
| simple_bpe | 28.12% | 90.00% | 42.86% | 0.00% |
| heuristic_morpheme | 100.00% | 100.00% | 100.00% | 100.00% |
| byte_patch | 0.00% | 0.00% | 0.00% | 0.00% |
| unigram_like | 20.45% | 90.00% | 33.33% | 0.00% |

## Tokenizer Decision

- Decision: auxiliary_only
- Gate passed: False
- Best LM baseline: unigram_like (NLL/token 4.810)
- Morphology LM: heuristic_morpheme (NLL/token 5.961)
- LM NLL gain: -1.151
- Best boundary baseline: simple_bpe (F1 42.86%)
- Morphology boundary F1: 100.00%
- Boundary F1 gain: 57.14%
- Recommendation: Use morphology for auxiliary supervision or analysis, not as the primary tokenizer yet.

## Tokenizer Stability

- Documents: 327
- Folds: 5
- Morphology win rate: 0.00%
- Stable primary candidate: False
- Best average baseline: unigram_like (NLL/token 4.735)
- Morphology average NLL/token: 5.614
- Average LM NLL gain: -0.879

| Tokenizer | Fold Wins | Average NLL/Token |
|---|---:|---:|
| byte_patch | 0 | 8.922 |
| heuristic_morpheme | 0 | 5.614 |
| simple_bpe | 0 | 5.385 |
| unigram_like | 5 | 4.735 |
| whitespace | 0 | 6.120 |

## Event Memory

- Cases: 200
- RAG recall: 46.50%
- Event memory recall: 93.50%
- Absolute gain: 47.00%
- RAG MRR: 43.50%
- Event memory MRR: 90.50%
- MRR gain: 47.00%
- Strict RAG recall@1: 40.50%
- Strict event memory recall@1: 90.50%
- Strict absolute gain@1: 50.00%
- Gate passed: True

### By Category

| Category | Cases | RAG | Event Memory | Gain | MRR Gain |
|---|---:|---:|---:|---:|---:|
| causal_source | 25 | 24.00% | 100.00% | 76.00% | 76.00% |
| direct | 50 | 74.00% | 74.00% | 0.00% | 0.00% |
| entity_conflict | 25 | 100.00% | 100.00% | 0.00% | 0.00% |
| entity_conflict_before_move | 25 | 0.00% | 100.00% | 100.00% | 100.00% |
| temporal_after | 25 | 0.00% | 100.00% | 100.00% | 100.00% |
| temporal_before | 25 | 0.00% | 100.00% | 100.00% | 100.00% |
| witness | 25 | 100.00% | 100.00% | 0.00% | 0.00% |

### Component Ablation

| Variant | Recall | Gain vs RAG | MRR | MRR Gain |
|---|---:|---:|---:|---:|
| entity_only | 46.50% | 0.00% | 43.50% | 0.00% |
| entity_action | 46.50% | 0.00% | 43.50% | 0.00% |
| entity_action_temporal | 71.50% | 25.00% | 68.50% | 25.00% |
| entity_action_causal | 68.50% | 22.00% | 65.50% | 22.00% |
| event_memory | 93.50% | 47.00% | 90.50% | 47.00% |

### Evidence Abstention

| Retriever | Threshold | Accuracy | Precision | Positive Recall | Negative Abstention | False Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.00 | 28.62% | 28.62% | 46.50% | 0.00% | 100.00% |
| rag | 0.50 | 46.15% | 37.50% | 37.50% | 60.00% | 40.00% |
| rag | 0.75 | 38.46% | 0.00% | 0.00% | 100.00% | 0.00% |
| rag | 1.00 | 38.46% | 0.00% | 0.00% | 100.00% | 0.00% |
| rag | 1.25 | 38.46% | 0.00% | 0.00% | 100.00% | 0.00% |
| rag | 1.50 | 38.46% | 0.00% | 0.00% | 100.00% | 0.00% |
| event_memory | 0.00 | 57.54% | 57.54% | 93.50% | 0.00% | 100.00% |
| event_memory | 0.50 | 57.54% | 57.54% | 93.50% | 0.00% | 100.00% |
| event_memory | 0.75 | 53.85% | 58.33% | 87.50% | 0.00% | 100.00% |
| event_memory | 1.00 | 69.23% | 70.00% | 87.50% | 40.00% | 60.00% |
| event_memory | 1.25 | 76.92% | 77.78% | 87.50% | 60.00% | 40.00% |
| event_memory | 1.50 | 69.23% | 83.33% | 62.50% | 80.00% | 20.00% |
| event_memory_hybrid | 0.00 | 57.54% | 57.54% | 93.50% | 0.00% | 100.00% |
| event_memory_hybrid | 0.50 | 76.00% | 75.35% | 81.00% | 68.00% | 32.00% |
| event_memory_hybrid | 0.75 | 72.31% | 78.95% | 75.00% | 68.00% | 32.00% |
| event_memory_hybrid | 1.00 | 72.31% | 78.95% | 75.00% | 68.00% | 32.00% |
| event_memory_hybrid | 1.25 | 80.00% | 90.91% | 75.00% | 88.00% | 12.00% |
| event_memory_hybrid | 1.50 | 75.08% | 100.00% | 59.50% | 100.00% | 0.00% |

Best abstention thresholds:

- RAG: threshold 0.50, accuracy 46.15%, negative abstention 60.00%.
- Event memory: threshold 1.25, accuracy 76.92%, negative abstention 60.00%.
- Event memory hybrid: threshold 1.25, accuracy 80.00%, negative abstention 88.00%.

Held-out calibrated thresholds:

- Top score: threshold 1.50, calibration cases 130, evaluation cases 195, eval accuracy 69.23%, eval precision 83.33%, eval positive recall 62.50%, eval negative abstention 80.00%.
- Score + evidence veto: threshold 1.25, calibration cases 130, evaluation cases 195, eval accuracy 78.97%, eval precision 89.11%, eval positive recall 75.00%, eval negative abstention 85.33%.
- Selected: threshold 1.25, calibration cases 130, evaluation cases 195, eval accuracy 78.97%, eval precision 89.11%, eval positive recall 75.00%, eval negative abstention 85.33%.
- Abstention gate passed: True (combined metric 1.00).

## Authored Dialogue Smoke

- Events: 12
- Recall cases: 10
- Evidence cases: 15
- RAG recall: 80.00%
- Event memory recall: 100.00%
- Absolute gain: 20.00%
- Event memory MRR: 85.00%
- Memory gate passed: True
- Evidence threshold: 1.25
- Evidence method: score_with_evidence_veto
- Evidence accuracy: 86.67%
- Evidence precision: 100.00%
- Evidence positive recall: 80.00%
- Evidence negative abstention: 100.00%
- Abstention gate passed: True

### Authored Dialogue By Category

| Category | Cases | RAG | Event Memory | Gain |
|---|---:|---:|---:|---:|
| causal_source | 1 | 100.00% | 100.00% | 0.00% |
| direct | 3 | 100.00% | 100.00% | 0.00% |
| entity_conflict | 1 | 100.00% | 100.00% | 0.00% |
| entity_conflict_before_move | 1 | 100.00% | 100.00% | 0.00% |
| temporal_after | 1 | 0.00% | 100.00% | 100.00% |
| temporal_before | 1 | 0.00% | 100.00% | 100.00% |
| witness | 2 | 100.00% | 100.00% | 0.00% |

## Annotated Transcript Smoke

- Annotation source: `benchmarks\sample_transcript_annotations.jsonl`
- Turns: 8
- Events: 7
- Recall cases: 6
- Evidence cases: 10
- State cases: 4
- RAG recall: 66.67%
- Event memory recall: 100.00%
- Absolute gain: 33.33%
- Evidence accuracy: 100.00%
- Evidence precision: 100.00%
- Evidence positive recall: 100.00%
- Evidence negative abstention: 100.00%
- Memory gate passed: True
- Abstention gate passed: True
- State resolution accuracy: 100.00%
- State resolution false positive rate: 0.00%

## State Grounding

- Cases: 4
- Accuracy: 100.00%

## State Resolution

- Cases: 100
- Accuracy: 100.00%
- Answer rate: 75.00%
- False positive rate: 0.00%

| Category | Cases | Accuracy | Answer Rate | False Positive Rate |
|---|---:|---:|---:|---:|
| at_initial_put | 25 | 100.00% | 100.00% | 0.00% |
| before_move | 25 | 100.00% | 100.00% | 0.00% |
| latest_after_move | 25 | 100.00% | 100.00% | 0.00% |
| unknown_object | 25 | 100.00% | 0.00% | 0.00% |
