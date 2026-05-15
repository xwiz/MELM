# State-Assisted BabyLM Entity Tracking

LM entity report: `reports/babylm_2026_small_model_stage_entity_tracking.json`
Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast`
Cases: `3152`

State memory gets first refusal. If the state parser cannot resolve a case, the evaluator falls back to the LM prediction.

| Rank | Run | Tokenizer | LM Accuracy | Assisted Accuracy | Lift | State Answer Rate | LM Fallbacks |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 40.42% | 100.00% | 59.58% | 100.00% | 0 |
| 2 | hf_bpe_seed13_1000step | hf_bpe | 39.94% | 100.00% | 60.06% | 100.00% | 0 |
| 3 | capped_morpheme_seed13_1000step | capped_morpheme | 38.58% | 100.00% | 61.42% | 100.00% | 0 |
| 4 | hf_unigram_seed13_1000step | hf_unigram | 31.12% | 100.00% | 68.88% | 100.00% | 0 |
