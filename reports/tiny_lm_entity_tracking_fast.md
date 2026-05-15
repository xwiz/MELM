# Tiny LM BabyLM Fast Entity Tracking

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `total_nll`
Cases: `3152`
Unique texts: `15760`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 41.88% | 1320 | 3152 |
| 2 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 41.62% | 1312 | 3152 |
| 3 | babylm_2026_tiered_morph_unigram_100step_seed13 | tiered_morph_unigram | 41.12% | 1296 | 3152 |
| 4 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 39.50% | 1245 | 3152 |
| 5 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 17.61% | 555 | 3152 |
