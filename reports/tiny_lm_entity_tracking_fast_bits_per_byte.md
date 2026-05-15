# Tiny LM BabyLM Fast Entity Tracking

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `bits_per_byte`
Cases: `3152`
Unique texts: `15760`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 31.22% | 984 | 3152 |
| 2 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 20.18% | 636 | 3152 |
| 3 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 12.53% | 395 | 3152 |
| 4 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 10.98% | 346 | 3152 |
