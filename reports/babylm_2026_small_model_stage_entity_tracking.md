# Tiny LM BabyLM Fast Entity Tracking

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/entity_tracking_fast`
Artifact root: `artifacts\babylm_2026_small_model_stage`
Score field: `total_nll`
Cases: `3152`
Unique texts: `15760`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 40.42% | 1274 | 3152 |
| 2 | hf_bpe_seed13_1000step | hf_bpe | 39.94% | 1259 | 3152 |
| 3 | capped_morpheme_seed13_1000step | capped_morpheme | 38.58% | 1216 | 3152 |
| 4 | hf_unigram_seed13_1000step | hf_unigram | 31.12% | 981 | 3152 |
