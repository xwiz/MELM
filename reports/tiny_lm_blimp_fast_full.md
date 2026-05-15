# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `mean_nll_per_token`
Cases: `13400`
Unique texts: `26794`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_tiered_morph_unigram_100step_seed13 | tiered_morph_unigram | 54.75% | 7337 | 13400 |
| 2 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 54.66% | 7324 | 13400 |
| 3 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 54.13% | 7254 | 13400 |
| 4 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 53.53% | 7173 | 13400 |
| 5 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 52.28% | 7006 | 13400 |
