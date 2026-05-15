# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `mean_nll_per_token`
Cases: `200`
Unique texts: `400`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 45.00% | 90 | 200 |
| 2 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 44.50% | 89 | 200 |
| 3 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 42.50% | 85 | 200 |
