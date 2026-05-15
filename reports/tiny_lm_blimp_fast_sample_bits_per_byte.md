# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `bits_per_byte`
Cases: `200`
Unique texts: `400`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 53.00% | 106 | 200 |
| 2 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 49.00% | 98 | 200 |
| 3 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 48.50% | 97 | 200 |
