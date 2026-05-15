# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `bits_per_byte`
Cases: `13400`
Unique texts: `26794`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_tiered_morph_unigram_100step_seed13 | tiered_morph_unigram | 52.66% | 7056 | 13400 |
| 2 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 50.29% | 6739 | 13400 |
| 3 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 49.80% | 6673 | 13400 |
| 4 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 49.57% | 6643 | 13400 |
| 5 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 44.74% | 5995 | 13400 |
