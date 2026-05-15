# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\tiny_lm`
Score field: `total_nll`
Cases: `1340`
Unique texts: `2680`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 54.55% | 731 | 1340 |
| 2 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 52.39% | 702 | 1340 |
| 3 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 49.70% | 666 | 1340 |
