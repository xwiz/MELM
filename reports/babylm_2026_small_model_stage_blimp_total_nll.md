# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\babylm_2026_small_model_stage`
Score field: `total_nll`
Cases: `13400`
Unique texts: `26794`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 53.67% | 7192 | 13400 |
| 2 | hf_unigram_seed13_1000step | hf_unigram | 52.60% | 7048 | 13400 |
| 3 | capped_morpheme_seed13_1000step | capped_morpheme | 52.57% | 7045 | 13400 |
| 4 | hf_bpe_seed13_1000step | hf_bpe | 52.38% | 7019 | 13400 |
