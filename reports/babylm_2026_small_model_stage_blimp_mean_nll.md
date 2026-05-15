# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\babylm_2026_small_model_stage`
Score field: `mean_nll_per_token`
Cases: `13400`
Unique texts: `26794`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 53.17% | 7125 | 13400 |
| 2 | hf_bpe_seed13_1000step | hf_bpe | 52.54% | 7040 | 13400 |
| 3 | capped_morpheme_seed13_1000step | capped_morpheme | 52.07% | 6978 | 13400 |
| 4 | hf_unigram_seed13_1000step | hf_unigram | 50.84% | 6813 | 13400 |
