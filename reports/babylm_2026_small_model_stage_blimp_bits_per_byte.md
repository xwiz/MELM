# Tiny LM BabyLM Fast BLiMP

Data dir: `local_data/babylm_eval_2026/strict/evaluation_data/fast_eval/blimp_fast`
Artifact root: `artifacts\babylm_2026_small_model_stage`
Score field: `bits_per_byte`
Cases: `13400`
Unique texts: `26794`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | capped_morpheme_seed13_1000step | capped_morpheme | 51.44% | 6893 | 13400 |
| 2 | hf_unigram_seed13_1000step | hf_unigram | 51.04% | 6840 | 13400 |
| 3 | hf_bpe_seed13_1000step | hf_bpe | 50.97% | 6830 | 13400 |
| 4 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 50.81% | 6809 | 13400 |
