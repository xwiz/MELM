# Tiny LM Artifact Evaluation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Artifact root: `artifacts\babylm_2026_small_model_stage`
Max validation bytes: `1000000`

| Rank | Run | Tokenizer | Steps | Params | Eval Bits/Byte | Delta vs Training Report |
|---:|---|---|---:|---:|---:|---:|
| 1 | capped_morpheme_seed13_1000step | capped_morpheme | 1000 | 23319808 | 1.680 | 0.000000 |
| 2 | tiered_morph_unigram_seed13_1000step | tiered_morph_unigram | 1000 | 23319808 | 1.937 | 0.000000 |
| 3 | hf_bpe_seed13_1000step | hf_bpe | 1000 | 23319808 | 1.980 | 0.000000 |
| 4 | hf_unigram_seed13_1000step | hf_unigram | 1000 | 23319808 | 1.994 | 0.000000 |
