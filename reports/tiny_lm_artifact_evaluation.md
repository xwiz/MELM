# Tiny LM Artifact Evaluation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Artifact root: `artifacts\tiny_lm`
Max validation bytes: `250000`

| Rank | Run | Tokenizer | Steps | Params | Eval Bits/Byte | Delta vs Training Report |
|---:|---|---|---:|---:|---:|---:|
| 1 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 100 | 582464 | 2.586 | 0.000000 |
| 2 | babylm_2026_tiered_morph_unigram_100step_seed13 | tiered_morph_unigram | 100 | 582464 | 3.027 | 0.000000 |
| 3 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 100 | 582464 | 3.081 | 0.000000 |
| 4 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 100 | 582464 | 3.240 | 0.000000 |
| 5 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 100 | 582464 | 3.501 | 0.000000 |
