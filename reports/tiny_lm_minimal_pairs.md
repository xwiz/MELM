# Tiny LM Minimal Pairs

Artifact root: `artifacts\tiny_lm`
Score field: `mean_nll_per_token`
Cases: `8`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 50.00% | 4 | 8 |
| 2 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 50.00% | 4 | 8 |
| 3 | babylm_2026_tiered_morph_unigram_100step_seed13 | tiered_morph_unigram | 50.00% | 4 | 8 |
| 4 | babylm_2026_hybrid_morph_unigram_100step_seed13 | hybrid_morph_unigram | 37.50% | 3 | 8 |
| 5 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 25.00% | 2 | 8 |

## Cases

### babylm_2026_hf_bpe_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | true | 0.3342 |
| agreement_plural_1 | agreement | false | -0.1538 |
| determiner_1 | determiner | true | 0.2544 |
| past_tense_1 | tense | false | -0.1578 |
| auxiliary_1 | auxiliary | true | 0.1747 |
| preposition_1 | preposition | true | 0.0381 |
| pronoun_1 | pronoun | false | -0.0296 |
| word_order_1 | word_order | false | -0.0064 |

### babylm_2026_hf_unigram_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.0110 |
| agreement_plural_1 | agreement | true | 0.0030 |
| determiner_1 | determiner | true | 0.2803 |
| past_tense_1 | tense | true | 0.0540 |
| auxiliary_1 | auxiliary | false | -0.2700 |
| preposition_1 | preposition | true | 0.0513 |
| pronoun_1 | pronoun | false | -0.0373 |
| word_order_1 | word_order | false | -0.0831 |

### babylm_2026_tiered_morph_unigram_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.0923 |
| agreement_plural_1 | agreement | true | 0.0665 |
| determiner_1 | determiner | true | 0.2304 |
| past_tense_1 | tense | true | 0.1924 |
| auxiliary_1 | auxiliary | false | -0.2250 |
| preposition_1 | preposition | true | 0.0393 |
| pronoun_1 | pronoun | false | -0.0286 |
| word_order_1 | word_order | false | -0.0524 |

### babylm_2026_hybrid_morph_unigram_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | true | 0.0639 |
| agreement_plural_1 | agreement | false | -0.0817 |
| determiner_1 | determiner | true | 0.0386 |
| past_tense_1 | tense | false | -0.1175 |
| auxiliary_1 | auxiliary | false | -0.2662 |
| preposition_1 | preposition | false | -0.0438 |
| pronoun_1 | pronoun | false | -0.0087 |
| word_order_1 | word_order | true | 0.0195 |

### babylm_2026_capped_morpheme_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.0648 |
| agreement_plural_1 | agreement | false | -0.0030 |
| determiner_1 | determiner | true | 0.0735 |
| past_tense_1 | tense | true | 0.2289 |
| auxiliary_1 | auxiliary | false | -0.2417 |
| preposition_1 | preposition | false | -0.0294 |
| pronoun_1 | pronoun | false | -0.0518 |
| word_order_1 | word_order | false | -0.0126 |
