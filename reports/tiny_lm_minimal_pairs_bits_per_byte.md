# Tiny LM Minimal Pairs

Artifact root: `artifacts\tiny_lm`
Score field: `bits_per_byte`
Cases: `8`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 62.50% | 5 | 8 |
| 2 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 62.50% | 5 | 8 |
| 3 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 37.50% | 3 | 8 |

## Cases

### babylm_2026_hf_bpe_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.0159 |
| agreement_plural_1 | agreement | true | 0.0947 |
| determiner_1 | determiner | false | -0.0208 |
| past_tense_1 | tense | true | 0.1417 |
| auxiliary_1 | auxiliary | true | 0.3323 |
| preposition_1 | preposition | true | 0.0183 |
| pronoun_1 | pronoun | true | 0.1409 |
| word_order_1 | word_order | false | -0.0029 |

### babylm_2026_hf_unigram_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.1906 |
| agreement_plural_1 | agreement | true | 0.1739 |
| determiner_1 | determiner | true | 0.0008 |
| past_tense_1 | tense | false | -0.0920 |
| auxiliary_1 | auxiliary | true | 0.0713 |
| preposition_1 | preposition | true | 0.0247 |
| pronoun_1 | pronoun | true | 0.1441 |
| word_order_1 | word_order | false | -0.0372 |

### babylm_2026_capped_morpheme_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.2253 |
| agreement_plural_1 | agreement | true | 0.1764 |
| determiner_1 | determiner | false | -0.1570 |
| past_tense_1 | tense | false | -0.0908 |
| auxiliary_1 | auxiliary | true | 0.1438 |
| preposition_1 | preposition | false | -0.0496 |
| pronoun_1 | pronoun | true | 0.1163 |
| word_order_1 | word_order | false | -0.0050 |
