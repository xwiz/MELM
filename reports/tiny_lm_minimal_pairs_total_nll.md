# Tiny LM Minimal Pairs

Artifact root: `artifacts\tiny_lm`
Score field: `total_nll`
Cases: `8`

| Rank | Run | Tokenizer | Accuracy | Correct | Cases |
|---:|---|---|---:|---:|---:|
| 1 | babylm_2026_hf_bpe_100step_seed13 | hf_bpe | 25.00% | 2 | 8 |
| 2 | babylm_2026_hf_unigram_100step_seed13 | hf_unigram | 25.00% | 2 | 8 |
| 3 | babylm_2026_capped_morpheme_100step_seed13 | capped_morpheme | 12.50% | 1 | 8 |

## Cases

### babylm_2026_hf_bpe_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | true | 2.0051 |
| agreement_plural_1 | agreement | false | -1.0766 |
| determiner_1 | determiner | false | -5.8036 |
| past_tense_1 | tense | false | -1.1048 |
| auxiliary_1 | auxiliary | true | 8.5852 |
| preposition_1 | preposition | false | -6.4837 |
| pronoun_1 | pronoun | false | -0.2370 |
| word_order_1 | word_order | false | -0.0580 |

### babylm_2026_hf_unigram_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.0767 |
| agreement_plural_1 | agreement | true | 0.0239 |
| determiner_1 | determiner | false | -5.1903 |
| past_tense_1 | tense | false | -6.6274 |
| auxiliary_1 | auxiliary | true | 4.0246 |
| preposition_1 | preposition | false | -6.4164 |
| pronoun_1 | pronoun | false | -0.3729 |
| word_order_1 | word_order | false | -0.7478 |

### babylm_2026_capped_morpheme_100step_seed13

| Case | Category | Correct? | Margin |
|---|---|---:|---:|
| agreement_singular_1 | agreement | false | -0.4538 |
| agreement_plural_1 | agreement | false | -0.0237 |
| determiner_1 | determiner | false | -6.7549 |
| past_tense_1 | tense | false | -5.7640 |
| auxiliary_1 | auxiliary | true | 4.8536 |
| preposition_1 | preposition | false | -7.1607 |
| pronoun_1 | pronoun | false | -0.4141 |
| word_order_1 | word_order | false | -0.1011 |
