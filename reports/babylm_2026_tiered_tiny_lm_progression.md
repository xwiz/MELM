# Tiny LM Progression

Mean bits/byte across multi-seed matched-parameter tiny LM ablations.

| Rank | Tokenizer | 10 steps | 50 steps | 100 steps | 200 steps | Relative Improvement |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.296 | 3.075 | 2.545 | 2.216 | 32.76% |
| 2 | tiered_morph_unigram | 3.763 | 3.539 | 2.989 | 2.679 | 28.82% |
| 3 | hf_bpe | 3.697 | 3.503 | 3.050 | 2.769 | 25.10% |
| 4 | hf_unigram | 4.168 | 3.884 | 3.204 | 2.851 | 31.61% |
