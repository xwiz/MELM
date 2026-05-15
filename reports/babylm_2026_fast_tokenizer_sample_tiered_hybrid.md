# Fast Tokenizer LM Probe

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Train bytes: `955858`
Validation bytes: `499998`
Elapsed seconds: `9.37`
Rank by: `bits_per_byte`

| Rank | Tokenizer | NLL/Token | Bits/Byte | Vocab | Train Tokens | Validation Tokens |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 5.618 | 2.222 | 4096 | 262664 | 137070 |
| 2 | tiered_morph_unigram | 5.873 | 2.612 | 3471 | 273448 | 154120 |
| 3 | hf_bpe | 6.224 | 2.729 | 3983 | 270910 | 151977 |
| 4 | hf_unigram | 5.691 | 2.790 | 4108 | 302559 | 169911 |
| 5 | hybrid_morph_unigram | 5.432 | 2.877 | 3936 | 305101 | 183560 |
