# Fast Tokenizer LM Probe

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`
Train bytes: `54399840`
Validation bytes: `56752783`
Elapsed seconds: `455.16`
Rank by: `bits_per_byte`

| Rank | Tokenizer | NLL/Token | Bits/Byte | Vocab | Train Tokens | Validation Tokens |
|---:|---|---:|---:|---:|---:|---:|
| 1 | capped_morpheme | 5.842 | 2.312 | 8192 | 14996860 | 15566580 |
| 2 | hf_bpe | 6.368 | 2.540 | 8188 | 14981675 | 15694331 |
| 3 | hf_unigram | 5.728 | 2.569 | 8288 | 16802441 | 17641522 |
| 4 | hybrid_morph_unigram | 5.888 | 2.657 | 26732 | 16777857 | 17752912 |
