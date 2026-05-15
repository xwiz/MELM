# Tiny LM Tokenizer Ablation

Source: `MELM_whitepaper.md`

This is a neural training smoke test, not a BabyLM-scale result.
Rank by bits per byte because token NLL is not directly comparable across segmentations.

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 2.323 | 4.960 | 1616 | 7727 |
| 2 | hf_bpe | 4.670 | 4.989 | 3230 | 7760 |
| 3 | hf_unigram | 4.734 | 5.034 | 3245 | 7760 |
