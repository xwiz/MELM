# Tiny LM Tokenizer Ablation

Source: `MELM_whitepaper.md`

This is a neural training smoke test, not a BabyLM-scale result.
Rank by bits per byte because token NLL is not directly comparable across segmentations.

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | heuristic_morpheme | 3.194 | 6.819 | 1616 | 61633 |
| 2 | simple_bpe | 4.195 | 6.220 | 2327 | 42263 |
| 3 | unigram_like | 4.533 | 6.432 | 2432 | 50973 |
