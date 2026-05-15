# Tiny LM Tokenizer Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`

This is a neural training smoke test, not a BabyLM-scale result.
Rank by bits per byte because token NLL is not directly comparable across segmentations.

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | heuristic_morpheme | 3.051 | 7.780 | 67958 | 146848 |
| 2 | simple_bpe | 3.938 | 6.985 | 97707 | 73983 |
| 3 | unigram_like | 4.404 | 7.070 | 107939 | 79183 |
