# Tiny LM Tokenizer Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`

This is a neural training smoke test, not a BabyLM-scale result.
Rank by bits per byte because token NLL is not directly comparable across segmentations.

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 3.301 | 8.418 | 67958 | 582464 |
| 2 | hf_bpe | 3.701 | 8.427 | 76095 | 582464 |
| 3 | hf_unigram | 4.170 | 8.427 | 85741 | 582464 |
