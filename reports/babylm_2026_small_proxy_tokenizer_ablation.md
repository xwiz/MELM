# Tiny LM Tokenizer Ablation

Source: `manifest:reports\babylm_2026_strict_small_manifest.json`

This is a neural training smoke test, not a BabyLM-scale result.
Rank by bits per byte because token NLL is not directly comparable across segmentations.

| Rank | Tokenizer | Bits/Byte | NLL/Token | Validation Tokens | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | capped_morpheme | 2.070 | 5.232 | 137081 | 1461504 |
| 2 | tiered_morph_unigram | 2.491 | 5.602 | 154131 | 1461504 |
| 3 | hf_bpe | 2.570 | 5.861 | 151988 | 1461504 |
| 4 | hf_unigram | 2.627 | 5.359 | 169922 | 1461504 |
