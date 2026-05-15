# BabyLM Reproduction Path

This repo now has a local-corpus adapter for BabyLM-style data. It does not
download data or assume a hosted dataset name; keep licensing and dataset
versioning explicit outside the code.

## Inputs

Expected local layout can be directory-based:

```text
C:\data\babylm\
  train\*.txt
  dev\*.txt
  test\*.txt
```

or filename-based:

```text
C:\data\babylm\
  train_10M.txt
  valid_10M.txt
  test_10M.txt
```

Recognized validation aliases include `dev`, `valid`, `validation`, and `val`.
If no split is visible in a path, the adapter assigns deterministic fallback
splits and reports how many files used fallback assignment.

## Commands

Build a manifest:

```powershell
python scripts\download_babylm_2026_strict_small.py
python scripts\build_babylm_manifest.py C:\data\babylm --track 10M --name babylm_10m --version local --license unknown --out reports\babylm_10m_manifest.json
```

Run tokenizer probes:

```powershell
python scripts\run_tokenizer_lm_probe.py --manifest reports\babylm_10m_manifest.json
python scripts\run_tokenizer_decision.py --manifest reports\babylm_10m_manifest.json
python scripts\run_tokenizer_stability.py --manifest reports\babylm_10m_manifest.json --json reports\babylm_10m_tokenizer_stability.json
python scripts\run_fast_tokenizer_lm_probe.py --manifest reports\babylm_10m_manifest.json --vocab-size 8192 --arms hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram
```

Profile the downloaded BabyLM 2026 fast evaluation subset:

```powershell
python scripts\profile_babylm_fast_eval.py
```

For the downloaded 2026 Strict-Small bundle, start with byte-capped probes:

```powershell
python scripts\run_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
python scripts\run_tokenizer_decision.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
python scripts\run_fast_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000 --vocab-size 4096 --arms hf_bpe,hf_unigram,capped_morpheme
```

Run the tiny neural smoke ablation:

```powershell
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_10m_manifest.json --tokenizers simple_bpe,unigram_like,heuristic_morpheme --steps 100 --sequence-length 128 --embedding-dim 128 --layers 2 --heads 4 --batch-size 32 --out-json reports\babylm_10m_tiny_lm_ablation.json --out-md reports\babylm_10m_tiny_lm_ablation.md
```

Save and reload tiny-LM artifacts once an arm deserves preservation:

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizer capped_morpheme --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --seed 13 --out-dir artifacts\tiny_lm\babylm_2026_capped_morpheme_100step_seed13
python scripts\summarize_tiny_lm_artifacts.py
python scripts\evaluate_tiny_lm_artifacts.py --manifest reports\babylm_2026_strict_small_manifest.json --root artifacts\tiny_lm --max-validation-bytes 250000
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_blimp_fast_full.json --out-md reports\tiny_lm_blimp_fast_full.md
python scripts\run_tiny_lm_entity_tracking_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_entity_tracking_fast.json --out-md reports\tiny_lm_entity_tracking_fast.md
python scripts\summarize_tokenizer_stage_gate.py
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --max-train-bytes 1000000 --max-validation-bytes 500000 --steps 200 --sequence-length 96 --embedding-dim 128 --layers 2 --heads 4 --batch-size 8 --max-vocab-size 4096 --pad-vocab-to-max-size --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_small_proxy_tokenizer_ablation.json --out-md reports\babylm_2026_small_proxy_tokenizer_ablation.md
python scripts\summarize_tiny_lm_ablation.py --report reports\babylm_2026_small_proxy_tokenizer_ablation.json --candidate tiered_morph_unigram --out-json reports\babylm_2026_small_proxy_tokenizer_decision.json --out-md reports\babylm_2026_small_proxy_tokenizer_decision.md
python scripts\summarize_tokenizer_stage_gate.py
python scripts\plan_small_model_stage.py
```

## Interpretation

The tiny neural ablation is only a pipeline check. A fast tokenizer
bits-per-byte win can promote an arm into neural ablation, but it cannot promote
morphology to primary status by itself. Final promotion still requires stable
downstream gains on a BabyLM-scale run against BPE/Unigram and byte/patch
baselines.
