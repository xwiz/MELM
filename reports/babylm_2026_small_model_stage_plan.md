# BabyLM 2026 Small-Model Stage Plan

Config: `experiments\babylm\small_model_tokenizer_stage.json`
Candidate: `tiered_morph_unigram`
Tokenizers: `hf_bpe, hf_unigram, capped_morpheme, tiered_morph_unigram`

## Dependency Check

- Gate decision: `advance_to_scaled_neural_ablation`
- Proxy decision: `promote_to_scaled_neural_ablation`
- Proxy candidate: `tiered_morph_unigram`
- Pass: `True`

## Training Profile

- Manifest: `reports/babylm_2026_strict_small_manifest.json`
- Seeds: `3, 13, 23`
- Train bytes cap: `5000000`
- Validation bytes cap: `1000000`
- Steps: `1000`
- Sequence length: `192`
- Embedding/layers/heads: `384/6/8`
- Batch size: `8`
- Vocab size: `16384`

## Estimates

- Parameters per arm: `23319808`
- Training tokens per arm: `1536000`
- Lower-bound FLOPs per arm: `214915350528000`
- Lower-bound FLOPs, all tokenizers one seed: `859661402112000`
- Lower-bound FLOPs, full multi-seed: `2578984206336000`

## Commands

### 1. Command

```powershell
python scripts\preflight_small_model_stage.py --config experiments\babylm\small_model_tokenizer_stage.json --out-json reports\babylm_2026_small_model_stage_preflight.json --out-md reports\babylm_2026_small_model_stage_preflight.md
```

### 2. Command

```powershell
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --seeds 3,13,23 --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --device auto --candidate tiered_morph_unigram --resume --run-cache-dir artifacts\babylm_2026_small_model_stage\multiseed_cache --out-json reports\babylm_2026_small_model_stage_multiseed.json --out-md reports\babylm_2026_small_model_stage_multiseed.md
```

### 3. Command

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizer hf_bpe --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --seed 13 --device auto --out-dir artifacts\babylm_2026_small_model_stage\hf_bpe_seed13_1000step
```

### 4. Command

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizer hf_unigram --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --seed 13 --device auto --out-dir artifacts\babylm_2026_small_model_stage\hf_unigram_seed13_1000step
```

### 5. Command

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizer capped_morpheme --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --seed 13 --device auto --out-dir artifacts\babylm_2026_small_model_stage\capped_morpheme_seed13_1000step
```

### 6. Command

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizer tiered_morph_unigram --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --seed 13 --device auto --out-dir artifacts\babylm_2026_small_model_stage\tiered_morph_unigram_seed13_1000step
```

### 7. Command

```powershell
python scripts\summarize_tiny_lm_artifacts.py --root artifacts\babylm_2026_small_model_stage --out-json reports\babylm_2026_small_model_stage_artifact_index.json --out-md reports\babylm_2026_small_model_stage_artifact_index.md
```

### 8. Command

```powershell
python scripts\evaluate_tiny_lm_artifacts.py --manifest reports/babylm_2026_strict_small_manifest.json --root artifacts\babylm_2026_small_model_stage --max-validation-bytes 1000000 --device auto --out-json reports\babylm_2026_small_model_stage_artifact_eval.json --out-md reports\babylm_2026_small_model_stage_artifact_eval.md
```

### 9. Command

```powershell
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\babylm_2026_small_model_stage --out-json reports\babylm_2026_small_model_stage_blimp_mean_nll.json --out-md reports\babylm_2026_small_model_stage_blimp_mean_nll.md
```

### 10. Command

```powershell
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\babylm_2026_small_model_stage --score-field bits_per_byte --out-json reports\babylm_2026_small_model_stage_blimp_bits_per_byte.json --out-md reports\babylm_2026_small_model_stage_blimp_bits_per_byte.md
```

### 11. Command

```powershell
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\babylm_2026_small_model_stage --score-field total_nll --out-json reports\babylm_2026_small_model_stage_blimp_total_nll.json --out-md reports\babylm_2026_small_model_stage_blimp_total_nll.md
```

### 12. Command

```powershell
python scripts\run_tiny_lm_entity_tracking_fast.py --root artifacts\babylm_2026_small_model_stage --out-json reports\babylm_2026_small_model_stage_entity_tracking.json --out-md reports\babylm_2026_small_model_stage_entity_tracking.md
```

### 13. Command

```powershell
python scripts\run_entity_tracking_symbolic.py --out-json reports\babylm_2026_small_model_stage_entity_tracking_symbolic.json --out-md reports\babylm_2026_small_model_stage_entity_tracking_symbolic.md
```

## Go/No-Go

- Run only if the dependency status is pass.
- Promote tiered_morph_unigram only if it beats the best HF baseline on bits/byte without losing fast-BLiMP/entity checks.
- Keep capped_morpheme as a compression/control arm even if it is not selected as the production tokenizer.
- If entity tracking regresses beyond the existing tolerance, prioritize event-memory-assisted state handling before larger model scale.
