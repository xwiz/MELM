# BabyLM 2026 Stage Execution Readiness

Date: 2026-05-11

## Decision

The checkpointed small-model tokenizer stage is locally executable, but should
be run with the resumable cache enabled. This machine is suitable for a pilot
run: it has CUDA on an NVIDIA GeForce RTX 4060 Laptop GPU with 8.6 GB device
memory, and the same-shape checkpoint smoke completed successfully.

## Preflight

Source: `reports/babylm_2026_small_model_stage_preflight.md`

| Check | Result |
|---|---:|
| Preflight status | pass |
| Free disk at preflight | 4,596,609,024 bytes |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8,585,216,000 bytes |
| Parameters per arm | 23,319,808 |
| Estimated checkpoint bytes for four arms | 373,116,928 |
| Training-memory lower bound | 656,232,448 bytes |

The disk margin is acceptable for the current checkpoint estimate, but still
tight. Avoid extra local corpus/cache downloads before the stage finishes.

## Same-Shape GPU Smoke

Command shape:

- tokenizer: `tiered_morph_unigram`
- model: 23.3M parameters
- sequence length: 192
- hidden/layers/heads: 384/6/8
- steps: 1
- device: CUDA

Result:

| Metric | Value |
|---|---:|
| Validation bits/byte | 4.429877 |
| Reload delta vs training report | 0.000000 |
| Checkpoint device | cuda |

Source: `reports/babylm_2026_stage_gpu_smoke_artifact_eval.md`

## Resume Smoke

The multiseed runner now supports `--resume` and `--run-cache-dir`. A tiny
two-arm run wrote per-run cache files for:

- `hf_bpe_seed13.json`
- `tiered_morph_unigram_seed13.json`

The immediate rerun reused those cache files and completed without retraining.
This is the right failure mode for the full stage: completed tokenizer/seed arms
survive interruption.

## Next Command

The generated run card now starts with preflight and then runs the resumable
multiseed stage:

```powershell
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports/babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --seeds 3,13,23 --max-train-bytes 5000000 --max-validation-bytes 1000000 --steps 1000 --sequence-length 192 --embedding-dim 384 --layers 6 --heads 8 --batch-size 8 --max-vocab-size 16384 --tokenizer-vocab-size 8192 --learning-rate 0.0003 --device auto --candidate tiered_morph_unigram --resume --run-cache-dir artifacts\babylm_2026_small_model_stage\multiseed_cache --out-json reports\babylm_2026_small_model_stage_multiseed.json --out-md reports\babylm_2026_small_model_stage_multiseed.md
```

Run this only when the machine can stay on long enough for the multiseed stage.
If it stops, rerun the same command; completed arms should be reused from the
cache directory.

## Artifact Retention

After downstream reports were generated, the temporary same-shape smoke artifact
and the 23.3M checkpoint artifacts were removed to recover disk space on the
local C: drive. The reports remain in `reports/`, and the checkpoints are
regenerable from `reports/babylm_2026_small_model_stage_plan.md`.
