# Small-Model Stage Preflight

Config: `experiments\babylm\small_model_tokenizer_stage.json`
Status: `pass`
Free disk bytes: `4596609024`
CUDA available: `True`
Estimated parameters per arm: `23319808`
Estimated checkpoint bytes: `373116928`
Estimated training memory lower bound bytes: `656232448`

## CUDA Devices

- `0` `NVIDIA GeForce RTX 4060 Laptop GPU` memory `8585216000`

## Checks

| Check | Status | Detail |
|---|---|---|
| manifest | pass | C:\Users\Son\cowork\MELM\reports\babylm_2026_strict_small_manifest.json |
| source_gate | pass | C:\Users\Son\cowork\MELM\reports\tokenizer_stage_gate.json |
| source_proxy_decision | pass | C:\Users\Son\cowork\MELM\reports\babylm_2026_small_proxy_tokenizer_decision.json |
| disk | pass | free=4596609024 required=2000000000 checkpoint_estimate=373116928 |
| device | pass | cuda_available=True largest_device_memory=8585216000 training_memory_lower_bound=656232448 |
