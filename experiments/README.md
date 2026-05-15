# MELM Experiments

This directory will hold experiment configs and run notes.

Planned groups:

- `babylm`: local BabyLM-style manifest and reproduction configs.
- `tokenizers`: BPE, Unigram, morphology-aware, and byte/patch comparisons.
- `memory`: RAG, temporal RAG, and event-memory retrieval comparisons.
- `models`: 125M-370M model baselines and integration runs.

Each experiment should record:

- dataset manifest;
- model/config version;
- random seed;
- compute environment;
- metrics;
- failure notes.

Current smoke command:

```powershell
python scripts\run_phase1_smoke.py
python scripts\build_corpus_manifest.py . --name melm_docs --source melm_docs --out reports\corpus_manifest.json
python scripts\run_tokenizer_lm_probe.py --manifest reports\corpus_manifest.json
python scripts\run_episodic_benchmark.py --stories 25 --k 2
python scripts\run_memory_ablation.py --stories 25 --k 2
python scripts\run_morphology_probe.py
python scripts\run_tokenizer_lm_probe.py MELM_whitepaper.md
python scripts\run_tokenizer_report.py MELM_whitepaper.md
python scripts\train_tiny_lm_baseline.py MELM_whitepaper.md --tokenizer simple_bpe --steps 2 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 2 --out-json reports\tiny_lm_baseline.json
python scripts\run_tiny_lm_tokenizer_ablation.py MELM_whitepaper.md --tokenizers simple_bpe,unigram_like,heuristic_morpheme --steps 2 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 2 --out-json reports\tiny_lm_tokenizer_ablation.json --out-md reports\tiny_lm_tokenizer_ablation.md
python scripts\run_phase1_report.py
python scripts\run_validation_suite.py
```

This prints tokenizer comparison metrics and the first event-memory-vs-RAG recall gate on a tiny controlled fixture.
The tiny LM trainer is a pipeline smoke test, not evidence for the final model-size hypothesis.

BabyLM local manifest setup is documented in `docs/babylm_reproduction.md` and
configured in `experiments/babylm/local_manifest_smoke.json`.

The current post-proxy tokenizer/model stage is configured in
`experiments/babylm/small_model_tokenizer_stage.json`. Generate the run card
with:

```powershell
python scripts\plan_small_model_stage.py
```

This produces `reports/babylm_2026_small_model_stage_plan.md` with exact
commands, dependency status, and lower-bound compute estimates.

Before running the full stage, run:

```powershell
python scripts\preflight_small_model_stage.py
```

The current local readiness report is
`reports/babylm_2026_stage_execution_readiness.md`. The multiseed command in
the generated run card uses `--resume` plus a per-run cache directory so
completed tokenizer/seed arms can survive interruption.

Memory integration experiments start in
`experiments/memory/entity_tracking_state_integration.json`. The first run uses
a state-first/LM-fallback policy on BabyLM fast entity tracking:

```powershell
python scripts\run_state_assisted_entity_tracking.py
```

This validates the integration wiring on a regular-format task. The next memory
experiments should use noisier authored dialogue and episodic fixtures.

The memory integration gate is generated with:

```powershell
python scripts\summarize_memory_integration_gate.py
```

Current decision: `advance_to_persistent_dialogue_demo`.

The first persistent dialogue scaffold is configured in
`experiments/demo/persistent_dialogue_demo.json` and runs with:

```powershell
python scripts\run_persistent_dialogue_demo.py
```

The reloadable persistent session demo is configured in
`experiments/demo/persistent_dialogue_session_demo.json` and runs with:

```powershell
python scripts\run_persistent_session_demo.py --reset
```

The transcript-derived persistent session smoke is configured in
`experiments/demo/transcript_session_demo.json` and runs with:

```powershell
python scripts\run_transcript_session_demo.py --reset --include-sample-distractors --include-sample-noisy-cases
```

The current resource-efficiency and morpheme-meaning MVP probes are configured
in `experiments/memory/resource_efficiency_probe.json` and
`experiments/semantics/morpheme_meaning_mvp.json`:

```powershell
python scripts\run_memory_resource_probe.py
python scripts\run_morpheme_meaning_mvp.py
```
