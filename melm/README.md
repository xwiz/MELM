# MELM Python Package

This package holds the Python implementation for the validation-first MELM build.

Planned modules:

- `tokenization`: BPE, Unigram, morphology-aware, and byte/patch tokenizer experiments.
- `data`: BabyLM and synthetic event-dialogue data preparation.
- `memory`: external event-memory prototype and RAG baselines.
- `training`: small-model training orchestration.
- `evaluation`: BabyLM, episodic recall, and conversation evaluation adapters.

Implemented now:

- dependency-free tokenizer smoke harness;
- trainable dependency-free BPE baseline;
- optional fast HF BPE/Unigram baselines;
- capped-vocabulary tokenizer wrapper for fairer morphology ablations;
- morphology boundary-F1 probe;
- tiny held-out token-LM probe;
- tokenizer decision gate for morphology as primary vs auxiliary signal;
- deterministic corpus manifest scanner;
- BabyLM-style local corpus manifest adapter;
- deterministic event-memory and RAG retrieval baselines;
- MELM Guard procedural working-memory rule engine for support/refund action validation;
- support/refunds benchmark fixture with prompt-only, schema-only, temporal/entity RAG, and MELM runtime baselines;
- authored support/refunds JSONL corpus, validator, and benchmark runner for the next publication-readiness dataset step;
- public LoCoMo memory benchmark adapter comparing vector RAG, Mem0-style, MemGPT-style, Zep-style, and MELM Memory OS local architecture families;
- Letta Evals-style LoCoMo export pack for future official Letta-vs-MELM runs;
- local MELM SLM Appliance runtime with JSONL memory, bounded context packing, cited extractive answers, and CLI commands;
- Memory OS support-state projection with indexed order/entity lookups;
- synthetic episodic recall fixture;
- JSONL import/export for episodic benchmark fixtures;
- hand-authored child-dialogue smoke benchmark;
- annotated transcript compiler for transcript-derived dialogue benchmarks;
- answerability and abstention calibration probes for event memory;
- held-out story-split calibration for abstention thresholds;
- state-grounding checks for preconditions and contradictions;
- explicit object-location state-resolution benchmark;
- child-level minimal-pair ranking smoke benchmark for saved checkpoints;
- official BabyLM 2026 fast-BLiMP adapter for saved-checkpoint ranking checks;
- tiny PyTorch causal-LM trainer for end-to-end baseline smoke tests;
- tiny-LM checkpoint saving, tokenizer artifact loading, and checkpoint re-evaluation;
- validation gate helpers;
- evidence-gated persistent dialogue demo and JSONL-backed session memory;
- morpheme/root/meaning MVP inference harness for novel-word and utterance-routing tests;
- unit tests.

Run the current smoke check from the repo root:

```powershell
python -m unittest discover -s tests
python scripts\run_tokenizer_decision.py
python scripts\build_babylm_manifest.py C:\data\babylm --track 10M --out reports\babylm_10m_manifest.json
python scripts\train_tiny_lm_baseline.py MELM_whitepaper.md --steps 2 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 2
python scripts\evaluate_tiny_lm_artifacts.py --manifest reports\babylm_2026_strict_small_manifest.json --root artifacts\tiny_lm --max-validation-bytes 250000
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20
python scripts\run_guard_benchmark.py
python scripts\run_memory_os_benchmark.py
python scripts\run_melm_runtime_benchmark.py
python scripts\validate_support_refund_dataset.py
python scripts\run_authored_support_refund_benchmark.py
python scripts\run_public_memory_benchmark.py --download
python scripts\export_letta_eval_pack.py --download --max-questions 250
python scripts\run_melm_letta_style_eval.py --dataset artifacts\letta_eval\locomo_letta_dataset.jsonl --memory artifacts\letta_eval\locomo_memory.jsonl
python scripts\melm_appliance_cli.py build-locomo --out artifacts\melm_appliance\locomo_memory.jsonl
python scripts\run_phase1_smoke.py
```

Keep the first implementation in Python for fast ML iteration. Treat `C:\dev\nameless_vector` as a design reference until its test health is confirmed.
