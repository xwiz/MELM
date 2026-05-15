# MELM Benchmarks

This directory will hold benchmark adapters and custom tasks.

Planned benchmark families:

- BabyLM/lm-evaluation-harness adapters.
- Controlled episodic recall.
- Temporal-neighbor recall.
- Cross-event causal question answering.
- State-grounding and contradiction checks inspired by `nameless_vector`.
- Child-level conversation probes.

All benchmarks should support both success and negative-results reporting.

Current runnable fixture:

- `melm.benchmarks.episodic_recall.episodic_memory_fixture()` provides a tiny deterministic event-recall benchmark used by tests and `scripts/run_phase1_smoke.py`.
- `melm.benchmarks.morphology.morphology_fixture()` provides a tiny gold segmentation set for boundary-F1 checks.
- `melm.benchmarks.synthetic_episodic.generate_synthetic_episodic_benchmark()` creates deterministic multi-story event chains for scalable event-memory-vs-RAG checks.
- `melm.benchmarks.generate_synthetic_evidence_benchmark()` adds controlled no-answer questions for score calibration and abstention testing.
- `melm.benchmarks.authored_child_dialogue_fixture()` provides a small hand-authored dialogue smoke test before transcript-derived evaluation.
- `melm.benchmarks.save_episodic_benchmark()` and `load_episodic_benchmark()` persist episodic tasks as JSONL fixtures for repeatable runs and future non-synthetic cases.
- `melm.benchmarks.save_dialogue_benchmark()` and `load_dialogue_benchmark()` persist full dialogue tasks with recall and no-answer evidence cases.
- `melm.benchmarks.load_annotated_transcript_benchmark()` compiles a single annotated transcript JSONL source into portable dialogue fixtures.
- `benchmarks/support_refunds_authored.jsonl` provides the first authored support/refunds corpus for MELM Guard + Memory OS. It is an internal seed batch with explicit provenance and an external-blind-batch flag, not a final publication dataset.
- `local_data/locomo10.json` can hold the public LoCoMo benchmark for session-evidence retrieval comparisons across vector RAG, Mem0-style, MemGPT-style, Zep-style, and MELM Memory OS local architecture families.
- `artifacts/letta_eval/` can hold the exported LoCoMo Letta Evals-style pack plus MELM's local results on the same JSONL dataset. This is a comparison harness, not an official Letta score until a real Letta target is configured.
- `melm.benchmarks.state_grounding.state_grounding_fixture()` provides simple precondition and contradiction cases inspired by `nameless_vector`.
- `melm.benchmarks.child_language_minimal_pairs_fixture()` provides a tiny grammatical minimal-pair ranking smoke test for saved LM checkpoints.
- `melm.benchmarks.load_blimp_fast_cases()` adapts local BabyLM 2026 fast-BLiMP JSONL files into the same minimal-pair interface.
- `benchmarks/morpheme_meaning_mvp.jsonl` provides the first constructed root/morpheme/meaning validation corpus for the full MELM whitepaper idea. Sound-symbolism is deferred from the active gate until it has stronger evidence.

The fixture is intentionally small. Its job is to validate retrieval mechanics before larger generated benchmarks are introduced. The JSONL format for dialogue fixtures is documented in `dialogue_fixture_schema.md`.

Export and rerun the current synthetic fixture with:

```powershell
python scripts\export_synthetic_benchmark.py
python scripts\validate_episodic_benchmark.py --events benchmarks\synthetic_episodic_events.jsonl --cases benchmarks\synthetic_episodic_cases.jsonl
python scripts\run_episodic_benchmark.py --events benchmarks\synthetic_episodic_events.jsonl --cases benchmarks\synthetic_episodic_cases.jsonl
python scripts\export_authored_dialogue_benchmark.py
python scripts\validate_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
python scripts\run_authored_dialogue_benchmark.py
python scripts\run_authored_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
python scripts\build_dialogue_benchmark_from_transcript.py --annotations benchmarks\sample_transcript_annotations.jsonl
python scripts\run_authored_dialogue_benchmark.py --events benchmarks\sample_transcript_events.jsonl --recall-cases benchmarks\sample_transcript_recall_cases.jsonl --evidence-cases benchmarks\sample_transcript_evidence_cases.jsonl
python scripts\analyze_dialogue_benchmark.py --events benchmarks\sample_transcript_events.jsonl --recall-cases benchmarks\sample_transcript_recall_cases.jsonl --evidence-cases benchmarks\sample_transcript_evidence_cases.jsonl
python scripts\run_memory_abstention.py
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm
python scripts\profile_babylm_fast_eval.py
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20
python scripts\run_morpheme_meaning_mvp.py
python scripts\validate_support_refund_dataset.py
python scripts\run_authored_support_refund_benchmark.py
python scripts\run_public_memory_benchmark.py --download
python scripts\export_letta_eval_pack.py --download --max-questions 250
python scripts\run_melm_letta_style_eval.py --dataset artifacts\letta_eval\locomo_letta_dataset.jsonl --memory artifacts\letta_eval\locomo_memory.jsonl
```
