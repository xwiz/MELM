# MELM Reports

Generated validation reports live here.

Current report command:

```powershell
python scripts\run_phase1_report.py
python scripts\run_validation_suite.py
```

The report includes tokenizer metrics, morphology boundary F1, event-memory-vs-RAG recall, and state-grounding accuracy.
It also includes an event-memory abstention probe so no-answer false positives are visible.
The validation suite converts the Phase 1 report into hard gates plus advisory research gates.

Run the tiny training smoke test with:

```powershell
python scripts\train_tiny_lm_baseline.py MELM_whitepaper.md --tokenizer simple_bpe --steps 2 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 2 --out-json reports\tiny_lm_baseline.json
python scripts\run_tiny_lm_tokenizer_ablation.py MELM_whitepaper.md --tokenizers simple_bpe,unigram_like,heuristic_morpheme --steps 2 --sequence-length 32 --embedding-dim 32 --layers 1 --heads 4 --batch-size 2 --out-json reports\tiny_lm_tokenizer_ablation.json --out-md reports\tiny_lm_tokenizer_ablation.md
```

Local corpus manifests can be generated with:

```powershell
python scripts\build_corpus_manifest.py . --name melm_docs --source melm_docs --out reports\corpus_manifest.json
```

Then run probes from the manifest:

```powershell
python scripts\run_tokenizer_lm_probe.py --manifest reports\corpus_manifest.json
python scripts\run_tokenizer_decision.py --manifest reports\corpus_manifest.json
python scripts\run_tokenizer_stability.py --manifest reports\corpus_manifest.json --json reports\tokenizer_stability.json
python scripts\run_phase1_report.py --manifest reports\corpus_manifest.json
```

BabyLM-style local corpora can be converted into the same manifest format:

```powershell
python scripts\download_babylm_2026_strict_small.py
python scripts\build_babylm_manifest.py C:\data\babylm --track 10M --name babylm_10m --version local --license unknown --out reports\babylm_10m_manifest.json
python scripts\run_tokenizer_lm_probe.py --manifest reports\babylm_10m_manifest.json
python scripts\run_tokenizer_decision.py --manifest reports\babylm_10m_manifest.json
python scripts\run_tokenizer_stability.py --manifest reports\babylm_10m_manifest.json --json reports\babylm_10m_tokenizer_stability.json
python scripts\run_fast_tokenizer_lm_probe.py --manifest reports\babylm_10m_manifest.json --vocab-size 8192 --arms hf_bpe,hf_unigram,capped_morpheme
```

For large corpora, use byte-capped probes first:

```powershell
python scripts\run_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
python scripts\run_tokenizer_decision.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000
python scripts\run_fast_tokenizer_lm_probe.py --manifest reports\babylm_2026_strict_small_manifest.json --max-train-bytes 1000000 --max-validation-bytes 500000 --vocab-size 4096 --arms hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram
python scripts\summarize_fast_tokenizer_decision.py --report reports\babylm_2026_fast_tokenizer_full.json
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --max-train-bytes 500000 --max-validation-bytes 250000 --steps 10 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --pad-vocab-to-max-size --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_matched_tiny_lm_ablation_tiered_hybrid.json --out-md reports\babylm_2026_matched_tiny_lm_ablation_tiered_hybrid.md
python scripts\summarize_tiny_lm_ablation.py --report reports\babylm_2026_matched_tiny_lm_ablation.json
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme --seeds 3,13,23 --max-train-bytes 500000 --max-validation-bytes 250000 --steps 10 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_multiseed_tiny_lm_ablation.json --out-md reports\babylm_2026_multiseed_tiny_lm_ablation.md
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme --seeds 3,13,23 --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_multiseed_tiny_lm_ablation_100step.json --out-md reports\babylm_2026_multiseed_tiny_lm_ablation_100step.md
python scripts\summarize_tiny_lm_progression.py
```

Save and verify reloadable tiny-LM artifacts with:

```powershell
python scripts\train_tiny_lm_checkpoint.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizer capped_morpheme --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --seed 13 --out-dir artifacts\tiny_lm\babylm_2026_capped_morpheme_100step_seed13
python scripts\train_tiny_lm_checkpoint.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizer hf_bpe --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --seed 13 --out-dir artifacts\tiny_lm\babylm_2026_hf_bpe_100step_seed13
python scripts\train_tiny_lm_checkpoint.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizer hf_unigram --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --seed 13 --out-dir artifacts\tiny_lm\babylm_2026_hf_unigram_100step_seed13
python scripts\train_tiny_lm_checkpoint.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizer tiered_morph_unigram --max-train-bytes 500000 --max-validation-bytes 250000 --steps 100 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --seed 13 --out-dir artifacts\tiny_lm\babylm_2026_tiered_morph_unigram_100step_seed13
python scripts\summarize_tiny_lm_artifacts.py
python scripts\evaluate_tiny_lm_artifacts.py --manifest reports\babylm_2026_strict_small_manifest.json --root artifacts\tiny_lm --max-validation-bytes 250000 --out-json reports\tiny_lm_artifact_evaluation.json --out-md reports\tiny_lm_artifact_evaluation.md
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --out-json reports\tiny_lm_minimal_pairs.json --out-md reports\tiny_lm_minimal_pairs.md
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --score-field bits_per_byte --out-json reports\tiny_lm_minimal_pairs_bits_per_byte.json --out-md reports\tiny_lm_minimal_pairs_bits_per_byte.md
python scripts\run_tiny_lm_minimal_pairs.py --root artifacts\tiny_lm --score-field total_nll --out-json reports\tiny_lm_minimal_pairs_total_nll.json --out-md reports\tiny_lm_minimal_pairs_total_nll.md
python scripts\summarize_checkpoint_validation_decision.py --artifact-report reports\tiny_lm_artifact_evaluation.json --minimal-pair-report reports\tiny_lm_minimal_pairs.json --out-json reports\tiny_lm_checkpoint_validation_decision.json --out-md reports\tiny_lm_checkpoint_validation_decision.md
python scripts\profile_babylm_fast_eval.py
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20 --out-json reports\tiny_lm_blimp_fast_20perfile.json --out-md reports\tiny_lm_blimp_fast_20perfile.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20 --score-field bits_per_byte --out-json reports\tiny_lm_blimp_fast_20perfile_bits_per_byte.json --out-md reports\tiny_lm_blimp_fast_20perfile_bits_per_byte.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --max-cases-per-file 20 --score-field total_nll --out-json reports\tiny_lm_blimp_fast_20perfile_total_nll.json --out-md reports\tiny_lm_blimp_fast_20perfile_total_nll.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_blimp_fast_full.json --out-md reports\tiny_lm_blimp_fast_full.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field bits_per_byte --out-json reports\tiny_lm_blimp_fast_full_bits_per_byte.json --out-md reports\tiny_lm_blimp_fast_full_bits_per_byte.md
python scripts\run_tiny_lm_blimp_fast.py --root artifacts\tiny_lm --score-field total_nll --out-json reports\tiny_lm_blimp_fast_full_total_nll.json --out-md reports\tiny_lm_blimp_fast_full_total_nll.md
python scripts\run_tiny_lm_entity_tracking_fast.py --root artifacts\tiny_lm --out-json reports\tiny_lm_entity_tracking_fast.json --out-md reports\tiny_lm_entity_tracking_fast.md
python scripts\run_multiseed_tiny_lm_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --candidate tiered_morph_unigram --seeds 3,13,23 --max-train-bytes 500000 --max-validation-bytes 250000 --steps 200 --sequence-length 64 --embedding-dim 64 --layers 1 --heads 4 --batch-size 8 --max-vocab-size 4096 --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_200step.json --out-md reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_200step.md
python scripts\summarize_tiny_lm_progression.py --reports reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_10step.json reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_50step.json reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_100step.json reports\babylm_2026_multiseed_tiny_lm_ablation_tiered_200step.json --out-json reports\babylm_2026_tiered_tiny_lm_progression.json --out-md reports\babylm_2026_tiered_tiny_lm_progression.md
python scripts\summarize_tokenizer_stage_gate.py
python scripts\run_tiny_lm_tokenizer_ablation.py --manifest reports\babylm_2026_strict_small_manifest.json --tokenizers hf_bpe,hf_unigram,capped_morpheme,tiered_morph_unigram --max-train-bytes 1000000 --max-validation-bytes 500000 --steps 200 --sequence-length 96 --embedding-dim 128 --layers 2 --heads 4 --batch-size 8 --max-vocab-size 4096 --pad-vocab-to-max-size --tokenizer-vocab-size 4096 --out-json reports\babylm_2026_small_proxy_tokenizer_ablation.json --out-md reports\babylm_2026_small_proxy_tokenizer_ablation.md
python scripts\summarize_tiny_lm_ablation.py --report reports\babylm_2026_small_proxy_tokenizer_ablation.json --candidate tiered_morph_unigram --out-json reports\babylm_2026_small_proxy_tokenizer_decision.json --out-md reports\babylm_2026_small_proxy_tokenizer_decision.md
python scripts\summarize_tokenizer_stage_gate.py
python scripts\plan_small_model_stage.py
python scripts\preflight_small_model_stage.py
python scripts\run_entity_tracking_symbolic.py
python scripts\summarize_small_model_stage_gate.py
python scripts\run_state_assisted_entity_tracking.py
python scripts\summarize_memory_integration_gate.py
python scripts\run_persistent_dialogue_demo.py
python scripts\run_persistent_session_demo.py --reset
python scripts\run_transcript_session_demo.py --reset --include-sample-distractors --include-sample-noisy-cases
python scripts\run_memory_resource_probe.py
python scripts\run_morpheme_meaning_mvp.py
```

Synthetic episodic fixtures can be exported and then evaluated from JSONL:

```powershell
python scripts\export_synthetic_benchmark.py
python scripts\validate_episodic_benchmark.py --events benchmarks\synthetic_episodic_events.jsonl --cases benchmarks\synthetic_episodic_cases.jsonl
python scripts\run_episodic_benchmark.py --events benchmarks\synthetic_episodic_events.jsonl --cases benchmarks\synthetic_episodic_cases.jsonl
```

Run the abstention probe directly with:

```powershell
python scripts\run_memory_abstention.py
```

Run the authored dialogue smoke test with:

```powershell
python scripts\export_authored_dialogue_benchmark.py
python scripts\validate_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
python scripts\run_authored_dialogue_benchmark.py
python scripts\run_authored_dialogue_benchmark.py --events benchmarks\authored_dialogue_events.jsonl --recall-cases benchmarks\authored_dialogue_recall_cases.jsonl --evidence-cases benchmarks\authored_dialogue_evidence_cases.jsonl
```

Compile a transcript annotation file into the same fixture format with:

```powershell
python scripts\build_dialogue_benchmark_from_transcript.py --annotations benchmarks\sample_transcript_annotations.jsonl
python scripts\analyze_dialogue_benchmark.py --events benchmarks\sample_transcript_events.jsonl --recall-cases benchmarks\sample_transcript_recall_cases.jsonl --evidence-cases benchmarks\sample_transcript_evidence_cases.jsonl
```

Run the state-resolution benchmark directly with:

```powershell
python scripts\run_state_resolution_benchmark.py --out-json reports\state_resolution.json
```
