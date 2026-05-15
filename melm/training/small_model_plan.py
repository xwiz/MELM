"""Run-card generation for the next BabyLM-style tokenizer stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmallModelStageSpec:
    name: str
    manifest: str
    tokenizers: tuple[str, ...]
    candidate: str
    seeds: tuple[int, ...]
    max_train_bytes: int
    max_validation_bytes: int
    steps: int
    sequence_length: int
    embedding_dim: int
    layers: int
    heads: int
    batch_size: int
    max_vocab_size: int
    tokenizer_vocab_size: int
    learning_rate: float
    checkpoint_seed: int
    artifact_root: str
    report_prefix: str
    device: str = "auto"
    source_gate: str | None = None
    source_proxy_decision: str | None = None


def small_model_spec_from_mapping(payload: dict) -> SmallModelStageSpec:
    """Build a stage spec from a JSON/YAML-like mapping."""

    required = [
        "name",
        "manifest",
        "tokenizers",
        "candidate",
        "seeds",
        "max_train_bytes",
        "max_validation_bytes",
        "steps",
        "sequence_length",
        "embedding_dim",
        "layers",
        "heads",
        "batch_size",
        "max_vocab_size",
        "tokenizer_vocab_size",
        "learning_rate",
        "checkpoint_seed",
        "artifact_root",
        "report_prefix",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing small-model stage fields: {', '.join(missing)}")

    tokenizers = tuple(str(item) for item in payload["tokenizers"])
    seeds = tuple(int(item) for item in payload["seeds"])
    if not tokenizers:
        raise ValueError("At least one tokenizer is required")
    if not seeds:
        raise ValueError("At least one seed is required")
    if str(payload["candidate"]) not in tokenizers:
        raise ValueError("Candidate tokenizer must be included in tokenizers")
    if int(payload["checkpoint_seed"]) not in seeds:
        raise ValueError("Checkpoint seed should be one of the scheduled seeds")
    if int(payload["embedding_dim"]) % int(payload["heads"]) != 0:
        raise ValueError("embedding_dim must be divisible by heads")

    return SmallModelStageSpec(
        name=str(payload["name"]),
        manifest=str(payload["manifest"]),
        tokenizers=tokenizers,
        candidate=str(payload["candidate"]),
        seeds=seeds,
        max_train_bytes=int(payload["max_train_bytes"]),
        max_validation_bytes=int(payload["max_validation_bytes"]),
        steps=int(payload["steps"]),
        sequence_length=int(payload["sequence_length"]),
        embedding_dim=int(payload["embedding_dim"]),
        layers=int(payload["layers"]),
        heads=int(payload["heads"]),
        batch_size=int(payload["batch_size"]),
        max_vocab_size=int(payload["max_vocab_size"]),
        tokenizer_vocab_size=int(payload["tokenizer_vocab_size"]),
        learning_rate=float(payload["learning_rate"]),
        checkpoint_seed=int(payload["checkpoint_seed"]),
        artifact_root=str(payload["artifact_root"]),
        report_prefix=str(payload["report_prefix"]),
        device=str(payload.get("device", "auto")),
        source_gate=payload.get("source_gate"),
        source_proxy_decision=payload.get("source_proxy_decision"),
    )


def estimate_tiny_decoder_parameters(
    *,
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
    layers: int,
) -> int:
    """Estimate parameters for the local TinyDecoderLM architecture."""

    embeddings = vocab_size * embedding_dim + sequence_length * embedding_dim
    output = embedding_dim * vocab_size + vocab_size
    layer_parameters = (12 * embedding_dim * embedding_dim) + (13 * embedding_dim)
    return embeddings + output + (layers * layer_parameters)


def estimate_training_flops(*, parameters: int, steps: int, batch_size: int, sequence_length: int) -> int:
    """Lower-bound dense LM training FLOPs using the common 6N token rule."""

    training_tokens = steps * batch_size * sequence_length
    return 6 * parameters * training_tokens


def build_small_model_stage_plan(
    spec: SmallModelStageSpec,
    *,
    gate_payload: dict | None = None,
    proxy_payload: dict | None = None,
    config_path: str | None = None,
) -> dict:
    """Create a deterministic run card for the next tokenizer/model stage."""

    parameters = estimate_tiny_decoder_parameters(
        vocab_size=spec.max_vocab_size,
        sequence_length=spec.sequence_length,
        embedding_dim=spec.embedding_dim,
        layers=spec.layers,
    )
    training_tokens_per_arm = spec.steps * spec.batch_size * spec.sequence_length
    flops_per_arm = estimate_training_flops(
        parameters=parameters,
        steps=spec.steps,
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
    )
    dependency_status = _dependency_status(
        gate_payload,
        proxy_payload,
        candidate=spec.candidate,
    )
    commands = _commands(spec, config_path=config_path)
    return {
        "name": spec.name,
        "candidate": spec.candidate,
        "tokenizers": list(spec.tokenizers),
        "dependency_status": dependency_status,
        "config": {
            "manifest": spec.manifest,
            "seeds": list(spec.seeds),
            "max_train_bytes": spec.max_train_bytes,
            "max_validation_bytes": spec.max_validation_bytes,
            "steps": spec.steps,
            "sequence_length": spec.sequence_length,
            "embedding_dim": spec.embedding_dim,
            "layers": spec.layers,
            "heads": spec.heads,
            "batch_size": spec.batch_size,
            "max_vocab_size": spec.max_vocab_size,
            "tokenizer_vocab_size": spec.tokenizer_vocab_size,
            "learning_rate": spec.learning_rate,
            "checkpoint_seed": spec.checkpoint_seed,
            "artifact_root": spec.artifact_root,
            "report_prefix": spec.report_prefix,
            "device": spec.device,
        },
        "estimates": {
            "parameters_per_arm": parameters,
            "training_tokens_per_arm": training_tokens_per_arm,
            "lower_bound_training_flops_per_arm": flops_per_arm,
            "lower_bound_training_flops_all_tokenizers_one_seed": flops_per_arm * len(spec.tokenizers),
            "lower_bound_training_flops_full_multiseed": flops_per_arm * len(spec.tokenizers) * len(spec.seeds),
        },
        "commands": commands,
        "go_no_go": [
            "Run only if the dependency status is pass.",
            "Promote tiered_morph_unigram only if it beats the best HF baseline on bits/byte without losing fast-BLiMP/entity checks.",
            "Keep capped_morpheme as a compression/control arm even if it is not selected as the production tokenizer.",
            "If entity tracking regresses beyond the existing tolerance, prioritize event-memory-assisted state handling before larger model scale.",
        ],
    }


def _dependency_status(
    gate_payload: dict | None,
    proxy_payload: dict | None,
    *,
    candidate: str,
) -> dict:
    gate_decision = _nested_decision(gate_payload)
    proxy_decision = _nested_decision(proxy_payload)
    gate_pass = gate_decision in {
        "advance_to_small_model_ablation",
        "advance_to_scaled_neural_ablation",
    }
    proxy_pass = proxy_decision == "promote_to_scaled_neural_ablation"
    proxy_candidate = None
    if proxy_payload is not None:
        decision = proxy_payload.get("decision", proxy_payload)
        if isinstance(decision, dict):
            proxy_candidate = decision.get("candidate_tokenizer")
    return {
        "gate_decision": gate_decision,
        "proxy_decision": proxy_decision,
        "proxy_candidate": proxy_candidate,
        "pass": bool(gate_pass and proxy_pass and proxy_candidate == candidate),
    }


def _nested_decision(payload: dict | None) -> str | None:
    if payload is None:
        return None
    decision = payload.get("decision", payload)
    if isinstance(decision, str):
        return decision
    return decision.get("decision")


def _commands(spec: SmallModelStageSpec, *, config_path: str | None = None) -> list[str]:
    tokenizers = ",".join(spec.tokenizers)
    seeds = ",".join(str(seed) for seed in spec.seeds)
    prefix = spec.report_prefix
    preflight_config = config_path or "experiments/babylm/small_model_tokenizer_stage.json"
    commands = [
        (
            "python scripts\\preflight_small_model_stage.py "
            f"--config {preflight_config} "
            f"--out-json {prefix}_preflight.json "
            f"--out-md {prefix}_preflight.md"
        ),
        (
            "python scripts\\run_multiseed_tiny_lm_ablation.py "
            f"--manifest {spec.manifest} "
            f"--tokenizers {tokenizers} "
            f"--seeds {seeds} "
            f"--max-train-bytes {spec.max_train_bytes} "
            f"--max-validation-bytes {spec.max_validation_bytes} "
            f"--steps {spec.steps} "
            f"--sequence-length {spec.sequence_length} "
            f"--embedding-dim {spec.embedding_dim} "
            f"--layers {spec.layers} "
            f"--heads {spec.heads} "
            f"--batch-size {spec.batch_size} "
            f"--max-vocab-size {spec.max_vocab_size} "
            f"--tokenizer-vocab-size {spec.tokenizer_vocab_size} "
            f"--learning-rate {spec.learning_rate} "
            f"--device {spec.device} "
            f"--candidate {spec.candidate} "
            "--resume "
            f"--run-cache-dir {spec.artifact_root}\\multiseed_cache "
            f"--out-json {prefix}_multiseed.json "
            f"--out-md {prefix}_multiseed.md"
        )
    ]
    for tokenizer in spec.tokenizers:
        run_name = f"{tokenizer}_seed{spec.checkpoint_seed}_{spec.steps}step"
        commands.append(
            "python scripts\\train_tiny_lm_checkpoint.py "
            f"--manifest {spec.manifest} "
            f"--tokenizer {tokenizer} "
            f"--max-train-bytes {spec.max_train_bytes} "
            f"--max-validation-bytes {spec.max_validation_bytes} "
            f"--steps {spec.steps} "
            f"--sequence-length {spec.sequence_length} "
            f"--embedding-dim {spec.embedding_dim} "
            f"--layers {spec.layers} "
            f"--heads {spec.heads} "
            f"--batch-size {spec.batch_size} "
            f"--max-vocab-size {spec.max_vocab_size} "
            f"--tokenizer-vocab-size {spec.tokenizer_vocab_size} "
            f"--learning-rate {spec.learning_rate} "
            f"--seed {spec.checkpoint_seed} "
            f"--device {spec.device} "
            f"--out-dir {spec.artifact_root}\\{run_name}"
        )
    commands.extend(
        [
            (
                "python scripts\\summarize_tiny_lm_artifacts.py "
                f"--root {spec.artifact_root} "
                f"--out-json {prefix}_artifact_index.json "
                f"--out-md {prefix}_artifact_index.md"
            ),
            (
                "python scripts\\evaluate_tiny_lm_artifacts.py "
                f"--manifest {spec.manifest} "
                f"--root {spec.artifact_root} "
                f"--max-validation-bytes {spec.max_validation_bytes} "
                f"--device {spec.device} "
                f"--out-json {prefix}_artifact_eval.json "
                f"--out-md {prefix}_artifact_eval.md"
            ),
            (
                "python scripts\\run_tiny_lm_blimp_fast.py "
                f"--root {spec.artifact_root} "
                f"--out-json {prefix}_blimp_mean_nll.json "
                f"--out-md {prefix}_blimp_mean_nll.md"
            ),
            (
                "python scripts\\run_tiny_lm_blimp_fast.py "
                f"--root {spec.artifact_root} "
                "--score-field bits_per_byte "
                f"--out-json {prefix}_blimp_bits_per_byte.json "
                f"--out-md {prefix}_blimp_bits_per_byte.md"
            ),
            (
                "python scripts\\run_tiny_lm_blimp_fast.py "
                f"--root {spec.artifact_root} "
                "--score-field total_nll "
                f"--out-json {prefix}_blimp_total_nll.json "
                f"--out-md {prefix}_blimp_total_nll.md"
            ),
            (
                "python scripts\\run_tiny_lm_entity_tracking_fast.py "
                f"--root {spec.artifact_root} "
                f"--out-json {prefix}_entity_tracking.json "
                f"--out-md {prefix}_entity_tracking.md"
            ),
            (
                "python scripts\\run_entity_tracking_symbolic.py "
                f"--out-json {prefix}_entity_tracking_symbolic.json "
                f"--out-md {prefix}_entity_tracking_symbolic.md"
            ),
        ]
    )
    return commands
