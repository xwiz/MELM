"""Synthetic benchmark fixtures for MELM validation."""

from .babylm_eval import (
    load_blimp_fast_cases,
    load_entity_tracking_fast_cases,
    profile_blimp_fast,
    profile_jsonl_directory,
)
from .dialogue import authored_child_dialogue_fixture
from .dialogue_noise import (
    sample_transcript_distractor_events,
    sample_transcript_noisy_evidence_cases,
)
from .entity_tracking import (
    BoxStatePrediction,
    BoxStateTracker,
    BoxStateTrackingReport,
    entity_tracking_events_from_prompt,
    evaluate_entity_tracking_symbolic,
    predict_entity_tracking_option,
)
from .evidence import EvidenceCase, generate_synthetic_evidence_benchmark
from .episodic_recall import episodic_memory_fixture
from .io import (
    load_dialogue_benchmark,
    load_episodic_benchmark,
    save_dialogue_benchmark,
    save_episodic_benchmark,
    validate_dialogue_benchmark,
    validate_episodic_benchmark,
    validate_evidence_benchmark,
)
from .minimal_pairs import MinimalPairCase, child_language_minimal_pairs_fixture
from .morphology import morphology_fixture
from .multiple_choice import MultipleChoiceCase
from .public_memory import (
    LOCOMO_URL,
    PublicArchitectureReport,
    PublicMemoryBenchmark,
    PublicMemoryComparisonReport,
    PublicContextBudgetPrediction,
    PublicContextBudgetReport,
    PublicMemoryDocument,
    PublicMemoryPrediction,
    PublicMemoryQuestion,
    evaluate_public_context_budget,
    evaluate_public_memory_architectures,
    load_locomo_public_memory_benchmark,
)
from .state_grounding import StateGroundingCase, state_grounding_fixture
from .state_resolution import StateResolutionCase, synthetic_state_resolution_fixture
from .support_refunds import (
    GuardBenchmarkCase,
    SupportMemoryCase,
    SupportRefundFixture,
    generate_support_refund_benchmark,
    support_refund_fixture,
)
from .support_refunds_authored import (
    AuthoredSupportRefundDataset,
    authored_support_refund_fixture,
    load_authored_support_refund_dataset,
    validate_authored_support_refund_dataset,
)
from .support_refunds_freeze import (
    DEFAULT_SUPPORT_REFUNDS_PREREGISTRATION_PATH,
    SupportRefundDatasetSummary,
    build_support_refunds_freeze_manifest,
    load_support_refunds_preregistration,
    sha256_file,
    summarize_support_refund_dataset,
    support_refunds_freeze_markdown,
    validate_support_refunds_preregistration,
    verify_support_refunds_freeze,
)
from .synthetic_episodic import EpisodicCase, generate_synthetic_episodic_benchmark
from .transcript import (
    AnnotatedTranscriptBenchmark,
    TranscriptTurn,
    load_annotated_transcript_benchmark,
    validate_annotated_transcript_benchmark,
)

__all__ = [
    "AnnotatedTranscriptBenchmark",
    "AuthoredSupportRefundDataset",
    "BoxStatePrediction",
    "BoxStateTracker",
    "BoxStateTrackingReport",
    "EpisodicCase",
    "EvidenceCase",
    "GuardBenchmarkCase",
    "MinimalPairCase",
    "MultipleChoiceCase",
    "PublicArchitectureReport",
    "PublicMemoryBenchmark",
    "PublicMemoryComparisonReport",
    "PublicContextBudgetPrediction",
    "PublicContextBudgetReport",
    "PublicMemoryDocument",
    "PublicMemoryPrediction",
    "PublicMemoryQuestion",
    "StateGroundingCase",
    "StateResolutionCase",
    "SupportMemoryCase",
    "SupportRefundDatasetSummary",
    "SupportRefundFixture",
    "TranscriptTurn",
    "authored_child_dialogue_fixture",
    "authored_support_refund_fixture",
    "episodic_memory_fixture",
    "entity_tracking_events_from_prompt",
    "evaluate_public_memory_architectures",
    "evaluate_public_context_budget",
    "evaluate_entity_tracking_symbolic",
    "child_language_minimal_pairs_fixture",
    "generate_synthetic_episodic_benchmark",
    "generate_synthetic_evidence_benchmark",
    "generate_support_refund_benchmark",
    "build_support_refunds_freeze_manifest",
    "load_blimp_fast_cases",
    "load_entity_tracking_fast_cases",
    "profile_jsonl_directory",
    "profile_blimp_fast",
    "predict_entity_tracking_option",
    "load_dialogue_benchmark",
    "load_annotated_transcript_benchmark",
    "load_episodic_benchmark",
    "load_authored_support_refund_dataset",
    "load_locomo_public_memory_benchmark",
    "load_support_refunds_preregistration",
    "LOCOMO_URL",
    "morphology_fixture",
    "save_dialogue_benchmark",
    "save_episodic_benchmark",
    "sample_transcript_distractor_events",
    "sample_transcript_noisy_evidence_cases",
    "sha256_file",
    "state_grounding_fixture",
    "support_refund_fixture",
    "support_refunds_freeze_markdown",
    "summarize_support_refund_dataset",
    "synthetic_state_resolution_fixture",
    "validate_annotated_transcript_benchmark",
    "validate_authored_support_refund_dataset",
    "validate_dialogue_benchmark",
    "validate_episodic_benchmark",
    "validate_evidence_benchmark",
    "validate_support_refunds_preregistration",
    "verify_support_refunds_freeze",
    "DEFAULT_SUPPORT_REFUNDS_PREREGISTRATION_PATH",
]
