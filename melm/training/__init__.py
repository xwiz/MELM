"""Training orchestration helpers for MELM validation."""

from .decision import TinyLMTokenizerDecision, decide_tiny_lm_tokenizer_ablation
from .multiseed import (
    MultiSeedTokenizerSummary,
    summaries_as_decision_reports,
    summarize_multiseed_reports,
)
from .progression import (
    StepTokenizerSummary,
    TokenizerProgression,
    summarize_progression,
)
from .small_model_plan import (
    SmallModelStageSpec,
    build_small_model_stage_plan,
    estimate_tiny_decoder_parameters,
    estimate_training_flops,
    small_model_spec_from_mapping,
)
from .small_model_preflight import (
    PreflightCheck,
    SmallModelPreflightReport,
    estimate_training_memory_lower_bound,
    preflight_small_model_stage,
)
from .tiny_lm import (
    SPECIAL_TOKENS,
    TinyLMCheckpointEvaluationReport,
    TinyLMConfig,
    TinyLMContinuationScore,
    TinyLMTextScore,
    TinyLMTrainingReport,
    TokenVocabulary,
    build_token_vocabulary,
    evaluate_tiny_lm_checkpoint,
    load_token_vocabulary,
    make_lm_sequences,
    score_tiny_lm_continuations,
    score_tiny_lm_texts,
    train_tiny_lm_baseline,
)

__all__ = [
    "SPECIAL_TOKENS",
    "TinyLMCheckpointEvaluationReport",
    "TinyLMConfig",
    "TinyLMContinuationScore",
    "TinyLMTextScore",
    "TinyLMTokenizerDecision",
    "TinyLMTrainingReport",
    "MultiSeedTokenizerSummary",
    "PreflightCheck",
    "StepTokenizerSummary",
    "SmallModelStageSpec",
    "SmallModelPreflightReport",
    "TokenizerProgression",
    "TokenVocabulary",
    "build_token_vocabulary",
    "build_small_model_stage_plan",
    "decide_tiny_lm_tokenizer_ablation",
    "evaluate_tiny_lm_checkpoint",
    "estimate_tiny_decoder_parameters",
    "estimate_training_flops",
    "estimate_training_memory_lower_bound",
    "load_token_vocabulary",
    "make_lm_sequences",
    "score_tiny_lm_continuations",
    "score_tiny_lm_texts",
    "small_model_spec_from_mapping",
    "preflight_small_model_stage",
    "summaries_as_decision_reports",
    "summarize_multiseed_reports",
    "summarize_progression",
    "train_tiny_lm_baseline",
]
