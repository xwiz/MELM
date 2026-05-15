"""Tokenizer prototypes and metrics for MELM validation."""

from .arms import DEFAULT_TOKENIZER_ARMS, FAST_TOKENIZER_ARMS, build_tokenizer_arms
from .artifacts import load_tokenizer_artifact, save_tokenizer_artifact
from .boundary import BoundaryReport, MorphemeExample, evaluate_boundary_f1
from .bpe import SimpleBPETokenizer, train_bpe
from .decision import TokenizerDecisionReport, decide_tokenizer_strategy
from .factory import build_default_tokenizers, simple_vocab
from .fast_decision import FastTokenizerDecision, decide_fast_tokenizer_ablation
from .hf_tokenizers import HFTokenizerWrapper, train_hf_bpe, train_hf_unigram
from .hybrid import (
    HybridMorphUnigramTokenizer,
    TieredMorphUnigramTokenizer,
    train_hybrid_morph_unigram,
    train_tiered_morph_unigram,
)
from .language_model import (
    TokenLMReport,
    compare_unigram_lms,
    evaluate_unigram_lm,
    split_for_lm,
)
from .metrics import TokenizationReport, compare_tokenizers, evaluate_tokenizer
from .simple_tokenizers import (
    BytePatchTokenizer,
    HeuristicMorphemeTokenizer,
    Tokenizer,
    UnigramLikeTokenizer,
    WhitespaceTokenizer,
)
from .stability import (
    TokenizerFoldReport,
    TokenizerStabilityReport,
    evaluate_tokenizer_lm_stability,
    make_lm_folds,
)
from .vocab import VocabCappedTokenizer, cap_tokenizer_vocab

__all__ = [
    "BytePatchTokenizer",
    "BoundaryReport",
    "DEFAULT_TOKENIZER_ARMS",
    "FAST_TOKENIZER_ARMS",
    "FastTokenizerDecision",
    "HeuristicMorphemeTokenizer",
    "HFTokenizerWrapper",
    "HybridMorphUnigramTokenizer",
    "TieredMorphUnigramTokenizer",
    "MorphemeExample",
    "SimpleBPETokenizer",
    "TokenizationReport",
    "Tokenizer",
    "TokenLMReport",
    "TokenizerDecisionReport",
    "TokenizerFoldReport",
    "TokenizerStabilityReport",
    "UnigramLikeTokenizer",
    "VocabCappedTokenizer",
    "WhitespaceTokenizer",
    "build_default_tokenizers",
    "build_tokenizer_arms",
    "cap_tokenizer_vocab",
    "compare_tokenizers",
    "compare_unigram_lms",
    "decide_tokenizer_strategy",
    "decide_fast_tokenizer_ablation",
    "evaluate_boundary_f1",
    "evaluate_tokenizer_lm_stability",
    "evaluate_unigram_lm",
    "evaluate_tokenizer",
    "make_lm_folds",
    "load_tokenizer_artifact",
    "save_tokenizer_artifact",
    "split_for_lm",
    "simple_vocab",
    "train_bpe",
    "train_hf_bpe",
    "train_hf_unigram",
    "train_hybrid_morph_unigram",
    "train_tiered_morph_unigram",
]
