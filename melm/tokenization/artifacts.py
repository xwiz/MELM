"""Save tokenizer artifacts for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .hf_tokenizers import HFTokenizerWrapper
from .hybrid import HybridMorphUnigramTokenizer, TieredMorphUnigramTokenizer
from .simple_tokenizers import (
    BytePatchTokenizer,
    HeuristicMorphemeTokenizer,
    UnigramLikeTokenizer,
    WhitespaceTokenizer,
)
from .vocab import VocabCappedTokenizer


def save_tokenizer_artifact(tokenizer, output_dir: str | Path) -> Path:
    """Save tokenizer metadata/files and return the metadata path."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": tokenizer.name,
        "class": tokenizer.__class__.__name__,
    }

    if isinstance(tokenizer, HFTokenizerWrapper):
        tokenizer_path = target / "tokenizer.json"
        tokenizer.tokenizer.save(str(tokenizer_path))
        metadata["type"] = "hf_tokenizers"
        metadata["tokenizer_json"] = tokenizer_path.name
    elif isinstance(tokenizer, VocabCappedTokenizer):
        vocab_path = target / "vocab.json"
        vocab_payload = {
            "name": tokenizer.name,
            "base": tokenizer.base.__class__.__name__,
            "unk_token": tokenizer.unk_token,
            "vocab": sorted(tokenizer.vocab),
        }
        vocab_path.write_text(json.dumps(vocab_payload, indent=2), encoding="utf-8")
        metadata["type"] = "vocab_capped"
        metadata["vocab_json"] = vocab_path.name
        metadata["vocab_size"] = len(tokenizer.vocab)
    elif isinstance(tokenizer, HybridMorphUnigramTokenizer):
        vocab_path = target / "hybrid_vocab.json"
        vocab_payload = {
            "name": tokenizer.name,
            "whole_words": sorted(tokenizer.whole_words),
            "morpheme_vocab": sorted(tokenizer.morpheme_vocab),
            "char_prefix": tokenizer.char_prefix,
            "number_token": tokenizer.number_token,
        }
        vocab_path.write_text(json.dumps(vocab_payload, indent=2), encoding="utf-8")
        metadata["type"] = "hybrid_morph_unigram"
        metadata["vocab_json"] = vocab_path.name
        metadata["whole_words"] = len(tokenizer.whole_words)
        metadata["morpheme_vocab"] = len(tokenizer.morpheme_vocab)
    elif isinstance(tokenizer, TieredMorphUnigramTokenizer):
        tokenizer_path = target / "unigram_tokenizer.json"
        tokenizer.unigram.tokenizer.save(str(tokenizer_path))
        vocab_path = target / "tiered_vocab.json"
        vocab_payload = {
            "name": tokenizer.name,
            "whole_words": sorted(tokenizer.whole_words),
            "morpheme_vocab": sorted(tokenizer.morpheme_vocab),
            "morph_length_slack": tokenizer.morph_length_slack,
            "number_token": tokenizer.number_token,
        }
        vocab_path.write_text(json.dumps(vocab_payload, indent=2), encoding="utf-8")
        metadata["type"] = "tiered_morph_unigram"
        metadata["tokenizer_json"] = tokenizer_path.name
        metadata["vocab_json"] = vocab_path.name
        metadata["whole_words"] = len(tokenizer.whole_words)
        metadata["morpheme_vocab"] = len(tokenizer.morpheme_vocab)
    else:
        metadata["type"] = "python_dataclass_or_heuristic"
        if hasattr(tokenizer, "__dataclass_fields__"):
            metadata["params"] = asdict(tokenizer)

    metadata_path = target / "tokenizer_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_tokenizer_artifact(metadata_path: str | Path):
    """Load a tokenizer previously saved with `save_tokenizer_artifact`."""

    metadata_target = Path(metadata_path)
    metadata = json.loads(metadata_target.read_text(encoding="utf-8"))
    artifact_dir = metadata_target.parent
    artifact_type = metadata["type"]

    if artifact_type == "hf_tokenizers":
        tokenizers = _import_tokenizers()
        tokenizer_path = artifact_dir / metadata["tokenizer_json"]
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        return HFTokenizerWrapper(tokenizer=tokenizer, name=metadata["name"])
    if artifact_type == "vocab_capped":
        vocab_path = artifact_dir / metadata["vocab_json"]
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        return VocabCappedTokenizer(
            base=_build_base_tokenizer(payload["base"]),
            vocab=frozenset(payload["vocab"]),
            name=payload["name"],
            unk_token=payload.get("unk_token", "<unk>"),
        )
    if artifact_type == "hybrid_morph_unigram":
        vocab_path = artifact_dir / metadata["vocab_json"]
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        return HybridMorphUnigramTokenizer(
            whole_words=frozenset(payload["whole_words"]),
            morpheme_vocab=frozenset(payload["morpheme_vocab"]),
            char_prefix=payload.get("char_prefix", "<char:"),
            number_token=payload.get("number_token", "<num>"),
            name=payload["name"],
        )
    if artifact_type == "tiered_morph_unigram":
        tokenizers = _import_tokenizers()
        tokenizer_path = artifact_dir / metadata["tokenizer_json"]
        vocab_path = artifact_dir / metadata["vocab_json"]
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        unigram = HFTokenizerWrapper(
            tokenizer=tokenizers.Tokenizer.from_file(str(tokenizer_path)),
            name="hf_unigram",
        )
        return TieredMorphUnigramTokenizer(
            unigram=unigram,
            whole_words=frozenset(payload["whole_words"]),
            morpheme_vocab=frozenset(payload["morpheme_vocab"]),
            morph_length_slack=int(payload.get("morph_length_slack", 1)),
            number_token=payload.get("number_token", "<num>"),
            name=payload["name"],
        )
    raise ValueError(f"Unsupported tokenizer artifact type: {artifact_type}")


def _build_base_tokenizer(class_name: str):
    mapping = {
        "BytePatchTokenizer": BytePatchTokenizer,
        "HeuristicMorphemeTokenizer": HeuristicMorphemeTokenizer,
        "UnigramLikeTokenizer": UnigramLikeTokenizer,
        "WhitespaceTokenizer": WhitespaceTokenizer,
    }
    try:
        return mapping[class_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported capped-tokenizer base: {class_name}") from exc


def _import_tokenizers():
    try:
        import tokenizers
    except ImportError as exc:
        raise RuntimeError(
            "Loading HF tokenizer artifacts requires the optional `tokenizers` package."
        ) from exc
    return tokenizers
