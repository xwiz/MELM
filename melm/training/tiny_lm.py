"""Tiny causal language-model trainer for pipeline validation.

This is not intended to compete with BabyLM baselines. It proves that a corpus
manifest, tokenizer arm, vocabulary, model config, and report can run end to
end before larger 125M-370M training jobs are scheduled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING

from melm.tokenization import Tokenizer

if TYPE_CHECKING:
    import torch


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass(frozen=True)
class TokenVocabulary:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


@dataclass(frozen=True)
class TinyLMConfig:
    tokenizer_name: str
    max_vocab_size: int = 2048
    pad_vocab_to_max_size: bool = False
    sequence_length: int = 64
    embedding_dim: int = 64
    layers: int = 1
    heads: int = 4
    dropout: float = 0.0
    batch_size: int = 8
    steps: int = 20
    learning_rate: float = 3e-4
    seed: int = 13
    device: str = "auto"


@dataclass(frozen=True)
class TinyLMTrainingReport:
    tokenizer: str
    train_documents: int
    validation_documents: int
    train_tokens: int
    validation_tokens: int
    validation_bytes: int
    train_sequences: int
    validation_sequences: int
    vocabulary_size: int
    parameters: int
    steps: int
    device: str
    final_train_loss: float
    validation_nll: float
    validation_bits_per_byte: float
    validation_perplexity: float


@dataclass(frozen=True)
class TinyLMCheckpointEvaluationReport:
    tokenizer: str
    checkpoint_dir: str
    validation_documents: int
    validation_tokens: int
    validation_bytes: int
    validation_sequences: int
    vocabulary_size: int
    parameters: int
    checkpoint_steps: int
    device: str
    validation_nll: float
    validation_bits_per_byte: float
    validation_perplexity: float


@dataclass(frozen=True)
class TinyLMTextScore:
    text: str
    tokens: int
    bytes: int
    sequences: int
    mean_nll_per_token: float
    total_nll: float
    bits_per_byte: float


@dataclass(frozen=True)
class TinyLMContinuationScore:
    prompt: str
    completion: str
    text: str
    tokens: int
    bytes: int
    mean_nll_per_token: float
    total_nll: float
    bits_per_byte: float


def build_token_vocabulary(
    tokenizer: Tokenizer,
    train_texts: list[str],
    *,
    max_vocab_size: int = 2048,
    pad_to_size: bool = False,
) -> TokenVocabulary:
    """Build a deterministic capped token vocabulary from training texts."""

    if max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("max_vocab_size must leave room for special tokens")

    counts: Counter[str] = Counter()
    for text in train_texts:
        counts.update(tokenizer.tokenize(text))

    remaining = max_vocab_size - len(SPECIAL_TOKENS)
    token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:remaining]:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    if pad_to_size:
        while len(token_to_id) < max_vocab_size:
            token_to_id[f"<extra:{len(token_to_id)}>"] = len(token_to_id)
    return TokenVocabulary(token_to_id)


def load_token_vocabulary(path: str | Path) -> TokenVocabulary:
    """Load a saved tiny-LM vocabulary."""

    token_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
    return TokenVocabulary({str(token): int(index) for token, index in token_to_id.items()})


def make_lm_sequences(
    texts: list[str],
    tokenizer: Tokenizer,
    vocabulary: TokenVocabulary,
    *,
    sequence_length: int,
) -> list[tuple[list[int], list[int]]]:
    """Create fixed-length causal LM input/target sequences."""

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    ids: list[int] = []
    for text in texts:
        ids.append(vocabulary.bos_id)
        ids.extend(vocabulary.encode(tokenizer.tokenize(text)))
        ids.append(vocabulary.eos_id)

    if len(ids) < 2:
        ids = [vocabulary.bos_id, vocabulary.eos_id]

    sequences: list[tuple[list[int], list[int]]] = []
    max_start = max(len(ids) - 1, 1)
    for start in range(0, max_start, sequence_length):
        x = ids[start : start + sequence_length]
        y = ids[start + 1 : start + sequence_length + 1]
        if not y:
            y = [vocabulary.eos_id]
        x = _pad(x, sequence_length, vocabulary.pad_id)
        y = _pad(y, sequence_length, vocabulary.pad_id)
        sequences.append((x, y))
    return sequences or [
        (
            _pad([vocabulary.bos_id], sequence_length, vocabulary.pad_id),
            _pad([vocabulary.eos_id], sequence_length, vocabulary.pad_id),
        )
    ]


def train_tiny_lm_baseline(
    train_texts: list[str],
    validation_texts: list[str],
    tokenizer: Tokenizer,
    config: TinyLMConfig,
    *,
    checkpoint_dir: str | Path | None = None,
) -> TinyLMTrainingReport:
    """Train and evaluate a tiny causal Transformer LM."""

    torch = _require_torch()
    _validate_config(config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    vocabulary = build_token_vocabulary(
        tokenizer,
        train_texts,
        max_vocab_size=config.max_vocab_size,
        pad_to_size=config.pad_vocab_to_max_size,
    )
    train_sequences = make_lm_sequences(
        train_texts,
        tokenizer,
        vocabulary,
        sequence_length=config.sequence_length,
    )
    validation_sequences = make_lm_sequences(
        validation_texts or train_texts,
        tokenizer,
        vocabulary,
        sequence_length=config.sequence_length,
    )

    device = _resolve_device(torch, config.device)
    model = _TinyDecoderLM(
        vocab_size=len(vocabulary),
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        pad_id=vocabulary.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_x, train_y = _tensorize(torch, train_sequences, device)
    final_train_loss = 0.0
    if config.steps > 0:
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)
        for _step in range(config.steps):
            indices = torch.randint(
                low=0,
                high=train_x.shape[0],
                size=(config.batch_size,),
                generator=generator,
                device=device,
            )
            batch_x = train_x.index_select(0, indices)
            batch_y = train_y.index_select(0, indices)
            logits = model(batch_x)
            loss = _cross_entropy(torch, logits, batch_y, vocabulary.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_train_loss = float(loss.detach().cpu())

    validation_nll = _evaluate_nll(
        torch,
        model,
        validation_sequences,
        config.batch_size,
        vocabulary.pad_id,
        device,
    )
    train_tokens = _target_token_count(train_sequences, vocabulary.pad_id)
    validation_tokens = _target_token_count(validation_sequences, vocabulary.pad_id)
    validation_bytes = sum(len(text.encode("utf-8")) for text in (validation_texts or train_texts))
    validation_bits_per_byte = (
        (validation_nll * validation_tokens) / (validation_bytes * math.log(2.0))
        if validation_bytes
        else 0.0
    )
    report = TinyLMTrainingReport(
        tokenizer=tokenizer.name,
        train_documents=len(train_texts),
        validation_documents=len(validation_texts),
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        validation_bytes=validation_bytes,
        train_sequences=len(train_sequences),
        validation_sequences=len(validation_sequences),
        vocabulary_size=len(vocabulary),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        steps=config.steps,
        device=str(device),
        final_train_loss=final_train_loss,
        validation_nll=validation_nll,
        validation_bits_per_byte=validation_bits_per_byte,
        validation_perplexity=math.exp(validation_nll) if validation_nll < 20 else float("inf"),
    )
    if checkpoint_dir is not None:
        _save_checkpoint(
            torch,
            model,
            vocabulary,
            config,
            report,
            checkpoint_dir,
        )
    return report


def evaluate_tiny_lm_checkpoint(
    validation_texts: list[str],
    tokenizer: Tokenizer,
    checkpoint_dir: str | Path,
    *,
    batch_size: int | None = None,
    device: str = "auto",
) -> TinyLMCheckpointEvaluationReport:
    """Reload a tiny LM checkpoint and evaluate it on validation texts."""

    torch = _require_torch()
    checkpoint = Path(checkpoint_dir)
    config, vocabulary, model, resolved_device = _load_checkpoint_model(
        torch,
        checkpoint,
        device,
    )
    sequences = make_lm_sequences(
        validation_texts,
        tokenizer,
        vocabulary,
        sequence_length=config.sequence_length,
    )

    evaluation_batch_size = batch_size or config.batch_size
    validation_nll = _evaluate_nll(
        torch,
        model,
        sequences,
        evaluation_batch_size,
        vocabulary.pad_id,
        resolved_device,
    )
    validation_tokens = _target_token_count(sequences, vocabulary.pad_id)
    validation_bytes = sum(len(text.encode("utf-8")) for text in validation_texts)
    validation_bits_per_byte = (
        (validation_nll * validation_tokens) / (validation_bytes * math.log(2.0))
        if validation_bytes
        else 0.0
    )
    return TinyLMCheckpointEvaluationReport(
        tokenizer=tokenizer.name,
        checkpoint_dir=str(checkpoint),
        validation_documents=len(validation_texts),
        validation_tokens=validation_tokens,
        validation_bytes=validation_bytes,
        validation_sequences=len(sequences),
        vocabulary_size=len(vocabulary),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_steps=config.steps,
        device=str(resolved_device),
        validation_nll=validation_nll,
        validation_bits_per_byte=validation_bits_per_byte,
        validation_perplexity=math.exp(validation_nll) if validation_nll < 20 else float("inf"),
    )


def score_tiny_lm_texts(
    texts: list[str],
    tokenizer: Tokenizer,
    checkpoint_dir: str | Path,
    *,
    batch_size: int | None = None,
    device: str = "auto",
) -> list[TinyLMTextScore]:
    """Score individual texts with a reloaded tiny LM checkpoint."""

    torch = _require_torch()
    checkpoint = Path(checkpoint_dir)
    config, vocabulary, model, resolved_device = _load_checkpoint_model(
        torch,
        checkpoint,
        device,
    )
    evaluation_batch_size = batch_size or config.batch_size
    all_sequences: list[tuple[list[int], list[int]]] = []
    group_ids: list[int] = []
    sequence_counts: list[int] = []
    token_counts: list[int] = []
    byte_counts: list[int] = []
    for text in texts:
        sequences = make_lm_sequences(
            [text],
            tokenizer,
            vocabulary,
            sequence_length=config.sequence_length,
        )
        group_id = len(sequence_counts)
        all_sequences.extend(sequences)
        group_ids.extend([group_id] * len(sequences))
        sequence_counts.append(len(sequences))
        token_counts.append(_target_token_count(sequences, vocabulary.pad_id))
        byte_counts.append(len(text.encode("utf-8")))

    loss_sums = _evaluate_grouped_nll_sums(
        torch,
        model,
        all_sequences,
        group_ids,
        group_count=len(texts),
        batch_size=evaluation_batch_size,
        pad_id=vocabulary.pad_id,
        device=resolved_device,
    )

    scores: list[TinyLMTextScore] = []
    for index, text in enumerate(texts):
        tokens = token_counts[index]
        byte_count = byte_counts[index]
        total_nll = loss_sums[index]
        mean_nll = total_nll / tokens if tokens else 0.0
        bits_per_byte = total_nll / (byte_count * math.log(2.0)) if byte_count else 0.0
        scores.append(
            TinyLMTextScore(
                text=text,
                tokens=tokens,
                bytes=byte_count,
                sequences=sequence_counts[index],
                mean_nll_per_token=mean_nll,
                total_nll=total_nll,
                bits_per_byte=bits_per_byte,
            )
        )
    return scores


def score_tiny_lm_continuations(
    prompts: list[str],
    completions: list[str],
    tokenizer: Tokenizer,
    checkpoint_dir: str | Path,
    *,
    batch_size: int | None = None,
    device: str = "auto",
) -> list[TinyLMContinuationScore]:
    """Score completions conditioned on prompts with a reloaded tiny LM."""

    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")

    torch = _require_torch()
    checkpoint = Path(checkpoint_dir)
    config, vocabulary, model, resolved_device = _load_checkpoint_model(
        torch,
        checkpoint,
        device,
    )
    evaluation_batch_size = batch_size or config.batch_size
    windows: list[list[int]] = []
    target_ids: list[int] = []
    active_indices: list[int] = []
    group_ids: list[int] = []
    token_counts: list[int] = []
    byte_counts: list[int] = []

    for group_id, (prompt, completion) in enumerate(zip(prompts, completions)):
        prompt_ids = vocabulary.encode(tokenizer.tokenize(prompt))
        completion_ids = vocabulary.encode(tokenizer.tokenize(completion)) + [vocabulary.eos_id]
        full_ids = [vocabulary.bos_id] + prompt_ids + completion_ids
        target_start = 1 + len(prompt_ids)
        token_counts.append(len(completion_ids))
        byte_counts.append(len(completion.encode("utf-8")))
        for target_pos in range(target_start, len(full_ids)):
            context = full_ids[max(0, target_pos - config.sequence_length) : target_pos]
            if not context:
                context = [vocabulary.bos_id]
            active_indices.append(len(context) - 1)
            windows.append(_pad(context, config.sequence_length, vocabulary.pad_id))
            target_ids.append(full_ids[target_pos])
            group_ids.append(group_id)

    loss_sums = _score_target_windows(
        torch,
        model,
        windows,
        target_ids,
        active_indices,
        group_ids,
        group_count=len(prompts),
        batch_size=evaluation_batch_size,
        device=resolved_device,
    )

    scores: list[TinyLMContinuationScore] = []
    for index, (prompt, completion) in enumerate(zip(prompts, completions)):
        tokens = token_counts[index]
        byte_count = byte_counts[index]
        total_nll = loss_sums[index]
        mean_nll = total_nll / tokens if tokens else 0.0
        bits_per_byte = total_nll / (byte_count * math.log(2.0)) if byte_count else 0.0
        scores.append(
            TinyLMContinuationScore(
                prompt=prompt,
                completion=completion,
                text=prompt + completion,
                tokens=tokens,
                bytes=byte_count,
                mean_nll_per_token=mean_nll,
                total_nll=total_nll,
                bits_per_byte=bits_per_byte,
            )
        )
    return scores


def _pad(values: list[int], length: int, pad_id: int) -> list[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_id] * (length - len(values))


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Tiny LM training requires PyTorch. Install torch or run tokenizer/memory probes only."
        ) from exc
    return torch


def _validate_config(config: TinyLMConfig) -> None:
    if config.embedding_dim % config.heads != 0:
        raise ValueError("embedding_dim must be divisible by heads")
    if config.layers < 1:
        raise ValueError("layers must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.steps < 0:
        raise ValueError("steps cannot be negative")


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _tensorize(torch, sequences: list[tuple[list[int], list[int]]], device):
    x = torch.tensor([item[0] for item in sequences], dtype=torch.long, device=device)
    y = torch.tensor([item[1] for item in sequences], dtype=torch.long, device=device)
    return x, y


def _cross_entropy(torch, logits, targets, pad_id: int):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def _evaluate_nll(torch, model, sequences, batch_size: int, pad_id: int, device) -> float:
    x, y = _tensorize(torch, sequences, device)
    weighted_loss = 0.0
    tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size]
            batch_y = y[start : start + batch_size]
            logits = model(batch_x)
            loss = _cross_entropy(torch, logits, batch_y, pad_id)
            token_count = int((batch_y != pad_id).sum().cpu())
            weighted_loss += float(loss.cpu()) * token_count
            tokens += token_count
    return weighted_loss / tokens if tokens else 0.0


def _evaluate_grouped_nll_sums(
    torch,
    model,
    sequences,
    group_ids: list[int],
    *,
    group_count: int,
    batch_size: int,
    pad_id: int,
    device,
) -> list[float]:
    x, y = _tensorize(torch, sequences, device)
    loss_sums = [0.0 for _ in range(group_count)]
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            batch_x = x[start:end]
            batch_y = y[start:end]
            logits = model(batch_x)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch_y.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            ).reshape(batch_y.shape)
            sequence_losses = losses.sum(dim=1).detach().cpu().tolist()
            for offset, sequence_loss in enumerate(sequence_losses):
                loss_sums[group_ids[start + offset]] += float(sequence_loss)
    return loss_sums


def _score_target_windows(
    torch,
    model,
    windows: list[list[int]],
    target_ids: list[int],
    active_indices: list[int],
    group_ids: list[int],
    *,
    group_count: int,
    batch_size: int,
    device,
) -> list[float]:
    x = torch.tensor(windows, dtype=torch.long, device=device)
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    active = torch.tensor(active_indices, dtype=torch.long, device=device)
    loss_sums = [0.0 for _ in range(group_count)]
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            batch_x = x[start:end]
            batch_targets = targets[start:end]
            batch_active = active[start:end]
            logits = model(batch_x)
            selected = logits[
                torch.arange(end - start, device=device),
                batch_active,
                :,
            ]
            losses = torch.nn.functional.cross_entropy(
                selected,
                batch_targets,
                reduction="none",
            ).detach().cpu().tolist()
            for offset, loss in enumerate(losses):
                loss_sums[group_ids[start + offset]] += float(loss)
    return loss_sums


def _target_token_count(sequences: list[tuple[list[int], list[int]]], pad_id: int) -> int:
    return sum(1 for _x, y in sequences for token_id in y if token_id != pad_id)


def _save_checkpoint(
    torch,
    model,
    vocabulary: TokenVocabulary,
    config: TinyLMConfig,
    report: TinyLMTrainingReport,
    checkpoint_dir: str | Path,
) -> None:
    target = Path(checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model_state.pt")
    (target / "vocab.json").write_text(
        json.dumps(vocabulary.token_to_id, indent=2),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (target / "training_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )


def _load_checkpoint_model(torch, checkpoint: Path, device: str):
    config = TinyLMConfig(
        **json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    vocabulary = load_token_vocabulary(checkpoint / "vocab.json")
    resolved_device = _resolve_device(torch, device)
    model = _TinyDecoderLM(
        vocab_size=len(vocabulary),
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        pad_id=vocabulary.pad_id,
    ).to(resolved_device)
    state_dict = torch.load(checkpoint / "model_state.pt", map_location=resolved_device)
    model.load_state_dict(state_dict)
    return config, vocabulary, model, resolved_device


class _TinyDecoderLM:
    def __new__(
        cls,
        *,
        vocab_size: int,
        sequence_length: int,
        embedding_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        pad_id: int,
    ):
        torch = _require_torch()

        class TinyDecoderLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token_embedding = torch.nn.Embedding(
                    vocab_size,
                    embedding_dim,
                    padding_idx=pad_id,
                )
                self.position_embedding = torch.nn.Embedding(
                    sequence_length,
                    embedding_dim,
                )
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.transformer = torch.nn.TransformerEncoder(layer, num_layers=layers)
                self.output = torch.nn.Linear(embedding_dim, vocab_size)

            def forward(self, input_ids):
                batch, length = input_ids.shape
                positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
                hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
                mask = torch.triu(
                    torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
                    diagonal=1,
                )
                hidden = self.transformer(hidden, mask=mask)
                return self.output(hidden)

        return TinyDecoderLM()
