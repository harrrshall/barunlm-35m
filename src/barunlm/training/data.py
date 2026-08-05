from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import Tensor

EXAMPLE_SCHEMA_VERSION = "barun-sft-example-v1"
IGNORE_INDEX = -100


class ManifestError(ValueError):
    """Raised when an SFT manifest violates the versioned input contract."""


@dataclass(frozen=True, slots=True)
class SFTExample:
    example_id: str
    prompt: str
    target: str
    metadata: Mapping[str, Any]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    example_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    target_tokens: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class RejectedExample:
    example_id: str
    reason: str
    encoded_tokens: int
    max_tokens: int
    prompt_tokens: int
    target_tokens: int
    content_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "id": self.example_id,
            "reason": self.reason,
            "encoded_tokens": self.encoded_tokens,
            "max_tokens": self.max_tokens,
            "prompt_tokens": self.prompt_tokens,
            "target_tokens": self.target_tokens,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class SFTBatch:
    example_ids: tuple[str, ...]
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    target_tokens: int

    def to(self, device: torch.device) -> SFTBatch:
        return SFTBatch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device=device, non_blocking=True),
            labels=self.labels.to(device=device, non_blocking=True),
            attention_mask=self.attention_mask.to(device=device, non_blocking=True),
            target_tokens=self.target_tokens,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"non-finite JSON constant {value!r} is not allowed")


def _canonical_record_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_record(payload: object, *, line_number: int) -> SFTExample:
    if not isinstance(payload, dict):
        raise ManifestError(f"line {line_number}: each JSONL row must be an object")
    allowed = {"schema_version", "id", "prompt", "target", "metadata"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManifestError(
            f"line {line_number}: unknown top-level fields {unknown}; put annotations in metadata"
        )
    missing = sorted({"schema_version", "id", "prompt", "target"} - set(payload))
    if missing:
        raise ManifestError(f"line {line_number}: missing required fields {missing}")
    if payload["schema_version"] != EXAMPLE_SCHEMA_VERSION:
        raise ManifestError(
            f"line {line_number}: schema_version must be {EXAMPLE_SCHEMA_VERSION!r}"
        )

    example_id = payload["id"]
    prompt = payload["prompt"]
    target = payload["target"]
    metadata = payload.get("metadata", {})
    if not isinstance(example_id, str) or not example_id.strip():
        raise ManifestError(f"line {line_number}: id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise ManifestError(f"line {line_number}: prompt must be a non-empty string")
    if not isinstance(target, str) or not target:
        raise ManifestError(f"line {line_number}: target must be a non-empty string")
    if not isinstance(metadata, dict):
        raise ManifestError(f"line {line_number}: metadata must be an object")
    try:
        content_sha256 = _canonical_record_hash(payload)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"line {line_number}: metadata is not strict JSON: {error}") from error
    return SFTExample(
        example_id=example_id,
        prompt=prompt,
        target=target,
        metadata=metadata,
        content_sha256=content_sha256,
    )


def _validate_training_split_metadata(
    example: SFTExample,
    *,
    expected_derived_split: str,
) -> None:
    """Reject known held-out rows before they can enter training or model selection.

    Generic manifests may omit split metadata. Dataset adapters that provide the
    versioned ``source_split``/``derived_split`` fields are held to them: an official
    evaluation row cannot be relabeled or passed as either the training or development
    manifest.
    """

    metadata = example.metadata
    source_split = metadata.get("source_split")
    derived_split = metadata.get("derived_split")
    for field, value in (("source_split", source_split), ("derived_split", derived_split)):
        if value is not None and not isinstance(value, str):
            raise ManifestError(
                f"example {example.example_id!r}: metadata.{field} must be a string"
            )
    held_out_names = {"eval", "evaluation", "final_eval", "test"}
    if source_split is not None and source_split.casefold() in held_out_names:
        raise ManifestError(
            f"example {example.example_id!r}: official held-out source_split "
            f"{source_split!r} is forbidden in SFT"
        )
    if derived_split is not None:
        if derived_split.casefold() in held_out_names:
            raise ManifestError(
                f"example {example.example_id!r}: held-out derived_split "
                f"{derived_split!r} is forbidden in SFT"
            )
        if derived_split != expected_derived_split:
            raise ManifestError(
                f"example {example.example_id!r}: expected metadata.derived_split "
                f"{expected_derived_split!r}, got {derived_split!r}"
            )


def load_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_derived_split: str | None = None,
) -> list[SFTExample]:
    """Load a strict, hash-pinned JSONL manifest without normalizing its text."""

    manifest_path = Path(path)
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256.lower():
        raise ManifestError(
            f"manifest SHA-256 mismatch for {manifest_path}: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    examples: list[SFTExample] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ManifestError(f"line {line_number}: blank JSONL rows are not allowed")
            try:
                payload = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ManifestError) as error:
                raise ManifestError(f"line {line_number}: invalid JSON: {error}") from error
            example = _validate_record(payload, line_number=line_number)
            if expected_derived_split is not None:
                _validate_training_split_metadata(
                    example,
                    expected_derived_split=expected_derived_split,
                )
            if example.example_id in seen_ids:
                raise ManifestError(f"line {line_number}: duplicate id {example.example_id!r}")
            seen_ids.add(example.example_id)
            examples.append(example)
    if not examples:
        raise ManifestError(f"manifest {manifest_path} contains no examples")
    return examples


def tokenize_examples(
    examples: Iterable[SFTExample],
    tokenizer: Tokenizer,
    *,
    eos_token_id: int,
    max_seq_len: int,
) -> tuple[list[TokenizedExample], list[RejectedExample]]:
    """Tokenize prompt and target separately and append EOS without truncation.

    Labels occupy the same positions as ``input_ids``. Prompt positions are set to
    ``IGNORE_INDEX``; the model's internal next-token shift therefore begins loss at
    the first target token. EOS is always a supervised target token.
    """

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if not 0 <= eos_token_id < tokenizer.get_vocab_size(with_added_tokens=True):
        raise ValueError("eos_token_id is outside the tokenizer vocabulary")

    accepted: list[TokenizedExample] = []
    rejected: list[RejectedExample] = []
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False).ids
        target_ids = tokenizer.encode(example.target, add_special_tokens=False).ids
        if not prompt_ids:
            raise ManifestError(f"example {example.example_id!r}: prompt encoded to zero tokens")
        if not target_ids:
            raise ManifestError(f"example {example.example_id!r}: target encoded to zero tokens")
        if eos_token_id in prompt_ids or eos_token_id in target_ids:
            raise ManifestError(
                f"example {example.example_id!r}: prompt/target already contains EOS; "
                "the trainer appends exactly one EOS target"
            )

        input_ids = tuple(prompt_ids + target_ids + [eos_token_id])
        labels = tuple([IGNORE_INDEX] * len(prompt_ids) + target_ids + [eos_token_id])
        target_tokens = len(target_ids) + 1
        if len(input_ids) > max_seq_len:
            rejected.append(
                RejectedExample(
                    example_id=example.example_id,
                    reason="overlength",
                    encoded_tokens=len(input_ids),
                    max_tokens=max_seq_len,
                    prompt_tokens=len(prompt_ids),
                    target_tokens=target_tokens,
                    content_sha256=example.content_sha256,
                )
            )
            continue
        accepted.append(
            TokenizedExample(
                example_id=example.example_id,
                input_ids=input_ids,
                labels=labels,
                prompt_tokens=len(prompt_ids),
                target_tokens=target_tokens,
                content_sha256=example.content_sha256,
            )
        )
    return accepted, rejected


def collate_sft(examples: Sequence[TokenizedExample], *, pad_token_id: int) -> SFTBatch:
    """Right-pad a response-only batch for causal training."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if pad_token_id < 0:
        raise ValueError("pad_token_id cannot be negative")
    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        if length != len(example.labels):
            raise ValueError(f"example {example.example_id!r} has misaligned inputs and labels")
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[row, :length] = True
    target_tokens = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
    if target_tokens != sum(example.target_tokens for example in examples):
        raise ValueError("response-only label alignment invariant failed")
    return SFTBatch(
        example_ids=tuple(example.example_id for example in examples),
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        target_tokens=target_tokens,
    )


def deterministic_batches(
    examples: Sequence[TokenizedExample],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[list[TokenizedExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(len(examples), generator=generator).tolist()
    else:
        indices = list(range(len(examples)))
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]
