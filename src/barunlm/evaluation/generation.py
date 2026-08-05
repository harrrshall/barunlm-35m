"""Reproducible greedy generation for BarunAction JSONL manifests."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer

from barunlm.config import BarunConfig
from barunlm.model import BarunLM
from barunlm.training.data import SFTExample, load_manifest, sha256_file

GENERATION_VERSION = "barun-greedy-generation-v1"
INT8_GENERATION_VERSION = "barun-greedy-generation-int8-v1"
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "barun_config.json", "tokenizer.json")


class GenerationError(RuntimeError):
    """The requested generation run is unsafe, ambiguous, or unreproducible."""


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    schema_version: str
    checkpoint_dir: str
    checkpoint_sha256: Mapping[str, str]
    manifest: str
    manifest_sha256: str
    predictions: str
    predictions_sha256: str
    device: str
    dtype: str
    batch_size: int
    max_new_tokens: int
    examples: int
    generated: int
    truncated: int
    failed: int
    elapsed_seconds: float
    checkpoint_format: str = "float"
    quantization_manifest_sha256: str | None = None
    source_checkpoint_sha256: Mapping[str, str] | None = None
    qengine: str | None = None
    runtime: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep the established v1 float receipt byte-compatible.  Representation-specific
        # provenance is emitted only for explicitly selected int8 checkpoints.
        if self.checkpoint_format == "float":
            for name in (
                "checkpoint_format",
                "quantization_manifest_sha256",
                "source_checkpoint_sha256",
                "qengine",
                "runtime",
            ):
                payload.pop(name)
        return payload


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"JSON artifact must be an object: {path}")
    return value


def verify_checkpoint(
    directory: str | Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify a trainer checkpoint manifest or an explicit immutable hash map."""

    checkpoint_dir = Path(directory).resolve()
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if manifest_path.is_file():
        manifest = _strict_json(manifest_path)
        if manifest.get("schema_version") not in {
            "barun-sft-checkpoint-v1",
            "barun-release-checkpoint-v1",
        }:
            raise GenerationError("unsupported checkpoint manifest version")
        recorded = manifest.get("file_sha256")
        if not isinstance(recorded, dict):
            raise GenerationError("checkpoint manifest lacks file_sha256")
        hashes = {str(name): str(digest) for name, digest in recorded.items()}
        if expected_sha256 is not None and dict(expected_sha256) != hashes:
            raise GenerationError("explicit checkpoint hashes differ from checkpoint manifest")
    elif expected_sha256 is not None:
        hashes = dict(expected_sha256)
    else:
        raise GenerationError(
            "checkpoint has no signed manifest; explicit expected_sha256 is required"
        )

    for required in REQUIRED_CHECKPOINT_FILES:
        if required not in hashes:
            raise GenerationError(f"checkpoint hash map omits {required!r}")
    for name, expected in sorted(hashes.items()):
        if not isinstance(expected, str) or not _is_sha256(expected):
            raise GenerationError(f"invalid SHA-256 for checkpoint file {name!r}")
        candidate = checkpoint_dir / name
        if not candidate.is_file():
            raise GenerationError(f"checkpoint is missing hashed file {name!r}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise GenerationError(
                f"checkpoint SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )
    return hashes


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_verified_model(
    directory: str | Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[BarunLM, Tokenizer, dict[str, str]]:
    checkpoint_dir = Path(directory).resolve()
    hashes = verify_checkpoint(checkpoint_dir, expected_sha256=expected_sha256)
    config = BarunConfig.from_json(checkpoint_dir / "barun_config.json")
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise GenerationError("tokenizer and model vocabulary sizes differ")
    model = BarunLM(config)
    missing, unexpected = load_model(model, checkpoint_dir / "model.safetensors", strict=False)
    if missing or unexpected:
        raise GenerationError(
            f"checkpoint tensors differ from BarunLM: missing={missing}, unexpected={unexpected}"
        )
    return model, tokenizer, hashes


def _device(name: str) -> torch.device:
    if name not in {"cpu", "cuda"}:
        raise GenerationError("device must be explicitly 'cpu' or 'cuda'")
    if name == "cuda" and not torch.cuda.is_available():
        raise GenerationError("CUDA was requested but is unavailable")
    if name == "cuda" and not torch.cuda.is_bf16_supported():
        raise GenerationError("the requested CUDA device does not support bfloat16")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_prompts(
    rows: Sequence[SFTExample], tokenizer: Tokenizer
) -> list[tuple[int, SFTExample, tuple[int, ...]]]:
    encoded: list[tuple[int, SFTExample, tuple[int, ...]]] = []
    for index, row in enumerate(rows):
        ids = tuple(tokenizer.encode(row.prompt, add_special_tokens=False).ids)
        if not ids:
            raise GenerationError(f"prompt {row.example_id!r} encodes to zero tokens")
        encoded.append((index, row, ids))
    return encoded


def _left_padded_batch(
    batch: Sequence[tuple[int, SFTExample, tuple[int, ...]]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(item[2]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
    for row_index, (_, _, ids) in enumerate(batch):
        length = len(ids)
        input_ids[row_index, -length:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_index, -length:] = True
    return input_ids, attention_mask


def _prediction_record(
    *,
    example_id: str,
    prediction_raw: str | None,
    truncated: bool,
    generation_failure: str | None,
    prompt_tokens: int,
    generated_tokens: int,
) -> dict[str, Any]:
    return {
        "generated_tokens": generated_tokens,
        "generation_failure": generation_failure,
        "id": example_id,
        "prediction_raw": prediction_raw,
        "prompt_tokens": prompt_tokens,
        "truncated": truncated,
    }


def generate_manifest(
    *,
    checkpoint_dir: str | Path,
    manifest_path: str | Path,
    manifest_sha256: str,
    predictions_path: str | Path,
    device_name: str,
    batch_size: int,
    max_new_tokens: int,
    expected_checkpoint_sha256: Mapping[str, str] | None = None,
    checkpoint_format: str = "float",
    expected_int8_manifest_sha256: str | None = None,
) -> GenerationSummary:
    """Run deterministic, unconstrained generation and persist every raw output.

    Apart from stopping at the checkpoint's EOS token, generated special tokens are decoded
    literally.  The function never trims whitespace, extracts JSON, or repairs malformed text.
    """

    if checkpoint_format not in {"float", "int8"}:
        raise GenerationError("checkpoint_format must be explicitly 'float' or 'int8'")
    if checkpoint_format == "float" and expected_int8_manifest_sha256 is not None:
        raise GenerationError(
            "expected_int8_manifest_sha256 is valid only with checkpoint_format='int8'"
        )
    if checkpoint_format == "int8":
        if device_name != "cpu":
            raise GenerationError("int8 generation requires device_name='cpu'")
        if expected_checkpoint_sha256 is not None:
            raise GenerationError(
                "expected_checkpoint_sha256 is valid only with checkpoint_format='float'"
            )
        if not isinstance(expected_int8_manifest_sha256, str) or not _is_sha256(
            expected_int8_manifest_sha256
        ):
            raise GenerationError(
                "int8 generation requires an expected_int8_manifest_sha256 lowercase SHA-256"
            )
    if batch_size < 1:
        raise GenerationError("batch_size must be positive")
    if max_new_tokens < 1:
        raise GenerationError("max_new_tokens must be positive")
    if not _is_sha256(manifest_sha256):
        raise GenerationError("manifest_sha256 must be a lowercase SHA-256")
    output_path = Path(predictions_path).resolve()
    if output_path.exists():
        raise GenerationError(f"refusing to overwrite predictions: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantization_manifest_sha256: str | None = None
    source_checkpoint_hashes: Mapping[str, str] | None = None
    qengine: str | None = None
    runtime: Mapping[str, str] | None = None
    if checkpoint_format == "int8":
        # Lazy import avoids a module cycle: quantization reuses the float verifier above.
        from barunlm.quantization import QuantizationError, load_verified_int8_model

        try:
            model, tokenizer, int8_info = load_verified_int8_model(
                checkpoint_dir,
                expected_manifest_sha256=expected_int8_manifest_sha256,
            )
        except QuantizationError as error:
            raise GenerationError(f"int8 checkpoint verification/load failed: {error}") from error
        checkpoint_hashes = dict(int8_info.artifact_sha256)
        quantization_manifest_sha256 = int8_info.manifest_sha256
        source_checkpoint_hashes = dict(int8_info.source_checkpoint_sha256)
        qengine = int8_info.qengine
        runtime = {
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "torch_version": str(torch.__version__),
        }
    else:
        model, tokenizer, checkpoint_hashes = load_verified_model(
            checkpoint_dir, expected_sha256=expected_checkpoint_sha256
        )
    rows = load_manifest(manifest_path, expected_sha256=manifest_sha256)
    encoded = _encode_prompts(rows, tokenizer)
    device = _device(device_name)
    if checkpoint_format == "int8":
        dtype_label = "torch.qint8_dynamic_with_float_remainder"
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model.to(device=device, dtype=dtype).eval()
        dtype_label = str(dtype)
    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    if eos_token_id is None or pad_token_id is None or eos_token_id == pad_token_id:
        raise GenerationError("tokenizer must define distinct <eos> and <pad> tokens")

    records: list[dict[str, Any] | None] = [None] * len(rows)
    eligible: list[tuple[int, SFTExample, tuple[int, ...]]] = []
    for item in encoded:
        index, row, ids = item
        if len(ids) + max_new_tokens > model.config.max_seq_len:
            records[index] = _prediction_record(
                example_id=row.example_id,
                prediction_raw=None,
                truncated=False,
                generation_failure="context_overflow",
                prompt_tokens=len(ids),
                generated_tokens=0,
            )
        else:
            eligible.append(item)
    # Similar-length batching avoids spending most prefill compute on left padding while
    # retaining original manifest order in the immutable output artifact.
    eligible.sort(key=lambda item: (len(item[2]), item[1].example_id))

    started = time.monotonic()
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        input_ids, attention_mask = _left_padded_batch(
            batch, pad_token_id=pad_token_id, device=device
        )
        prompt_width = input_ids.shape[1]
        try:
            _sync(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    attention_mask=attention_mask,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            _sync(device)
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for index, row, ids in batch:
                records[index] = _prediction_record(
                    example_id=row.example_id,
                    prediction_raw=None,
                    truncated=False,
                    generation_failure="oom",
                    prompt_tokens=len(ids),
                    generated_tokens=0,
                )
            continue

        for batch_index, (index, row, ids) in enumerate(batch):
            continuation = output_ids[batch_index, prompt_width:].tolist()
            eos_position = next(
                (
                    position
                    for position, token_id in enumerate(continuation)
                    if token_id == eos_token_id
                ),
                None,
            )
            truncated = eos_position is None
            content_ids = continuation if eos_position is None else continuation[:eos_position]
            generated_tokens = len(continuation) if eos_position is None else eos_position + 1
            raw = tokenizer.decode(content_ids, skip_special_tokens=False)
            records[index] = _prediction_record(
                example_id=row.example_id,
                prediction_raw=raw,
                truncated=truncated,
                generation_failure=None,
                prompt_tokens=len(ids),
                generated_tokens=generated_tokens,
            )
    elapsed = time.monotonic() - started

    if any(record is None for record in records):  # pragma: no cover - internal invariant
        raise GenerationError("generation did not account for every manifest row")
    finalized = [record for record in records if record is not None]
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in finalized:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = GenerationSummary(
        schema_version=(
            INT8_GENERATION_VERSION if checkpoint_format == "int8" else GENERATION_VERSION
        ),
        checkpoint_dir=str(Path(checkpoint_dir).resolve()),
        checkpoint_sha256=checkpoint_hashes,
        manifest=str(Path(manifest_path).resolve()),
        manifest_sha256=manifest_sha256,
        predictions=str(output_path),
        predictions_sha256=sha256_file(output_path),
        device=str(device),
        dtype=dtype_label,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        examples=len(rows),
        generated=sum(record["generation_failure"] is None for record in finalized),
        truncated=sum(bool(record["truncated"]) for record in finalized),
        failed=sum(record["generation_failure"] is not None for record in finalized),
        elapsed_seconds=elapsed,
        checkpoint_format=checkpoint_format,
        quantization_manifest_sha256=quantization_manifest_sha256,
        source_checkpoint_sha256=source_checkpoint_hashes,
        qengine=qengine,
        runtime=runtime,
    )
    summary_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    summary_path.write_text(
        json.dumps(summary.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "GENERATION_VERSION",
    "INT8_GENERATION_VERSION",
    "GenerationError",
    "GenerationSummary",
    "generate_manifest",
    "load_verified_model",
    "verify_checkpoint",
]
