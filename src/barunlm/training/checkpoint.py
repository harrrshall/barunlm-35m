from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from barunlm.config import BarunConfig
from barunlm.model import BarunLM

from .config import BaseCheckpointConfig
from .data import sha256_file


class CheckpointError(RuntimeError):
    """Raised when pinned checkpoint provenance or contents do not match."""


@dataclass(slots=True)
class LoadedCheckpoint:
    model: BarunLM
    tokenizer: Tokenizer
    provenance: dict[str, Any]


def _resolve_directory(spec: BaseCheckpointConfig) -> Path:
    if spec.source == "local":
        if spec.local_dir is None:  # pragma: no cover - protected by config validation
            raise CheckpointError("local checkpoint has no directory")
        return spec.local_dir
    if spec.repo_id is None or spec.revision is None:  # pragma: no cover
        raise CheckpointError("Hugging Face checkpoint is missing its pinned revision")
    return Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=sorted(spec.expected_sha256),
        )
    ).resolve()


def load_pinned_checkpoint(spec: BaseCheckpointConfig) -> LoadedCheckpoint:
    """Load BarunLM only after every required file matches its configured digest."""

    directory = _resolve_directory(spec)
    if not directory.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {directory}")
    actual_hashes: dict[str, str] = {}
    for name, expected in sorted(spec.expected_sha256.items()):
        path = directory / name
        if not path.is_file():
            raise CheckpointError(f"checkpoint is missing required file {name!r}")
        actual = sha256_file(path)
        actual_hashes[name] = actual
        if actual != expected:
            raise CheckpointError(
                f"checkpoint SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )

    config = BarunConfig.from_json(directory / "barun_config.json")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != config.vocab_size:
        raise CheckpointError(
            f"tokenizer vocabulary {tokenizer_vocab} != model vocabulary {config.vocab_size}"
        )
    model = BarunLM(config)
    missing, unexpected = load_model(model, directory / "model.safetensors", strict=False)
    if missing or unexpected:
        raise CheckpointError(
            f"checkpoint tensors do not match BarunLM: missing={missing}, unexpected={unexpected}"
        )

    source: dict[str, str] = {"kind": spec.source}
    if spec.source == "local":
        source["local_dir"] = str(directory)
    else:
        source["repo_id"] = spec.repo_id or ""
        source["revision"] = spec.revision or ""
        source["snapshot_dir"] = str(directory)
    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        provenance={
            "source": source,
            "file_sha256": actual_hashes,
            "parameter_counts": model.parameter_counts(),
        },
    )
