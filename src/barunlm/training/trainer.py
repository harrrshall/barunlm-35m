from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model, save_model
from tokenizers import Tokenizer
from torch import Tensor

from barunlm.model import BarunLM

from .checkpoint import LoadedCheckpoint, load_pinned_checkpoint
from .config import TrainingRunConfig
from .data import (
    RejectedExample,
    TokenizedExample,
    collate_sft,
    deterministic_batches,
    load_manifest,
    sha256_file,
    tokenize_examples,
)

RUN_MANIFEST_VERSION = "barun-sft-run-v1"
TRAINER_STATE_VERSION = "barun-sft-state-v1"


class TrainingError(RuntimeError):
    """Raised when training cannot continue without changing the registered run."""


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    run_id: str
    run_dir: str
    status: str
    global_steps: int
    initial_dev_loss: float | None
    best_dev_loss: float | None
    final_dev_loss: float | None
    best_checkpoint: str | None
    final_checkpoint: str
    duration_seconds: float
    estimated_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "global_steps": self.global_steps,
            "initial_dev_loss": self.initial_dev_loss,
            "best_dev_loss": self.best_dev_loss,
            "final_dev_loss": self.final_dev_loss,
            "best_checkpoint": self.best_checkpoint,
            "final_checkpoint": self.final_checkpoint,
            "duration_seconds": self.duration_seconds,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(slots=True)
class TrainerState:
    epoch: int
    next_batch_index: int
    global_step: int
    initial_dev_loss: float | None
    best_dev_loss: float | None
    final_dev_loss: float | None
    bad_evaluations: int
    last_eval_step: int
    best_checkpoint: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_output(arguments: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path.cwd(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _git_provenance() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    root = _git_output(["rev-parse", "--show-toplevel"])
    status = _git_output(["status", "--porcelain=v1", "-z"])
    diff = _git_output(["diff", "--binary", "HEAD"])
    if commit is None or root is None or status is None or diff is None:
        return {"available": False}

    digest = hashlib.sha256()
    digest.update(status)
    digest.update(diff)
    repo_root = Path(root.decode("utf-8").strip())
    untracked = _git_output(["ls-files", "--others", "--exclude-standard", "-z"])
    untracked_hashes: dict[str, str] = {}
    if untracked is not None:
        for raw_name in filter(None, untracked.split(b"\0")):
            name = raw_name.decode("utf-8", errors="surrogateescape")
            candidate = repo_root / name
            if candidate.is_file():
                file_hash = sha256_file(candidate)
                untracked_hashes[name] = file_hash
                digest.update(raw_name)
                digest.update(file_hash.encode("ascii"))
    return {
        "available": True,
        "commit": commit.decode("ascii").strip(),
        "dirty": bool(status),
        "dirty_patch_sha256": digest.hexdigest(),
        "untracked_file_sha256": untracked_hashes,
    }


def _environment_provenance(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        payload["device_name"] = torch.cuda.get_device_name(device)
        payload["device_capability"] = list(torch.cuda.get_device_capability(device))
        payload["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        payload["processor"] = platform.processor()
    return payload


def _select_device(config: TrainingRunConfig) -> torch.device:
    requested = config.execution.device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("CUDA was explicitly requested but is unavailable")
        if not torch.cuda.is_bf16_supported():
            raise TrainingError(
                "bf16 CUDA training was requested but the GPU does not support bf16"
            )
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int, *, deterministic: bool, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_optimizer(model: BarunLM, config: TrainingRunConfig) -> torch.optim.AdamW:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.optimization.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.optimization.learning_rate,
        betas=(config.optimization.beta1, config.optimization.beta2),
        eps=config.optimization.adam_epsilon,
        fused=False,
    )


def _learning_rate(config: TrainingRunConfig, *, step: int, total_steps: int) -> float:
    warmup = config.optimization.warmup_steps
    base = config.optimization.learning_rate
    if warmup and step <= warmup:
        return base * step / warmup
    decay_steps = total_steps - warmup
    if decay_steps <= 0:
        raise TrainingError("warmup_steps must be smaller than the total optimizer steps")
    progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = config.optimization.min_learning_rate_ratio
    return base * (floor + (1.0 - floor) * cosine)


def _content_identity(prompt: str, target: str) -> str:
    encoded = json.dumps(
        {"prompt": prompt, "target": target},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_summary(
    accepted: Sequence[TokenizedExample], rejected: Sequence[RejectedExample]
) -> dict[str, Any]:
    lengths = [len(example.input_ids) for example in accepted]
    targets = [example.target_tokens for example in accepted]
    return {
        "accepted_examples": len(accepted),
        "rejected_examples": len(rejected),
        "encoded_tokens": sum(lengths),
        "target_tokens": sum(targets),
        "min_tokens": min(lengths) if lengths else None,
        "max_tokens": max(lengths) if lengths else None,
    }


def _write_rejections(
    path: Path,
    train_rejected: Sequence[RejectedExample],
    dev_rejected: Sequence[RejectedExample],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for split, rows in (("train", train_rejected), ("dev", dev_rejected)):
            for rejection in rows:
                payload = {"split": split, **rejection.to_dict()}
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.no_grad()
def evaluate_response_loss(
    model: BarunLM,
    examples: Sequence[TokenizedExample],
    *,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Measure token-weighted response NLL; prompt and padding labels remain ignored."""

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    target_tokens = 0
    for rows in deterministic_batches(
        examples, batch_size=batch_size, seed=0, epoch=0, shuffle=False
    ):
        batch = collate_sft(rows, pad_token_id=pad_token_id).to(device)
        with _autocast(device):
            output = model(
                batch.input_ids,
                labels=batch.labels,
                attention_mask=batch.attention_mask,
            )
        if output.loss is None or not torch.isfinite(output.loss).item():
            raise TrainingError("non-finite or missing development loss")
        loss_sum += float(output.loss.detach().float().item()) * batch.target_tokens
        target_tokens += batch.target_tokens
    if was_training:
        model.train()
    if target_tokens == 0:  # pragma: no cover - protected by tokenization invariants
        raise TrainingError("development set has no supervised response tokens")
    mean_loss = loss_sum / target_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "target_tokens": target_tokens,
        "examples": len(examples),
    }


def _random_state_payload(*, include_cuda: bool) -> dict[str, Any]:
    version, internal, gaussian = random.getstate()
    payload: dict[str, Any] = {
        "python": {"version": version, "internal": list(internal), "gaussian": gaussian},
        "torch_cpu": base64.b64encode(bytes(torch.get_rng_state().tolist())).decode("ascii"),
        "torch_cuda": None,
    }
    if include_cuda:
        payload["torch_cuda"] = base64.b64encode(
            bytes(torch.cuda.get_rng_state().cpu().tolist())
        ).decode("ascii")
    return payload


def _restore_random_state(payload: Mapping[str, Any]) -> None:
    python_state = payload["python"]
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["internal"]),
            python_state["gaussian"],
        )
    )
    cpu_bytes = base64.b64decode(payload["torch_cpu"])
    torch.set_rng_state(torch.tensor(list(cpu_bytes), dtype=torch.uint8))
    cuda_payload = payload.get("torch_cuda")
    if cuda_payload:
        if not torch.cuda.is_available():
            raise TrainingError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        cuda_state = torch.tensor(list(base64.b64decode(cuda_payload)), dtype=torch.uint8)
        torch.cuda.set_rng_state(cuda_state)


def _state_payload(state: TrainerState, config: TrainingRunConfig) -> dict[str, Any]:
    return {
        "schema_version": TRAINER_STATE_VERSION,
        "config_compatibility_sha256": config.compatibility_sha256,
        "train_manifest_sha256": config.data.train_sha256,
        "dev_manifest_sha256": config.data.dev_sha256,
        "epoch": state.epoch,
        "next_batch_index": state.next_batch_index,
        "global_step": state.global_step,
        "initial_dev_loss": state.initial_dev_loss,
        "best_dev_loss": state.best_dev_loss,
        "final_dev_loss": state.final_dev_loss,
        "bad_evaluations": state.bad_evaluations,
        "last_eval_step": state.last_eval_step,
        "best_checkpoint": state.best_checkpoint,
        "rng": _random_state_payload(include_cuda=config.execution.device == "cuda"),
    }


def _save_checkpoint(
    run_dir: Path,
    model: BarunLM,
    tokenizer: Tokenizer,
    optimizer: torch.optim.AdamW,
    state: TrainerState,
    config: TrainingRunConfig,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"step-{state.global_step:08d}"
    if checkpoint_dir.exists():
        _verify_checkpoint(checkpoint_dir)
        return checkpoint_dir
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoint_root))
    try:
        save_model(model, temporary / "model.safetensors")
        model.config.save_json(temporary / "barun_config.json")
        tokenizer.save(str(temporary / "tokenizer.json"))
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        _atomic_json(temporary / "trainer_state.json", _state_payload(state, config))
        hashes = {
            path.name: sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        _atomic_json(
            temporary / "checkpoint_manifest.json",
            {
                "schema_version": "barun-sft-checkpoint-v1",
                "created_at": _utc_now(),
                "run_id": config.run_id,
                "global_step": state.global_step,
                "file_sha256": hashes,
            },
        )
        temporary.replace(checkpoint_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint_dir


def _verify_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"resume checkpoint lacks {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "barun-sft-checkpoint-v1":
        raise TrainingError("unsupported resume checkpoint manifest version")
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        raise TrainingError("resume checkpoint has no file hash map")
    for name, expected in hashes.items():
        path = checkpoint_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise TrainingError(f"resume checkpoint file failed SHA-256 verification: {name}")
    return manifest


def _load_resume(
    config: TrainingRunConfig,
    model: BarunLM,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> TrainerState:
    checkpoint_dir = config.resume_from
    if checkpoint_dir is None:  # pragma: no cover - caller invariant
        raise TrainingError("resume path is missing")
    expected_root = (config.run_dir / "checkpoints").resolve()
    if checkpoint_dir.parent.resolve() != expected_root:
        raise TrainingError("resume_from must name a checkpoint inside this exact immutable run")
    _verify_checkpoint(checkpoint_dir)
    state_payload = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    if state_payload.get("schema_version") != TRAINER_STATE_VERSION:
        raise TrainingError("unsupported trainer state version")
    if state_payload.get("config_compatibility_sha256") != config.compatibility_sha256:
        raise TrainingError("resume config differs from the checkpoint's registered config")
    if state_payload.get("train_manifest_sha256") != config.data.train_sha256:
        raise TrainingError("resume training manifest digest changed")
    if state_payload.get("dev_manifest_sha256") != config.data.dev_sha256:
        raise TrainingError("resume development manifest digest changed")

    missing, unexpected = load_model(model, checkpoint_dir / "model.safetensors", strict=False)
    if missing or unexpected:
        raise TrainingError(f"resume model mismatch: missing={missing}, unexpected={unexpected}")
    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt", map_location=device, weights_only=True
    )
    optimizer.load_state_dict(optimizer_state)
    _restore_random_state(state_payload["rng"])
    return TrainerState(
        epoch=int(state_payload["epoch"]),
        next_batch_index=int(state_payload["next_batch_index"]),
        global_step=int(state_payload["global_step"]),
        initial_dev_loss=float(state_payload["initial_dev_loss"]),
        best_dev_loss=float(state_payload["best_dev_loss"]),
        final_dev_loss=float(state_payload["final_dev_loss"]),
        bad_evaluations=int(state_payload["bad_evaluations"]),
        last_eval_step=int(state_payload["last_eval_step"]),
        best_checkpoint=state_payload.get("best_checkpoint"),
    )


def _prepare_run_directory(config: TrainingRunConfig) -> None:
    if config.resume_from is None:
        config.run_dir.mkdir(parents=True, exist_ok=False)
        (config.run_dir / "checkpoints").mkdir()
    else:
        if not config.run_dir.is_dir():
            raise TrainingError("cannot resume because the immutable run directory is missing")
        if not (config.run_dir / "run_manifest.json").is_file():
            raise TrainingError("cannot resume a run without its run_manifest.json")


def _record_metric(metrics_path: Path, event: str, **values: Any) -> None:
    _append_jsonl(metrics_path, {"timestamp": _utc_now(), "event": event, **values})


def _evaluate_and_update(
    *,
    model: BarunLM,
    dev_examples: Sequence[TokenizedExample],
    pad_token_id: int,
    device: torch.device,
    config: TrainingRunConfig,
    state: TrainerState,
    metrics_path: Path,
    diagnostic_only: bool = False,
) -> bool:
    if state.best_dev_loss is None and not diagnostic_only:
        raise TrainingError("development evaluation is disabled for this completion-only run")
    metrics = evaluate_response_loss(
        model,
        dev_examples,
        batch_size=config.optimization.eval_batch_size,
        pad_token_id=pad_token_id,
        device=device,
    )
    dev_loss = float(metrics["loss"])
    state.final_dev_loss = dev_loss
    state.last_eval_step = state.global_step
    if diagnostic_only:
        improved = False
    else:
        assert state.best_dev_loss is not None
        improved = dev_loss < state.best_dev_loss - config.optimization.early_stopping_min_delta
        if improved:
            state.best_dev_loss = dev_loss
            state.bad_evaluations = 0
        else:
            state.bad_evaluations += 1
    _record_metric(
        metrics_path,
        "dev",
        global_step=state.global_step,
        epoch=state.epoch,
        improved=improved,
        diagnostic_only=diagnostic_only,
        **metrics,
    )
    return improved


def _train_group(
    *,
    model: BarunLM,
    optimizer: torch.optim.AdamW,
    rows: Sequence[Sequence[TokenizedExample]],
    pad_token_id: int,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float, int]:
    total_target_tokens = sum(
        example.target_tokens for microbatch in rows for example in microbatch
    )
    if total_target_tokens < 1:  # pragma: no cover - tokenization invariant
        raise TrainingError("optimizer group has no response targets")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for microbatch in rows:
        batch = collate_sft(microbatch, pad_token_id=pad_token_id).to(device)
        with _autocast(device):
            output = model(
                batch.input_ids,
                labels=batch.labels,
                attention_mask=batch.attention_mask,
            )
            if output.loss is None:
                raise TrainingError("model did not return a training loss")
            scaled_loss = output.loss * (batch.target_tokens / total_target_tokens)
        if not torch.isfinite(output.loss).item():
            raise TrainingError("non-finite training loss")
        scaled_loss.backward()
        weighted_loss += float(output.loss.detach().float().item()) * batch.target_tokens
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not torch.isfinite(gradient_norm).item():
        raise TrainingError("non-finite gradient norm")
    optimizer.step()
    return weighted_loss / total_target_tokens, float(gradient_norm.item()), total_target_tokens


def _run_training_loop(
    *,
    model: BarunLM,
    optimizer: torch.optim.AdamW,
    tokenizer: Tokenizer,
    train_examples: Sequence[TokenizedExample],
    dev_examples: Sequence[TokenizedExample] | None,
    pad_token_id: int,
    device: torch.device,
    config: TrainingRunConfig,
    state: TrainerState,
    metrics_path: Path,
    total_steps: int,
    diagnostic_dev_only: bool = False,
) -> tuple[TrainerState, str, Path]:
    model.train()
    stop_reason = "max_epochs"
    final_checkpoint: Path | None = None
    for epoch in range(state.epoch, config.optimization.epochs):
        batches = list(
            deterministic_batches(
                train_examples,
                batch_size=config.optimization.batch_size,
                seed=config.optimization.seed,
                epoch=epoch,
                shuffle=True,
            )
        )
        cursor = state.next_batch_index if epoch == state.epoch else 0
        if cursor < 0 or cursor > len(batches):
            raise TrainingError("resume batch cursor is outside the deterministic epoch")
        while cursor < len(batches):
            end = min(cursor + config.optimization.gradient_accumulation_steps, len(batches))
            next_step = state.global_step + 1
            learning_rate = _learning_rate(config, step=next_step, total_steps=total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            train_loss, gradient_norm, target_tokens = _train_group(
                model=model,
                optimizer=optimizer,
                rows=batches[cursor:end],
                pad_token_id=pad_token_id,
                device=device,
                gradient_clip_norm=config.optimization.gradient_clip_norm,
            )
            state.global_step = next_step
            cursor = end
            state.epoch = epoch + 1 if cursor == len(batches) else epoch
            state.next_batch_index = 0 if cursor == len(batches) else cursor
            _record_metric(
                metrics_path,
                "train",
                global_step=state.global_step,
                epoch=epoch,
                next_batch_index=state.next_batch_index,
                loss=train_loss,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                target_tokens=target_tokens,
            )

            reached_limit = state.global_step >= total_steps
            end_of_epoch = cursor == len(batches)
            should_evaluate = dev_examples is not None and (
                state.global_step % config.optimization.eval_every_steps == 0
                or end_of_epoch
                or reached_limit
            )
            improved = False
            if should_evaluate and state.last_eval_step != state.global_step:
                improved = _evaluate_and_update(
                    model=model,
                    dev_examples=dev_examples,
                    pad_token_id=pad_token_id,
                    device=device,
                    config=config,
                    state=state,
                    metrics_path=metrics_path,
                    diagnostic_only=diagnostic_dev_only,
                )
            if improved:
                state.best_checkpoint = str(
                    config.run_dir / "checkpoints" / f"step-{state.global_step:08d}"
                )
            should_save = (
                improved
                or state.global_step % config.optimization.save_every_steps == 0
                or reached_limit
                or end_of_epoch
            )
            if should_save:
                final_checkpoint = _save_checkpoint(
                    config.run_dir, model, tokenizer, optimizer, state, config
                )
                if improved:
                    _atomic_json(
                        config.run_dir / "best_checkpoint.json",
                        {
                            "checkpoint": str(final_checkpoint),
                            "global_step": state.global_step,
                            "dev_loss": state.best_dev_loss,
                        },
                    )
            if (
                dev_examples is not None
                and not diagnostic_dev_only
                and state.bad_evaluations >= config.optimization.early_stopping_patience
            ):
                stop_reason = "early_stopping"
                reached_limit = True
            if reached_limit:
                if state.global_step >= total_steps and stop_reason != "early_stopping":
                    stop_reason = (
                        "max_steps" if config.optimization.max_steps is not None else "max_epochs"
                    )
                break
        if stop_reason in {"early_stopping", "max_steps"}:
            break
        state.next_batch_index = 0

    if final_checkpoint is None:
        final_checkpoint = _save_checkpoint(
            config.run_dir, model, tokenizer, optimizer, state, config
        )
    return state, stop_reason, final_checkpoint


def train_sft(
    config: TrainingRunConfig,
    *,
    completion_only: bool = False,
    diagnostic_dev_only: bool = False,
) -> TrainingSummary:
    """Run deterministic full-parameter response-only SFT for one immutable run ID.

    ``completion_only=True`` is an explicit no-development mode for a final refit whose
    training population exhausts the frozen source data.  It never opens or tokenizes the
    configured development path, never evaluates development loss, forbids resume, and
    saves only checkpoints selected by the unconditional training schedule.  The default
    path retains the established development diagnostics and resume behavior.

    ``diagnostic_dev_only=True`` logs initial and final development loss but cannot
    create a best checkpoint, early-stop, or alter the unconditional training schedule.
    It is mutually exclusive with completion-only mode and resume.
    """

    started = time.monotonic()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    run_prepared = False
    try:
        if completion_only and diagnostic_dev_only:
            raise TrainingError("completion-only and diagnostic-dev-only modes are exclusive")
        if (completion_only or diagnostic_dev_only) and config.resume_from is not None:
            raise TrainingError("completion-only/diagnostic-dev-only training forbids resume")
        device = _select_device(config)
        _set_seed(
            config.optimization.seed,
            deterministic=config.execution.deterministic,
            device=device,
        )
        loaded: LoadedCheckpoint = load_pinned_checkpoint(config.base_checkpoint)
        model = loaded.model
        tokenizer = loaded.tokenizer
        if config.data.max_seq_len > model.config.max_seq_len:
            raise TrainingError(
                f"data max_seq_len={config.data.max_seq_len} exceeds model limit "
                f"{model.config.max_seq_len}"
            )
        if model.config.mtp_loss_weight != 0:
            raise TrainingError(
                "response-only SFT requires mtp_loss_weight=0; ablate MTP in a separate trainer"
            )
        eos_token_id = tokenizer.token_to_id(config.data.eos_token)
        pad_token_id = tokenizer.token_to_id(config.data.pad_token)
        if eos_token_id is None:
            raise TrainingError(f"tokenizer has no configured EOS token {config.data.eos_token!r}")
        if pad_token_id is None:
            raise TrainingError(f"tokenizer has no configured pad token {config.data.pad_token!r}")
        if eos_token_id == pad_token_id:
            raise TrainingError("EOS and padding token IDs must differ")

        train_raw = load_manifest(
            config.data.train_manifest,
            expected_sha256=config.data.train_sha256,
            expected_derived_split="train",
        )
        if completion_only:
            dev_raw = ()
        else:
            dev_raw = load_manifest(
                config.data.dev_manifest,
                expected_sha256=config.data.dev_sha256,
                expected_derived_split="dev",
            )
            shared_ids = {row.example_id for row in train_raw} & {row.example_id for row in dev_raw}
            if shared_ids:
                raise TrainingError(
                    f"train/dev ID overlap is forbidden; first IDs: {sorted(shared_ids)[:5]}"
                )
            train_content = {_content_identity(row.prompt, row.target) for row in train_raw}
            dev_content = {_content_identity(row.prompt, row.target) for row in dev_raw}
            if train_content & dev_content:
                raise TrainingError("train/dev contain exact prompt-target duplicates")
        train_examples, train_rejected = tokenize_examples(
            train_raw,
            tokenizer,
            eos_token_id=eos_token_id,
            max_seq_len=config.data.max_seq_len,
        )
        if completion_only:
            dev_examples = None
            dev_rejected: list[RejectedExample] = []
        else:
            dev_examples, dev_rejected = tokenize_examples(
                dev_raw,
                tokenizer,
                eos_token_id=eos_token_id,
                max_seq_len=config.data.max_seq_len,
            )

        _prepare_run_directory(config)
        run_prepared = True
        rejections_path = config.run_dir / "data_rejections.jsonl"
        if config.resume_from is None:
            _write_rejections(rejections_path, train_rejected, dev_rejected)
        if (train_rejected or dev_rejected) and config.data.overlength_policy == "error":
            raise TrainingError(
                "overlength examples were recorded and overlength_policy='error'; "
                f"see {rejections_path}"
            )
        if not train_examples or (not completion_only and not dev_examples):
            raise TrainingError("required training/development examples were not retained")

        batches_per_epoch = math.ceil(len(train_examples) / config.optimization.batch_size)
        steps_per_epoch = math.ceil(
            batches_per_epoch / config.optimization.gradient_accumulation_steps
        )
        planned_steps = steps_per_epoch * config.optimization.epochs
        total_steps = (
            planned_steps
            if config.optimization.max_steps is None
            else min(planned_steps, config.optimization.max_steps)
        )
        if config.optimization.warmup_steps >= total_steps:
            raise TrainingError("warmup_steps must be smaller than the planned optimizer steps")

        model.to(device=device, dtype=torch.float32)
        optimizer = _build_optimizer(model, config)
        metrics_path = config.run_dir / "metrics.jsonl"
        if config.resume_from is None:
            if completion_only:
                initial_metrics = None
                initial_loss = None
            else:
                if dev_examples is None:  # pragma: no cover - guarded above
                    raise TrainingError("development examples are unexpectedly unavailable")
                if diagnostic_dev_only:
                    _atomic_json(
                        config.run_dir / "heldout_access_started.json",
                        {
                            "schema_version": "barun-heldout-access-marker-v1",
                            "population": "development",
                            "reason": "diagnostic_dev_loss_read_imminent",
                            "retry_lock": "forbidden",
                        },
                    )
                initial_metrics = evaluate_response_loss(
                    model,
                    dev_examples,
                    batch_size=config.optimization.eval_batch_size,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                initial_loss = float(initial_metrics["loss"])
            state = TrainerState(
                epoch=0,
                next_batch_index=0,
                global_step=0,
                initial_dev_loss=initial_loss,
                best_dev_loss=(None if diagnostic_dev_only else initial_loss),
                final_dev_loss=initial_loss,
                bad_evaluations=0,
                last_eval_step=0,
                best_checkpoint=None,
            )
            _atomic_json(config.run_dir / "resolved_config.json", config.compatibility_payload())
            if completion_only:
                data_manifest = {
                    "train_manifest": str(config.data.train_manifest),
                    "train_manifest_sha256": config.data.train_sha256,
                    "dev_manifest": None,
                    "dev_manifest_sha256": None,
                    "dev_config_path_ignored": True,
                    "dev_labels_read": False,
                    "max_seq_len": config.data.max_seq_len,
                    "overlength_policy": config.data.overlength_policy,
                    "train": _dataset_summary(train_examples, train_rejected),
                    "dev": None,
                    "rejections_file": str(rejections_path),
                    "rejections_sha256": sha256_file(rejections_path),
                }
            else:
                if dev_examples is None:  # pragma: no cover - guarded above
                    raise TrainingError("development examples are unexpectedly unavailable")
                data_manifest = {
                    "train_manifest": str(config.data.train_manifest),
                    "train_manifest_sha256": config.data.train_sha256,
                    "dev_manifest": str(config.data.dev_manifest),
                    "dev_manifest_sha256": config.data.dev_sha256,
                    "max_seq_len": config.data.max_seq_len,
                    "overlength_policy": config.data.overlength_policy,
                    "train": _dataset_summary(train_examples, train_rejected),
                    "dev": _dataset_summary(dev_examples, dev_rejected),
                    "rejections_file": str(rejections_path),
                    "rejections_sha256": sha256_file(rejections_path),
                }
            run_manifest = {
                "schema_version": RUN_MANIFEST_VERSION,
                "created_at": _utc_now(),
                "run_id": config.run_id,
                "hypothesis": config.hypothesis,
                "decision": config.decision,
                "config_source": str(config.source_path),
                "config_source_sha256": config.source_sha256,
                "config_compatibility_sha256": config.compatibility_sha256,
                "command": sys.argv,
                "git": _git_provenance(),
                "environment": _environment_provenance(device),
                "execution": {
                    "precision": config.execution.precision,
                    "deterministic": config.execution.deterministic,
                    "jarvis_resource_id": config.execution.jarvis_resource_id,
                    "estimated_hourly_cost": config.execution.estimated_hourly_cost,
                },
                "base_checkpoint": loaded.provenance,
                "data": data_manifest,
                "tokens": {
                    "eos_token": config.data.eos_token,
                    "eos_token_id": eos_token_id,
                    "pad_token": config.data.pad_token,
                    "pad_token_id": pad_token_id,
                },
                "planned_optimizer_steps": total_steps,
            }
            if completion_only:
                run_manifest.update(
                    {
                        "checkpoint_policy": "unconditional_final_only",
                        "training_mode": "completion_only_no_dev",
                    }
                )
            elif diagnostic_dev_only:
                data_manifest.update(
                    {
                        "dev_labels_read": True,
                        "dev_loss_role": "diagnostic_only",
                    }
                )
                run_manifest.update(
                    {
                        "checkpoint_policy": "unconditional_final_only",
                        "training_mode": "diagnostic_dev_loss_unconditional_final",
                    }
                )
            _atomic_json(
                config.run_dir / "run_manifest.json",
                run_manifest,
            )
            if completion_only:
                _record_metric(
                    metrics_path,
                    "completion_only_contract",
                    checkpoint_policy="unconditional_final_only",
                    dev_labels_read=False,
                )
            else:
                if initial_metrics is None:  # pragma: no cover - guarded above
                    raise TrainingError("initial development metrics are unavailable")
                _record_metric(
                    metrics_path,
                    "dev",
                    global_step=0,
                    epoch=0,
                    improved=False,
                    baseline=True,
                    diagnostic_only=diagnostic_dev_only,
                    **initial_metrics,
                )
        else:
            state = _load_resume(config, model, optimizer, device)
            _record_metric(
                metrics_path,
                "resume",
                global_step=state.global_step,
                epoch=state.epoch,
                next_batch_index=state.next_batch_index,
                checkpoint=str(config.resume_from),
            )
            if state.global_step >= total_steps:
                raise TrainingError(
                    "resume checkpoint has already reached the configured step limit"
                )

        state, stop_reason, final_checkpoint = _run_training_loop(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            train_examples=train_examples,
            dev_examples=dev_examples,
            pad_token_id=pad_token_id,
            device=device,
            config=config,
            state=state,
            metrics_path=metrics_path,
            total_steps=total_steps,
            diagnostic_dev_only=diagnostic_dev_only,
        )
        duration = time.monotonic() - started
        cost = (
            None
            if config.execution.estimated_hourly_cost is None
            else duration / 3600 * config.execution.estimated_hourly_cost
        )
        summary = TrainingSummary(
            run_id=config.run_id,
            run_dir=str(config.run_dir),
            status=stop_reason,
            global_steps=state.global_step,
            initial_dev_loss=state.initial_dev_loss,
            best_dev_loss=state.best_dev_loss,
            final_dev_loss=state.final_dev_loss,
            best_checkpoint=state.best_checkpoint,
            final_checkpoint=str(final_checkpoint),
            duration_seconds=duration,
            estimated_cost=cost,
        )
        _atomic_json(config.run_dir / "summary.json", summary.to_dict())
        _record_metric(metrics_path, "complete", **summary.to_dict())
        return summary
    except Exception as error:
        if run_prepared:
            _atomic_json(
                config.run_dir / "failure.json",
                {
                    "timestamp": _utc_now(),
                    "run_id": config.run_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        raise
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
