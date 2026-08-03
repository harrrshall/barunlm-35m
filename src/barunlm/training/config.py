from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "barun-sft-config-v1"
_RUN_ID = re.compile(r"^\d{8}-\d{4}-[a-z0-9][a-z0-9-]*-s\d+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_CHECKPOINT_FILES = {"model.safetensors", "barun_config.json", "tokenizer.json"}


class TrainingConfigError(ValueError):
    """Raised when a typed training configuration is incomplete or ambiguous."""


def _check_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - optional)
    if missing:
        raise TrainingConfigError(f"{context}: missing required fields {missing}")
    if unknown:
        raise TrainingConfigError(f"{context}: unknown fields {unknown}")


def _as_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise TrainingConfigError(f"{name} must be at least {minimum}")
    return value


def _as_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrainingConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise TrainingConfigError(f"{name} must be at least {minimum}")
    return result


def _reject_json_constant(value: str) -> None:
    raise TrainingConfigError(f"non-finite JSON constant {value!r} is not allowed")


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.lower()) is None:
        raise TrainingConfigError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _resolve_path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True, slots=True)
class BaseCheckpointConfig:
    source: str
    local_dir: Path | None
    repo_id: str | None
    revision: str | None
    expected_sha256: Mapping[str, str]

    @classmethod
    def from_dict(cls, payload: object, *, base_dir: Path) -> BaseCheckpointConfig:
        if not isinstance(payload, dict):
            raise TrainingConfigError("base_checkpoint must be an object")
        _check_keys(
            payload,
            required={"source", "expected_sha256"},
            optional={"local_dir", "repo_id", "revision"},
            context="base_checkpoint",
        )
        source = payload["source"]
        if source not in {"local", "huggingface"}:
            raise TrainingConfigError("base_checkpoint.source must be 'local' or 'huggingface'")
        hashes = payload["expected_sha256"]
        if not isinstance(hashes, dict) or set(hashes) != _CHECKPOINT_FILES:
            raise TrainingConfigError(
                f"base_checkpoint.expected_sha256 must contain exactly {sorted(_CHECKPOINT_FILES)}"
            )
        expected = {
            name: _sha256(value, name=f"expected_sha256.{name}") for name, value in hashes.items()
        }

        if source == "local":
            if (
                "local_dir" not in payload
                or payload.get("repo_id") is not None
                or payload.get("revision") is not None
            ):
                raise TrainingConfigError(
                    "a local checkpoint requires local_dir and forbids repo_id/revision"
                )
            return cls(
                source=source,
                local_dir=_resolve_path(
                    payload["local_dir"], base_dir=base_dir, name="base_checkpoint.local_dir"
                ),
                repo_id=None,
                revision=None,
                expected_sha256=expected,
            )

        repo_id = payload.get("repo_id")
        revision = payload.get("revision")
        if payload.get("local_dir") is not None:
            raise TrainingConfigError("a Hugging Face checkpoint forbids local_dir")
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise TrainingConfigError("a Hugging Face checkpoint requires repo_id")
        if not isinstance(revision, str) or _GIT_COMMIT.fullmatch(revision) is None:
            raise TrainingConfigError(
                "a Hugging Face revision must be a full 40-character commit hash, not a branch"
            )
        return cls(
            source=source,
            local_dir=None,
            repo_id=repo_id,
            revision=revision.lower(),
            expected_sha256=expected,
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    train_manifest: Path
    train_sha256: str
    dev_manifest: Path
    dev_sha256: str
    max_seq_len: int
    eos_token: str
    pad_token: str
    overlength_policy: str

    @classmethod
    def from_dict(cls, payload: object, *, base_dir: Path) -> DataConfig:
        if not isinstance(payload, dict):
            raise TrainingConfigError("data must be an object")
        _check_keys(
            payload,
            required={
                "train_manifest",
                "train_sha256",
                "dev_manifest",
                "dev_sha256",
                "max_seq_len",
                "eos_token",
                "pad_token",
                "overlength_policy",
            },
            optional=set(),
            context="data",
        )
        eos_token = payload["eos_token"]
        pad_token = payload["pad_token"]
        if not isinstance(eos_token, str) or not eos_token:
            raise TrainingConfigError("data.eos_token must be a non-empty string")
        if not isinstance(pad_token, str) or not pad_token:
            raise TrainingConfigError("data.pad_token must be a non-empty string")
        policy = payload["overlength_policy"]
        if policy not in {"error", "record_and_exclude"}:
            raise TrainingConfigError(
                "data.overlength_policy must be 'error' or 'record_and_exclude'"
            )
        return cls(
            train_manifest=_resolve_path(
                payload["train_manifest"], base_dir=base_dir, name="data.train_manifest"
            ),
            train_sha256=_sha256(payload["train_sha256"], name="data.train_sha256"),
            dev_manifest=_resolve_path(
                payload["dev_manifest"], base_dir=base_dir, name="data.dev_manifest"
            ),
            dev_sha256=_sha256(payload["dev_sha256"], name="data.dev_sha256"),
            max_seq_len=_as_int(payload["max_seq_len"], name="data.max_seq_len", minimum=2),
            eos_token=eos_token,
            pad_token=pad_token,
            overlength_policy=policy,
        )


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    seed: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_learning_rate_ratio: float
    weight_decay: float
    beta1: float
    beta2: float
    adam_epsilon: float
    warmup_steps: int
    max_steps: int | None
    gradient_clip_norm: float
    eval_every_steps: int
    save_every_steps: int
    early_stopping_patience: int
    early_stopping_min_delta: float

    @classmethod
    def from_dict(cls, payload: object) -> OptimizationConfig:
        if not isinstance(payload, dict):
            raise TrainingConfigError("optimization must be an object")
        required = {
            "seed",
            "epochs",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "min_learning_rate_ratio",
            "weight_decay",
            "beta1",
            "beta2",
            "adam_epsilon",
            "warmup_steps",
            "max_steps",
            "gradient_clip_norm",
            "eval_every_steps",
            "save_every_steps",
            "early_stopping_patience",
            "early_stopping_min_delta",
        }
        _check_keys(payload, required=required, optional=set(), context="optimization")
        max_steps_raw = payload["max_steps"]
        max_steps = (
            None
            if max_steps_raw is None
            else _as_int(max_steps_raw, name="optimization.max_steps", minimum=1)
        )
        result = cls(
            seed=_as_int(payload["seed"], name="optimization.seed", minimum=0),
            epochs=_as_int(payload["epochs"], name="optimization.epochs", minimum=1),
            batch_size=_as_int(payload["batch_size"], name="optimization.batch_size", minimum=1),
            eval_batch_size=_as_int(
                payload["eval_batch_size"], name="optimization.eval_batch_size", minimum=1
            ),
            gradient_accumulation_steps=_as_int(
                payload["gradient_accumulation_steps"],
                name="optimization.gradient_accumulation_steps",
                minimum=1,
            ),
            learning_rate=_as_float(
                payload["learning_rate"], name="optimization.learning_rate", minimum=0.0
            ),
            min_learning_rate_ratio=_as_float(
                payload["min_learning_rate_ratio"],
                name="optimization.min_learning_rate_ratio",
                minimum=0.0,
            ),
            weight_decay=_as_float(
                payload["weight_decay"], name="optimization.weight_decay", minimum=0.0
            ),
            beta1=_as_float(payload["beta1"], name="optimization.beta1", minimum=0.0),
            beta2=_as_float(payload["beta2"], name="optimization.beta2", minimum=0.0),
            adam_epsilon=_as_float(
                payload["adam_epsilon"], name="optimization.adam_epsilon", minimum=0.0
            ),
            warmup_steps=_as_int(
                payload["warmup_steps"], name="optimization.warmup_steps", minimum=0
            ),
            max_steps=max_steps,
            gradient_clip_norm=_as_float(
                payload["gradient_clip_norm"],
                name="optimization.gradient_clip_norm",
                minimum=0.0,
            ),
            eval_every_steps=_as_int(
                payload["eval_every_steps"], name="optimization.eval_every_steps", minimum=1
            ),
            save_every_steps=_as_int(
                payload["save_every_steps"], name="optimization.save_every_steps", minimum=1
            ),
            early_stopping_patience=_as_int(
                payload["early_stopping_patience"],
                name="optimization.early_stopping_patience",
                minimum=1,
            ),
            early_stopping_min_delta=_as_float(
                payload["early_stopping_min_delta"],
                name="optimization.early_stopping_min_delta",
                minimum=0.0,
            ),
        )
        if result.learning_rate <= 0:
            raise TrainingConfigError("optimization.learning_rate must be positive")
        if not 0 <= result.min_learning_rate_ratio <= 1:
            raise TrainingConfigError("optimization.min_learning_rate_ratio must be in [0, 1]")
        if not 0 <= result.beta1 < 1 or not 0 <= result.beta2 < 1:
            raise TrainingConfigError("optimization beta values must be in [0, 1)")
        if result.adam_epsilon <= 0:
            raise TrainingConfigError("optimization.adam_epsilon must be positive")
        if result.gradient_clip_norm <= 0:
            raise TrainingConfigError("optimization.gradient_clip_norm must be positive")
        return result


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    device: str
    precision: str
    deterministic: bool
    jarvis_resource_id: str | None
    estimated_hourly_cost: float | None

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionConfig:
        if not isinstance(payload, dict):
            raise TrainingConfigError("execution must be an object")
        _check_keys(
            payload,
            required={
                "device",
                "precision",
                "deterministic",
                "jarvis_resource_id",
                "estimated_hourly_cost",
            },
            optional=set(),
            context="execution",
        )
        device = payload["device"]
        precision = payload["precision"]
        deterministic = payload["deterministic"]
        resource_id = payload["jarvis_resource_id"]
        hourly_cost_raw = payload["estimated_hourly_cost"]
        if device not in {"cpu", "cuda"}:
            raise TrainingConfigError("execution.device must be explicitly 'cpu' or 'cuda'")
        required_precision = "fp32" if device == "cpu" else "bf16"
        if precision != required_precision:
            raise TrainingConfigError(
                f"execution.precision must be {required_precision!r} for device {device!r}"
            )
        if not isinstance(deterministic, bool):
            raise TrainingConfigError("execution.deterministic must be a boolean")
        if resource_id is not None and (
            not isinstance(resource_id, str) or not resource_id.strip()
        ):
            raise TrainingConfigError(
                "execution.jarvis_resource_id must be null or a non-empty string"
            )
        hourly_cost = (
            None
            if hourly_cost_raw is None
            else _as_float(hourly_cost_raw, name="execution.estimated_hourly_cost", minimum=0.0)
        )
        return cls(
            device=device,
            precision=precision,
            deterministic=deterministic,
            jarvis_resource_id=resource_id,
            estimated_hourly_cost=hourly_cost,
        )


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    schema_version: str
    run_id: str
    output_root: Path
    hypothesis: str
    decision: str
    base_checkpoint: BaseCheckpointConfig
    data: DataConfig
    optimization: OptimizationConfig
    execution: ExecutionConfig
    resume_from: Path | None
    source_path: Path
    source_sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingRunConfig:
        source_path = Path(path).expanduser().resolve()
        raw = source_path.read_bytes()
        try:
            payload = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TrainingConfigError) as error:
            raise TrainingConfigError(f"invalid training config JSON: {error}") from error
        if not isinstance(payload, dict):
            raise TrainingConfigError("training config must be a JSON object")
        _check_keys(
            payload,
            required={
                "schema_version",
                "run_id",
                "output_root",
                "hypothesis",
                "decision",
                "base_checkpoint",
                "data",
                "optimization",
                "execution",
            },
            optional={"resume_from"},
            context="training config",
        )
        if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise TrainingConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
        run_id = payload["run_id"]
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise TrainingConfigError(
                "run_id must match YYYYMMDD-HHMM-short-name-sN using lowercase letters/digits"
            )
        hypothesis = payload["hypothesis"]
        decision = payload["decision"]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise TrainingConfigError("hypothesis must be a non-empty string")
        if not isinstance(decision, str) or not decision.strip():
            raise TrainingConfigError("decision must be a non-empty string")
        base_dir = source_path.parent
        resume_raw = payload.get("resume_from")
        return cls(
            schema_version=CONFIG_SCHEMA_VERSION,
            run_id=run_id,
            output_root=_resolve_path(
                payload["output_root"], base_dir=base_dir, name="output_root"
            ),
            hypothesis=hypothesis,
            decision=decision,
            base_checkpoint=BaseCheckpointConfig.from_dict(
                payload["base_checkpoint"], base_dir=base_dir
            ),
            data=DataConfig.from_dict(payload["data"], base_dir=base_dir),
            optimization=OptimizationConfig.from_dict(payload["optimization"]),
            execution=ExecutionConfig.from_dict(payload["execution"]),
            resume_from=(
                None
                if resume_raw is None
                else _resolve_path(resume_raw, base_dir=base_dir, name="resume_from")
            ),
            source_path=source_path,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    def compatibility_payload(self) -> dict[str, Any]:
        """Return the effective config used to verify an exact resume."""

        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in sorted(value.items())}
            if isinstance(value, tuple | list):
                return [encode(item) for item in value]
            return value

        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "output_root": self.output_root,
            "hypothesis": self.hypothesis,
            "decision": self.decision,
            "base_checkpoint": asdict(self.base_checkpoint),
            "data": asdict(self.data),
            "optimization": asdict(self.optimization),
            "execution": asdict(self.execution),
        }
        return encode(payload)

    @property
    def compatibility_sha256(self) -> str:
        encoded = json.dumps(
            self.compatibility_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
