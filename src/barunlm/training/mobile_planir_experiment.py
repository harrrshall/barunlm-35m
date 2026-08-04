"""Fixed backend for the construction-internal PlanIR mechanism screen.

The irreversible phase order is part of the scientific contract: all nine
completion-only fits finish and their final checkpoints are frozen before the
first screen manifest is decoded; all nine raw generations are then frozen
before the pinned scorer or gate is invoked.  This module has no API for old
selection/confirmation data, reused Mobile development data, the official
Mobile evaluation set, Qwen, refits, checkpoint selection, or model promotion.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch
from tokenizers import Tokenizer

from barunlm.datasets.mobile_planir_screen import (
    EXPECTED_OUTPUT_SHA256,
    OUTPUT_FILENAMES,
    SCREEN_ROWS,
    SPLIT_VERSION,
    TRAIN_ROWS,
)
from barunlm.evaluation.generation import GENERATION_VERSION, GenerationSummary, generate_manifest
from barunlm.evaluation.mobile_actions import read_jsonl
from barunlm.evaluation.mobile_planir_screen import (
    SCREEN_PREDICTION_ROW_VERSION,
    build_evidence_receipt,
    evaluate_screen_gate,
    score_screen,
    validate_evidence_receipt,
)
from barunlm.training.config import TrainingRunConfig
from barunlm.training.data import sha256_file
from barunlm.training.trainer import TrainingSummary, train_sft

BACKEND_VERSION = "barun-mobile-planir-construction-screen-execution-v1"
CONFIG_VERSION = "barun-mobile-planir-construction-screen-config-v1"
RUN_ID = "20260804-0545-mobile-planir-construction-screen-s17"
ARMS = ("A", "B", "C")
SEEDS = (17, 29, 43)
FIT_ORDER = tuple((arm, seed) for seed in SEEDS for arm in ARMS)
EXPECTED_STEPS = 73
GENERATION_BATCH_SIZE = 128
MAX_NEW_TOKENS = 256
PARAMETER_COUNT = 35_072_768
BASE_REPO = "harrrshall/BarunLM-35M"
BASE_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
BASE_HASHES = {
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "model.safetensors": "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
}
PROTECTED_RESOURCE_IDS = frozenset({463058, 463689, 463697, 463719, 463786, 463788, 463793, 463802})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MobilePlanIRExperimentError(RuntimeError):
    """The frozen mechanism-screen execution contract was violated."""


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Training-only capability passed into the pre-screen phase.

    Deliberately contains no screen path, screen hash, scorer handle, or label object.
    """

    arm: str
    seed: int
    run_id: str
    train_manifest: Path
    train_sha256: str


@dataclass(frozen=True, slots=True)
class FitResult:
    spec: FitSpec
    final_checkpoint: Path
    checkpoint_file_sha256: Mapping[str, str]
    model_sha256: str
    training_receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RawGenerationResult:
    fit: FitResult
    screen_manifest_sha256: str
    predictions_path: Path
    predictions_sha256: str
    summary_path: Path
    summary_sha256: str
    generation: Mapping[str, object]


class ScreenOperations(Protocol):
    """Injectable seam that makes the irreversible phase graph CPU-testable."""

    def train_fit(self, spec: FitSpec) -> FitResult: ...

    def freeze_checkpoints(self, fits: Sequence[FitResult]) -> None: ...

    def begin_screen_access(self) -> None: ...

    def generate_raw(self, fit: FitResult) -> RawGenerationResult: ...

    def freeze_raw_predictions(self, rows: Sequence[RawGenerationResult]) -> None: ...

    def enrich_and_score(self, rows: Sequence[RawGenerationResult]) -> Mapping[str, object]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MobilePlanIRExperimentError(f"refusing to overwrite immutable artifact {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MobilePlanIRExperimentError(f"refusing to overwrite immutable artifact {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MobilePlanIRExperimentError(f"invalid {label}: {path}") from error
    if type(value) is not dict:
        raise MobilePlanIRExperimentError(f"{label} must be one exact JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise MobilePlanIRExperimentError(
            f"{label} fields changed; missing={sorted(expected - set(value))!r}, "
            f"extra={sorted(set(value) - expected)!r}"
        )


def load_frozen_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and strictly validate the registered construction-screen config."""

    config_path = Path(path).resolve()
    payload = _load_json(config_path, label="experiment config")
    _exact_keys(
        payload,
        {
            "arms",
            "base_checkpoint",
            "claims",
            "data",
            "decision",
            "decoding",
            "execution",
            "gate",
            "hypothesis",
            "optimization",
            "resource_policy",
            "retry_policy",
            "run_id",
            "schema_version",
            "seeds",
            "token_compute_audit",
            "use",
        },
        label="experiment config",
    )
    if (
        payload["schema_version"] != CONFIG_VERSION
        or payload["run_id"] != RUN_ID
        or payload["use"] != "construction-internal-mechanism-screen-only"
    ):
        raise MobilePlanIRExperimentError("config identity or construction-only use changed")
    if payload["seeds"] != list(SEEDS):
        raise MobilePlanIRExperimentError("seed set/order changed")
    expected_arms = [
        {"arm_id": "A", "model_output": "ACTION_IR_V1", "prompt_reference_table": False},
        {"arm_id": "B", "model_output": "ACTION_IR_V1", "prompt_reference_table": True},
        {"arm_id": "C", "model_output": "PLAN_IR_V2", "prompt_reference_table": True},
    ]
    if payload["arms"] != expected_arms:
        raise MobilePlanIRExperimentError("A/B/C representation contract changed")
    if payload["base_checkpoint"] != {
        "repo_id": BASE_REPO,
        "revision": BASE_REVISION,
        "parameter_count": PARAMETER_COUNT,
        "file_sha256": BASE_HASHES,
    }:
        raise MobilePlanIRExperimentError("canonical BarunLM-35M base binding changed")

    optimization = payload["optimization"]
    if optimization != {
        "adam_epsilon": 1e-8,
        "batch_size": 63,
        "beta1": 0.9,
        "beta2": 0.95,
        "early_stopping": False,
        "effective_batch_size": 63,
        "epochs": 1,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "learning_rate": 1e-4,
        "loss": "response_only_cross_entropy",
        "max_seq_len": 2_048,
        "min_learning_rate_ratio": 0.1,
        "precision": "bf16",
        "warmup_steps": 12,
        "weight_decay": 0.1,
    }:
        raise MobilePlanIRExperimentError("optimization contract changed")
    if math.ceil(TRAIN_ROWS / int(optimization["effective_batch_size"])) != EXPECTED_STEPS:
        raise MobilePlanIRExperimentError("row/batch budget no longer yields exactly 73 steps")
    decoding = payload["decoding"]
    expected_decoding = {
        "algorithm": "unconstrained_greedy",
        "batch_size": GENERATION_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "output_repair": False,
        "runtime_gold_target_length_used": False,
        "stop_on_eos": True,
        "temperature": 0,
    }
    if decoding != expected_decoding:
        raise MobilePlanIRExperimentError("unconstrained deterministic decoding changed")

    data = payload["data"]
    if type(data) is not dict:
        raise MobilePlanIRExperimentError("config data contract must be an object")
    _exact_keys(
        data,
        {
            "artifact_sha256",
            "forbidden_populations",
            "materialization_relative_path",
            "screen_label_access",
            "screen_rows_per_arm",
            "source_provenance",
            "split_version",
            "train_rows_per_arm",
            "training_labels",
        },
        label="config data",
    )
    if (
        data.get("artifact_sha256") != dict(EXPECTED_OUTPUT_SHA256)
        or data.get("split_version") != SPLIT_VERSION
        or data.get("train_rows_per_arm") != TRAIN_ROWS
        or data.get("screen_rows_per_arm") != SCREEN_ROWS
        or data.get("materialization_relative_path")
        != "experiments/runs/20260804-0545-mobile-planir-construction-screen-s17/materialized-v1"
        or data.get("training_labels") != "arm_specific_construction_train_manifests_only"
        or data.get("screen_label_access") != "only_after_all_nine_final_checkpoints_are_frozen"
    ):
        raise MobilePlanIRExperimentError("materialized data or label-access contract changed")
    forbidden = data.get("forbidden_populations")
    if (
        type(forbidden) is not dict
        or set(forbidden)
        != {
            "human_confirmation_rows_read",
            "human_selection_rows_read",
            "official_mobile_961_rows_read",
            "old_confirmation_rows_read",
            "old_selection_rows_read",
            "reused_mobile_756_rows_read",
        }
        or any(value != 0 for value in forbidden.values())
    ):
        raise MobilePlanIRExperimentError("every nonconstruction population must remain at zero")
    if data.get("source_provenance") != {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "construction_manifest_membership_sha256": (
            "b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"
        ),
        "construction_manifest_rows": 5_745,
        "construction_manifest_sha256": (
            "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
        ),
        "dataset": "google/mobile-actions",
        "dataset_revision": "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
        "license": "CC BY 4.0",
        "original_train_membership_sha256": (
            "4cdfc3649c21a9f0d3dc4d8d9cffcae3cb5c6abc4414a9fb09636e9220afe96f"
        ),
        "original_train_rows": 7_937,
        "original_train_sha256": (
            "131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e"
        ),
        "role": "former_v3_construction_rows_for_internal_mechanism_evidence_only",
        "source_is_fresh_evaluation_evidence": False,
    }:
        raise MobilePlanIRExperimentError("construction source provenance changed")

    execution = payload["execution"]
    if execution != {
        "automatic_retry": False,
        "backend": BACKEND_VERSION,
        "base_reset_per_fit": True,
        "checkpoint_policy": "unconditional_final_only",
        "cuda_runtime": "13.0",
        "device": "cuda",
        "max_runtime_minutes": 45,
        "phase_order": [
            "complete_all_nine_completion_only_fits",
            "freeze_all_nine_final_step_73_checkpoints",
            "begin_construction_screen_access",
            "generate_all_nine_raw_prediction_populations",
            "freeze_all_nine_raw_prediction_populations",
            "score_once_and_evaluate_the_frozen_gate",
        ],
        "precision": "bf16",
        "provider_template": "axolotl",
        "python_implementation": "CPython",
        "python_version": "3.11.10",
        "region": "IN2",
        "required_gpu": "H200",
        "requirements_sha256": ("6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9"),
        "screen_scoring_after_all_raw_predictions_frozen": True,
        "spot": False,
        "storage_gb": 40,
        "torch_version": "2.13.0",
        "visible_gpus": 1,
        "wandb": "disabled",
    }:
        raise MobilePlanIRExperimentError("execution contract changed")
    claims = payload["claims"]
    if claims != {
        "fresh_evidence": False,
        "human_collection_unlocked": False,
        "larger_model_comparison": "forbidden",
        "maximum_positive_claim": "licenses_generalized_planir_development_only",
        "model_promotion": "forbidden",
        "release_result": False,
    }:
        raise MobilePlanIRExperimentError("internal-only claim boundary changed")
    if payload["resource_policy"] != {
        "fresh_instance_required": True,
        "protected_resource_ids": sorted(PROTECTED_RESOURCE_IDS),
        "provider": "JarvisLabs",
    }:
        raise MobilePlanIRExperimentError("fresh-resource/protected-ID contract changed")
    if payload["gate"] != {
        "argument_value_margin_points_vs_each_control": 5,
        "ast_margin_points_vs_each_control": 3,
        "candidate_compiler_success_minimum_per_thousand": 995,
        "candidate_conditional_schema_valid_per_thousand": 1000,
        "candidate_raw_parse_minimum_per_thousand": 995,
        "hard_conjunction": True,
        "minimum_seed_wins_vs_both_controls": 2,
        "subset_maximum_loss_points": 2,
        "zero_failures": [
            "truncation",
            "missing",
            "generation_failure",
            "catastrophic_unauthorized_action",
        ],
    }:
        raise MobilePlanIRExperimentError("scientific gate changed")
    retry = payload["retry_policy"]
    if retry != {
        "any_new_attempt": "forbidden",
        "rescue_or_threshold_change_after_result": "forbidden",
        "same_attempt_automatic_retry": "forbidden",
    }:
        raise MobilePlanIRExperimentError("no-rescue/no-retry contract changed")
    token_audit = payload["token_compute_audit"]
    train_audit = token_audit.get("train") if type(token_audit) is dict else None
    screen_audit = token_audit.get("screen") if type(token_audit) is dict else None
    proxy = token_audit.get("dense_training_flops_proxy") if type(token_audit) is dict else None
    if (
        type(token_audit) is not dict
        or _canonical_sha256(token_audit)
        != "17d5b514b1c31b2b23ca86789f5f8db9f832c1c8f6b25d4f8c8d6d714e2a4ae4"
        or token_audit.get("tokenizer_sha256") != BASE_HASHES["tokenizer.json"]
        or type(train_audit) is not dict
        or type(screen_audit) is not dict
        or type(proxy) is not dict
        or [(train_audit[arm]["rows"], train_audit[arm]["full_sequence_tokens"]) for arm in ARMS]
        != [(4596, 1_570_646), (4596, 1_660_067), (4596, 1_653_640)]
        or [(screen_audit[arm]["rows"], screen_audit[arm]["prompt_tokens"]) for arm in ARMS]
        != [(1148, 289_118), (1148, 312_161), (1148, 312_161)]
        or train_audit.get("rows_over_2048") != 0
        or train_audit.get("token_matched_sensitivity_triggered") is not False
        or screen_audit.get("rows_over_2048") != 0
        or proxy.get("nine_fit_total") != 3_083_540_032_783_872
        or proxy.get("formula") != "6 * parameter_count * unpadded_full_sequence_tokens"
    ):
        raise MobilePlanIRExperimentError("pinned token/compute audit changed")
    return payload, sha256_file(config_path)


def _binary_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise MobilePlanIRExperimentError(f"blank row in pinned manifest {path}")
            rows += 1
    return rows


def build_fit_specs(
    *, frozen_config: Mapping[str, Any], materialization_dir: str | Path
) -> tuple[FitSpec, ...]:
    """Hash materialized bytes without decoding any screen target."""

    materialized = Path(materialization_dir).resolve()
    hashes = frozen_config["data"]["artifact_sha256"]
    for name, filename in OUTPUT_FILENAMES.items():
        artifact = materialized / filename
        if not artifact.is_file() or sha256_file(artifact) != hashes[name]:
            raise MobilePlanIRExperimentError(f"materialized artifact changed: {name}")
        if name.startswith("train_") and _binary_jsonl_rows(artifact) != TRAIN_ROWS:
            raise MobilePlanIRExperimentError(f"{name} must contain exactly {TRAIN_ROWS} rows")
        if name.startswith("screen_") and _binary_jsonl_rows(artifact) != SCREEN_ROWS:
            raise MobilePlanIRExperimentError(f"{name} must contain exactly {SCREEN_ROWS} rows")

    prefix = RUN_ID.rsplit("-s", 1)[0]
    return tuple(
        FitSpec(
            arm=arm,
            seed=seed,
            run_id=f"{prefix}-{arm.lower()}-s{seed}",
            train_manifest=materialized / OUTPUT_FILENAMES[f"train_{arm.lower()}"],
            train_sha256=str(hashes[f"train_{arm.lower()}"]),
        )
        for arm, seed in FIT_ORDER
    )


def _resource_id(value: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise MobilePlanIRExperimentError("JarvisLabs resource ID must be explicit numeric text")
    parsed = int(value)
    if parsed < 1 or parsed in PROTECTED_RESOURCE_IDS:
        raise MobilePlanIRExperimentError("refusing a missing or durably protected resource ID")
    return value


def build_training_payload(
    *,
    frozen_config: Mapping[str, Any],
    spec: FitSpec,
    output_root: Path,
    ignored_dev_sentinel: Path,
    jarvis_resource_id: str,
) -> dict[str, object]:
    """Build one independent, base-reset, completion-only trainer payload."""

    if (spec.arm, spec.seed) not in FIT_ORDER:
        raise MobilePlanIRExperimentError("fit is outside the registered arm/seed set")
    if ignored_dev_sentinel.exists():
        raise MobilePlanIRExperimentError("ignored development sentinel must not exist")
    resource_id = _resource_id(jarvis_resource_id)
    optimization = frozen_config["optimization"]
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": spec.run_id,
        "output_root": str(output_root.resolve()),
        "hypothesis": frozen_config["hypothesis"],
        "decision": frozen_config["decision"],
        "base_checkpoint": {
            "source": "huggingface",
            "repo_id": BASE_REPO,
            "revision": BASE_REVISION,
            "expected_sha256": dict(BASE_HASHES),
        },
        "data": {
            "train_manifest": str(spec.train_manifest.resolve()),
            "train_sha256": spec.train_sha256,
            "dev_manifest": str(ignored_dev_sentinel.resolve()),
            "dev_sha256": "0" * 64,
            "max_seq_len": 2_048,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": {
            "seed": spec.seed,
            "epochs": 1,
            "batch_size": 63,
            "eval_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "min_learning_rate_ratio": optimization["min_learning_rate_ratio"],
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": optimization["adam_epsilon"],
            "warmup_steps": 12,
            "max_steps": None,
            "gradient_clip_norm": optimization["gradient_clip_norm"],
            "eval_every_steps": EXPECTED_STEPS,
            "save_every_steps": EXPECTED_STEPS,
            "early_stopping_patience": EXPECTED_STEPS + 1,
            "early_stopping_min_delta": 0.0,
        },
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": resource_id,
            "estimated_hourly_cost": None,
        },
    }


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    manifest = _load_json(checkpoint / "checkpoint_manifest.json", label="checkpoint manifest")
    if (
        manifest.get("schema_version") != "barun-sft-checkpoint-v1"
        or manifest.get("global_step") != EXPECTED_STEPS
        or type(manifest.get("file_sha256")) is not dict
    ):
        raise MobilePlanIRExperimentError("final checkpoint manifest changed")
    hashes = manifest["file_sha256"]
    for name, expected in hashes.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or not (checkpoint / name).is_file()
            or sha256_file(checkpoint / name) != expected
        ):
            raise MobilePlanIRExperimentError(f"final checkpoint file failed verification: {name}")
    if not {"model.safetensors", "barun_config.json", "tokenizer.json"}.issubset(hashes):
        raise MobilePlanIRExperimentError("checkpoint omits an inference-required file")
    return {str(name): str(value) for name, value in sorted(hashes.items())}


def _decoding_payload(tokenizer: Tokenizer) -> dict[str, object]:
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    if eos_id is None or pad_id is None or eos_id == pad_id:
        raise MobilePlanIRExperimentError("checkpoint tokenizer lacks distinct EOS/pad tokens")
    return {
        "schema_version": "barun-mobile-planir-decoding-v1",
        "algorithm": "unconstrained_greedy",
        "temperature": 0,
        "batch_size": GENERATION_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop_on_eos": True,
        "eos_token": "<eos>",
        "eos_token_id": eos_id,
        "pad_token": "<pad>",
        "pad_token_id": pad_id,
        "output_repair": False,
        "runtime_gold_target_length_used": False,
        "generation_implementation_version": GENERATION_VERSION,
        "tokenizer_sha256": BASE_HASHES["tokenizer.json"],
    }


def _validate_fit_result(result: FitResult, expected: FitSpec) -> None:
    if not isinstance(result, FitResult) or result.spec != expected:
        raise MobilePlanIRExperimentError("training result is not bound to its exact fit spec")
    if (
        result.final_checkpoint.name != f"step-{EXPECTED_STEPS:08d}"
        or result.model_sha256 != result.checkpoint_file_sha256.get("model.safetensors")
        or _SHA256.fullmatch(result.model_sha256) is None
    ):
        raise MobilePlanIRExperimentError("training result does not bind the final step-73 model")
    receipt = result.training_receipt
    required = {
        "arm": expected.arm,
        "seed": expected.seed,
        "run_id": expected.run_id,
        "train_rows": TRAIN_ROWS,
        "train_sha256": expected.train_sha256,
        "completed_optimizer_steps": EXPECTED_STEPS,
        "training_mode": "completion_only_no_dev",
        "checkpoint_policy": "unconditional_final_only",
        "development_labels_read": 0,
        "rejected_examples": 0,
        "base_reset_from_canonical_checkpoint": True,
    }
    if type(receipt) is not dict or any(
        receipt.get(key) != value for key, value in required.items()
    ):
        raise MobilePlanIRExperimentError(
            "training receipt violates completion-only/base-reset budget"
        )
    observed_hashes = _checkpoint_hashes(result.final_checkpoint)
    if observed_hashes != dict(result.checkpoint_file_sha256) or receipt.get(
        "checkpoint_file_sha256"
    ) != dict(result.checkpoint_file_sha256):
        raise MobilePlanIRExperimentError("training result/checkpoint receipt hashes differ")


def _validate_raw_result(result: RawGenerationResult, expected: FitResult) -> None:
    if not isinstance(result, RawGenerationResult) or result.fit != expected:
        raise MobilePlanIRExperimentError("raw generation is not bound to its final checkpoint")
    if (
        _SHA256.fullmatch(result.predictions_sha256) is None
        or _SHA256.fullmatch(result.summary_sha256) is None
        or not result.predictions_path.is_file()
        or not result.summary_path.is_file()
        or sha256_file(result.predictions_path) != result.predictions_sha256
        or sha256_file(result.summary_path) != result.summary_sha256
        or _binary_jsonl_rows(result.predictions_path) != SCREEN_ROWS
    ):
        raise MobilePlanIRExperimentError("raw generation artifacts changed or have wrong count")
    generation = result.generation
    expected_screen_sha256 = str(EXPECTED_OUTPUT_SHA256[f"screen_{expected.spec.arm.lower()}"])
    if type(generation) is not dict or any(
        generation.get(key) != value
        for key, value in {
            "schema_version": GENERATION_VERSION,
            "manifest_sha256": expected_screen_sha256,
            "predictions_sha256": result.predictions_sha256,
            "batch_size": GENERATION_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "examples": SCREEN_ROWS,
        }.items()
    ):
        raise MobilePlanIRExperimentError("generic generation summary changed")
    if result.screen_manifest_sha256 != expected_screen_sha256:
        raise MobilePlanIRExperimentError("raw generation used the wrong arm screen manifest")
    if _load_json(result.summary_path, label="generic generation summary") != dict(generation):
        raise MobilePlanIRExperimentError("generic GenerationSummary bytes differ from receipt")
    if generation.get("checkpoint_sha256") != dict(expected.checkpoint_file_sha256):
        raise MobilePlanIRExperimentError("generic generation used the wrong checkpoint tree")
    generic = read_jsonl(result.predictions_path)
    ids = [row.get("id") for row in generic]
    if (
        len(ids) != SCREEN_ROWS
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != SCREEN_ROWS
    ):
        raise MobilePlanIRExperimentError("generic prediction IDs/count are invalid")


def _verify_training_run(
    *, run_manifest: Mapping[str, Any], summary: TrainingSummary, spec: FitSpec
) -> dict[str, int]:
    if (
        summary.global_steps != EXPECTED_STEPS
        or summary.status != "max_epochs"
        or any(
            value is not None
            for value in (
                summary.initial_dev_loss,
                summary.best_dev_loss,
                summary.final_dev_loss,
                summary.best_checkpoint,
            )
        )
    ):
        raise MobilePlanIRExperimentError("fit did not complete exact completion-only epoch")
    if (
        run_manifest.get("run_id") != spec.run_id
        or run_manifest.get("planned_optimizer_steps") != EXPECTED_STEPS
        or run_manifest.get("training_mode") != "completion_only_no_dev"
        or run_manifest.get("checkpoint_policy") != "unconditional_final_only"
    ):
        raise MobilePlanIRExperimentError("trainer run manifest changed phase/checkpoint policy")
    base = run_manifest.get("base_checkpoint")
    if type(base) is not dict:
        raise MobilePlanIRExperimentError("trainer omitted base provenance")
    source = base.get("source")
    counts = base.get("parameter_counts")
    if (
        type(source) is not dict
        or source.get("kind") != "huggingface"
        or source.get("repo_id") != BASE_REPO
        or source.get("revision") != BASE_REVISION
        or base.get("file_sha256") != BASE_HASHES
        or type(counts) is not dict
        or counts.get("total") != PARAMETER_COUNT
    ):
        raise MobilePlanIRExperimentError("fit did not independently reset from canonical base")
    data = run_manifest.get("data")
    train = data.get("train") if type(data) is dict else None
    if (
        type(data) is not dict
        or data.get("train_manifest_sha256") != spec.train_sha256
        or data.get("dev_labels_read") is not False
        or data.get("dev") is not None
        or data.get("dev_manifest") is not None
        or type(train) is not dict
        or train.get("accepted_examples") != TRAIN_ROWS
        or train.get("rejected_examples") != 0
    ):
        raise MobilePlanIRExperimentError("fit read development labels or changed training rows")
    encoded = train.get("encoded_tokens")
    target = train.get("target_tokens")
    if type(encoded) is not int or type(target) is not int:
        raise MobilePlanIRExperimentError("trainer omitted exact token accounting")
    return {"encoded_tokens": encoded, "target_tokens": target}


def _validate_training_token_audit(
    frozen_config: Mapping[str, Any], *, spec: FitSpec, observed: Mapping[str, int]
) -> None:
    arm_audit = frozen_config["token_compute_audit"]["train"][spec.arm]
    if (
        observed.get("encoded_tokens") != arm_audit["full_sequence_tokens"]
        or observed.get("target_tokens") != arm_audit["target_tokens"]
    ):
        raise MobilePlanIRExperimentError(
            f"fit {spec.arm}/s{spec.seed} token presentation differs from frozen audit"
        )


def _raw_freeze_payload(rows: Sequence[RawGenerationResult]) -> dict[str, object]:
    if tuple((row.fit.spec.arm, row.fit.spec.seed) for row in rows) != FIT_ORDER:
        raise MobilePlanIRExperimentError("raw freeze requires all nine generation runs")
    records: list[dict[str, object]] = []
    for row in rows:
        _validate_raw_result(row, row.fit)
        records.append(
            {
                "arm": row.fit.spec.arm,
                "seed": row.fit.spec.seed,
                "manifest_sha256": row.screen_manifest_sha256,
                "model_sha256": row.fit.model_sha256,
                "generic_predictions_sha256": row.predictions_sha256,
                "generation_summary_sha256": row.summary_sha256,
                "rows": SCREEN_ROWS,
            }
        )
    return {
        "schema_version": "barun-mobile-planir-raw-freeze-v1",
        "run_count": 9,
        "row_count": 9 * SCREEN_ROWS,
        "scorer_invoked": False,
        "gate_invoked": False,
        "runs": records,
    }


def _validate_raw_freeze_receipt(path: Path, rows: Sequence[RawGenerationResult]) -> None:
    if _load_json(path, label="raw prediction freeze") != _raw_freeze_payload(rows):
        raise MobilePlanIRExperimentError(
            "raw predictions or their freeze receipt changed before scoring"
        )


class FixedScreenOperations:
    """Real train/generate/score implementation for one already provisioned exact ID."""

    def __init__(
        self,
        *,
        frozen_config: Mapping[str, Any],
        config_path: Path,
        config_sha256: str,
        materialization_dir: Path,
        execution_dir: Path,
        jarvis_resource_id: str,
    ) -> None:
        self.frozen_config = frozen_config
        self.config_path = config_path.resolve()
        self.config_sha256 = config_sha256
        self.materialization_dir = materialization_dir.resolve()
        self.execution_dir = execution_dir.resolve()
        self.jarvis_resource_id = _resource_id(jarvis_resource_id)
        self.fit_root = self.execution_dir / "fits"
        self.training_root = self.execution_dir / "training"
        self.config_root = self.execution_dir / "training-configs"
        self.fit_root.mkdir(parents=True)
        self.config_root.mkdir()
        self._checkpoints_frozen = False
        self._screen_access_started = False
        self._raw_frozen = False
        # Least privilege: individual screen paths do not exist in object state yet.
        self._screen_manifests: dict[str, tuple[Path, str]] | None = None

    def train_fit(self, spec: FitSpec) -> FitResult:
        if self._checkpoints_frozen or self._screen_access_started:
            raise MobilePlanIRExperimentError("training is forbidden after checkpoint freeze")
        if sha256_file(spec.train_manifest) != spec.train_sha256:
            raise MobilePlanIRExperimentError("training manifest changed before fit")
        if _binary_jsonl_rows(spec.train_manifest) != TRAIN_ROWS:
            raise MobilePlanIRExperimentError("training manifest row count changed")
        fit_dir = self.fit_root / f"{spec.arm.lower()}-s{spec.seed}"
        fit_dir.mkdir()
        sentinel = fit_dir / "NO-SCREEN-OR-DEVELOPMENT-LABELS-ACCESSED.jsonl"
        payload = build_training_payload(
            frozen_config=self.frozen_config,
            spec=spec,
            output_root=self.training_root,
            ignored_dev_sentinel=sentinel,
            jarvis_resource_id=self.jarvis_resource_id,
        )
        config_path = self.config_root / f"{spec.run_id}.json"
        _write_new_json(config_path, payload)
        training_config = TrainingRunConfig.from_json(config_path)
        summary = train_sft(training_config, completion_only=True)
        run_dir = Path(summary.run_dir).resolve()
        if sentinel.exists() or (run_dir / "best_checkpoint.json").exists():
            raise MobilePlanIRExperimentError(
                "completion-only fit accessed dev or selected checkpoint"
            )
        run_manifest = _load_json(run_dir / "run_manifest.json", label="training run manifest")
        token_counts = _verify_training_run(run_manifest=run_manifest, summary=summary, spec=spec)
        _validate_training_token_audit(
            self.frozen_config,
            spec=spec,
            observed=token_counts,
        )
        metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
        train_steps = [
            row.get("global_step") for row in metrics_rows if row.get("event") == "train"
        ]
        if train_steps != list(range(1, EXPECTED_STEPS + 1)):
            raise MobilePlanIRExperimentError(
                "trainer did not record exactly optimizer steps 1..73"
            )
        checkpoint = Path(summary.final_checkpoint).resolve()
        hashes = _checkpoint_hashes(checkpoint)
        receipt: dict[str, object] = {
            "schema_version": "barun-mobile-planir-fit-receipt-v1",
            "arm": spec.arm,
            "seed": spec.seed,
            "run_id": spec.run_id,
            "train_manifest": str(spec.train_manifest),
            "train_sha256": spec.train_sha256,
            "train_rows": TRAIN_ROWS,
            "rejected_examples": 0,
            "completed_optimizer_steps": EXPECTED_STEPS,
            "training_mode": "completion_only_no_dev",
            "checkpoint_policy": "unconditional_final_only",
            "development_labels_read": 0,
            "screen_labels_read": 0,
            "base_reset_from_canonical_checkpoint": True,
            "base_checkpoint": self.frozen_config["base_checkpoint"],
            "final_checkpoint": str(checkpoint),
            "checkpoint_file_sha256": hashes,
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "model_sha256": hashes["model.safetensors"],
            "encoded_tokens": token_counts["encoded_tokens"],
            "target_tokens": token_counts["target_tokens"],
            "training_config": str(config_path),
            "training_config_sha256": sha256_file(config_path),
            "training_summary": asdict(summary),
        }
        _write_new_json(fit_dir / "training-receipt.json", receipt)
        result = FitResult(
            spec=spec,
            final_checkpoint=checkpoint,
            checkpoint_file_sha256=hashes,
            model_sha256=hashes["model.safetensors"],
            training_receipt=receipt,
        )
        _validate_fit_result(result, spec)
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def freeze_checkpoints(self, fits: Sequence[FitResult]) -> None:
        if self._checkpoints_frozen or self._screen_access_started:
            raise MobilePlanIRExperimentError("checkpoint freeze is not repeatable")
        if tuple((fit.spec.arm, fit.spec.seed) for fit in fits) != FIT_ORDER:
            raise MobilePlanIRExperimentError("checkpoint freeze requires all nine exact fits")
        records = []
        for fit in fits:
            _validate_fit_result(fit, fit.spec)
            if _checkpoint_hashes(fit.final_checkpoint) != dict(fit.checkpoint_file_sha256):
                raise MobilePlanIRExperimentError("checkpoint changed before phase freeze")
            records.append(
                {
                    "arm": fit.spec.arm,
                    "seed": fit.spec.seed,
                    "run_id": fit.spec.run_id,
                    "model_sha256": fit.model_sha256,
                    "checkpoint_file_sha256": dict(fit.checkpoint_file_sha256),
                }
            )
        _write_new_json(
            self.execution_dir / "all-checkpoints-frozen.json",
            {
                "schema_version": "barun-mobile-planir-checkpoint-freeze-v1",
                "fit_count": 9,
                "completed_optimizer_steps": 9 * EXPECTED_STEPS,
                "screen_json_decoded": False,
                "screen_labels_read": 0,
                "checkpoint_selection": False,
                "fits": records,
            },
        )
        self._checkpoints_frozen = True

    def begin_screen_access(self) -> None:
        if not self._checkpoints_frozen or self._screen_access_started:
            raise MobilePlanIRExperimentError(
                "screen access requires one completed checkpoint freeze"
            )
        freeze = self.execution_dir / "all-checkpoints-frozen.json"
        if not freeze.is_file():
            raise MobilePlanIRExperimentError("checkpoint freeze receipt is missing")
        _write_new_json(
            self.execution_dir / "screen-access-started.json",
            {
                "schema_version": "barun-mobile-planir-screen-access-v1",
                "started_at": _utc_now(),
                "all_nine_final_checkpoints_frozen": True,
                "checkpoint_freeze_sha256": sha256_file(freeze),
                "generic_generation_decodes_sft_rows_including_targets": True,
                "reason": "raw_generation_from_exact_screen_prompt_manifests",
                "retry_after_this_boundary": "forbidden",
            },
        )
        # Resolve the three label-bearing paths only after the durable boundary above.
        hashes = self.frozen_config["data"]["artifact_sha256"]
        self._screen_manifests = {
            arm: (
                self.materialization_dir / OUTPUT_FILENAMES[f"screen_{arm.lower()}"],
                str(hashes[f"screen_{arm.lower()}"]),
            )
            for arm in ARMS
        }
        self._screen_access_started = True

    def generate_raw(self, fit: FitResult) -> RawGenerationResult:
        if not self._screen_access_started or self._raw_frozen or self._screen_manifests is None:
            raise MobilePlanIRExperimentError("raw generation is outside its irreversible phase")
        screen_manifest, screen_sha256 = self._screen_manifests[fit.spec.arm]
        if sha256_file(screen_manifest) != screen_sha256:
            raise MobilePlanIRExperimentError("screen manifest changed at generation boundary")
        fit_dir = self.fit_root / f"{fit.spec.arm.lower()}-s{fit.spec.seed}"
        generation_dir = fit_dir / "raw-generation"
        generation_dir.mkdir()
        predictions = generation_dir / "predictions.jsonl"
        summary: GenerationSummary = generate_manifest(
            checkpoint_dir=fit.final_checkpoint,
            manifest_path=screen_manifest,
            manifest_sha256=screen_sha256,
            predictions_path=predictions,
            device_name="cuda",
            batch_size=GENERATION_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            expected_checkpoint_sha256=fit.checkpoint_file_sha256,
        )
        summary_path = predictions.with_suffix(predictions.suffix + ".manifest.json")
        summary_record = summary.to_dict()
        if (
            summary.schema_version != GENERATION_VERSION
            or summary.examples != SCREEN_ROWS
            or summary.manifest_sha256 != screen_sha256
            or dict(summary.checkpoint_sha256) != dict(fit.checkpoint_file_sha256)
            or summary.batch_size != GENERATION_BATCH_SIZE
            or summary.max_new_tokens != MAX_NEW_TOKENS
            or summary.device != "cuda"
            or summary.dtype != "torch.bfloat16"
            or summary.predictions_sha256 != sha256_file(predictions)
        ):
            raise MobilePlanIRExperimentError("generic generation violated its frozen contract")
        result = RawGenerationResult(
            fit=fit,
            screen_manifest_sha256=screen_sha256,
            predictions_path=predictions,
            predictions_sha256=summary.predictions_sha256,
            summary_path=summary_path,
            summary_sha256=sha256_file(summary_path),
            generation=summary_record,
        )
        _validate_raw_result(result, fit)
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def freeze_raw_predictions(self, rows: Sequence[RawGenerationResult]) -> None:
        if not self._screen_access_started or self._raw_frozen:
            raise MobilePlanIRExperimentError("raw prediction freeze is outside its phase")
        payload = _raw_freeze_payload(rows)
        _write_new_json(
            self.execution_dir / "all-raw-predictions-frozen.json",
            payload,
        )
        self._raw_frozen = True

    def _enrich_one_run(
        self,
        raw: RawGenerationResult,
        manifest_rows: Sequence[Mapping[str, Any]],
        *,
        generation_source_sha256: str,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        fit = raw.fit
        generic_rows = read_jsonl(raw.predictions_path)
        manifest_by_id: dict[str, Mapping[str, Any]] = {}
        for row in manifest_rows:
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in manifest_by_id:
                raise MobilePlanIRExperimentError("screen manifest has invalid/duplicate IDs")
            manifest_by_id[sample_id] = row
        generic_by_id: dict[str, Mapping[str, Any]] = {}
        for row in generic_rows:
            if type(row) is not dict or set(row) != {
                "generated_tokens",
                "generation_failure",
                "id",
                "prediction_raw",
                "prompt_tokens",
                "truncated",
            }:
                raise MobilePlanIRExperimentError("generic generation row schema changed")
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in generic_by_id:
                raise MobilePlanIRExperimentError("generic generation has invalid/duplicate IDs")
            generic_by_id[sample_id] = row
        if set(generic_by_id) != set(manifest_by_id) or len(generic_by_id) != SCREEN_ROWS:
            raise MobilePlanIRExperimentError("generic prediction IDs differ from exact screen IDs")

        tokenizer_path = fit.final_checkpoint / "tokenizer.json"
        if sha256_file(tokenizer_path) != BASE_HASHES["tokenizer.json"]:
            raise MobilePlanIRExperimentError("fit tokenizer differs from pinned base tokenizer")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        decoding = _decoding_payload(tokenizer)
        decoding_sha256 = _canonical_sha256(decoding)
        enriched: list[dict[str, object]] = []
        token_bindings: list[dict[str, object]] = []
        finish_reasons: dict[str, int] = {"eos": 0, "max_new_tokens": 0, "failure": 0}
        generated_tokens = 0
        prompt_lengths: list[tuple[str, int]] = []

        for sample_id in sorted(manifest_by_id):
            manifest = manifest_by_id[sample_id]
            generic = generic_by_id[sample_id]
            prompt = manifest.get("prompt")
            metadata = manifest.get("metadata")
            if not isinstance(prompt, str) or type(metadata) is not dict:
                raise MobilePlanIRExperimentError("screen row lacks prompt/metadata")
            screen = metadata.get("mobile_planir_screen")
            evidence = screen.get("prompt_evidence") if type(screen) is dict else None
            if type(screen) is not dict or type(evidence) is not dict:
                raise MobilePlanIRExperimentError("screen row lacks PlanIR prompt evidence")
            if screen.get("arm") != fit.spec.arm or screen.get("source_id") != sample_id:
                raise MobilePlanIRExperimentError("screen arm/source identity changed")
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            observed_prompt_tokens = generic["prompt_tokens"]
            if type(observed_prompt_tokens) is not int or observed_prompt_tokens != len(ids):
                raise MobilePlanIRExperimentError("generic prompt-token count is not reproducible")
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if (
                screen.get("output_prompt_sha256") != prompt_sha256
                or evidence.get("prompt_sha256") != prompt_sha256
                or not isinstance(evidence.get("table_sha256"), str)
            ):
                raise MobilePlanIRExperimentError("model-visible prompt evidence changed")
            prediction_raw = generic["prediction_raw"]
            if prediction_raw is not None and not isinstance(prediction_raw, str):
                raise MobilePlanIRExperimentError("prediction_raw must be exact string or null")
            truncated = generic["truncated"]
            generation_failure = generic["generation_failure"]
            if type(truncated) is not bool or (
                generation_failure is not None
                and (
                    not isinstance(generation_failure, str)
                    or not generation_failure
                    or any(character.isspace() for character in generation_failure)
                )
            ):
                raise MobilePlanIRExperimentError("generic finish/failure evidence is invalid")
            generated = generic["generated_tokens"]
            if type(generated) is not int or generated < 0:
                raise MobilePlanIRExperimentError("generated-token evidence is invalid")
            finish = (
                "failure"
                if generation_failure is not None
                else "max_new_tokens"
                if truncated
                else "eos"
            )
            finish_reasons[finish] += 1
            generated_tokens += generated
            prompt_lengths.append((sample_id, len(ids)))
            token_bindings.append(
                {
                    "id": sample_id,
                    "prompt_sha256": prompt_sha256,
                    "prompt_tokens": len(ids),
                    "prompt_token_ids_sha256": _canonical_sha256(ids),
                }
            )
            enriched.append(
                {
                    "schema_version": SCREEN_PREDICTION_ROW_VERSION,
                    "id": sample_id,
                    "source_id": sample_id,
                    "arm": fit.spec.arm,
                    "seed": fit.spec.seed,
                    "prompt_sha256": prompt_sha256,
                    "table_sha256": evidence["table_sha256"],
                    # The scorer contract binds the actual final model tensor file.
                    "checkpoint_sha256": fit.model_sha256,
                    "decoding_sha256": decoding_sha256,
                    "prediction_raw": prediction_raw,
                    "prediction_raw_sha256": (
                        None
                        if prediction_raw is None
                        else hashlib.sha256(prediction_raw.encode("utf-8")).hexdigest()
                    ),
                    "missing": prediction_raw is None,
                    # These are copied from immutable generic rows, never inferred by caller.
                    "truncated": truncated,
                    "generation_failure": generation_failure,
                }
            )

        # generate_manifest sorts by (length, id) and batches without prompt truncation.
        ordered_lengths = [
            length for _, length in sorted(prompt_lengths, key=lambda item: (item[1], item[0]))
        ]
        padded_prompt_tokens = sum(
            max(batch) * len(batch)
            for start in range(0, len(ordered_lengths), GENERATION_BATCH_SIZE)
            if (batch := ordered_lengths[start : start + GENERATION_BATCH_SIZE])
        )
        presentation = {
            "schema_version": "barun-mobile-planir-model-presentation-v1",
            "arm": fit.spec.arm,
            "seed": fit.spec.seed,
            "rows": SCREEN_ROWS,
            "screen_manifest_sha256": raw.screen_manifest_sha256,
            "generic_predictions_sha256": raw.predictions_sha256,
            "generation_summary_sha256": raw.summary_sha256,
            "generation_implementation_version": GENERATION_VERSION,
            "generation_source_sha256": generation_source_sha256,
            "checkpoint_model_sha256": fit.model_sha256,
            "checkpoint_file_sha256": dict(fit.checkpoint_file_sha256),
            "checkpoint_manifest_sha256": sha256_file(
                fit.final_checkpoint / "checkpoint_manifest.json"
            ),
            "decoding": decoding,
            "decoding_sha256": decoding_sha256,
            "prompt_token_binding_sha256": _canonical_sha256(token_bindings),
            "prompt_tokens_unpadded": sum(length for _, length in prompt_lengths),
            "prompt_tokens_left_padded": padded_prompt_tokens,
            "generated_tokens_reported": generated_tokens,
            "finish_reasons": finish_reasons,
            "inference_dense_flops_proxy": 2
            * PARAMETER_COUNT
            * (sum(length for _, length in prompt_lengths) + generated_tokens),
            "inference_proxy_formula": (
                "2 * parameter_count * (unpadded_prompt_tokens + reported_generated_tokens); "
                "excludes attention quadratic, padding, and runtime kernel effects"
            ),
        }
        return enriched, presentation

    def enrich_and_score(self, rows: Sequence[RawGenerationResult]) -> Mapping[str, object]:
        if not self._raw_frozen or self._screen_manifests is None:
            raise MobilePlanIRExperimentError(
                "pinned scoring requires frozen nine-run raw evidence"
            )
        if tuple((row.fit.spec.arm, row.fit.spec.seed) for row in rows) != FIT_ORDER:
            raise MobilePlanIRExperimentError("scoring population is not the frozen nine runs")
        raw_freeze = self.execution_dir / "all-raw-predictions-frozen.json"
        if not raw_freeze.is_file():
            raise MobilePlanIRExperimentError("raw prediction freeze receipt is missing")
        _validate_raw_freeze_receipt(raw_freeze, rows)

        evaluation_dir = self.execution_dir / "evaluation"
        evaluation_dir.mkdir()
        manifest_by_arm: dict[str, list[Mapping[str, Any]]] = {}
        for arm in ARMS:
            path, digest = self._screen_manifests[arm]
            if sha256_file(path) != digest:
                raise MobilePlanIRExperimentError("screen manifest changed before pinned scoring")
            manifest_by_arm[arm] = list(read_jsonl(path))
        generation_source = Path(generate_manifest.__code__.co_filename).resolve()
        generation_source_sha256 = sha256_file(generation_source)

        combined_predictions: list[dict[str, object]] = []
        presentation_receipts: list[dict[str, object]] = []
        for raw in rows:
            enriched, presentation = self._enrich_one_run(
                raw,
                manifest_by_arm[raw.fit.spec.arm],
                generation_source_sha256=generation_source_sha256,
            )
            per_run = (
                self.fit_root
                / f"{raw.fit.spec.arm.lower()}-s{raw.fit.spec.seed}"
                / "model-presentation-receipt.json"
            )
            _write_new_json(per_run, presentation)
            presentation_receipts.append(
                {
                    "arm": raw.fit.spec.arm,
                    "seed": raw.fit.spec.seed,
                    "path": str(per_run),
                    "sha256": sha256_file(per_run),
                }
            )
            combined_predictions.extend(enriched)
        if len(combined_predictions) != 9 * SCREEN_ROWS:
            raise MobilePlanIRExperimentError("enriched prediction population changed")
        predictions_path = evaluation_dir / "predictions.jsonl"
        _write_new_jsonl(predictions_path, combined_predictions)

        combined_manifest = [row for arm in ARMS for row in manifest_by_arm[arm]]
        if len(combined_manifest) != len(ARMS) * SCREEN_ROWS:
            raise MobilePlanIRExperimentError("combined pinned screen manifest changed")
        # Production uses the default enforce_pinned=True.  Never pass the synthetic opt-out.
        evaluation = score_screen(combined_manifest, combined_predictions)
        if evaluation.pinned_manifest_enforced is not True:
            raise MobilePlanIRExperimentError("scientific scorer did not enforce pinned manifests")
        gate = evaluate_screen_gate(evaluation)

        repository_root = Path(__file__).resolve().parents[3]
        compiler_source = repository_root / "src/barunlm/evaluation/grounded_planir_v2.py"
        evaluator_source = Path(score_screen.__code__.co_filename).resolve()
        compiler_sha256 = sha256_file(compiler_source)
        evaluator_sha256 = sha256_file(evaluator_source)
        receipt = build_evidence_receipt(
            combined_manifest,
            combined_predictions,
            config_sha256=self.config_sha256,
            compiler_source_sha256=compiler_sha256,
            evaluator_source_sha256=evaluator_sha256,
            renderer_source_sha256=compiler_sha256,
        )
        validate_evidence_receipt(
            receipt,
            combined_manifest,
            combined_predictions,
            config_sha256=self.config_sha256,
            compiler_source_sha256=compiler_sha256,
            evaluator_source_sha256=evaluator_sha256,
            renderer_source_sha256=compiler_sha256,
        )
        _write_new_json(evaluation_dir / "scores.json", evaluation.to_record())
        _write_new_json(evaluation_dir / "gate.json", gate.to_record())
        _write_new_json(evaluation_dir / "evidence-receipt.json", receipt)
        result: dict[str, object] = {
            "schema_version": BACKEND_VERSION,
            "run_id": RUN_ID,
            "status": "passed_internal_gate" if gate.passed else "rejected_internal_gate",
            "gate_passed": gate.passed,
            "decision": (
                "license_generalized_planir_development_only"
                if gate.passed
                else "close_narrow_placeholder_v2_without_rescue"
            ),
            "construction_internal_only": True,
            "fresh_evidence": False,
            "checkpoint_promoted": False,
            "larger_model_comparison_performed": False,
            "fit_count": 9,
            "optimizer_steps": 9 * EXPECTED_STEPS,
            "screen_rows_per_run": SCREEN_ROWS,
            "prediction_rows": len(combined_predictions),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "score_sha256": sha256_file(evaluation_dir / "scores.json"),
            "gate_sha256": sha256_file(evaluation_dir / "gate.json"),
            "evidence_receipt_sha256": sha256_file(evaluation_dir / "evidence-receipt.json"),
            "model_presentation_receipts": presentation_receipts,
            "all_checkpoints_frozen_before_screen_access": True,
            "all_raw_predictions_frozen_before_scoring": True,
            "data_firewall": {
                **self.frozen_config["data"]["forbidden_populations"],
                "construction_train_rows_read_per_fit": TRAIN_ROWS,
                "construction_screen_rows_read_per_run": SCREEN_ROWS,
            },
            "retry_or_rescue_authorized": False,
        }
        return result


def _force_math_sdpa() -> dict[str, bool | None]:
    backend = torch.backends.cuda
    settings = (
        ("flash", "enable_flash_sdp", "flash_sdp_enabled", False),
        ("memory_efficient", "enable_mem_efficient_sdp", "mem_efficient_sdp_enabled", False),
        ("math", "enable_math_sdp", "math_sdp_enabled", True),
    )
    evidence: dict[str, bool | None] = {}
    for label, setter_name, getter_name, expected in settings:
        setter = getattr(backend, setter_name, None)
        getter = getattr(backend, getter_name, None)
        if not callable(setter) or not callable(getter):
            raise MobilePlanIRExperimentError(f"cannot verify {label} SDPA backend")
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            raise MobilePlanIRExperimentError(f"failed to fix {label} SDPA={expected}")
        evidence[label] = actual
    cudnn_setter = getattr(backend, "enable_cudnn_sdp", None)
    cudnn_getter = getattr(backend, "cudnn_sdp_enabled", None)
    if cudnn_setter is None and cudnn_getter is None:
        evidence["cudnn"] = None
    elif callable(cudnn_setter) and callable(cudnn_getter):
        cudnn_setter(False)
        if cudnn_getter() is not False:
            raise MobilePlanIRExperimentError("failed to disable cuDNN SDPA")
        evidence["cudnn"] = False
    else:
        raise MobilePlanIRExperimentError("cuDNN SDPA interface is incomplete")
    return evidence


def _cuda_preflight(*, configured_before_torch_import: bool) -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise MobilePlanIRExperimentError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if configured_before_torch_import is not True:
        raise MobilePlanIRExperimentError("cuBLAS configuration before torch import is unproven")
    if torch.cuda.is_initialized():
        raise MobilePlanIRExperimentError("CUDA initialized before deterministic preflight")
    sdpa = _force_math_sdpa()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise MobilePlanIRExperimentError("one BF16-capable CUDA H200 is required")
    if torch.cuda.device_count() != 1:
        raise MobilePlanIRExperimentError("exactly one visible CUDA device is required")
    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name.upper():
        raise MobilePlanIRExperimentError(f"expected H200, observed {device_name!r}")
    return {
        "cublas_workspace_config": ":4096:8",
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "cuda_available": True,
        "bf16_supported": True,
        "device_count": 1,
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "sdpa_backends": sdpa,
    }


def _validate_controller_preflight(
    receipt: Mapping[str, object], *, config_sha256: str, jarvis_resource_id: str
) -> None:
    protected = receipt.get("protected_resource_ids")
    if (
        receipt.get("model_or_cuda_loaded") is not False
        or receipt.get("screen_json_decoded") is not False
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("jarvis_resource_id") != int(jarvis_resource_id)
        or receipt.get("fresh_instance") is not True
        or type(protected) is not list
        or any(type(value) is not int for value in protected)
        or not PROTECTED_RESOURCE_IDS.issubset(protected)
        or int(jarvis_resource_id) in protected
    ):
        raise MobilePlanIRExperimentError(
            "controller preflight does not prove a fresh safe boundary"
        )


def execute_mobile_planir_experiment(
    *,
    config_path: str | Path,
    materialization_dir: str | Path,
    output_dir: str | Path,
    preflight_receipt: Mapping[str, object],
    jarvis_resource_id: str,
    cublas_configured_before_torch_import: bool,
) -> Mapping[str, object]:
    """Execute the fixed screen on an already-created, controller-attested H200."""

    resource_id = _resource_id(jarvis_resource_id)
    frozen, config_sha256 = load_frozen_config(config_path)
    _validate_controller_preflight(
        preflight_receipt,
        config_sha256=config_sha256,
        jarvis_resource_id=resource_id,
    )
    specs = build_fit_specs(
        frozen_config=frozen,
        materialization_dir=materialization_dir,
    )
    destination = Path(output_dir).resolve()
    execution_dir = destination / "execution"
    execution_dir.mkdir(parents=True, exist_ok=False)
    try:
        environment = _cuda_preflight(
            configured_before_torch_import=cublas_configured_before_torch_import
        )
        _write_new_json(
            execution_dir / "environment-receipt.json",
            {
                "schema_version": "barun-mobile-planir-environment-v1",
                "captured_at": _utc_now(),
                "jarvis_resource_id": int(resource_id),
                "device": "cuda",
                "precision": "bf16",
                "wandb": "disabled",
                "automatic_retry": False,
                "environment": environment,
            },
        )
        operations = FixedScreenOperations(
            frozen_config=frozen,
            config_path=Path(config_path),
            config_sha256=config_sha256,
            materialization_dir=Path(materialization_dir),
            execution_dir=execution_dir,
            jarvis_resource_id=resource_id,
        )
        result = dict(run_fixed_schedule(specs, operations))
        result.update(
            {
                "completed_at": _utc_now(),
                "jarvis_resource_id": int(resource_id),
                "config_sha256": config_sha256,
                "backend_source_sha256": sha256_file(Path(__file__)),
            }
        )
        _write_new_json(execution_dir / "result.json", result)
        return result
    except Exception as error:
        failure_path = execution_dir / "failure.json"
        if not failure_path.exists():
            _write_new_json(
                failure_path,
                {
                    "schema_version": "barun-mobile-planir-execution-failure-v1",
                    "failed_at": _utc_now(),
                    "jarvis_resource_id": int(resource_id),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "artifacts_preserved": True,
                    "automatic_retry_attempted": False,
                    "screen_access_started": (
                        execution_dir.joinpath("screen-access-started.json").is_file()
                    ),
                    "raw_model_output_exists": any(
                        execution_dir.glob("fits/*/raw-generation/predictions.jsonl")
                    ),
                    "retry_authorized": False,
                    "official_mobile_961_rows_read": 0,
                    "reused_mobile_756_rows_read": 0,
                    "human_selection_rows_read": 0,
                    "human_confirmation_rows_read": 0,
                },
            )
        raise


def run_fixed_schedule(
    specs: Sequence[FitSpec], operations: ScreenOperations
) -> Mapping[str, object]:
    """Run the only legal phase graph; any reordering requires different code/version."""

    frozen_specs = tuple(specs)
    identities = tuple((item.arm, item.seed) for item in frozen_specs)
    if identities != FIT_ORDER or len({item.run_id for item in frozen_specs}) != 9:
        raise MobilePlanIRExperimentError(
            "fit order/membership must be A/B/C within seeds 17/29/43"
        )

    # No screen path is opened by this phase.  Each fit is completion-only and base-reset.
    fits = tuple(operations.train_fit(spec) for spec in frozen_specs)
    if tuple((item.spec.arm, item.spec.seed) for item in fits) != FIT_ORDER:
        raise MobilePlanIRExperimentError("training results changed fit identity/order")
    for result, expected in zip(fits, frozen_specs, strict=True):
        _validate_fit_result(result, expected)
    operations.freeze_checkpoints(fits)

    # This explicit state transition is the first permission to decode a screen manifest.
    operations.begin_screen_access()
    raw = tuple(operations.generate_raw(fit) for fit in fits)
    if tuple((item.fit.spec.arm, item.fit.spec.seed) for item in raw) != FIT_ORDER:
        raise MobilePlanIRExperimentError("generation results changed fit identity/order")
    for result, expected in zip(raw, fits, strict=True):
        _validate_raw_result(result, expected)
    operations.freeze_raw_predictions(raw)

    # Scoring, gate evaluation, and model-output enrichment are forbidden above this line.
    return operations.enrich_and_score(raw)


__all__ = [
    "ARMS",
    "BACKEND_VERSION",
    "CONFIG_VERSION",
    "EXPECTED_STEPS",
    "FIT_ORDER",
    "MAX_NEW_TOKENS",
    "SEEDS",
    "FitResult",
    "FitSpec",
    "MobilePlanIRExperimentError",
    "RawGenerationResult",
    "ScreenOperations",
    "run_fixed_schedule",
]
