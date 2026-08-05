"""Fixed execution backend for Month-Boundary Counterfactual SFT v1.

The stdlib-only entrypoint performs byte-level preflight before importing this module.
This backend then runs exactly nine independent base-initialized screening fits, applies
the frozen selection gate, evaluates confirmation only after selection passes, and runs
one completion-only full refit only after confirmation passes.  A successful full refit
is scored once on the pinned reused-756 compatibility population.  The official 961-row
evaluation population remains unavailable throughout.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch

from barunlm.evaluation.generation import generate_manifest
from barunlm.evaluation.mobile_actions import read_jsonl, write_scores
from barunlm.evaluation.mobile_temporal_counterfactual import (
    AST_EXACT_METRIC,
    CALENDAR_CROSS_MONTH_SUBSET,
    CALENDAR_DATETIME_EXACT_METRIC,
    CALENDAR_SAME_MONTH_SUBSET,
    NON_CALENDAR_SUBSET,
    compare_seeded_arms,
    evaluate_temporal_gates,
    paired_cluster_bootstrap,
    score_temporal_subsets,
)
from barunlm.training.config import TrainingRunConfig
from barunlm.training.data import sha256_file
from barunlm.training.trainer import TrainingSummary, train_sft

BACKEND_VERSION = "barun-mobile-temporal-execution-v1"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_SEEDS = (17, 29, 43)
EXPECTED_ARM_IDS = ("A", "B", "C")
EXPECTED_BASE_REPO = "harrrshall/BarunLM-35M"
EXPECTED_BASE_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
EXPECTED_BASE_HASHES = {
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "model.safetensors": "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
}
EXPECTED_PARAMETER_COUNT = 35_072_768
PINNED_REUSED_756_SHA256 = "988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55"
PINNED_REUSED_756_ROWS = 756
GENERATION_BATCH_SIZE = 128
SHADOW_MAX_NEW_TOKENS = 256
TERMINAL_MAX_NEW_TOKENS = 192
EXPECTED_COMPARISON_NAMES = ("C_minus_A", "C_minus_B")
EXPECTED_BOOTSTRAP_CONTRACT = {
    "estimator": "mean_of_fixed_seed_within_seed_cluster_rates",
    "expected_training_seeds": [17, 29, 43],
    "quantile_method": "linear",
    "resamples": 10_000,
    "resampling_unit": "semantic_cluster_within_each_fixed_training_seed",
    "rng": {
        "library": "numpy",
        "api": "Generator",
        "bit_generator": "PCG64",
        "seed": 17,
    },
}


class TemporalExecutionError(RuntimeError):
    """The fixed execution contract could not be proved or completed."""


class TemporalPhaseOperations(Protocol):
    """Small seam used to test the irreversible phase ordering without a GPU."""

    def train_screening_fit(self, fit: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def score_screening_fit(
        self, fit_result: Mapping[str, Any], *, population: str
    ) -> Sequence[Mapping[str, Any]]: ...

    def evaluate_phase_gate(
        self,
        *,
        phase: str,
        evidence: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    ) -> Mapping[str, Any]: ...

    def train_conditional_full_refit(
        self, specification: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def evaluate_terminal_compatibility(
        self, full_refit: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TemporalExecutionError(f"refusing to overwrite immutable artifact {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(payload))
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
        raise TemporalExecutionError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TemporalExecutionError(f"{label} must be one JSON object")
    return value


def _force_math_sdpa() -> dict[str, bool | None]:
    backend = torch.backends.cuda
    required = (
        ("flash", "enable_flash_sdp", "flash_sdp_enabled", False),
        (
            "memory_efficient",
            "enable_mem_efficient_sdp",
            "mem_efficient_sdp_enabled",
            False,
        ),
        ("math", "enable_math_sdp", "math_sdp_enabled", True),
    )
    evidence: dict[str, bool | None] = {}
    for label, setter_name, getter_name, expected in required:
        setter = getattr(backend, setter_name, None)
        getter = getattr(backend, getter_name, None)
        if not callable(setter) or not callable(getter):
            raise TemporalExecutionError(
                f"PyTorch cannot configure and verify the {label} SDPA backend"
            )
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            raise TemporalExecutionError(
                f"failed to force {label} SDPA to {expected}: observed {actual!r}"
            )
        evidence[label] = actual

    cudnn_setter = getattr(backend, "enable_cudnn_sdp", None)
    cudnn_getter = getattr(backend, "cudnn_sdp_enabled", None)
    if cudnn_setter is None and cudnn_getter is None:
        evidence["cudnn"] = None
    elif callable(cudnn_setter) and callable(cudnn_getter):
        cudnn_setter(False)
        actual = cudnn_getter()
        if type(actual) is not bool or actual is not False:
            raise TemporalExecutionError(f"failed to disable cuDNN SDPA: observed {actual!r}")
        evidence["cudnn"] = actual
    else:
        raise TemporalExecutionError("PyTorch exposes an unverifiable cuDNN SDPA interface")
    return evidence


def _cuda_preflight(*, configured_before_torch_import: bool) -> dict[str, Any]:
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise TemporalExecutionError(
            "CUBLAS_WORKSPACE_CONFIG was not fixed to :4096:8 before backend import"
        )
    if configured_before_torch_import is not True:
        raise TemporalExecutionError("cuBLAS configuration before torch import cannot be proved")
    if torch.cuda.is_initialized():
        raise TemporalExecutionError("CUDA was initialized before determinism preflight")
    sdpa = _force_math_sdpa()
    if not torch.cuda.is_available():
        raise TemporalExecutionError("explicit CUDA execution was requested but is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise TemporalExecutionError("the selected CUDA device does not support bfloat16")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise TemporalExecutionError(
            f"frozen experiment requires exactly one visible CUDA device, observed {device_count}"
        )
    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name.upper():
        raise TemporalExecutionError(
            f"frozen experiment requires an H200 CUDA device, observed {device_name!r}"
        )
    return {
        "cublas_workspace_config": actual,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "cuda_available": True,
        "bf16_supported": True,
        "device_count": device_count,
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "sdpa_backends": sdpa,
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "wandb_mode": os.environ.get("WANDB_MODE"),
    }


def _validate_jarvis_resource_id(value: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise TemporalExecutionError("jarvis_resource_id must be an explicit positive numeric ID")
    return value


def _validate_base(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("base_checkpoint")
    if not isinstance(raw, Mapping):
        raise TemporalExecutionError("frozen config lacks base_checkpoint")
    expected = {
        "repo_id": EXPECTED_BASE_REPO,
        "revision": EXPECTED_BASE_REVISION,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "file_sha256": EXPECTED_BASE_HASHES,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise TemporalExecutionError(f"frozen base_checkpoint.{key} changed")
    return {
        "source": "huggingface",
        "repo_id": EXPECTED_BASE_REPO,
        "revision": EXPECTED_BASE_REVISION,
        "expected_sha256": dict(EXPECTED_BASE_HASHES),
    }


def _optimization_payload(*, seed: int, expected_steps: int) -> dict[str, Any]:
    if expected_steps <= 12:
        raise TemporalExecutionError("expected optimizer steps must exceed the frozen warmup")
    return {
        "seed": seed,
        "epochs": 1,
        "batch_size": 63,
        "eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "min_learning_rate_ratio": 0.1,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_epsilon": 1e-8,
        "warmup_steps": 12,
        "max_steps": None,
        "gradient_clip_norm": 1.0,
        # One diagnostic evaluation and one checkpoint at the unconditional final step.
        "eval_every_steps": expected_steps,
        "save_every_steps": expected_steps,
        "early_stopping_patience": expected_steps + 1,
        "early_stopping_min_delta": 0.0,
    }


def build_screening_training_payload(
    *,
    frozen_config: Mapping[str, Any],
    fit: Mapping[str, Any],
    train_manifest: Path,
    selection_manifest: Path,
    selection_sha256: str,
    output_root: Path,
    jarvis_resource_id: str,
) -> dict[str, Any]:
    """Build the only allowed screening trainer configuration."""

    arm_id = fit.get("arm_id")
    seed = fit.get("seed")
    if arm_id not in EXPECTED_ARM_IDS or seed not in EXPECTED_SEEDS:
        raise TemporalExecutionError("screening fit has an unregistered arm or seed")
    expected_steps = fit.get("expected_optimizer_steps")
    if type(expected_steps) is not int:
        raise TemporalExecutionError("screening fit lacks expected optimizer steps")
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": fit["run_id"],
        "output_root": str(output_root),
        "hypothesis": frozen_config["hypothesis"],
        "decision": frozen_config["decision"],
        "base_checkpoint": _validate_base(frozen_config),
        "data": {
            "train_manifest": str(train_manifest),
            "train_sha256": fit["train_sha256"],
            "dev_manifest": str(selection_manifest),
            "dev_sha256": selection_sha256,
            "max_seq_len": 2_048,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": _optimization_payload(seed=seed, expected_steps=expected_steps),
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": jarvis_resource_id,
            "estimated_hourly_cost": None,
        },
    }


def _build_full_refit_payload(
    *,
    frozen_config: Mapping[str, Any],
    specification: Mapping[str, Any],
    train_manifest: Path,
    output_root: Path,
    no_dev_sentinel: Path,
    jarvis_resource_id: str,
) -> dict[str, Any]:
    expected_steps = specification.get("expected_optimizer_steps")
    seed = specification.get("seed")
    if type(expected_steps) is not int or seed != 17:
        raise TemporalExecutionError("conditional full-refit optimizer contract changed")
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": specification["run_id"],
        "output_root": str(output_root),
        "hypothesis": frozen_config["hypothesis"],
        "decision": frozen_config["decision"],
        "base_checkpoint": _validate_base(frozen_config),
        "data": {
            "train_manifest": str(train_manifest),
            "train_sha256": specification["train_sha256"],
            "dev_manifest": str(no_dev_sentinel),
            "dev_sha256": "0" * 64,
            "max_seq_len": 2_048,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": _optimization_payload(seed=17, expected_steps=expected_steps),
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": jarvis_resource_id,
            "estimated_hourly_cost": None,
        },
    }


def _verify_training_base(run_manifest: Mapping[str, Any]) -> None:
    base = run_manifest.get("base_checkpoint")
    if not isinstance(base, Mapping):
        raise TemporalExecutionError("trainer run manifest lacks base provenance")
    source = base.get("source")
    if not isinstance(source, Mapping) or source.get("kind") != "huggingface":
        raise TemporalExecutionError("fit did not initialize from pinned Hugging Face base")
    if (
        source.get("repo_id") != EXPECTED_BASE_REPO
        or source.get("revision") != EXPECTED_BASE_REVISION
    ):
        raise TemporalExecutionError("fit initialized from the wrong base revision")
    if base.get("file_sha256") != EXPECTED_BASE_HASHES:
        raise TemporalExecutionError("fit base checkpoint hashes changed")
    counts = base.get("parameter_counts")
    if not isinstance(counts, Mapping) or counts.get("total") != EXPECTED_PARAMETER_COUNT:
        raise TemporalExecutionError("fit base parameter count changed")


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    manifest = _load_json(checkpoint / "checkpoint_manifest.json", label="checkpoint manifest")
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        raise TemporalExecutionError("checkpoint manifest lacks file hashes")
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        if not isinstance(hashes.get(name), str) or sha256_file(checkpoint / name) != hashes[name]:
            raise TemporalExecutionError(f"final checkpoint failed hash verification: {name}")
    return {str(name): str(digest) for name, digest in hashes.items()}


def _validate_decoding_contract(frozen: Mapping[str, Any]) -> None:
    expected = {
        "algorithm": "unconstrained_greedy",
        "batch_size": GENERATION_BATCH_SIZE,
        "shadow_max_new_tokens": SHADOW_MAX_NEW_TOKENS,
        "terminal_max_new_tokens": TERMINAL_MAX_NEW_TOKENS,
        "stop_on_eos": True,
        "runtime_gold_target_length_used_to_set_limit": False,
        "shadow_cap_selection_policy": (
            "preregistered_static_constant_chosen_before_predictions_to_exceed_the_frozen_shadow_length_audit"
        ),
        "terminal_cap_selection_policy": (
            "fixed_192_token_candidate_v2_and_qwen_matched_protocol_with_zero_truncations_required"
        ),
    }
    if frozen.get("decoding_contract") != expected:
        raise TemporalExecutionError("frozen decoding_contract changed")


def _validate_training_contract(frozen: Mapping[str, Any]) -> None:
    contract = frozen.get("training_contract")
    required = {
        "screening_selection_labels_read_for_diagnostic_loss": True,
        "screening_development_loss_used_for_selection": False,
        "screening_checkpoint_choice": "unconditional_final_only",
        "conditional_full_refit_mode": "completion_only_no_development_manifest_read",
        "conditional_full_refit_development_labels_read": 0,
        "early_stopping": False,
        "attention_backend": "deterministic_math_sdpa_only",
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(name) != value for name, value in required.items()
    ):
        raise TemporalExecutionError("frozen training_contract changed")


def _validate_token_length_audit(
    frozen: Mapping[str, Any], execution_plan: Mapping[str, Any]
) -> None:
    audit = execution_plan.get("token_length_audit")
    if not isinstance(audit, Mapping) or frozen.get("token_length_audit") != audit:
        raise TemporalExecutionError("config/plan token_length_audit binding changed")
    populations = audit.get("populations")
    expected = {
        "selection": (1_024, 445, 194, 195, 594),
        "confirmation": (1_168, 471, 192, 193, 637),
        "construction": (5_745, 487, 196, 197, 660),
        "repeat": (6_838, 487, 196, 197, 660),
        "mbcf": (6_838, 487, 196, 197, 660),
        "conditional_full_refit": (9_483, 487, 196, 197, 660),
    }
    if (
        audit.get("schema_version") != "barun-mobile-temporal-token-length-audit-v1"
        or audit.get("tokenizer_sha256") != EXPECTED_BASE_HASHES["tokenizer.json"]
        or audit.get("max_seq_len") != 2_048
        or audit.get("eos_tokens") != 1
        or audit.get("role")
        != "pre_score_evidence_used_to_choose_a_static_nontruncating_cap_never_derived_at_runtime"
        or not isinstance(populations, Mapping)
        or set(populations) != set(expected)
    ):
        raise TemporalExecutionError("frozen token-length audit metadata changed")
    for name, values in expected.items():
        row = populations.get(name)
        if (
            not isinstance(row, Mapping)
            or (
                row.get("rows"),
                row.get("max_prompt_tokens"),
                row.get("max_target_tokens"),
                row.get("max_target_plus_eos_tokens"),
                row.get("max_full_sequence_tokens"),
            )
            != values
            or row.get("rows_over_max_seq_len") != 0
        ):
            raise TemporalExecutionError(f"frozen token-length audit changed for {name}")


def _validate_retry_policy(frozen: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = frozen.get("retry_policy")
    expected_lock = (
        "forbidden after any usable held-out signal, including a selection diagnostic loss, "
        "selection/confirmation/terminal prediction, or action-evaluation score; a retry is "
        "allowed only when the previous attempt produced no usable held-out signal of any kind"
    )
    expected_execution = (
        "fresh_instance_new_output_directory_new_attempt_id_incremented_ordinal_and_prior_zero_"
        "signal_receipt_only; safe_run_owned_resume_retry_forbidden"
    )
    if (
        not isinstance(policy, Mapping)
        or policy.get("retry_after_any_held_out_signal") != expected_lock
        or policy.get("retry_execution_policy") != expected_execution
        or policy.get("scientific_change_on_retry") != "forbidden"
    ):
        raise TemporalExecutionError("frozen held-out retry lock changed")
    return policy


def _max_new_tokens_for_phase(phase: str) -> int:
    if phase in {"selection", "confirmation"}:
        return SHADOW_MAX_NEW_TOKENS
    if phase == "terminal":
        return TERMINAL_MAX_NEW_TOKENS
    raise TemporalExecutionError(f"unsupported decoding phase {phase!r}")


def _ratio(value: object, *, label: str) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or type(value.get("numerator")) is not int
        or type(value.get("denominator")) is not int
        or int(value["denominator"]) <= 0
    ):
        raise TemporalExecutionError(f"{label} is not an exact ratio")
    return {"numerator": int(value["numerator"]), "denominator": int(value["denominator"])}


def _gate_thresholds(frozen: Mapping[str, Any], *, confirmation: bool) -> dict[str, Any]:
    name = "confirmation_gate" if confirmation else "selection_gate"
    gate = frozen.get(name)
    if not isinstance(gate, Mapping) or gate.get("hard_conjunction") is not True:
        raise TemporalExecutionError(f"{name} is missing its hard conjunction")
    same_loss = _ratio(
        gate.get("maximum_mean_calendar_same_month_loss"),
        label=f"{name}.maximum_mean_calendar_same_month_loss",
    )
    non_calendar_loss = _ratio(
        gate.get("maximum_mean_non_calendar_loss"),
        label=f"{name}.maximum_mean_non_calendar_loss",
    )
    if same_loss != non_calendar_loss:
        raise TemporalExecutionError("v1 evaluator requires equal frozen subgroup loss caps")
    thresholds: dict[str, Any] = {
        "target_subset": CALENDAR_CROSS_MONTH_SUBSET,
        "target_metric": CALENDAR_DATETIME_EXACT_METRIC,
        "overall_metric": AST_EXACT_METRIC,
        "regression_subsets": [CALENDAR_SAME_MONTH_SUBSET, NON_CALENDAR_SUBSET],
        "regression_metrics": {
            CALENDAR_SAME_MONTH_SUBSET: CALENDAR_DATETIME_EXACT_METRIC,
            NON_CALENDAR_SUBSET: AST_EXACT_METRIC,
        },
        "minimum_target_mean_gain": _ratio(
            gate.get("minimum_mean_calendar_cross_month_gain"),
            label=f"{name}.minimum_mean_calendar_cross_month_gain",
        ),
        "minimum_overall_mean_gain": _ratio(
            gate.get("minimum_mean_overall_ast_exact_gain"),
            label=f"{name}.minimum_mean_overall_ast_exact_gain",
        ),
        "minimum_positive_seed_count": gate.get(
            "minimum_positive_seed_matched_comparisons_out_of_three"
        ),
        "maximum_regression_mean_loss": same_loss,
        "minimum_candidate_parse_valid": _ratio(
            gate.get("minimum_pooled_candidate_parse_valid"),
            label=f"{name}.minimum_pooled_candidate_parse_valid",
        ),
        "minimum_candidate_schema_valid": _ratio(
            gate.get("minimum_pooled_candidate_schema_valid"),
            label=f"{name}.minimum_pooled_candidate_schema_valid",
        ),
    }
    for key in (
        "maximum_missing_predictions",
        "maximum_generation_failures",
        "maximum_truncations",
        "maximum_catastrophic_unauthorized_actions",
    ):
        value = gate.get(key)
        if type(value) is not int or value < 0:
            raise TemporalExecutionError(f"{name}.{key} must be non-negative")
        thresholds[key] = value
    if confirmation:
        bootstrap = gate.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise TemporalExecutionError("confirmation bootstrap contract is missing")
        thresholds["bootstrap_lower_bound_strictly_greater_than"] = _ratio(
            bootstrap.get("required_one_sided_fifth_percentile_strictly_greater_than"),
            label="confirmation bootstrap threshold",
        )
        thresholds["bootstrap_lower_probability"] = {"numerator": 1, "denominator": 20}
    if gate.get("comparisons") != list(EXPECTED_COMPARISON_NAMES):
        raise TemporalExecutionError(f"{name}.comparisons changed")
    direct = gate.get("evaluator_thresholds")
    if not isinstance(direct, Mapping):
        raise TemporalExecutionError(f"{name}.evaluator_thresholds is required")
    expected_direct = {
        "comparison_names": list(EXPECTED_COMPARISON_NAMES),
        "expected_seeds": list(EXPECTED_SEEDS),
        **thresholds,
    }
    if confirmation:
        expected_direct["bootstrap_contract"] = EXPECTED_BOOTSTRAP_CONTRACT
    if dict(direct) != expected_direct:
        raise TemporalExecutionError(
            f"{name}.evaluator_thresholds differs from the legacy preregistration mapping"
        )
    return dict(direct)


def _minimum_ratio_check(
    observed: object, minimum: object, *, label: str
) -> tuple[bool, dict[str, int]]:
    if (
        not isinstance(observed, Mapping)
        or type(observed.get("numerator")) is not int
        or type(observed.get("denominator")) is not int
    ):
        raise TemporalExecutionError(f"terminal observation {label} is not an exact ratio")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes, bytearray))
        or len(minimum) != 2
        or any(type(value) is not int for value in minimum)
    ):
        raise TemporalExecutionError(f"terminal threshold {label} is not [numerator, denominator]")
    numerator = int(observed["numerator"])
    denominator = int(observed["denominator"])
    minimum_numerator, expected_denominator = (int(minimum[0]), int(minimum[1]))
    if denominator != expected_denominator:
        raise TemporalExecutionError(
            f"terminal {label} denominator changed: {denominator} != {expected_denominator}"
        )
    return numerator >= minimum_numerator, {
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate_terminal_veto(
    *,
    frozen_config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    temporal_subsets: Mapping[str, Any],
    sample_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply every frozen reused-756 compatibility threshold without repair."""

    gate = frozen_config.get("terminal_compatibility_veto")
    if not isinstance(gate, Mapping):
        raise TemporalExecutionError("terminal_compatibility_veto is missing")
    if gate.get("post_score_recipe_or_threshold_change") != "forbidden":
        raise TemporalExecutionError("terminal post-score mutation policy changed")
    if aggregate.get("sample_count") != PINNED_REUSED_756_ROWS:
        raise TemporalExecutionError("terminal aggregate does not contain exactly 756 rows")
    if len(sample_scores) != PINNED_REUSED_756_ROWS:
        raise TemporalExecutionError("terminal sample evidence does not contain exactly 756 rows")
    sample_ids = [row.get("sample_id") for row in sample_scores]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise TemporalExecutionError("terminal sample evidence has an invalid sample ID")
    if len(set(sample_ids)) != PINNED_REUSED_756_ROWS:
        raise TemporalExecutionError("terminal sample evidence has duplicate IDs")

    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    for threshold_name, aggregate_name in (
        ("minimum_ast_exact", "ast_exact_match"),
        ("minimum_parse_valid", "parse_valid"),
        ("minimum_schema_valid", "schema_valid"),
    ):
        passed, value = _minimum_ratio_check(
            aggregate.get(aggregate_name), gate.get(threshold_name), label=aggregate_name
        )
        checks[f"{aggregate_name}_at_least_minimum"] = passed
        observed[aggregate_name] = value

    subsets = temporal_subsets.get("subsets")
    if not isinstance(subsets, Mapping):
        raise TemporalExecutionError("terminal temporal score lacks subsets")
    cross = subsets.get(CALENDAR_CROSS_MONTH_SUBSET)
    if not isinstance(cross, Mapping):
        raise TemporalExecutionError("terminal temporal score lacks cross-month rows")
    passed, value = _minimum_ratio_check(
        cross.get(CALENDAR_DATETIME_EXACT_METRIC),
        gate.get("minimum_cross_month_calendar_datetime_exact"),
        label="cross_month_calendar_datetime_exact",
    )
    checks["cross_month_calendar_datetime_exact_at_least_minimum"] = passed
    observed["cross_month_calendar_datetime_exact"] = value

    expected_scenarios = gate.get("minimum_scenario_ast_exact")
    scenarios = aggregate.get("per_scenario_success")
    if not isinstance(expected_scenarios, Mapping) or not isinstance(scenarios, Mapping):
        raise TemporalExecutionError("terminal per-scenario thresholds or observations are missing")
    if set(scenarios) != set(expected_scenarios):
        raise TemporalExecutionError("terminal scenario population changed")
    observed_scenarios: dict[str, Any] = {}
    for scenario in sorted(expected_scenarios):
        scenario_passed, scenario_value = _minimum_ratio_check(
            scenarios.get(scenario),
            expected_scenarios[scenario],
            label=f"scenario.{scenario}",
        )
        checks[f"scenario_{scenario}_at_least_minimum"] = scenario_passed
        observed_scenarios[str(scenario)] = scenario_value
    observed["per_scenario_ast_exact"] = observed_scenarios

    missing = aggregate.get("missing_prediction")
    truncation = aggregate.get("truncation")
    if not isinstance(missing, Mapping) or not isinstance(truncation, Mapping):
        raise TemporalExecutionError("terminal failure-count ratios are missing")
    counts = {
        "missing_predictions": missing.get("numerator"),
        "generation_failures": sum(
            row.get("generation_failure") is not None for row in sample_scores
        ),
        "truncations": truncation.get("numerator"),
        "catastrophic_unauthorized_actions": aggregate.get("catastrophic_unauthorized_actions"),
    }
    sample_counts = {
        "missing_predictions": sum(row.get("prediction_raw") is None for row in sample_scores),
        "truncations": sum(row.get("truncated") is True for row in sample_scores),
        "catastrophic_unauthorized_actions": sum(
            row.get("catastrophic_unauthorized_action") is True for row in sample_scores
        ),
    }
    for name, sample_count in sample_counts.items():
        if type(counts[name]) is not int or counts[name] != sample_count:
            raise TemporalExecutionError(f"terminal aggregate/sample count differs for {name}")
    maxima = {
        "missing_predictions": gate.get("maximum_missing_predictions"),
        "generation_failures": gate.get("maximum_generation_failures"),
        "truncations": gate.get("maximum_truncations"),
        "catastrophic_unauthorized_actions": gate.get("maximum_catastrophic_unauthorized_actions"),
    }
    for name, maximum in maxima.items():
        if type(maximum) is not int or maximum < 0 or type(counts[name]) is not int:
            raise TemporalExecutionError(f"terminal maximum/count is invalid for {name}")
        checks[f"{name}_at_most_maximum"] = int(counts[name]) <= maximum
    observed["failure_counts"] = counts
    return {
        "schema_version": "barun-mobile-temporal-terminal-veto-v1",
        "passed": all(checks.values()),
        "status": "pass" if all(checks.values()) else "reject",
        "checks": checks,
        "observed": observed,
        "thresholds": dict(gate),
        "post_score_training_or_repair_allowed": False,
    }


def _export_inference_checkpoint(source: Path, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        shutil.copy2(source / name, destination / name)
        hashes[name] = sha256_file(destination / name)
    _write_new_json(
        destination / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "source_checkpoint": str(source),
            "file_sha256": hashes,
        },
    )
    if any(path.name == "optimizer.pt" for path in destination.rglob("*")):
        raise TemporalExecutionError("optimizer state entered inference-only export")
    return hashes


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)


def _build_essential_bundle(*, execution_dir: Path, controller_dir: Path) -> dict[str, Any]:
    """Copy inspectable evidence while excluding screening weights and all optimizers."""

    final_essential = execution_dir / "essential"
    essential = execution_dir / f".essential-building-{os.getpid()}-{time.time_ns()}"
    essential.mkdir()
    try:
        return _populate_essential_bundle(
            execution_dir=execution_dir,
            controller_dir=controller_dir,
            essential=essential,
            final_essential=final_essential,
        )
    finally:
        if essential.exists():
            shutil.rmtree(essential)


def _populate_essential_bundle(
    *, execution_dir: Path, controller_dir: Path, essential: Path, final_essential: Path
) -> dict[str, Any]:
    controller_evidence = essential / "controller"
    controller_evidence.mkdir()
    for name in (
        "preflight-receipt.json",
        "execution-plan.json",
        "attempt-preregistration.json",
        "source-snapshot.json",
        "pre-cuda-validation.json",
        "pre-cuda-validation.log",
    ):
        source = controller_dir / name
        if source.is_file():
            shutil.copy2(source, controller_evidence / name)
    plan = _load_json(controller_dir / "execution-plan.json", label="controller execution plan")
    provenance = plan.get("launch_provenance")
    if not isinstance(provenance, Mapping):
        raise TemporalExecutionError("execution plan lacks launch provenance")
    attempt_binding = provenance.get("attempt_preregistration")
    snapshot_binding = provenance.get("source_snapshot")
    if not isinstance(attempt_binding, Mapping) or not isinstance(snapshot_binding, Mapping):
        raise TemporalExecutionError("execution plan launch provenance is incomplete")
    for name, binding in (
        ("attempt-preregistration.json", attempt_binding),
        ("source-snapshot.json", snapshot_binding),
    ):
        copied = controller_evidence / name
        if not copied.is_file() or sha256_file(copied) != binding.get("sha256"):
            raise TemporalExecutionError(f"bound launch provenance changed before export: {name}")
    validation = plan.get("pre_cuda_validation")
    validation_log = controller_evidence / "pre-cuda-validation.log"
    validation_receipt = controller_evidence / "pre-cuda-validation.json"
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "passed"
        or not validation_log.is_file()
        or sha256_file(validation_log) != validation.get("log_sha256")
        or not validation_receipt.is_file()
    ):
        raise TemporalExecutionError("bound CPU-only pre-CUDA validation evidence changed")
    for name in ("environment-receipt.json", "result.json", "failure.json"):
        source = execution_dir / name
        if source.is_file():
            shutil.copy2(source, essential / name)
    for name in (
        "gates",
        "fits",
        "training-configs",
        "conditional-full-refit",
        "terminal-compatibility-veto",
    ):
        _copy_tree(execution_dir / name, essential / name)

    training_source = execution_dir / "training"
    training_destination = essential / "training"
    if training_source.is_dir():
        training_destination.mkdir()
        allowed = {
            "data_rejections.jsonl",
            "failure.json",
            "heldout_access_started.json",
            "metrics.jsonl",
            "resolved_config.json",
            "run_manifest.json",
            "summary.json",
        }
        for run_dir in sorted(training_source.iterdir()):
            if not run_dir.is_dir():
                continue
            destination = training_destination / run_dir.name
            destination.mkdir()
            for name in sorted(allowed):
                source = run_dir / name
                if source.is_file():
                    shutil.copy2(source, destination / name)

    files = [path for path in sorted(essential.rglob("*")) if path.is_file()]
    if any(path.name == "optimizer.pt" for path in files):
        raise TemporalExecutionError("optimizer state entered compact essential evidence")
    model_paths = [
        path.relative_to(essential).as_posix() for path in files if path.name == "model.safetensors"
    ]
    allowed_model = "conditional-full-refit/inference-checkpoint/model.safetensors"
    if any(path != allowed_model for path in model_paths) or len(model_paths) > 1:
        raise TemporalExecutionError(
            "screening or unexpected model weights entered essential evidence"
        )
    entries = {
        path.relative_to(essential).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    manifest = {
        "schema_version": "barun-mobile-temporal-essential-v1",
        "created_at": _utc_now(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries.values()),
        "optimizer_files": 0,
        "screening_model_weight_files": 0,
        "promoted_model_weight_files": len(model_paths),
        "files": entries,
    }
    _write_new_json(essential / "artifact-manifest.json", manifest)
    try:
        if final_essential.exists():
            if (final_essential / "artifact-manifest.json").is_file():
                raise TemporalExecutionError("refusing to replace a completed essential bundle")
            quarantine = execution_dir / "essential-prior-incomplete"
            if quarantine.exists():
                raise TemporalExecutionError(
                    "both partial essential and prior-incomplete quarantine exist"
                )
            final_essential.replace(quarantine)
        essential.replace(final_essential)
        return {
            "path": str(final_essential),
            "manifest": str(final_essential / "artifact-manifest.json"),
            "manifest_sha256": sha256_file(final_essential / "artifact-manifest.json"),
            "file_count": len(entries),
            "total_bytes": manifest["total_bytes"],
        }
    finally:
        if essential.exists():
            shutil.rmtree(essential)


def _usable_held_out_artifacts(execution_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for path in sorted(execution_dir.rglob("*")):
        if (
            path.is_file()
            and path.stat().st_size > 0
            and path.name
            in {
                "predictions.jsonl",
                "sample_scores.jsonl",
                "heldout_access_started.json",
                "selection-heldout-access-started.json",
                "terminal-heldout-access-started.json",
            }
            and (
                path.name
                in {"heldout_access_started.json", "selection-heldout-access-started.json"}
                or any(
                    part in {"selection", "confirmation", "terminal-compatibility-veto"}
                    for part in path.parts
                )
            )
        ):
            artifacts.append(path.relative_to(execution_dir).as_posix())
        elif path.is_file() and path.name == "metrics.jsonl" and path.stat().st_size > 0:
            try:
                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
                rows = [json.loads(line) for line in lines]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                # Uncertainty must fail closed: a torn metric write may already contain a
                # selection loss even when the final JSON line cannot be parsed.
                artifacts.append(path.relative_to(execution_dir).as_posix())
                continue
            known_events = {
                "checkpoint",
                "complete",
                "completion_only_contract",
                "dev",
                "resume",
                "train",
            }
            if any(
                not isinstance(row, Mapping)
                or row.get("event") not in known_events
                or row.get("event") == "dev"
                for row in rows
            ):
                artifacts.append(path.relative_to(execution_dir).as_posix())
    return artifacts


def _data_firewall_receipt(
    *, terminal: object, reused_rows_read: int, reused_rows_scored: int
) -> dict[str, Any]:
    if reused_rows_read not in {0, PINNED_REUSED_756_ROWS} or reused_rows_scored not in {
        0,
        PINNED_REUSED_756_ROWS,
    }:
        raise TemporalExecutionError("reused-756 firewall counts are invalid")
    if reused_rows_scored > reused_rows_read:
        raise TemporalExecutionError("reused-756 scored rows exceed read rows")
    if terminal is None:
        if reused_rows_read or reused_rows_scored:
            raise TemporalExecutionError("terminal rows were accessed without a terminal result")
        terminal_status = "not_reached_due_to_shadow_gate"
    else:
        if not isinstance(terminal, Mapping):
            raise TemporalExecutionError("terminal result is not an object")
        terminal_status = str(terminal.get("status"))
    return {
        "official_961_rows_read": 0,
        "official_961_rows_scored": 0,
        "reused_756_rows_read": reused_rows_read,
        "reused_756_rows_scored": reused_rows_scored,
        "terminal_compatibility_veto": terminal_status,
    }


def run_gated_schedule(
    execution_plan: Mapping[str, Any], operations: TemporalPhaseOperations
) -> dict[str, Any]:
    """Execute the fixed phase graph; gate failures are successful terminal outcomes."""

    raw_fits = execution_plan.get("fits")
    if not isinstance(raw_fits, Sequence) or len(raw_fits) != 9:
        raise TemporalExecutionError("execution plan must contain exactly nine screening fits")
    fits = list(raw_fits)
    identities = [(fit.get("arm_id"), fit.get("seed")) for fit in fits]
    expected = [(arm, seed) for seed in EXPECTED_SEEDS for arm in EXPECTED_ARM_IDS]
    if identities != expected:
        raise TemporalExecutionError("screening fit order or membership changed")

    trained = [operations.train_screening_fit(fit) for fit in fits]
    phases: dict[str, Any] = {}
    for phase in ("selection", "confirmation"):
        evidence: dict[str, dict[int, Sequence[Mapping[str, Any]]]] = {
            arm: {} for arm in EXPECTED_ARM_IDS
        }
        for fit, result in zip(fits, trained, strict=True):
            arm = str(fit["arm_id"])
            seed = int(fit["seed"])
            evidence[arm][seed] = operations.score_screening_fit(result, population=phase)
        gate = operations.evaluate_phase_gate(phase=phase, evidence=evidence)
        if type(gate.get("passed")) is not bool:
            raise TemporalExecutionError(f"{phase} gate lacks a strict boolean result")
        phases[phase] = gate
        if not gate["passed"]:
            return {
                "schema_version": BACKEND_VERSION,
                "status": f"stopped_after_{phase}_gate_failure",
                "screening_fit_count": len(trained),
                "phases": phases,
                "conditional_full_refit": None,
                "terminal_compatibility_veto": None,
            }

    full_spec = execution_plan.get("conditional_full_refit")
    if not isinstance(full_spec, Mapping):
        raise TemporalExecutionError("execution plan lacks conditional full-refit specification")
    full = operations.train_conditional_full_refit(full_spec)
    terminal = operations.evaluate_terminal_compatibility(full)
    if type(terminal.get("passed")) is not bool:
        raise TemporalExecutionError("terminal compatibility veto lacks a boolean result")
    return {
        "schema_version": BACKEND_VERSION,
        "status": (
            "terminal_compatibility_passed"
            if terminal["passed"]
            else "terminal_compatibility_rejected"
        ),
        "screening_fit_count": len(trained),
        "phases": phases,
        "conditional_full_refit": full,
        "terminal_compatibility_veto": terminal,
    }


class _FixedOperations:
    def __init__(
        self,
        *,
        frozen_config: Mapping[str, Any],
        execution_plan: Mapping[str, Any],
        screening_dir: Path,
        execution_dir: Path,
        jarvis_resource_id: str,
    ) -> None:
        self.frozen_config = frozen_config
        self.execution_plan = execution_plan
        self.screening_dir = screening_dir
        self.execution_dir = execution_dir
        self.jarvis_resource_id = jarvis_resource_id
        _validate_decoding_contract(frozen_config)
        _validate_training_contract(frozen_config)
        _validate_token_length_audit(frozen_config, execution_plan)
        self.retry_policy = dict(_validate_retry_policy(frozen_config))
        self.training_root = execution_dir / "training"
        self.config_root = execution_dir / "training-configs"
        self.fit_root = execution_dir / "fits"
        self.fit_root.mkdir()
        self.config_root.mkdir()
        self._manifest_paths = {
            "selection": screening_dir / str(execution_plan["selection_manifest"]),
            "confirmation": screening_dir / str(execution_plan["confirmation_manifest"]),
        }
        materialization = frozen_config.get("materialization")
        if not isinstance(materialization, Mapping):
            raise TemporalExecutionError("frozen materialization contract is missing")
        hashes = materialization.get("artifact_sha256")
        if not isinstance(hashes, Mapping):
            raise TemporalExecutionError("frozen materialization hashes are missing")
        self._manifest_hashes = {
            phase: str(hashes[phase]) for phase in ("selection", "confirmation")
        }
        terminal = execution_plan.get("terminal_compatibility_veto")
        if not isinstance(terminal, Mapping):
            raise TemporalExecutionError("execution plan lacks terminal compatibility binding")
        if (
            terminal.get("status") != "bound_for_conditional_terminal_read"
            or terminal.get("manifest_sha256") != PINNED_REUSED_756_SHA256
            or terminal.get("rows") != PINNED_REUSED_756_ROWS
            or terminal.get("max_new_tokens") != TERMINAL_MAX_NEW_TOKENS
            or not isinstance(terminal.get("manifest"), str)
        ):
            raise TemporalExecutionError("terminal reused-756 binding changed or is incomplete")
        self._terminal_manifest = Path(str(terminal["manifest"])).resolve()
        self.terminal_rows_read = 0
        self.terminal_rows_scored = 0
        self._fit_results: dict[tuple[str, int], dict[str, Any]] = {}

    def train_screening_fit(self, fit: Mapping[str, Any]) -> Mapping[str, Any]:
        arm = str(fit["arm_id"])
        seed = int(fit["seed"])
        fit_dir = self.fit_root / f"{arm.lower()}-s{seed}"
        fit_dir.mkdir()
        train_manifest = self.screening_dir / str(fit["train_manifest"])
        payload = build_screening_training_payload(
            frozen_config=self.frozen_config,
            fit=fit,
            train_manifest=train_manifest,
            selection_manifest=self._manifest_paths["selection"],
            selection_sha256=self._manifest_hashes["selection"],
            output_root=self.training_root,
            jarvis_resource_id=self.jarvis_resource_id,
        )
        config_path = self.config_root / f"{fit['run_id']}.json"
        _write_new_json(config_path, payload)
        config = TrainingRunConfig.from_json(config_path)
        _write_new_json(
            fit_dir / "selection-heldout-access-started.json",
            {
                "schema_version": "barun-heldout-access-marker-v1",
                "population": "selection",
                "reason": "trainer_dev_manifest_access_imminent",
                "retry_lock": "forbidden",
            },
        )
        summary = train_sft(config, diagnostic_dev_only=True)
        if summary.global_steps != fit["expected_optimizer_steps"]:
            raise TemporalExecutionError(
                f"fit {arm}/s{seed} completed {summary.global_steps} optimizer steps"
            )
        if summary.status != "max_epochs":
            raise TemporalExecutionError(f"fit {arm}/s{seed} stopped as {summary.status!r}")
        if (
            summary.initial_dev_loss is None
            or summary.final_dev_loss is None
            or summary.best_dev_loss is not None
            or summary.best_checkpoint is not None
        ):
            raise TemporalExecutionError(
                f"fit {arm}/s{seed} did not preserve diagnostic-only dev-loss semantics"
            )
        final_checkpoint = Path(summary.final_checkpoint).resolve()
        # Development loss is diagnostic only.  The final checkpoint is unconditional.
        if final_checkpoint.name != f"step-{summary.global_steps:08d}":
            raise TemporalExecutionError("trainer did not return the unconditional final step")
        checkpoint_hashes = _checkpoint_hashes(final_checkpoint)
        run_manifest_path = Path(summary.run_dir) / "run_manifest.json"
        run_manifest = _load_json(run_manifest_path, label="training run manifest")
        _verify_training_base(run_manifest)
        if (
            run_manifest.get("training_mode") != "diagnostic_dev_loss_unconditional_final"
            or run_manifest.get("checkpoint_policy") != "unconditional_final_only"
            or (Path(summary.run_dir) / "best_checkpoint.json").exists()
        ):
            raise TemporalExecutionError("screening trainer used diagnostic loss for selection")
        receipt = {
            "schema_version": "barun-mobile-temporal-fit-receipt-v1",
            "arm_id": arm,
            "seed": seed,
            "run_id": fit["run_id"],
            "base_checkpoint": {
                "repo_id": EXPECTED_BASE_REPO,
                "revision": EXPECTED_BASE_REVISION,
                "file_sha256": EXPECTED_BASE_HASHES,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
            },
            "train_manifest": str(train_manifest),
            "train_sha256": fit["train_sha256"],
            "train_rows": fit["train_rows"],
            "expected_optimizer_steps": fit["expected_optimizer_steps"],
            "completed_optimizer_steps": summary.global_steps,
            "selection_dev_loss_role": "diagnostic_only",
            "checkpoint_selection": "unconditional_final_checkpoint",
            "best_checkpoint_consumed": False,
            "final_checkpoint": str(final_checkpoint),
            "checkpoint_file_sha256": checkpoint_hashes,
            "training_config": str(config_path),
            "training_summary": asdict(summary),
        }
        _write_new_json(fit_dir / "training-receipt.json", receipt)
        result = {**receipt, "fit_dir": str(fit_dir)}
        self._fit_results[(arm, seed)] = result
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def score_screening_fit(
        self, fit_result: Mapping[str, Any], *, population: str
    ) -> Sequence[Mapping[str, Any]]:
        if population not in {"selection", "confirmation"}:
            raise TemporalExecutionError(f"unsupported evaluation population {population!r}")
        manifest = self._manifest_paths[population]
        manifest_sha256 = self._manifest_hashes[population]
        if sha256_file(manifest) != manifest_sha256:
            raise TemporalExecutionError(f"{population} manifest changed before scoring")
        fit_dir = Path(str(fit_result["fit_dir"]))
        phase_dir = fit_dir / population
        phase_dir.mkdir()
        checkpoint = Path(str(fit_result["final_checkpoint"]))
        predictions = phase_dir / "predictions.jsonl"
        generation = generate_manifest(
            checkpoint_dir=checkpoint,
            manifest_path=manifest,
            manifest_sha256=manifest_sha256,
            predictions_path=predictions,
            device_name="cuda",
            batch_size=GENERATION_BATCH_SIZE,
            max_new_tokens=_max_new_tokens_for_phase(population),
        )
        score_paths = write_scores(manifest, predictions, phase_dir / "scores")
        manifest_rows = read_jsonl(manifest)
        score_rows = read_jsonl(score_paths["samples"])
        temporal = score_temporal_subsets(manifest_rows, score_rows)
        _write_new_json(phase_dir / "temporal-subsets.json", temporal)
        _write_new_json(
            phase_dir / "evaluation-receipt.json",
            {
                "schema_version": "barun-mobile-temporal-fit-evaluation-v1",
                "population": population,
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha256,
                "generation": generation.to_dict(),
                "decoding_contract": self.frozen_config["decoding_contract"],
                "effective_max_new_tokens": _max_new_tokens_for_phase(population),
                "sample_scores": str(score_paths["samples"]),
                "sample_scores_sha256": sha256_file(score_paths["samples"]),
                "aggregate": str(score_paths["aggregate"]),
                "aggregate_sha256": sha256_file(score_paths["aggregate"]),
                "temporal_subsets": str(phase_dir / "temporal-subsets.json"),
            },
        )
        gc.collect()
        torch.cuda.empty_cache()
        return score_rows

    def evaluate_phase_gate(
        self,
        *,
        phase: str,
        evidence: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    ) -> Mapping[str, Any]:
        if phase not in {"selection", "confirmation"}:
            raise TemporalExecutionError(f"unsupported gate phase {phase!r}")
        manifest_path = self._manifest_paths[phase]
        manifest_rows = read_jsonl(manifest_path)
        comparisons = {
            "C_minus_A": compare_seeded_arms(
                manifest_rows,
                evidence["C"],
                evidence["A"],
                expected_seeds=EXPECTED_SEEDS,
            ),
            "C_minus_B": compare_seeded_arms(
                manifest_rows,
                evidence["C"],
                evidence["B"],
                expected_seeds=EXPECTED_SEEDS,
            ),
        }
        phase_dir = self.execution_dir / "gates" / phase
        phase_dir.mkdir(parents=True)
        _write_new_json(phase_dir / "comparisons.json", comparisons)
        bootstraps: dict[str, Mapping[str, Any]] | None = None
        if phase == "confirmation":
            gate_config = self.frozen_config["confirmation_gate"]
            bootstrap_config = gate_config["bootstrap"]
            if (
                bootstrap_config.get("resamples") != 10_000
                or bootstrap_config.get("rng") != "NumPy PCG64"
                or bootstrap_config.get("seed") != 17
            ):
                raise TemporalExecutionError("confirmation bootstrap recipe changed")
            direct = gate_config.get("evaluator_thresholds")
            if (
                not isinstance(direct, Mapping)
                or direct.get("bootstrap_contract") != EXPECTED_BOOTSTRAP_CONTRACT
            ):
                raise TemporalExecutionError("confirmation direct bootstrap contract changed")
            bootstraps = {
                "C_minus_A": paired_cluster_bootstrap(
                    manifest_rows,
                    evidence["C"],
                    evidence["A"],
                    subset=CALENDAR_CROSS_MONTH_SUBSET,
                    metric=CALENDAR_DATETIME_EXACT_METRIC,
                    expected_seeds=EXPECTED_SEEDS,
                    resamples=10_000,
                    rng_seed=17,
                ),
                "C_minus_B": paired_cluster_bootstrap(
                    manifest_rows,
                    evidence["C"],
                    evidence["B"],
                    subset=CALENDAR_CROSS_MONTH_SUBSET,
                    metric=CALENDAR_DATETIME_EXACT_METRIC,
                    expected_seeds=EXPECTED_SEEDS,
                    resamples=10_000,
                    rng_seed=17,
                ),
            }
            _write_new_json(phase_dir / "bootstraps.json", bootstraps)
        thresholds = _gate_thresholds(self.frozen_config, confirmation=phase == "confirmation")
        gate = evaluate_temporal_gates(
            comparisons,
            thresholds=thresholds,
            bootstraps=bootstraps,
        )
        _write_new_json(phase_dir / "gate.json", gate)
        return gate

    def train_conditional_full_refit(self, specification: Mapping[str, Any]) -> Mapping[str, Any]:
        if specification.get("status") != "conditional_disabled_until_confirmation_gate_passes":
            raise TemporalExecutionError("conditional full-refit plan status changed")
        full_dir = self.execution_dir / "conditional-full-refit"
        full_dir.mkdir()
        train_manifest = (self.screening_dir / str(specification["train_manifest"])).resolve()
        no_dev_sentinel = full_dir / "NO-DEVELOPMENT-LABELS-ACCESSED.jsonl"
        if no_dev_sentinel.exists():
            raise TemporalExecutionError("no-development sentinel unexpectedly exists")
        payload = _build_full_refit_payload(
            frozen_config=self.frozen_config,
            specification=specification,
            train_manifest=train_manifest,
            output_root=self.training_root,
            no_dev_sentinel=no_dev_sentinel,
            jarvis_resource_id=self.jarvis_resource_id,
        )
        config_path = self.config_root / f"{specification['run_id']}.json"
        _write_new_json(config_path, payload)
        config = TrainingRunConfig.from_json(config_path)
        summary: TrainingSummary = train_sft(config, completion_only=True)
        if summary.global_steps != specification["expected_optimizer_steps"]:
            raise TemporalExecutionError("conditional full refit completed the wrong step count")
        if any(
            value is not None
            for value in (
                summary.initial_dev_loss,
                summary.best_dev_loss,
                summary.final_dev_loss,
                summary.best_checkpoint,
            )
        ):
            raise TemporalExecutionError("completion-only refit emitted development selection")
        run_dir = Path(summary.run_dir)
        run_manifest = _load_json(run_dir / "run_manifest.json", label="full-refit run manifest")
        _verify_training_base(run_manifest)
        data_receipt = run_manifest.get("data")
        if (
            run_manifest.get("training_mode") != "completion_only_no_dev"
            or run_manifest.get("checkpoint_policy") != "unconditional_final_only"
            or not isinstance(data_receipt, Mapping)
            or data_receipt.get("dev_labels_read") is not False
            or data_receipt.get("dev") is not None
            or (run_dir / "best_checkpoint.json").exists()
            or no_dev_sentinel.exists()
        ):
            raise TemporalExecutionError("full-refit zero-development-read receipt failed")
        final_checkpoint = Path(summary.final_checkpoint).resolve()
        checkpoint_hashes = _checkpoint_hashes(final_checkpoint)
        export_hashes = _export_inference_checkpoint(
            final_checkpoint, full_dir / "inference-checkpoint"
        )
        receipt = {
            "schema_version": "barun-mobile-temporal-full-refit-receipt-v1",
            "run_id": specification["run_id"],
            "seed": 17,
            "train_manifest": str(train_manifest),
            "train_sha256": specification["train_sha256"],
            "train_rows": specification["train_rows"],
            "completed_optimizer_steps": summary.global_steps,
            "checkpoint_policy": "unconditional_final_only",
            "development_labels_read": 0,
            "selection_rows_read": 0,
            "confirmation_rows_read": 0,
            "reused_756_rows_read": 0,
            "official_961_rows_read": 0,
            "final_checkpoint": str(final_checkpoint),
            "checkpoint_file_sha256": checkpoint_hashes,
            "inference_checkpoint": str(full_dir / "inference-checkpoint"),
            "inference_file_sha256": export_hashes,
            "training_summary": asdict(summary),
        }
        _write_new_json(full_dir / "full-refit-receipt.json", receipt)
        return receipt

    def evaluate_terminal_compatibility(self, full_refit: Mapping[str, Any]) -> Mapping[str, Any]:
        """Score the reused-756 veto once, only after the full refit is immutable."""

        terminal_dir = self.execution_dir / "terminal-compatibility-veto"
        terminal_dir.mkdir()
        manifest = self._terminal_manifest
        if not manifest.is_file():
            raise TemporalExecutionError(f"terminal reused-756 manifest is missing: {manifest}")
        _write_new_json(
            terminal_dir / "terminal-heldout-access-started.json",
            {
                "schema_version": "barun-heldout-access-marker-v1",
                "population": "reused_756_terminal",
                "reason": "manifest_hash_read_imminent",
                "retry_lock": "forbidden",
            },
        )
        # Hashing opens the sealed population. Account for that access conservatively before
        # the first byte is read so even a hash/read failure cannot produce a false zero.
        self.terminal_rows_read = PINNED_REUSED_756_ROWS
        if sha256_file(manifest) != PINNED_REUSED_756_SHA256:
            raise TemporalExecutionError("terminal reused-756 manifest failed pinned SHA-256")
        checkpoint_value = full_refit.get("inference_checkpoint")
        if not isinstance(checkpoint_value, str):
            raise TemporalExecutionError("full refit lacks its inference-only checkpoint")
        checkpoint = Path(checkpoint_value).resolve()
        predictions = terminal_dir / "predictions.jsonl"
        generation = generate_manifest(
            checkpoint_dir=checkpoint,
            manifest_path=manifest,
            manifest_sha256=PINNED_REUSED_756_SHA256,
            predictions_path=predictions,
            device_name="cuda",
            batch_size=GENERATION_BATCH_SIZE,
            max_new_tokens=_max_new_tokens_for_phase("terminal"),
        )
        if generation.examples != PINNED_REUSED_756_ROWS:
            raise TemporalExecutionError("terminal generation did not contain exactly 756 rows")
        score_paths = write_scores(manifest, predictions, terminal_dir / "scores")
        manifest_rows = read_jsonl(manifest)
        sample_scores = read_jsonl(score_paths["samples"])
        temporal = score_temporal_subsets(manifest_rows, sample_scores)
        _write_new_json(terminal_dir / "temporal-subsets.json", temporal)
        aggregate = _load_json(score_paths["aggregate"], label="terminal aggregate")
        gate = evaluate_terminal_veto(
            frozen_config=self.frozen_config,
            aggregate=aggregate,
            temporal_subsets=temporal,
            sample_scores=sample_scores,
        )
        self.terminal_rows_scored = PINNED_REUSED_756_ROWS
        result = {
            **gate,
            "population": "pinned_reused_756",
            "logical_population_accesses": 1,
            "manifest": str(manifest),
            "manifest_sha256": PINNED_REUSED_756_SHA256,
            "rows_read": PINNED_REUSED_756_ROWS,
            "rows_scored": PINNED_REUSED_756_ROWS,
            "decoding_contract": self.frozen_config["decoding_contract"],
            "effective_max_new_tokens": _max_new_tokens_for_phase("terminal"),
            "generation": generation.to_dict(),
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "sample_scores": str(score_paths["samples"]),
            "sample_scores_sha256": sha256_file(score_paths["samples"]),
            "aggregate": str(score_paths["aggregate"]),
            "aggregate_sha256": sha256_file(score_paths["aggregate"]),
            "temporal_subsets": str(terminal_dir / "temporal-subsets.json"),
            "official_961_rows_read": 0,
            "training_or_checkpoint_mutation_after_score": False,
        }
        _write_new_json(terminal_dir / "terminal-result.json", result)
        gc.collect()
        torch.cuda.empty_cache()
        return result


def execute_temporal_experiment(
    *,
    config_path: Path,
    materialization_receipt: Path,
    output_dir: Path,
    preflight_receipt: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    jarvis_resource_id: str,
    cublas_configured_before_torch_import: bool,
) -> dict[str, Any]:
    """Run the frozen experiment locally on an already provisioned exact Jarvis ID."""

    resource_id = _validate_jarvis_resource_id(jarvis_resource_id)
    execution_dir = output_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=False)
    operations: _FixedOperations | None = None
    try:
        determinism = _cuda_preflight(
            configured_before_torch_import=cublas_configured_before_torch_import
        )
        _write_new_json(
            execution_dir / "environment-receipt.json",
            {
                "schema_version": "barun-mobile-temporal-environment-v1",
                "captured_at": _utc_now(),
                "jarvis_resource_id": resource_id,
                "device": "cuda",
                "precision": "bf16",
                "determinism": determinism,
                "wandb": "disabled",
                "automatic_retry": False,
            },
        )
        frozen = _load_json(config_path, label="frozen experiment config")
        if execution_plan.get("execution_backend") != BACKEND_VERSION:
            raise TemporalExecutionError("execution plan does not name the frozen backend")
        if preflight_receipt.get("model_or_cuda_loaded") is not False:
            raise TemporalExecutionError(
                "stdlib preflight does not prove zero prior model/CUDA use"
            )
        operations = _FixedOperations(
            frozen_config=frozen,
            execution_plan=execution_plan,
            screening_dir=materialization_receipt.parent,
            execution_dir=execution_dir,
            jarvis_resource_id=resource_id,
        )
        result = run_gated_schedule(execution_plan, operations)
        terminal = result.get("terminal_compatibility_veto")
        result = {
            **result,
            "completed_at": _utc_now(),
            "jarvis_resource_id": resource_id,
            "retry_policy": operations.retry_policy,
            "automatic_retry_attempted": False,
            "essential_bundle": {
                "path": str(execution_dir / "essential"),
                "manifest": str(execution_dir / "essential" / "artifact-manifest.json"),
                "screening_weights_included": False,
                "optimizer_state_included": False,
            },
            "data_firewall": _data_firewall_receipt(
                terminal=terminal,
                reused_rows_read=operations.terminal_rows_read,
                reused_rows_scored=operations.terminal_rows_scored,
            ),
        }
        _write_new_json(execution_dir / "result.json", result)
        _build_essential_bundle(execution_dir=execution_dir, controller_dir=output_dir)
        return result
    except Exception as error:
        usable_held_out = _usable_held_out_artifacts(execution_dir)
        terminal_rows_read = 0 if operations is None else operations.terminal_rows_read
        terminal_rows_scored = 0 if operations is None else operations.terminal_rows_scored
        heldout_signal = bool(usable_held_out or terminal_rows_read or terminal_rows_scored)
        failure = {
            "schema_version": "barun-mobile-temporal-execution-failure-v1",
            "failed_at": _utc_now(),
            "jarvis_resource_id": resource_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "artifacts_preserved": True,
            "automatic_retry_attempted": False,
            "usable_held_out_artifacts": usable_held_out,
            "retry_after_this_attempt": (
                "forbidden" if heldout_signal else "external_controller_must_verify_eligibility"
            ),
            "official_961_rows_read": 0,
            "reused_756_rows_read": terminal_rows_read,
            "reused_756_rows_scored": terminal_rows_scored,
        }
        failure_path = execution_dir / "failure.json"
        if not failure_path.exists():
            _write_new_json(failure_path, failure)
        if not (execution_dir / "essential" / "artifact-manifest.json").is_file():
            try:
                _build_essential_bundle(execution_dir=execution_dir, controller_dir=output_dir)
            except (OSError, TemporalExecutionError, ValueError, KeyError) as bundle_error:
                bundle_failure = execution_dir / "essential-bundle-failure.json"
                if not bundle_failure.exists():
                    _write_new_json(
                        bundle_failure,
                        {
                            "schema_version": "barun-essential-bundle-failure-v1",
                            "error_type": type(bundle_error).__name__,
                            "message": str(bundle_error),
                        },
                    )
        raise


__all__ = [
    "BACKEND_VERSION",
    "DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG",
    "EXPECTED_BOOTSTRAP_CONTRACT",
    "TemporalExecutionError",
    "build_screening_training_payload",
    "execute_temporal_experiment",
    "run_gated_schedule",
]
