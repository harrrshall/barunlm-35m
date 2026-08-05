"""Run the two preregistered Mobile Actions follow-up arms on one fresh GPU instance."""

from __future__ import annotations

# This must precede torch and every project import that can transitively import torch.
import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT = (
    os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
)

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from barunlm.datasets.mobile_actions import (
    MOBILE_ACTIONS_REVISION,
    TokenizerIdentity,
    download_pinned_source,
    prepare_mobile_actions,
)
from barunlm.evaluation.generation import generate_manifest, verify_checkpoint
from barunlm.evaluation.mobile_actions import write_scores
from barunlm.training import TrainingRunConfig, train_sft
from barunlm.training.data import sha256_file

BASE_REPO = "harrrshall/BarunLM-35M"
BASE_REVISION = "ef3e483a9fd7d906ecf2a7929babeffaf82d1d16"
BASE_HASHES = {
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "model.safetensors": "f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6",
}
PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "mobile_followup_sweep_v1.json"
)
PREREGISTRATION_SHA256 = "67396bf8e40d8076a54232946eb7817d34a7239d172cf32782fd5f5989172d92"
EXPECTED_ARM_IDS = ("batch63", "hardmix70")
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")


class ExistingSweepError(FileExistsError):
    """Raised when an immutable sweep ID already owns its artifact directory."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def _export_inference_checkpoint(source: Path, destination: Path, *, arm_id: str) -> None:
    """Copy only inference files and issue a manifest that does not require optimizer state."""

    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        source_path = source / name
        target = destination / name
        shutil.copy2(source_path, target)
        hashes[name] = sha256_file(target)
    _write_json(
        destination / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "arm_id": arm_id,
            "source_checkpoint": str(source),
            "file_sha256": hashes,
        },
    )


def _export_arm_essential(
    *, arm_id: str, arm_dir: Path, final_checkpoint: Path, essential: Path
) -> None:
    destination = essential / "arms" / arm_id
    destination.mkdir(parents=True, exist_ok=False)
    _export_inference_checkpoint(
        final_checkpoint,
        destination / "checkpoint",
        arm_id=arm_id,
    )
    shutil.copytree(arm_dir / "final-eval", destination / "final-eval")
    shutil.copy2(arm_dir / "result.json", destination / "result.json")
    shutil.copy2(arm_dir / "training-config.json", destination / "training-config.json")
    source_run = final_checkpoint.parent.parent
    training_evidence = destination / "training"
    training_evidence.mkdir()
    for name in ("metrics.jsonl", "run_manifest.json", "summary.json"):
        shutil.copy2(source_run / name, training_evidence / name)
    best_checkpoint = source_run / "best_checkpoint.json"
    if best_checkpoint.is_file():
        shutil.copy2(best_checkpoint, training_evidence / best_checkpoint.name)
    if any(path.name == "optimizer.pt" for path in destination.rglob("*")):
        raise RuntimeError("optimizer state entered the compact essential bundle")
    _write_json(
        destination / "bundle-manifest.json",
        {
            "schema_version": "barun-mobile-arm-essential-v1",
            "arm_id": arm_id,
            "checkpoint_policy": "final_only",
            "contents_sha256": _tree_manifest(destination),
        },
    )


def _stage_shared_essential(
    *, export: Path, essential: Path, audit_path: Path, audit_sha256: str
) -> None:
    shutil.copy2(export / "repository-tests.log", essential / "repository-tests.log")
    essential_data = essential / "data"
    essential_data.mkdir(exist_ok=False)
    shutil.copy2(audit_path, essential_data / "audit.json")
    if sha256_file(essential_data / "audit.json") != audit_sha256:
        raise RuntimeError("Mobile adapter audit changed while entering essential evidence")


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
            raise TypeError(f"PyTorch cannot configure or verify the {label} SDPA backend")
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(
                f"failed to force {label} SDPA backend to {expected}: observed {actual!r}"
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
            raise RuntimeError(f"failed to disable the cuDNN SDPA backend: observed {actual!r}")
        evidence["cudnn"] = actual
    else:
        raise TypeError("PyTorch exposes an unverifiable partial cuDNN SDPA interface")
    return evidence


def _cuda_determinism_preflight() -> dict[str, Any]:
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG changed after import: "
            f"expected {DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r}, got {actual!r}"
        )
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        raise RuntimeError("deterministic cuBLAS configuration preceded torch cannot be proved")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the deterministic sweep preflight")
    return {
        "cublas_workspace_config": actual,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "sdpa_backends": _force_math_sdpa(),
    }


def _environment(preflight: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "determinism_preflight": preflight,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return payload


def _run_tests(export: Path, *, essential: Path | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = export / "repository-tests.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if essential is not None:
        shutil.copy2(log_path, essential / "repository-tests.log")
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"repository tests failed with exit code {result.returncode}")


def _load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise RuntimeError("the frozen Mobile follow-up preregistration hash changed")
    payload = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "barun-mobile-followup-sweep-v1":
        raise ValueError("unsupported Mobile follow-up preregistration")
    budget = payload.get("dev_trial_budget")
    arms = payload.get("arms")
    selection = payload.get("selection")
    if budget != {"total": 3, "consumed_by_reference": 1, "remaining_arms": 2}:
        raise ValueError("the Mobile development budget must remain exactly three trials")
    if not isinstance(arms, list) or tuple(arm.get("arm_id") for arm in arms) != EXPECTED_ARM_IDS:
        raise ValueError("the preregistration must contain exactly the two frozen arms")
    if not isinstance(selection, dict) or selection != {
        "metric_path": "ast_exact_match.value",
        "direction": "maximize",
        "minimum_absolute_gain_over_reference": 0.01,
        "tie_policy": "inconclusive_keep_reference",
        "checkpoint_policy": "final_only",
        "non_selection_metrics": "report_only",
    }:
        raise ValueError("the preregistered final-only AST-exact selection rule changed")
    return payload


def _official_eval_firewall(prepared: Any, audit: dict[str, Any]) -> dict[str, bool]:
    expected = {
        "opaque_unparsed": True,
        "prompts_parsed": False,
        "labels_parsed": False,
        "materialized": False,
        "lengths_computed": False,
        "overlaps_computed": False,
        "summaries_computed": False,
        "targets_parsed": False,
        "tool_schemas_parsed": False,
    }
    evidence = audit.get("official_eval")
    if not isinstance(evidence, dict) or any(
        evidence.get(name) is not value for name, value in expected.items()
    ):
        raise RuntimeError("official Mobile eval prompts and labels did not remain opaque")
    if set(prepared.hashes) != {"train", "dev", "audit"}:
        raise RuntimeError("prepared hashes must contain train/dev/audit and no official eval")
    return expected


def _stage_prepared_data(prepared: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "train.jsonl": (Path(prepared.train_manifest), prepared.hashes["train"]),
        "dev.jsonl": (Path(prepared.dev_manifest), prepared.hashes["dev"]),
        "audit.json": (Path(prepared.audit_path), prepared.hashes["audit"]),
    }
    for name, (source, expected) in artifacts.items():
        if sha256_file(source) != expected:
            raise RuntimeError(f"prepared artifact changed before export: {source}")
        target = destination / name
        shutil.copy2(source, target)
        if sha256_file(target) != expected:
            raise RuntimeError(f"prepared artifact changed during export: {source}")


def _strict_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"training-view input hash mismatch: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty training-view source: {path}")
    return rows


def _is_hard_mobile_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    call_names = metadata.get("call_names") if isinstance(metadata, dict) else None
    if (
        not isinstance(call_names, list)
        or not call_names
        or any(not isinstance(name, str) for name in call_names)
    ):
        raise RuntimeError("Mobile training row lacks a valid metadata.call_names list")
    return "create_calendar_event" in call_names or "show_map" in call_names or len(call_names) > 1


def _stable_order(rows: list[dict[str, Any]], *, seed: int, label: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        example_id = row.get("id")
        if not isinstance(example_id, str):
            raise TypeError("Mobile training row lacks an ID")
        digest = hashlib.sha256(f"{seed}:{label}:{example_id}".encode()).hexdigest()
        return digest, example_id

    return sorted(rows, key=key)


def _materialize_hard_mix(
    *,
    source: Path,
    source_sha256: str,
    destination: Path,
    audit_path: Path,
    hard_fraction: float,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    if not 0 < hard_fraction < 1:
        raise ValueError("hard_fraction must be strictly between zero and one")
    rows = _strict_rows(source, source_sha256)
    hard = _stable_order([row for row in rows if _is_hard_mobile_row(row)], seed=seed, label="hard")
    easy = _stable_order(
        [row for row in rows if not _is_hard_mobile_row(row)], seed=seed, label="easy"
    )
    if not hard or not easy:
        raise RuntimeError("hard-mix curriculum requires non-empty hard and easy pools")
    hard_slots = math.floor(len(rows) * hard_fraction + 0.5)
    slots = [(hard[index % len(hard)], "hard") for index in range(hard_slots)]
    slots.extend((easy[index % len(easy)], "easy") for index in range(len(rows) - hard_slots))
    slots.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:slot:{item[1]}:{item[0]['id']}".encode()
        ).hexdigest()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    selected_ids: list[str] = []
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for slot, (source_row, bucket) in enumerate(slots):
            clone = dict(source_row)
            metadata = dict(clone.get("metadata", {}))
            source_id = str(source_row["id"])
            clone["id"] = f"{source_id}--hardmix70-{slot:05d}"
            metadata["sweep_curriculum"] = {
                "bucket": bucket,
                "selector_version": "barun-hard-mix-v1",
                "source_id": source_id,
            }
            clone["metadata"] = metadata
            selected_ids.append(source_id)
            handle.write(
                json.dumps(
                    clone,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    output_sha256 = sha256_file(destination)
    audit = {
        "schema_version": "barun-hard-mix-audit-v1",
        "source_manifest": str(source),
        "source_sha256": source_sha256,
        "output_manifest": str(destination),
        "output_sha256": output_sha256,
        "selector_version": "barun-hard-mix-v1",
        "seed": seed,
        "hard_predicate": "contains_create_calendar_event_or_show_map_or_is_multicall",
        "target_hard_fraction": hard_fraction,
        "source_rows": len(rows),
        "output_rows": len(slots),
        "hard_pool_rows": len(hard),
        "easy_pool_rows": len(easy),
        "hard_slots": hard_slots,
        "easy_slots": len(rows) - hard_slots,
        "unique_source_rows_selected": len(set(selected_ids)),
        "repeated_slots": len(selected_ids) - len(set(selected_ids)),
        "official_eval_rows_read": 0,
        "development_rows_read": 0,
    }
    _write_json(audit_path, audit)
    return output_sha256, audit


def _arm_training_run_id(sweep_run_id: str, arm_id: str, seed: int) -> str:
    parts = sweep_run_id.split("-", 2)
    if len(parts) < 3 or len(parts[0]) != 8 or len(parts[1]) != 4:
        raise ValueError("sweep run ID must start with YYYYMMDD-HHMM")
    return f"{parts[0]}-{parts[1]}-mob-{arm_id}-s{seed}"


def _training_config(
    *,
    args: argparse.Namespace,
    arm: dict[str, Any],
    arm_dir: Path,
    train_manifest: Path,
    train_sha256: str,
    dev_manifest: Path,
    dev_sha256: str,
) -> dict[str, Any]:
    optimization = dict(arm["optimization"])
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": _arm_training_run_id(args.run_id, arm["arm_id"], optimization["seed"]),
        "output_root": str(arm_dir / "training"),
        "hypothesis": arm["hypothesis"],
        "decision": arm["decision"],
        "base_checkpoint": {
            "source": "huggingface",
            "repo_id": BASE_REPO,
            "revision": BASE_REVISION,
            "expected_sha256": BASE_HASHES,
        },
        "data": {
            "train_manifest": str(train_manifest),
            "train_sha256": train_sha256,
            "dev_manifest": str(dev_manifest),
            "dev_sha256": dev_sha256,
            "max_seq_len": 2048,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": optimization,
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": str(args.jarvis_machine_id),
            "estimated_hourly_cost": args.hourly_cost,
        },
    }


def _metric(aggregate: dict[str, Any]) -> dict[str, int | float]:
    metric = aggregate.get("ast_exact_match")
    if not isinstance(metric, dict):
        raise TypeError("scores lack ast_exact_match")
    numerator = metric.get("numerator")
    denominator = metric.get("denominator")
    value = metric.get("value")
    if (
        not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator < 1
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not math.isclose(value, numerator / denominator, rel_tol=0, abs_tol=1e-15)
    ):
        raise RuntimeError("invalid ast_exact_match metric")
    return {"numerator": numerator, "denominator": denominator, "value": float(value)}


def _select_by_preregistered_metric(
    preregistration: dict[str, Any], arm_metrics: dict[str, dict[str, int | float]]
) -> dict[str, Any]:
    reference = preregistration["reference_trial"]
    reference_metric = dict(reference["ast_exact_match"])
    candidates = {"reference": reference_metric, **arm_metrics}
    best_value = max(float(metric["value"]) for metric in candidates.values())
    winners = sorted(
        name for name, metric in candidates.items() if float(metric["value"]) == best_value
    )
    margin = preregistration["selection"]["minimum_absolute_gain_over_reference"]
    gain = best_value - float(reference_metric["value"])
    if len(winners) != 1:
        decision = "inconclusive_tie_keep_reference"
        selected = "reference"
    elif winners[0] == "reference":
        decision = "keep_reference"
        selected = "reference"
    elif gain < margin:
        decision = "inconclusive_below_practical_margin_keep_reference"
        selected = "reference"
    else:
        decision = "promote_followup_arm"
        selected = winners[0]
    return {
        "metric_path": "ast_exact_match.value",
        "inputs": candidates,
        "winner_before_practical_margin": winners[0] if len(winners) == 1 else None,
        "gain_over_reference": gain,
        "minimum_absolute_gain_over_reference": margin,
        "decision": decision,
        "selected": selected,
        "non_selection_metrics_consulted": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_root = args.artifact_root / args.run_id
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingSweepError(f"refusing to overwrite sweep root {run_root}") from error
    export = run_root / "export"
    export.mkdir()
    essential = export / "essential"
    essential.mkdir()
    preflight = _cuda_determinism_preflight()
    preregistration = _load_preregistration()
    shutil.copy2(PREREGISTRATION_PATH, export / "preregistration.json")
    shutil.copy2(PREREGISTRATION_PATH, essential / "preregistration.json")
    environment = _environment(preflight)
    _write_json(export / "environment.json", environment)
    _write_json(essential / "environment.json", environment)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the Mobile sweep requires a CUDA GPU with bfloat16 support")
    _run_tests(export, essential=essential)

    base_dir = Path(
        snapshot_download(
            repo_id=BASE_REPO,
            revision=BASE_REVISION,
            allow_patterns=sorted(BASE_HASHES),
        )
    ).resolve()
    verify_checkpoint(base_dir, expected_sha256=BASE_HASHES)
    tokenizer = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    source = download_pinned_source(run_root / "cache")
    prepared = prepare_mobile_actions(
        source,
        run_root / "data",
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=BASE_REPO,
            revision=BASE_REVISION,
            sha256=BASE_HASHES["tokenizer.json"],
        ),
    )
    audit = json.loads(prepared.audit_path.read_text(encoding="utf-8"))
    firewall = _official_eval_firewall(prepared, audit)
    expected_train = preregistration["reference_trial"]["train_manifest_sha256"]
    expected_dev = preregistration["reference_trial"]["dev_manifest_sha256"]
    if prepared.hashes["train"] != expected_train or prepared.hashes["dev"] != expected_dev:
        raise RuntimeError("prepared train/dev hashes differ from the frozen reference trial")
    data_dir = export / "data"
    _stage_prepared_data(prepared, data_dir)
    _stage_shared_essential(
        export=export,
        essential=essential,
        audit_path=data_dir / "audit.json",
        audit_sha256=prepared.hashes["audit"],
    )
    train_manifest = data_dir / "train.jsonl"
    dev_manifest = data_dir / "dev.jsonl"
    target_max = int(audit["tokenization"]["per_split"]["dev"]["target"]["max"])
    generation_limit = min(256, target_max + 16)

    base_eval = export / "base-eval"
    base_predictions = base_eval / "predictions.jsonl"
    base_generation = generate_manifest(
        checkpoint_dir=base_dir,
        manifest_path=dev_manifest,
        manifest_sha256=expected_dev,
        predictions_path=base_predictions,
        device_name="cuda",
        batch_size=args.generation_batch_size,
        max_new_tokens=generation_limit,
        expected_checkpoint_sha256=BASE_HASHES,
    )
    write_scores(dev_manifest, base_predictions, base_eval / "scores")
    base_metrics = json.loads((base_eval / "scores" / "aggregate.json").read_text())

    views_dir = export / "training-views"
    views_dir.mkdir()
    arm_results: dict[str, Any] = {}
    selection_inputs: dict[str, dict[str, int | float]] = {}
    for arm in preregistration["arms"]:
        arm_id = arm["arm_id"]
        view = arm["train_view"]
        if view["kind"] == "uniform":
            arm_train = train_manifest
            arm_train_sha256 = expected_train
            view_audit = {
                "schema_version": "barun-uniform-view-audit-v1",
                "source_manifest": str(train_manifest),
                "source_sha256": expected_train,
                "output_manifest": str(train_manifest),
                "output_sha256": expected_train,
                "official_eval_rows_read": 0,
                "development_rows_read": 0,
            }
            _write_json(views_dir / f"{arm_id}-audit.json", view_audit)
        elif view["kind"] == "fixed_size_hard_mix":
            arm_train = views_dir / f"{arm_id}.jsonl"
            arm_train_sha256, view_audit = _materialize_hard_mix(
                source=train_manifest,
                source_sha256=expected_train,
                destination=arm_train,
                audit_path=views_dir / f"{arm_id}-audit.json",
                hard_fraction=float(view["hard_fraction"]),
                seed=int(arm["optimization"]["seed"]),
            )
        else:  # pragma: no cover - frozen preregistration invariant
            raise RuntimeError(f"unsupported train view {view['kind']!r}")

        arm_dir = export / "arms" / arm_id
        config_path = arm_dir / "training-config.json"
        _write_json(
            config_path,
            _training_config(
                args=args,
                arm=arm,
                arm_dir=arm_dir,
                train_manifest=arm_train,
                train_sha256=arm_train_sha256,
                dev_manifest=dev_manifest,
                dev_sha256=expected_dev,
            ),
        )
        training = train_sft(TrainingRunConfig.from_json(config_path))
        final_checkpoint = Path(training.final_checkpoint)
        evaluation = arm_dir / "final-eval"
        predictions = evaluation / "predictions.jsonl"
        generation = generate_manifest(
            checkpoint_dir=final_checkpoint,
            manifest_path=dev_manifest,
            manifest_sha256=expected_dev,
            predictions_path=predictions,
            device_name="cuda",
            batch_size=args.generation_batch_size,
            max_new_tokens=generation_limit,
        )
        write_scores(dev_manifest, predictions, evaluation / "scores")
        aggregate = json.loads((evaluation / "scores" / "aggregate.json").read_text())
        primary = _metric(aggregate)
        selection_inputs[arm_id] = primary
        arm_result = {
            "arm_id": arm_id,
            "train_view": view_audit,
            "training": training.to_dict(),
            "evaluated_checkpoint_policy": "final_only",
            "evaluated_checkpoint": str(final_checkpoint),
            "generation": generation.to_dict(),
            "primary_metric": primary,
            "report_only_metrics": aggregate,
        }
        arm_results[arm_id] = arm_result
        _write_json(arm_dir / "result.json", arm_result)
        _export_arm_essential(
            arm_id=arm_id,
            arm_dir=arm_dir,
            final_checkpoint=final_checkpoint,
            essential=essential,
        )
        _write_json(essential / "artifact-sha256.json", _tree_manifest(essential))
        torch.cuda.empty_cache()

    selection = _select_by_preregistered_metric(preregistration, selection_inputs)
    _write_json(export / "selection.json", selection)
    result = {
        "schema_version": "barun-mobile-followup-sweep-result-v1",
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "mobile_actions_revision": MOBILE_ACTIONS_REVISION,
        "preregistration_sha256": sha256_file(export / "preregistration.json"),
        "official_eval_firewall": firewall,
        "official_eval_rows_opaque_unparsed": prepared.final_eval_rows,
        "prepared_hashes": dict(prepared.hashes),
        "generation_limit": generation_limit,
        "base_generation": base_generation.to_dict(),
        "base_report_only_metrics": base_metrics,
        "arms": arm_results,
        "selection": selection,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(export / "result.json", result)
    shutil.copy2(export / "selection.json", essential / "selection.json")
    shutil.copy2(export / "result.json", essential / "result.json")
    _write_json(
        essential / "bundle-manifest.json",
        {
            "schema_version": "barun-mobile-sweep-essential-v1",
            "run_id": args.run_id,
            "arms": list(EXPECTED_ARM_IDS),
            "optimizer_state_included": False,
            "source_export": str(export),
        },
    )
    if any(path.name == "optimizer.pt" for path in essential.rglob("*")):
        raise RuntimeError("optimizer state entered the compact essential bundle")
    _write_json(essential / "artifact-sha256.json", _tree_manifest(essential))
    _write_json(export / "artifact-sha256.json", _tree_manifest(export))
    return result


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/root/barun-artifacts"))
    parser.add_argument("--generation-batch-size", type=_positive_int, default=128)
    parser.add_argument("--hourly-cost", type=_nonnegative_float, default=378.27)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export = args.artifact_root / args.run_id / "export"
    try:
        result = run(args)
    except ExistingSweepError:
        raise
    except BaseException as error:
        export.mkdir(parents=True, exist_ok=True)
        _write_json(
            export / "failure.json",
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        _write_json(export / "artifact-sha256.json", _tree_manifest(export))
        raise
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
