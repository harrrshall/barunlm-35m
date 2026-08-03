"""Run one preregistered PRESTO English SFT stage from an exact local checkpoint."""

from __future__ import annotations

# cuBLAS determinism must be configured before importing torch or project modules
# that can transitively import torch.
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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT = (
    os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
)

import torch
from tokenizers import Tokenizer

from barunlm.datasets.presto import (
    PRESTO_REVISION,
    TokenizerIdentity,
    download_pinned_archive,
    prepare_presto,
)
from barunlm.evaluation.generation import generate_manifest, verify_checkpoint
from barunlm.evaluation.presto import phenomenon_group, write_scores
from barunlm.training import TrainingRunConfig, train_sft
from barunlm.training.data import sha256_file

RECIPE_PATH = Path(__file__).resolve().parents[1] / "configs" / "presto_stage_v1.json"
# Updated only when the complete preregistered recipe is intentionally revised.
RECIPE_SHA256 = "62c1222e8b9d348fa6b9aa9582001a5a06675f541e630a08b0d35f0efcd53cd9"
EXPECTED_CATEGORIES = ("abstain", "confirm", "revision", "disfluency", "contextual")
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*-s\d+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ExistingPrestoRunError(FileExistsError):
    """An immutable run ID already owns the requested artifact directory."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_progress(export: Path, essential: Path, phase: str, **details: Any) -> None:
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        **details,
    }
    _append_jsonl(export / "progress.jsonl", event)
    _append_jsonl(essential / "progress.jsonl", event)


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def _load_recipe() -> dict[str, Any]:
    actual = sha256_file(RECIPE_PATH)
    if actual != RECIPE_SHA256:
        raise RuntimeError(f"PRESTO recipe hash changed: expected {RECIPE_SHA256}, got {actual}")
    payload = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "barun-presto-stage-preregistration-v1":
        raise ValueError("unsupported PRESTO stage preregistration")
    if payload.get("recipe_id") != "presto-context-safety-sft-v1":
        raise ValueError("unexpected PRESTO recipe ID")
    dataset = payload.get("dataset")
    tokenizer = payload.get("tokenizer_identity")
    view = payload.get("train_view")
    optimization = payload.get("optimization")
    evaluation = payload.get("evaluation")
    if not isinstance(dataset, dict) or dataset.get("revision") != PRESTO_REVISION:
        raise ValueError("PRESTO dataset revision changed")
    if not isinstance(tokenizer, dict) or not SHA256_PATTERN.fullmatch(
        str(tokenizer.get("sha256", ""))
    ):
        raise ValueError("PRESTO recipe lacks a pinned tokenizer")
    if (
        not isinstance(view, dict)
        or view.get("kind") != "dev_deduplicated_train_plus_stratified_focus_replay"
        or tuple(view.get("categories", ())) != EXPECTED_CATEGORIES
        or view.get("selector_version") != "barun-presto-focus-replay-v1"
        or not isinstance(view.get("maximum_replays_per_category"), int)
        or view["maximum_replays_per_category"] < 1
        or view.get("development_rows_read_for_exact_duplicate_exclusion") != 14_288
        or view.get("development_rows_read_for_focus_selection") != 0
        or view.get("official_test_rows_read") != 0
    ):
        raise ValueError("PRESTO focus-replay recipe changed")
    exclusion = view.get("exact_prompt_target_dev_exclusion")
    if not isinstance(exclusion, dict) or exclusion.get("enabled") is not True:
        raise ValueError("PRESTO exact train/development exclusion changed")
    if not isinstance(optimization, dict) or optimization.get("epochs") != 1:
        raise ValueError("PRESTO stage must remain one recipe with one epoch")
    if not isinstance(evaluation, dict) or evaluation.get("native_presto_metric") is not False:
        raise ValueError("PRESTO derived/native metric distinction changed")
    return payload


def _force_math_sdpa() -> dict[str, bool | None]:
    backend = torch.backends.cuda
    required = (
        ("flash", "enable_flash_sdp", "flash_sdp_enabled", False),
        ("memory_efficient", "enable_mem_efficient_sdp", "mem_efficient_sdp_enabled", False),
        ("math", "enable_math_sdp", "math_sdp_enabled", True),
    )
    evidence: dict[str, bool | None] = {}
    for label, setter_name, getter_name, expected in required:
        setter = getattr(backend, setter_name, None)
        getter = getattr(backend, getter_name, None)
        if not callable(setter) or not callable(getter):
            raise TypeError(f"PyTorch cannot configure and verify {label} SDPA")
        setter(expected)
        actual = getter()
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(f"failed to set {label} SDPA to {expected}: got {actual!r}")
        evidence[label] = actual
    cudnn_setter = getattr(backend, "enable_cudnn_sdp", None)
    cudnn_getter = getattr(backend, "cudnn_sdp_enabled", None)
    if cudnn_setter is None and cudnn_getter is None:
        evidence["cudnn"] = None
    elif callable(cudnn_setter) and callable(cudnn_getter):
        cudnn_setter(False)
        actual = cudnn_getter()
        if type(actual) is not bool or actual is not False:
            raise RuntimeError(f"failed to disable cuDNN SDPA: got {actual!r}")
        evidence["cudnn"] = actual
    else:
        raise TypeError("PyTorch exposes a partial, unverifiable cuDNN SDPA interface")
    return evidence


def _cuda_determinism_preflight() -> dict[str, Any]:
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG changed: expected "
            f"{DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r}, got {actual!r}"
        )
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        raise RuntimeError("cuBLAS determinism was not configured before importing torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the PRESTO stage preflight")
    return {
        "cublas_workspace_config": actual,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "sdpa_backends": _force_math_sdpa(),
    }


def _environment(preflight: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "determinism_preflight": preflight,
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


def _run_tests(export: Path, essential: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    log = export / "repository-tests.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    shutil.copy2(log, essential / log.name)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"repository tests failed with exit code {result.returncode}")


def _checkpoint_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "barun_config.json": args.input_config_sha256,
        "model.safetensors": args.input_model_sha256,
        "tokenizer.json": args.input_tokenizer_sha256,
    }


def _effective_preregistration(
    args: argparse.Namespace, recipe: dict[str, Any], checkpoint_hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "barun-presto-effective-preregistration-v1",
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "recipe_path": str(RECIPE_PATH),
        "recipe_sha256": RECIPE_SHA256,
        "recipe": recipe,
        "input_checkpoint": {
            "source": "local_exact_directory",
            "directory": str(args.input_checkpoint_dir.resolve()),
            "file_sha256": checkpoint_hashes,
        },
        "runtime_only": {
            "artifact_root": str(args.artifact_root),
            "generation_batch_size": args.generation_batch_size,
            "estimated_hourly_cost": args.hourly_cost,
        },
        "created_before_dataset_preparation": True,
        "official_test_member_accessed": False,
    }


def _official_test_firewall(prepared: Any, audit: Mapping[str, Any]) -> dict[str, Any]:
    evidence = audit.get("official_test")
    expected = {
        "archive_payload_hashing_only": True,
        "bytes_read_from_sensitive_members": 0,
        "combined_dataset_member_opened": False,
        "labels_parsed": False,
        "materialized": False,
        "member_opened": False,
        "opaque_unparsed": True,
        "prompts_parsed": False,
        "sensitive_members_opened": [],
        "targets_parsed": False,
        "test_partition_members_opened": False,
    }
    if not isinstance(evidence, Mapping) or any(
        evidence.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("official PRESTO test/combined members did not remain opaque")
    if set(prepared.hashes) != {"train", "dev", "audit"}:
        raise RuntimeError("PRESTO preparation must export only train/dev/audit hashes")
    source = audit.get("source")
    members = source.get("members") if isinstance(source, Mapping) else None
    if not isinstance(members, Mapping):
        raise TypeError("PRESTO audit lacks member-level access evidence")
    for name in ("presto_dataset.jsonl", "presto_test.jsonl"):
        member = members.get(name)
        if (
            not isinstance(member, Mapping)
            or member.get("runtime_open_count") != 0
            or member.get("runtime_bytes_read") != 0
        ):
            raise RuntimeError(f"sealed PRESTO member was read: {name}")
    return {
        **expected,
        "official_test_rows_opaque_unparsed": prepared.official_test_rows,
    }


def _stage_prepared_data(prepared: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    sources = {
        "train.jsonl": (Path(prepared.train_manifest), prepared.hashes["train"]),
        "dev.jsonl": (Path(prepared.dev_manifest), prepared.hashes["dev"]),
        "audit.json": (Path(prepared.audit_path), prepared.hashes["audit"]),
    }
    for name, (source, expected) in sources.items():
        if sha256_file(source) != expected:
            raise RuntimeError(f"prepared artifact changed before export: {source}")
        target = destination / name
        shutil.copy2(source, target)
        if sha256_file(target) != expected:
            raise RuntimeError(f"prepared artifact changed during export: {source}")


def _strict_manifest_rows(path: Path, expected_sha256: str, *, split: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"manifest hash mismatch before training-view construction: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid manifest JSON at {path}:{line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise TypeError(f"invalid manifest row at {path}:{line_number}")
            sample_id = row.get("id")
            metadata = row["metadata"]
            if not isinstance(sample_id, str) or sample_id in seen:
                raise RuntimeError(f"invalid or duplicate manifest ID at {path}:{line_number}")
            if metadata.get("source_split") != split or metadata.get("derived_split") != split:
                raise RuntimeError(f"unexpected split metadata at {path}:{line_number}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _content_identity(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise TypeError("PRESTO manifest row lacks exact prompt/target strings")
    encoded = json.dumps(
        {"prompt": prompt, "target": target},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _focus_categories(row: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = row["metadata"]
    assert isinstance(metadata, Mapping)
    phenomenon = metadata.get("linguistic_phenomenon")
    decision = metadata.get("policy_decision")
    prompt = row.get("prompt")
    if not isinstance(phenomenon, str) or decision not in {"ABSTAIN", "CALL", "CONFIRM"}:
        raise RuntimeError("PRESTO training row lacks focus metadata")
    if not isinstance(prompt, str):
        raise TypeError("PRESTO training row lacks prompt")
    categories: list[str] = []
    if decision == "ABSTAIN":
        categories.append("abstain")
    if decision == "CONFIRM":
        categories.append("confirm")
    group = phenomenon_group(phenomenon)
    if group in {"revision", "disfluency"}:
        categories.append(group)
    try:
        context_text = prompt.split("\nCONTEXT ", 1)[1].split("\nDIALOGUE ", 1)[0]
        dialogue_text = prompt.split("\nDIALOGUE ", 1)[1].split("\n<user>\n", 1)[0]
        context = json.loads(context_text)
        dialogue = json.loads(dialogue_text)
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("PRESTO training prompt has malformed context fields") from error
    if not isinstance(context, dict) or not isinstance(dialogue, list):
        raise TypeError("PRESTO training prompt context fields have unexpected types")
    if dialogue or any(context.get(name) for name in ("contacts", "lists", "notes")):
        categories.append("contextual")
    return tuple(categories)


def _materialize_focus_view(
    *,
    source: Path,
    source_sha256: str,
    dev_source: Path,
    dev_sha256: str,
    destination: Path,
    audit_path: Path,
    seed: int,
    maximum_per_category: int,
) -> tuple[str, dict[str, Any]]:
    """Exclude exact dev duplicates, then build a focus replay without reading test rows."""

    source_rows = _strict_manifest_rows(source, source_sha256, split="train")
    dev_rows = _strict_manifest_rows(dev_source, dev_sha256, split="dev")
    dev_content = {_content_identity(row) for row in dev_rows}
    excluded = [row for row in source_rows if _content_identity(row) in dev_content]
    rows = [row for row in source_rows if _content_identity(row) not in dev_content]
    if not rows:
        raise RuntimeError("exact train/development exclusion removed every training row")
    eligible: dict[str, list[dict[str, Any]]] = {category: [] for category in EXPECTED_CATEGORIES}
    for row in rows:
        for category in _focus_categories(row):
            eligible[category].append(row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for category in EXPECTED_CATEGORIES:
        candidates = eligible[category]
        selected[category] = sorted(
            candidates,
            key=lambda row: (
                hashlib.sha256(f"{seed}:{category}:{row['id']}".encode()).hexdigest(),
                str(row["id"]),
            ),
        )[:maximum_per_category]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
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
                + "\n"
            )
        for category in EXPECTED_CATEGORIES:
            for index, row in enumerate(selected[category]):
                replay = dict(row)
                metadata = dict(replay["metadata"])
                source_id = str(row["id"])
                replay["id"] = f"{source_id}--presto-focus-{category}-{index:05d}"
                metadata["focus_replay"] = {
                    "category": category,
                    "selector_version": "barun-presto-focus-replay-v1",
                    "source_id": source_id,
                }
                replay["metadata"] = metadata
                handle.write(
                    json.dumps(
                        replay,
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
    replay_counts = {category: len(selected[category]) for category in EXPECTED_CATEGORIES}
    selected_ids = {
        category: [str(row["id"]) for row in selected[category]] for category in EXPECTED_CATEGORIES
    }
    audit = {
        "schema_version": "barun-presto-focus-replay-audit-v1",
        "selector_version": "barun-presto-focus-replay-v1",
        "source_manifest": str(source),
        "source_sha256": source_sha256,
        "source_rows": len(source_rows),
        "rows_after_exact_dev_exclusion": len(rows),
        "exact_prompt_target_train_rows_excluded": len(excluded),
        "excluded_train_ids": sorted(str(row["id"]) for row in excluded),
        "excluded_train_id_membership_sha256": hashlib.sha256(
            ("\n".join(sorted(str(row["id"]) for row in excluded)) + "\n").encode()
        ).hexdigest(),
        "output_manifest": str(destination),
        "output_sha256": output_sha256,
        "output_rows": len(rows) + sum(replay_counts.values()),
        "seed": seed,
        "maximum_replays_per_category": maximum_per_category,
        "eligible_counts": {category: len(eligible[category]) for category in EXPECTED_CATEGORIES},
        "replay_counts": replay_counts,
        "selected_source_membership_sha256": {
            category: hashlib.sha256(
                ("\n".join(selected_ids[category]) + "\n").encode()
            ).hexdigest()
            for category in EXPECTED_CATEGORIES
        },
        "unique_replayed_source_rows": len(
            {sample_id for values in selected_ids.values() for sample_id in values}
        ),
        "duplicate_across_categories": sum(replay_counts.values())
        - len({sample_id for values in selected_ids.values() for sample_id in values}),
        "development_rows_read_for_exact_duplicate_exclusion": len(dev_rows),
        "development_rows_read_for_focus_selection": 0,
        "official_test_rows_read": 0,
    }
    if {_content_identity(row) for row in rows}.intersection(dev_content):
        raise RuntimeError("train/development exact prompt-target exclusion failed")
    _write_json(audit_path, audit)
    return output_sha256, audit


def _training_config(
    *,
    args: argparse.Namespace,
    recipe: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    train_manifest: Path,
    train_sha256: str,
    dev_manifest: Path,
    dev_sha256: str,
    export: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": args.run_id,
        "output_root": str(export / "training"),
        "hypothesis": recipe["hypothesis"],
        "decision": recipe["decision"],
        "base_checkpoint": {
            "source": "local",
            "local_dir": str(args.input_checkpoint_dir.resolve()),
            "expected_sha256": dict(input_hashes),
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
        "optimization": dict(recipe["optimization"]),
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": str(args.jarvis_machine_id),
            "estimated_hourly_cost": args.hourly_cost,
        },
    }


def _export_checkpoint(
    source: Path,
    destination: Path,
    *,
    run_id: str,
    input_hashes: Mapping[str, str],
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        target = destination / name
        shutil.copy2(source / name, target)
        hashes[name] = sha256_file(target)
    _write_json(
        destination / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "run_id": run_id,
            "source_checkpoint": str(source),
            "input_checkpoint_sha256": dict(input_hashes),
            "file_sha256": hashes,
        },
    )
    return hashes


def _copy_training_evidence(selected_checkpoint: Path, destination: Path) -> None:
    run_dir = selected_checkpoint.parent.parent
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.jsonl", "run_manifest.json", "summary.json"):
        shutil.copy2(run_dir / name, destination / name)
    best = run_dir / "best_checkpoint.json"
    if best.is_file():
        shutil.copy2(best, destination / best.name)


def _metric_value(aggregate: Mapping[str, Any], path: tuple[str, ...]) -> float:
    current: Any = aggregate
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise RuntimeError(f"PRESTO aggregate lacks metric {'.'.join(path)}")
        current = current[key]
    if (
        isinstance(current, bool)
        or not isinstance(current, int | float)
        or not math.isfinite(current)
    ):
        raise RuntimeError(f"PRESTO aggregate metric {'.'.join(path)} is invalid")
    return float(current)


def _gate(post: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = recipe["gate"]
    exact = _metric_value(post, ("ast_exact_match", "value"))
    schema = _metric_value(post, ("schema_valid", "value"))
    abstention = _metric_value(post, ("abstention", "f1"))
    false_call = _metric_value(post, ("false_call_on_gate", "value"))
    buckets = post.get("headline_phenomenon_buckets")
    if not isinstance(buckets, Mapping):
        raise TypeError("PRESTO aggregate lacks headline phenomenon buckets")
    simple = buckets.get("no_phenomenon")
    represented = isinstance(simple, Mapping) and int(simple.get("count", 0)) > 0
    gaps: dict[str, float | None] = {}
    simple_value = _metric_value(simple, ("ast_exact_match", "value")) if represented else None
    for name in ("revision", "disfluency"):
        bucket = buckets.get(name)
        if (
            not isinstance(bucket, Mapping)
            or int(bucket.get("count", 0)) < 1
            or simple_value is None
        ):
            gaps[name] = None
            represented = False
        else:
            gaps[name] = simple_value - _metric_value(bucket, ("ast_exact_match", "value"))
    maximum_gap = float(thresholds["maximum_no_phenomenon_to_revision_or_disfluency_gap"])
    checks = {
        "derived_ast_exact_at_least_threshold": exact
        >= float(thresholds["derived_ast_exact_at_least"]),
        "schema_valid_at_least_threshold": schema >= float(thresholds["schema_valid_at_least"]),
        "abstention_f1_at_least_threshold": abstention
        >= float(thresholds["abstention_f1_at_least"]),
        "false_call_on_gate_at_most_threshold": false_call
        <= float(thresholds["false_call_on_gate_at_most"]),
        "gap_buckets_represented": represented,
        "revision_disfluency_gap_at_most_threshold": represented
        and all(gap is not None and gap <= maximum_gap for gap in gaps.values()),
    }
    return {
        "thresholds": dict(thresholds),
        "observed": {
            "derived_ast_exact": exact,
            "schema_valid": schema,
            "abstention_f1": abstention,
            "false_call_on_gate": false_call,
            "no_phenomenon_minus_hard_bucket_gap": gaps,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_root = args.artifact_root / args.run_id
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingPrestoRunError(f"refusing to overwrite PRESTO run root {run_root}") from error
    export = run_root / "export"
    essential = export / "essential"
    essential.mkdir(parents=True)
    _record_progress(export, essential, "run_root_claimed", run_root=str(run_root))

    preflight = _cuda_determinism_preflight()
    _record_progress(export, essential, "determinism_preflight_passed")
    environment = _environment(preflight)
    environment["jarvis_machine_id"] = args.jarvis_machine_id
    _write_json(export / "environment.json", environment)
    shutil.copy2(export / "environment.json", essential / "environment.json")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the PRESTO stage requires a CUDA GPU with bfloat16 support")

    recipe = _load_recipe()
    input_hashes = _checkpoint_hashes(args)
    if args.input_tokenizer_sha256 != recipe["tokenizer_identity"]["sha256"]:
        raise RuntimeError("input checkpoint tokenizer differs from the frozen PRESTO tokenizer")
    effective_preregistration = _effective_preregistration(args, recipe, input_hashes)
    _write_json(export / "preregistration.json", effective_preregistration)
    shutil.copy2(export / "preregistration.json", essential / "preregistration.json")
    shutil.copy2(RECIPE_PATH, essential / "recipe.json")
    _record_progress(export, essential, "effective_preregistration_frozen")

    _run_tests(export, essential)
    _record_progress(export, essential, "repository_tests_passed")

    checkpoint_dir = args.input_checkpoint_dir.resolve()
    verify_checkpoint(checkpoint_dir, expected_sha256=input_hashes)
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    _record_progress(export, essential, "input_checkpoint_verified")

    archive = download_pinned_archive(run_root / "cache")
    prepared = prepare_presto(
        archive,
        run_root / "prepared",
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=recipe["tokenizer_identity"]["identifier"],
            revision=recipe["tokenizer_identity"]["revision"],
            sha256=args.input_tokenizer_sha256,
        ),
    )
    expected_prepared = {
        "train": recipe["dataset"]["train_manifest_sha256"],
        "dev": recipe["dataset"]["dev_manifest_sha256"],
        "audit": recipe["dataset"]["audit_sha256"],
    }
    if dict(prepared.hashes) != expected_prepared:
        raise RuntimeError(
            f"prepared PRESTO hashes differ from preregistration: {dict(prepared.hashes)}"
        )
    audit = json.loads(prepared.audit_path.read_text(encoding="utf-8"))
    firewall = _official_test_firewall(prepared, audit)
    _record_progress(export, essential, "presto_data_prepared_and_firewall_verified")

    data_dir = export / "data"
    _stage_prepared_data(prepared, data_dir)
    essential_data = essential / "data"
    essential_data.mkdir()
    shutil.copy2(data_dir / "audit.json", essential_data / "audit.json")
    train_view = data_dir / "train-focus.jsonl"
    focus_audit_path = data_dir / "train-focus-audit.json"
    focus_sha256, focus_audit = _materialize_focus_view(
        source=data_dir / "train.jsonl",
        source_sha256=prepared.hashes["train"],
        dev_source=data_dir / "dev.jsonl",
        dev_sha256=prepared.hashes["dev"],
        destination=train_view,
        audit_path=focus_audit_path,
        seed=int(recipe["optimization"]["seed"]),
        maximum_per_category=int(recipe["train_view"]["maximum_replays_per_category"]),
    )
    shutil.copy2(focus_audit_path, essential_data / focus_audit_path.name)
    _record_progress(
        export,
        essential,
        "train_only_focus_view_materialized",
        output_rows=focus_audit["output_rows"],
        output_sha256=focus_sha256,
        exact_prompt_target_train_rows_excluded=focus_audit[
            "exact_prompt_target_train_rows_excluded"
        ],
    )

    dev_manifest = data_dir / "dev.jsonl"
    target_max = int(audit["tokenization"]["per_split"]["dev"]["target"]["max"])
    generation_limit = min(384, target_max + 24)
    base_eval = export / "base-eval"
    base_predictions = base_eval / "predictions.jsonl"
    base_generation = generate_manifest(
        checkpoint_dir=checkpoint_dir,
        manifest_path=dev_manifest,
        manifest_sha256=prepared.hashes["dev"],
        predictions_path=base_predictions,
        device_name="cuda",
        batch_size=args.generation_batch_size,
        max_new_tokens=generation_limit,
        expected_checkpoint_sha256=input_hashes,
    )
    write_scores(dev_manifest, base_predictions, base_eval / "scores")
    _record_progress(export, essential, "input_checkpoint_dev_scored")

    training_config_path = export / "training-config.json"
    _write_json(
        training_config_path,
        _training_config(
            args=args,
            recipe=recipe,
            input_hashes=input_hashes,
            train_manifest=train_view,
            train_sha256=focus_sha256,
            dev_manifest=dev_manifest,
            dev_sha256=prepared.hashes["dev"],
            export=export,
        ),
    )
    training = train_sft(TrainingRunConfig.from_json(training_config_path))
    selected_checkpoint = Path(training.best_checkpoint or training.final_checkpoint)
    _record_progress(
        export,
        essential,
        "full_parameter_response_only_sft_completed",
        selected_checkpoint=str(selected_checkpoint),
    )

    post_eval = export / "post-eval"
    post_predictions = post_eval / "predictions.jsonl"
    post_generation = generate_manifest(
        checkpoint_dir=selected_checkpoint,
        manifest_path=dev_manifest,
        manifest_sha256=prepared.hashes["dev"],
        predictions_path=post_predictions,
        device_name="cuda",
        batch_size=args.generation_batch_size,
        max_new_tokens=generation_limit,
    )
    write_scores(dev_manifest, post_predictions, post_eval / "scores")
    _record_progress(export, essential, "post_sft_dev_scored")

    base_metrics = json.loads((base_eval / "scores" / "aggregate.json").read_text())
    post_metrics = json.loads((post_eval / "scores" / "aggregate.json").read_text())
    checkpoint_hashes = _export_checkpoint(
        selected_checkpoint,
        essential / "checkpoint",
        run_id=args.run_id,
        input_hashes=input_hashes,
    )
    shutil.copytree(base_eval, essential / "base-eval")
    shutil.copytree(post_eval, essential / "post-eval")
    shutil.copy2(training_config_path, essential / training_config_path.name)
    _copy_training_evidence(selected_checkpoint, essential / "training")

    result = {
        "schema_version": "barun-presto-stage-result-v1",
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "recipe_sha256": RECIPE_SHA256,
        "input_checkpoint": {
            "directory": str(checkpoint_dir),
            "file_sha256": input_hashes,
        },
        "output_checkpoint_sha256": checkpoint_hashes,
        "data": {
            "revision": PRESTO_REVISION,
            "train_rows": prepared.train_rows,
            "focused_train_rows": focus_audit["output_rows"],
            "dev_rows": prepared.dev_rows,
            "official_test_rows_opaque_unparsed": prepared.official_test_rows,
            "official_test_firewall": firewall,
            "prepared_hashes": dict(prepared.hashes),
            "focus_view_sha256": focus_sha256,
        },
        "metric_scope": "derived_action_ir_not_native_presto_semantic_parse",
        "generation_limit": generation_limit,
        "base": base_metrics,
        "post_sft": post_metrics,
        "delta": {
            "ast_exact_match": _metric_value(post_metrics, ("ast_exact_match", "value"))
            - _metric_value(base_metrics, ("ast_exact_match", "value")),
            "schema_valid": _metric_value(post_metrics, ("schema_valid", "value"))
            - _metric_value(base_metrics, ("schema_valid", "value")),
            "abstention_f1": _metric_value(post_metrics, ("abstention", "f1"))
            - _metric_value(base_metrics, ("abstention", "f1")),
        },
        "gate": _gate(post_metrics, recipe),
        "training": training.to_dict(),
        "generation": {
            "base": base_generation.to_dict(),
            "post_sft": post_generation.to_dict(),
        },
        "limitations": list(recipe["limitations"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(export / "result.json", result)
    shutil.copy2(export / "result.json", essential / "result.json")
    _write_json(
        essential / "bundle-manifest.json",
        {
            "schema_version": "barun-presto-essential-bundle-v1",
            "run_id": args.run_id,
            "optimizer_state_included": False,
            "sample_level_base_and_post_evidence_included": True,
            "source_export": str(export),
        },
    )
    if any(path.name == "optimizer.pt" for path in essential.rglob("*")):
        raise RuntimeError("optimizer state entered the compact PRESTO essential bundle")
    _record_progress(export, essential, "essential_bundle_completed")
    _record_progress(export, essential, "run_completed", gate_passed=result["gate"]["passed"])
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


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _sha256(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match YYYYMMDD-HHMM-lowercase-name-sN")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument("--input-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-model-sha256", type=_sha256, required=True)
    parser.add_argument("--input-config-sha256", type=_sha256, required=True)
    parser.add_argument("--input-tokenizer-sha256", type=_sha256, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/barun-artifacts"))
    parser.add_argument("--generation-batch-size", type=_positive_int, default=128)
    parser.add_argument("--hourly-cost", type=_nonnegative_float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export = args.artifact_root / args.run_id / "export"
    essential = export / "essential"
    try:
        result = run(args)
    except ExistingPrestoRunError:
        raise
    except BaseException as error:
        export.mkdir(parents=True, exist_ok=True)
        essential.mkdir(parents=True, exist_ok=True)
        _record_progress(
            export,
            essential,
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        failure = {
            "at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        _write_json(export / "failure.json", failure)
        _write_json(essential / "failure.json", failure)
        _write_json(essential / "artifact-sha256.json", _tree_manifest(essential))
        _write_json(export / "artifact-sha256.json", _tree_manifest(export))
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
