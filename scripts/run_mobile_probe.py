"""One decision-bearing remote probe: audit, base score, SFT, and post-SFT score."""

from __future__ import annotations

# PyTorch requires this variable to be set before CUDA/cuBLAS is initialized when
# deterministic algorithms are enabled.  The probe performs a base CUDA evaluation
# before entering the trainer, so setting it inside the trainer would be too late.
# Keep this assignment before importing torch or any barunlm module.
import argparse
import json
import os
import platform
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


class ExistingRunError(FileExistsError):
    """Raised when an immutable run ID has already claimed its artifact directory."""


def _cuda_determinism_preflight() -> dict[str, Any]:
    """Fail before the first CUDA query if deterministic cuBLAS setup is unprovable."""

    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG changed after import: "
            f"expected {DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r}, got {actual!r}"
        )
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        raise RuntimeError("deterministic cuBLAS configuration was not set before importing torch")
    cuda_already_initialized = torch.cuda.is_initialized()
    if cuda_already_initialized:
        raise RuntimeError("CUDA was initialized before the deterministic probe preflight")
    sdpa_backends = _force_math_sdpa()
    return {
        "cublas_workspace_config": actual,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
        "sdpa_backends": sdpa_backends,
    }


def _force_math_sdpa() -> dict[str, bool | None]:
    """Disable fused CUDA SDPA backends and verify math-only execution."""

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
        cudnn_actual = cudnn_getter()
        if type(cudnn_actual) is not bool or cudnn_actual is not False:
            raise RuntimeError(
                f"failed to disable the cuDNN SDPA backend: observed {cudnn_actual!r}"
            )
        evidence["cudnn"] = cudnn_actual
    else:
        raise TypeError("PyTorch exposes an unverifiable partial cuDNN SDPA interface")
    return evidence


def _official_eval_firewall(prepared: Any, audit: dict[str, Any]) -> dict[str, bool]:
    """Require proof that official eval prompts and labels stayed opaque."""

    official_eval = audit.get("official_eval")
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
    if not isinstance(official_eval, dict) or any(
        official_eval.get(name) is not value for name, value in expected.items()
    ):
        raise RuntimeError(
            "Mobile Actions adapter did not prove that official eval prompts/labels "
            "remained opaque and unmaterialized"
        )
    if "final_eval" in prepared.hashes:
        raise RuntimeError("official eval content hash must not enter train/dev probe artifacts")
    return expected


def _stage_prepared_data(prepared: Any, destination: Path) -> None:
    """Export only verified train/dev artifacts; never copy the combined source cache."""

    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "train.jsonl": (Path(prepared.train_manifest), prepared.hashes["train"]),
        "dev.jsonl": (Path(prepared.dev_manifest), prepared.hashes["dev"]),
        "audit.json": (Path(prepared.audit_path), prepared.hashes["audit"]),
    }
    for name, (source, expected_sha256) in artifacts.items():
        if sha256_file(source) != expected_sha256:
            raise RuntimeError(f"prepared artifact changed before export: {source}")
        target = destination / name
        shutil.copy2(source, target)
        if sha256_file(target) != expected_sha256:
            raise RuntimeError(f"prepared artifact changed while exporting: {source}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_tests(export: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    (export / "repository-tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"repository tests failed with exit code {result.returncode}")


def _environment(determinism_preflight: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "determinism_preflight": determinism_preflight,
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


def _training_config(
    *,
    args: argparse.Namespace,
    workspace: Path,
    prepared: Any,
    prepared_data: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "barun-sft-config-v1",
        "run_id": args.run_id,
        "output_root": str(workspace / "training"),
        "hypothesis": (
            "One full pass of completion-only Mobile Actions SFT should teach Action IR syntax "
            "and improve grouped-development executable exact match by at least 25 points."
        ),
        "decision": (
            "Keep this recipe only if schema validity reaches 95% and exact match improves by "
            "at least 25 percentage points over the untouched base."
        ),
        "base_checkpoint": {
            "source": "huggingface",
            "repo_id": BASE_REPO,
            "revision": BASE_REVISION,
            "expected_sha256": BASE_HASHES,
        },
        "data": {
            "train_manifest": str(prepared_data / "train.jsonl"),
            "train_sha256": prepared.hashes["train"],
            "dev_manifest": str(prepared_data / "dev.jsonl"),
            "dev_sha256": prepared.hashes["dev"],
            "max_seq_len": 2048,
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "overlength_policy": "error",
        },
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "min_learning_rate_ratio": 0.1,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": 1e-8,
            "warmup_steps": args.warmup_steps,
            "max_steps": None,
            "gradient_clip_norm": 1.0,
            "eval_every_steps": args.eval_every_steps,
            # Preserve resumable evidence during the paid run, including before
            # the first full-development evaluation at the default step 100.
            "save_every_steps": 50,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.001,
        },
        "execution": {
            "device": "cuda",
            "precision": "bf16",
            "deterministic": True,
            "jarvis_resource_id": str(args.jarvis_machine_id),
            "estimated_hourly_cost": args.hourly_cost,
        },
    }


def _export_checkpoint(source: Path, destination: Path, run_id: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in ("model.safetensors", "barun_config.json", "tokenizer.json"):
        shutil.copy2(source / name, destination / name)
        hashes[name] = sha256_file(destination / name)
    _write_json(
        destination / "checkpoint_manifest.json",
        {
            "schema_version": "barun-release-checkpoint-v1",
            "run_id": run_id,
            "source_checkpoint": str(source),
            "file_sha256": hashes,
        },
    )


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_root = args.artifact_root / args.run_id
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ExistingRunError(f"refusing to overwrite run root {run_root}") from error
    export = run_root / "export"
    export.mkdir()
    determinism_preflight = _cuda_determinism_preflight()
    _write_json(export / "environment.json", _environment(determinism_preflight))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the remote probe requires a CUDA GPU with bfloat16 support")
    _run_tests(export)

    base_dir = Path(
        snapshot_download(
            repo_id=BASE_REPO,
            revision=BASE_REVISION,
            allow_patterns=sorted(BASE_HASHES),
        )
    ).resolve()
    verify_checkpoint(base_dir, expected_sha256=BASE_HASHES)
    tokenizer = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    data_dir = run_root / "data"
    source = download_pinned_source(run_root / "cache")
    prepared = prepare_mobile_actions(
        source,
        data_dir,
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=BASE_REPO,
            revision=BASE_REVISION,
            sha256=BASE_HASHES["tokenizer.json"],
        ),
    )
    audit = json.loads(prepared.audit_path.read_text(encoding="utf-8"))
    official_eval_firewall = _official_eval_firewall(prepared, audit)
    exported_data = export / "data"
    _stage_prepared_data(prepared, exported_data)
    dev_manifest = exported_data / "dev.jsonl"
    target_max = int(audit["tokenization"]["per_split"]["dev"]["target"]["max"])
    generation_limit = min(256, target_max + 16)

    base_eval = export / "base-eval"
    base_predictions = base_eval / "predictions.jsonl"
    base_generation = generate_manifest(
        checkpoint_dir=base_dir,
        manifest_path=dev_manifest,
        manifest_sha256=prepared.hashes["dev"],
        predictions_path=base_predictions,
        device_name="cuda",
        batch_size=args.generation_batch_size,
        max_new_tokens=generation_limit,
        expected_checkpoint_sha256=BASE_HASHES,
    )
    write_scores(dev_manifest, base_predictions, base_eval / "scores")

    config_path = export / "training-config.json"
    _write_json(
        config_path,
        _training_config(
            args=args,
            workspace=export,
            prepared=prepared,
            prepared_data=exported_data,
        ),
    )
    training_summary = train_sft(TrainingRunConfig.from_json(config_path))
    selected_checkpoint = Path(
        training_summary.best_checkpoint or training_summary.final_checkpoint
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

    _export_checkpoint(selected_checkpoint, export / "checkpoint", args.run_id)

    base_metrics = json.loads((base_eval / "scores" / "aggregate.json").read_text())
    post_metrics = json.loads((post_eval / "scores" / "aggregate.json").read_text())
    result = {
        "schema_version": "barun-mobile-probe-result-v1",
        "run_id": args.run_id,
        "jarvis_machine_id": args.jarvis_machine_id,
        "data": {
            "revision": MOBILE_ACTIONS_REVISION,
            "train_rows": prepared.train_rows,
            "dev_rows": prepared.dev_rows,
            "official_eval_rows_opaque_unparsed": prepared.final_eval_rows,
            "official_eval_firewall": official_eval_firewall,
            "hashes": dict(prepared.hashes),
        },
        "generation_limit": generation_limit,
        "base": base_metrics,
        "post_sft": post_metrics,
        "delta": {
            "ast_exact_match": (
                post_metrics["ast_exact_match"]["value"] - base_metrics["ast_exact_match"]["value"]
            ),
            "schema_valid": (
                post_metrics["schema_valid"]["value"] - base_metrics["schema_valid"]["value"]
            ),
        },
        "gate": {
            "schema_valid_at_least_0_95": post_metrics["schema_valid"]["value"] >= 0.95,
            "exact_gain_at_least_0_25": (
                post_metrics["ast_exact_match"]["value"] - base_metrics["ast_exact_match"]["value"]
                >= 0.25
            ),
        },
        "training": training_summary.to_dict(),
        "generation": {
            "base": base_generation.to_dict(),
            "post_sft": post_generation.to_dict(),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result["gate"]["passed"] = all(result["gate"].values())
    _write_json(export / "result.json", result)
    _write_json(export / "artifact-sha256.json", _tree_manifest(export))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jarvis-machine-id", type=_positive_int, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/root/barun-artifacts"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--hourly-cost", type=float, default=255.15)
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    args = build_parser().parse_args()
    export = args.artifact_root / args.run_id / "export"
    try:
        result = run(args)
    except ExistingRunError:
        # A refused duplicate run is not this invocation's artifact directory.
        # Do not overwrite its failure record or hash manifest while reporting it.
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
