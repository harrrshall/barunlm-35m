"""Score one exact checkpoint with the frozen candidate-v2 Mobile regression gate."""

from __future__ import annotations

# Match the candidate-v2 generation run before importing torch or project modules.
import argparse
import json
import os
from pathlib import Path
from typing import Any

DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
_CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT = (
    os.environ.get("CUBLAS_WORKSPACE_CONFIG") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
)

import torch

from barunlm.evaluation.mobile_regression import run_mobile_regression

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = REPOSITORY_ROOT / "configs" / "mobile_regression_v1.json"


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
        raise RuntimeError("PyTorch exposes a partial, unverifiable cuDNN SDPA interface")
    return evidence


def _runtime_environment() -> dict[str, Any]:
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG changed after import")
    if not _CUBLAS_CONFIG_SET_BEFORE_TORCH_IMPORT:
        raise RuntimeError("cuBLAS determinism was not configured before importing torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before the Mobile regression preflight")
    sdpa = _force_math_sdpa()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frozen Mobile regression gate requires CUDA with bfloat16")
    return {
        "device": "cuda",
        "dtype": "torch.bfloat16",
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": actual_cublas,
        "sdpa_backends": sdpa,
        "configured_before_torch_import": True,
        "cuda_initialized_before_preflight": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-hashes",
        type=Path,
        required=True,
        help="Exact JSON hash map or checkpoint manifest containing file_sha256.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional relocated copy of the frozen 756-row development manifest.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Optional relocated copy of the frozen Mobile adapter audit.",
    )
    parser.add_argument(
        "--reference-aggregate",
        type=Path,
        help="Optional relocated copy of the exact candidate-v2 aggregate evidence.",
    )
    parser.add_argument(
        "--reference-samples",
        type=Path,
        help="Optional relocated copy of the exact candidate-v2 sample evidence.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_mobile_regression(
        checkpoint_dir=args.checkpoint,
        checkpoint_hashes_path=args.checkpoint_hashes,
        output_dir=args.output,
        repository_root=args.repository_root,
        preregistration_path=args.preregistration,
        runtime_environment=_runtime_environment(),
        manifest_path=args.manifest,
        audit_path=args.audit,
        reference_aggregate_path=args.reference_aggregate,
        reference_samples_path=args.reference_samples,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
