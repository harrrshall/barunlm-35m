"""Versioned, hash-verified CPU dynamic-int8 checkpoints for BarunLM."""

from __future__ import annotations

import json
import platform
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import torch
from tokenizers import Tokenizer
from torch import nn

from barunlm.config import BarunConfig
from barunlm.evaluation.generation import (
    REQUIRED_CHECKPOINT_FILES,
    load_verified_model,
    verify_checkpoint,
)
from barunlm.model import BarunLM
from barunlm.training.data import sha256_file

INT8_FORMAT_VERSION = "barun-cpu-dynamic-int8-v1"
INT8_MANIFEST_FILE = "quantization_manifest.json"
INT8_WEIGHTS_FILE = "model.int8.pt"
INT8_ARTIFACT_FILES = ("barun_config.json", INT8_WEIGHTS_FILE, "tokenizer.json")
MAX_INT8_WEIGHTS_BYTES = 8 * 1024**3
_QENGINE_LOCK = threading.RLock()


class QuantizationError(RuntimeError):
    """An int8 export or load request failed closed."""


@dataclass(frozen=True, slots=True)
class Int8CheckpointInfo:
    directory: Path
    manifest_sha256: str
    source_checkpoint_sha256: Mapping[str, str]
    artifact_sha256: Mapping[str, str]
    qengine: str
    torch_version: str
    quantized_modules: tuple[str, ...]
    float_linear_modules: tuple[str, ...]
    source_payload_bytes: int
    quantized_payload_bytes: int
    package_bytes: int
    reduction_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": dict(sorted(self.artifact_sha256.items())),
            "directory": str(self.directory),
            "float_linear_modules": list(self.float_linear_modules),
            "format_version": INT8_FORMAT_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "package_bytes": self.package_bytes,
            "qengine": self.qengine,
            "quantized_modules": list(self.quantized_modules),
            "quantized_payload_bytes": self.quantized_payload_bytes,
            "reduction_fraction": self.reduction_fraction,
            "source_checkpoint_sha256": dict(sorted(self.source_checkpoint_sha256.items())),
            "source_payload_bytes": self.source_payload_bytes,
            "torch_version": self.torch_version,
        }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise QuantizationError(f"duplicate JSON key in int8 manifest: {key!r}")
        value[key] = child
    return value


def _reject_constant(value: str) -> NoReturn:
    raise QuantizationError(f"non-finite JSON value in int8 manifest: {value!r}")


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except QuantizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QuantizationError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, Mapping):
        raise QuantizationError(f"JSON artifact must be an object: {path}")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing or unknown:
        raise QuantizationError(
            f"{path} fields differ: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )


def _hash_map(value: Any, *, expected_names: set[str], path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise QuantizationError(f"{path} must be an object")
    if set(value) != expected_names:
        raise QuantizationError(
            f"{path} file set differs: expected={sorted(expected_names)!r}, "
            f"got={sorted(str(name) for name in value)!r}"
        )
    hashes: dict[str, str] = {}
    for raw_name, digest in value.items():
        if not isinstance(raw_name, str) or not _is_sha256(digest):
            raise QuantizationError(f"{path} contains an invalid file name or SHA-256")
        hashes[raw_name] = digest
    return hashes


def _name_list(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(name, str) or not name for name in value):
        raise QuantizationError(f"{path} must be an array of non-empty strings")
    names = tuple(value)
    if names != tuple(sorted(set(names))):
        raise QuantizationError(f"{path} must be sorted and unique")
    return names


def _positive_int(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 1:
        raise QuantizationError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise QuantizationError(f"{path} must be a non-negative integer")
    return value


def _module_plan(model: BarunLM) -> tuple[tuple[str, ...], tuple[str, ...]]:
    quantized: list[str] = []
    kept_float: list[str] = []
    embedding_weight = model.embedding.weight
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight is embedding_weight:
            kept_float.append(name)
        else:
            quantized.append(name)
    if not quantized:
        raise QuantizationError("BarunLM contains no eligible Linear modules")
    return tuple(sorted(quantized)), tuple(sorted(kept_float))


def _parameter_counts(
    model: BarunLM,
    quantized_modules: tuple[str, ...],
    float_linear_modules: tuple[str, ...],
) -> dict[str, int]:
    modules = dict(model.named_modules())
    return {
        "float_linear_weight_elements": sum(
            modules[name].weight.numel() for name in float_linear_modules
        ),
        "quantized_linear_weight_elements": sum(
            modules[name].weight.numel() for name in quantized_modules
        ),
        "total_unique_parameters": model.parameter_counts()["total"],
    }


def _set_qengine(qengine: str) -> None:
    if not isinstance(qengine, str) or not qengine or qengine == "none":
        raise QuantizationError("qengine must be explicitly named and cannot be 'none'")
    supported = tuple(torch.backends.quantized.supported_engines)
    if qengine not in supported:
        raise QuantizationError(
            f"qengine {qengine!r} is unavailable; supported engines are {supported!r}"
        )
    try:
        torch.backends.quantized.engine = qengine
    except RuntimeError as error:
        raise QuantizationError(f"failed to activate qengine {qengine!r}") from error
    if torch.backends.quantized.engine != qengine:
        raise QuantizationError(f"qengine activation did not select {qengine!r}")


def _quantize_structure(
    model: BarunLM,
    *,
    qengine: str,
    expected_modules: tuple[str, ...],
) -> BarunLM:
    with _QENGINE_LOCK:
        _set_qengine(qengine)
        qconfig_spec = {
            name: torch.ao.quantization.default_dynamic_qconfig for name in expected_modules
        }
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                model.cpu().eval(),
                qconfig_spec=qconfig_spec,
                dtype=torch.qint8,
                inplace=False,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise QuantizationError(
                f"dynamic int8 conversion failed for qengine {qengine!r}"
            ) from error
    return quantized


def _verify_quantized_modules(
    model: BarunLM,
    *,
    expected_modules: tuple[str, ...],
    expected_weight_elements: int,
) -> None:
    modules = dict(model.named_modules())
    observed_elements = 0
    for name in expected_modules:
        module = modules.get(name)
        if not isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            raise QuantizationError(f"module {name!r} is not a dynamic quantized Linear")
        weight = module.weight()
        if weight.dtype is not torch.qint8:
            raise QuantizationError(f"module {name!r} does not contain qint8 weights")
        if weight.q_zero_point() != 0:
            raise QuantizationError(f"module {name!r} does not use symmetric zero-point weights")
        observed_elements += weight.numel()
    if observed_elements != expected_weight_elements:
        raise QuantizationError(
            "quantized weight element count differs from the versioned manifest"
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_dynamic_int8_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    expected_source_sha256: Mapping[str, str],
    qengine: str,
) -> Int8CheckpointInfo:
    """Export a new CPU dynamic-int8 checkpoint without modifying its source."""

    source = Path(source_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise QuantizationError(f"refusing to overwrite int8 checkpoint directory: {output}")
    with _QENGINE_LOCK:
        _set_qengine(qengine)
    source_hashes = verify_checkpoint(source, expected_sha256=expected_source_sha256)
    model, _, loaded_hashes = load_verified_model(source, expected_sha256=source_hashes)
    if loaded_hashes != source_hashes:  # pragma: no cover - defensive invariant
        raise QuantizationError("source checkpoint hashes changed between verification and load")
    source_model_hashes = {name: source_hashes[name] for name in REQUIRED_CHECKPOINT_FILES}
    quantized_modules, float_linear_modules = _module_plan(model)
    counts = _parameter_counts(model, quantized_modules, float_linear_modules)
    quantized = _quantize_structure(
        model,
        qengine=qengine,
        expected_modules=quantized_modules,
    )
    _verify_quantized_modules(
        quantized,
        expected_modules=quantized_modules,
        expected_weight_elements=counts["quantized_linear_weight_elements"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        shutil.copyfile(source / "barun_config.json", output / "barun_config.json")
        shutil.copyfile(source / "tokenizer.json", output / "tokenizer.json")
        torch.save(quantized.state_dict(), output / INT8_WEIGHTS_FILE)
        artifact_hashes = {name: sha256_file(output / name) for name in INT8_ARTIFACT_FILES}
        if artifact_hashes["barun_config.json"] != source_model_hashes["barun_config.json"]:
            raise QuantizationError("exported config hash differs from the source checkpoint")
        if artifact_hashes["tokenizer.json"] != source_model_hashes["tokenizer.json"]:
            raise QuantizationError("exported tokenizer hash differs from the source checkpoint")
        source_payload_bytes = sum(
            (source / name).stat().st_size for name in REQUIRED_CHECKPOINT_FILES
        )
        quantized_payload_bytes = sum(
            (output / name).stat().st_size for name in INT8_ARTIFACT_FILES
        )
        reduction_fraction = 1.0 - quantized_payload_bytes / source_payload_bytes
        manifest = {
            "algorithm": {
                "activation_quantization": "dynamic",
                "implementation": "torch.ao.quantization.quantize_dynamic",
                "qengine": qengine,
                "tied_embedding_linear_policy": "keep_float",
                "weight_dtype": "qint8",
                "weight_qscheme": "per_tensor_affine_symmetric_zero_point",
            },
            "architecture": "barunlm.model.BarunLM",
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "float_linear_modules": list(float_linear_modules),
            "parameter_counts": counts,
            "quantized_modules": list(quantized_modules),
            "runtime": {
                "platform_machine": platform.machine(),
                "platform_system": platform.system(),
                "torch_version": torch.__version__,
            },
            "schema_version": INT8_FORMAT_VERSION,
            "size_bytes": {
                "quantized_payload": quantized_payload_bytes,
                "reduction_fraction": reduction_fraction,
                "source_payload": source_payload_bytes,
            },
            "source_checkpoint_sha256": dict(sorted(source_model_hashes.items())),
        }
        _write_json(output / INT8_MANIFEST_FILE, manifest)
        manifest_sha256 = sha256_file(output / INT8_MANIFEST_FILE)
        return verify_int8_checkpoint(output, expected_manifest_sha256=manifest_sha256)
    except Exception:
        shutil.rmtree(output)
        raise


def verify_int8_checkpoint(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> Int8CheckpointInfo:
    """Verify a versioned int8 checkpoint and its externally pinned manifest hash."""

    checkpoint = Path(directory).resolve()
    if not _is_sha256(expected_manifest_sha256):
        raise QuantizationError("expected_manifest_sha256 must be a lowercase SHA-256")
    manifest_path = checkpoint / INT8_MANIFEST_FILE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise QuantizationError(f"int8 checkpoint is missing {INT8_MANIFEST_FILE!r}")
    try:
        children = tuple(checkpoint.iterdir())
    except OSError as error:
        raise QuantizationError(
            f"cannot inventory int8 checkpoint directory: {checkpoint}"
        ) from error
    expected_children = set(INT8_ARTIFACT_FILES) | {INT8_MANIFEST_FILE}
    observed_children = {child.name for child in children}
    if observed_children != expected_children or any(not child.is_file() for child in children):
        raise QuantizationError(
            "int8 checkpoint file set differs: "
            f"expected={sorted(expected_children)!r}, got={sorted(observed_children)!r}"
        )
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise QuantizationError(
            "int8 manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    manifest = _strict_json(manifest_path)
    _exact_fields(
        manifest,
        {
            "algorithm",
            "architecture",
            "artifact_sha256",
            "float_linear_modules",
            "parameter_counts",
            "quantized_modules",
            "runtime",
            "schema_version",
            "size_bytes",
            "source_checkpoint_sha256",
        },
        path="$",
    )
    if manifest["schema_version"] != INT8_FORMAT_VERSION:
        raise QuantizationError("unsupported int8 checkpoint format")
    if manifest["architecture"] != "barunlm.model.BarunLM":
        raise QuantizationError("unsupported int8 checkpoint architecture")

    algorithm = manifest["algorithm"]
    if not isinstance(algorithm, Mapping):
        raise QuantizationError("$.algorithm must be an object")
    expected_algorithm = {
        "activation_quantization": "dynamic",
        "implementation": "torch.ao.quantization.quantize_dynamic",
        "tied_embedding_linear_policy": "keep_float",
        "weight_dtype": "qint8",
        "weight_qscheme": "per_tensor_affine_symmetric_zero_point",
    }
    _exact_fields(algorithm, set(expected_algorithm) | {"qengine"}, path="$.algorithm")
    for name, expected in expected_algorithm.items():
        if algorithm[name] != expected:
            raise QuantizationError(f"unsupported int8 algorithm field {name!r}")
    qengine = algorithm["qengine"]
    if not isinstance(qengine, str) or not qengine:
        raise QuantizationError("$.algorithm.qengine must be a non-empty string")

    artifact_hashes = _hash_map(
        manifest["artifact_sha256"],
        expected_names=set(INT8_ARTIFACT_FILES),
        path="$.artifact_sha256",
    )
    source_hashes = _hash_map(
        manifest["source_checkpoint_sha256"],
        expected_names=set(REQUIRED_CHECKPOINT_FILES),
        path="$.source_checkpoint_sha256",
    )
    if artifact_hashes["barun_config.json"] != source_hashes["barun_config.json"]:
        raise QuantizationError("int8 config provenance differs from the source checkpoint")
    if artifact_hashes["tokenizer.json"] != source_hashes["tokenizer.json"]:
        raise QuantizationError("int8 tokenizer provenance differs from the source checkpoint")
    for name, expected in sorted(artifact_hashes.items()):
        path = checkpoint / name
        if not path.is_file() or path.is_symlink():
            raise QuantizationError(f"int8 checkpoint is missing hashed file {name!r}")
        actual = sha256_file(path)
        if actual != expected:
            raise QuantizationError(
                f"int8 checkpoint SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )

    runtime = manifest["runtime"]
    if not isinstance(runtime, Mapping):
        raise QuantizationError("$.runtime must be an object")
    _exact_fields(
        runtime,
        {"platform_machine", "platform_system", "torch_version"},
        path="$.runtime",
    )
    torch_version = runtime["torch_version"]
    if not isinstance(torch_version, str) or not torch_version:
        raise QuantizationError("$.runtime.torch_version must be a non-empty string")
    for name in ("platform_machine", "platform_system"):
        if not isinstance(runtime[name], str) or not runtime[name]:
            raise QuantizationError(f"$.runtime.{name} must be a non-empty string")

    counts = manifest["parameter_counts"]
    if not isinstance(counts, Mapping):
        raise QuantizationError("$.parameter_counts must be an object")
    _exact_fields(
        counts,
        {
            "float_linear_weight_elements",
            "quantized_linear_weight_elements",
            "total_unique_parameters",
        },
        path="$.parameter_counts",
    )
    _nonnegative_int(
        counts["float_linear_weight_elements"],
        path="$.parameter_counts.float_linear_weight_elements",
    )
    _positive_int(
        counts["quantized_linear_weight_elements"],
        path="$.parameter_counts.quantized_linear_weight_elements",
    )
    _positive_int(
        counts["total_unique_parameters"],
        path="$.parameter_counts.total_unique_parameters",
    )

    sizes = manifest["size_bytes"]
    if not isinstance(sizes, Mapping):
        raise QuantizationError("$.size_bytes must be an object")
    _exact_fields(
        sizes,
        {"quantized_payload", "reduction_fraction", "source_payload"},
        path="$.size_bytes",
    )
    source_payload_bytes = _positive_int(
        sizes["source_payload"], path="$.size_bytes.source_payload"
    )
    recorded_quantized_payload = _positive_int(
        sizes["quantized_payload"], path="$.size_bytes.quantized_payload"
    )
    observed_quantized_payload = sum(
        (checkpoint / name).stat().st_size for name in INT8_ARTIFACT_FILES
    )
    if recorded_quantized_payload != observed_quantized_payload:
        raise QuantizationError("recorded int8 payload size differs from the verified files")
    weight_bytes = (checkpoint / INT8_WEIGHTS_FILE).stat().st_size
    source_payload_bound = max(16 * 1024**2, 2 * source_payload_bytes)
    if weight_bytes > MAX_INT8_WEIGHTS_BYTES or weight_bytes > source_payload_bound:
        raise QuantizationError("verified int8 weight artifact exceeds the format size bound")
    expected_reduction = 1.0 - observed_quantized_payload / source_payload_bytes
    reduction = sizes["reduction_fraction"]
    if (
        not isinstance(reduction, (int, float))
        or not abs(float(reduction) - expected_reduction) < 1e-12
    ):
        raise QuantizationError("recorded int8 size reduction is inconsistent")

    quantized_modules = _name_list(manifest["quantized_modules"], path="$.quantized_modules")
    float_linear_modules = _name_list(
        manifest["float_linear_modules"], path="$.float_linear_modules"
    )
    package_bytes = observed_quantized_payload + manifest_path.stat().st_size
    return Int8CheckpointInfo(
        directory=checkpoint,
        manifest_sha256=actual_manifest_sha256,
        source_checkpoint_sha256=MappingProxyType(source_hashes),
        artifact_sha256=MappingProxyType(artifact_hashes),
        qengine=qengine,
        torch_version=torch_version,
        quantized_modules=quantized_modules,
        float_linear_modules=float_linear_modules,
        source_payload_bytes=source_payload_bytes,
        quantized_payload_bytes=observed_quantized_payload,
        package_bytes=package_bytes,
        reduction_fraction=expected_reduction,
    )


def load_verified_int8_model(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[BarunLM, Tokenizer, Int8CheckpointInfo]:
    """Load only the explicitly requested, hash-verified dynamic-int8 representation."""

    info = verify_int8_checkpoint(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if info.torch_version != torch.__version__:
        raise QuantizationError(
            "int8 artifact requires exact torch version "
            f"{info.torch_version!r}, current runtime is {torch.__version__!r}"
        )
    manifest = _strict_json(info.directory / INT8_MANIFEST_FILE)
    runtime = manifest["runtime"]
    if (
        runtime["platform_machine"] != platform.machine()
        or runtime["platform_system"] != platform.system()
    ):
        raise QuantizationError(
            "int8 artifact platform differs from the current CPU runtime; "
            "cross-platform packed-weight loading is not claimed by format v1"
        )
    config = BarunConfig.from_json(info.directory / "barun_config.json")
    tokenizer = Tokenizer.from_file(str(info.directory / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise QuantizationError("int8 tokenizer and model vocabulary sizes differ")
    float_model = BarunLM(config).cpu().eval()
    quantized_modules, float_linear_modules = _module_plan(float_model)
    if quantized_modules != info.quantized_modules:
        raise QuantizationError("int8 quantized module plan differs from the current architecture")
    if float_linear_modules != info.float_linear_modules:
        raise QuantizationError("int8 float-module plan differs from the current architecture")
    counts = _parameter_counts(float_model, quantized_modules, float_linear_modules)
    if counts != manifest["parameter_counts"]:
        raise QuantizationError("int8 parameter counts differ from the current architecture")
    with _QENGINE_LOCK:
        quantized_model = _quantize_structure(
            float_model,
            qengine=info.qengine,
            expected_modules=quantized_modules,
        )
        try:
            state = torch.load(
                info.directory / INT8_WEIGHTS_FILE,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise QuantizationError("failed to deserialize verified int8 weights") from error
        if not isinstance(state, Mapping):
            raise QuantizationError("verified int8 weight artifact is not a state mapping")
        if config.tie_embeddings:
            embedding_weight = state.get("embedding.weight")
            head_weight = state.get("lm_head.weight")
            if (
                not isinstance(embedding_weight, torch.Tensor)
                or not isinstance(head_weight, torch.Tensor)
                or embedding_weight.dtype is not torch.float32
                or head_weight.dtype is not torch.float32
                or not torch.equal(embedding_weight, head_weight)
            ):
                raise QuantizationError(
                    "verified int8 state does not preserve equal FP32 tied embedding/head weights"
                )
        try:
            missing, unexpected = quantized_model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise QuantizationError("verified int8 weights do not match BarunLM") from error
        if missing or unexpected:
            raise QuantizationError(
                "verified int8 weights differ from BarunLM: "
                f"missing={missing}, unexpected={unexpected}"
            )
    if config.tie_embeddings and (
        not isinstance(quantized_model.lm_head, nn.Linear)
        or quantized_model.lm_head.weight is not quantized_model.embedding.weight
        or quantized_model.embedding.weight.dtype is not torch.float32
    ):
        raise QuantizationError("loaded int8 model did not preserve the FP32 tied output head")
    _verify_quantized_modules(
        quantized_model,
        expected_modules=quantized_modules,
        expected_weight_elements=counts["quantized_linear_weight_elements"],
    )
    return quantized_model.eval(), tokenizer, info


__all__ = [
    "INT8_ARTIFACT_FILES",
    "INT8_FORMAT_VERSION",
    "INT8_MANIFEST_FILE",
    "INT8_WEIGHTS_FILE",
    "MAX_INT8_WEIGHTS_BYTES",
    "Int8CheckpointInfo",
    "QuantizationError",
    "export_dynamic_int8_checkpoint",
    "load_verified_int8_model",
    "verify_int8_checkpoint",
]
