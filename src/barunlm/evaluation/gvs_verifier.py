"""CPU-testable verifier architecture contract for proposed BarunAction GVS-v1.

The production proposal freezes the BarunLM generator/backbone, installs bias-free
rank-eight LoRA branches on every q/v projection, and trains one biased scalar head.
For the release configuration this is exactly 135,617 trainable parameters.  This
module implements and audits that architecture, deterministic multi-positive listwise
loss, and rank selection.  It never loads a checkpoint or dataset and never authorizes
training, model access, CUDA, or Jarvis use; external receipts must bind actual weights.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
import math
import platform
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import safetensors.torch as safetensors_torch
import torch
import torch.nn.functional as F
from safetensors import SafetensorError
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from torch import Tensor, nn

import barunlm.model as barun_model_module
from barunlm.config import BarunConfig
from barunlm.model import (
    BarunBlock,
    BarunLM,
    BoundedSwiGLU,
    GroupedAttention,
    PartialRotaryEmbedding,
    ResidualSelector,
    RMSNorm,
)

from .gvs_support import FROZEN_CANDIDATE_COUNT, count_verifier_parameters

# Runtime behavior must not be able to change while the receipt remains identical.
# Keep import-time references for the external operations used by scoring, selection,
# loss, serialization, hashing, and Unicode validation, then verify both the captured
# aliases and their originating module slots at every public execution boundary.
_F_LINEAR = F.linear
_F_DROPOUT = F.dropout
_F_EMBEDDING = F.embedding
_F_SCALED_DOT_PRODUCT_ATTENTION = F.scaled_dot_product_attention
_F_SILU = F.silu
_HASH_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_MARSHAL_DUMPS = marshal.dumps
_MATH_SQRT = math.sqrt
_MODEL_LOCAL_BIDIRECTIONAL_MASK = barun_model_module._local_bidirectional_mask
_MODEL_LOCAL_BLOCK_MASK = barun_model_module._local_block_mask
_MODEL_LOCAL_CAUSAL_MASK = barun_model_module._local_causal_mask
_MODEL_ROTATE_HALF = barun_model_module._rotate_half
_NN_INIT_KAIMING_UNIFORM = nn.init.kaiming_uniform_
_SAFETENSORS_LOAD = safetensors_load
_SAFETENSORS_SAVE = safetensors_save
_TORCH_ARANGE = torch.arange
_TORCH_CAT = torch.cat
_TORCH_EMPTY = torch.empty
_TORCH_EQUAL = torch.equal
_TORCH_GENERATOR = torch.Generator
_TORCH_ISFINITE = torch.isfinite
_TORCH_IS_FLOATING_POINT = torch.is_floating_point
_TORCH_LOGSUMEXP = torch.logsumexp
_TORCH_RSQRT = torch.rsqrt
_TORCH_SIGMOID = torch.sigmoid
_TORCH_STACK = torch.stack
_TORCH_TENSOR = torch.tensor
_TORCH_WHERE = torch.where
_TORCH_ZEROS = torch.zeros
_UNICODE_NORMALIZE = unicodedata.normalize

GVS_VERIFIER_CONTRACT_VERSION = "barun-gvs-verifier-contract-v3"
GVS_VERIFIER_RECEIPT_VERSION = "barun-gvs-verifier-receipt-v2"
GVS_VERIFIER_ADAPTER_VERSION = "barun-gvs-verifier-adapter-v1"

PRODUCTION_RANK = 8
PRODUCTION_ALPHA = 8
PRODUCTION_SEED = 17
PRODUCTION_GENERATOR_PARAMETERS = 35_072_768
PRODUCTION_VERIFIER_PARAMETERS = 135_617
PRODUCTION_SYSTEM_PARAMETERS = 35_208_385

_MAX_SEED = 2**63 - 1
_MAX_RECEIPT_JSON_BYTES = 512 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_INTEGER_DIGITS = 128
_MAX_JSON_NODES = 100_000
_MAX_JSON_TEXT_BYTES = 512 * 1024
_MAX_ADAPTER_BYTES = 64 * 1024 * 1024
_ADAPTER_MANIFEST_TENSOR = "__barun_gvs_adapter_manifest_utf8__"
_VERIFIER_SOURCE_PATH = Path(__file__)
_MODEL_SOURCE_PATH = Path(inspect.getsourcefile(BarunLM) or "")
if not _VERIFIER_SOURCE_PATH.is_file() or not _MODEL_SOURCE_PATH.is_file():  # pragma: no cover
    raise RuntimeError("GVS verifier requires inspectable verifier and BarunLM source files")
_VERIFIER_SOURCE_SHA256_AT_IMPORT = _HASH_SHA256(_VERIFIER_SOURCE_PATH.read_bytes()).hexdigest()
_MODEL_SOURCE_SHA256_AT_IMPORT = _HASH_SHA256(_MODEL_SOURCE_PATH.read_bytes()).hexdigest()


class GVSVerifierError(ValueError):
    """The verifier architecture, loss, or audit contract is invalid."""


def _behavior_value_record(value: object, *, label: str, depth: int = 0) -> object:
    """Encode bounded callable state without address-bearing ``repr`` output."""

    if depth > 8:
        raise GVSVerifierError(f"{label} exceeds the callable-state depth bound")
    if value is None or type(value) in {bool, int, str}:
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise GVSVerifierError(f"{label} contains a non-finite callable-state float")
        return {"kind": "float", "value": value.hex()}
    if type(value) is bytes:
        return {"kind": "bytes", "sha256": _HASH_SHA256(value).hexdigest(), "size": len(value)}
    if type(value) in {tuple, list}:
        if len(value) > 256:
            raise GVSVerifierError(f"{label} exceeds the callable-state sequence bound")
        return {
            "items": [
                _behavior_value_record(child, label=f"{label}[]", depth=depth + 1)
                for child in value
            ],
            "kind": type(value).__name__,
        }
    if type(value) is dict:
        if len(value) > 256 or any(type(key) is not str for key in value):
            raise GVSVerifierError(f"{label} has unsupported callable-state mapping keys")
        return {
            "items": {
                key: _behavior_value_record(
                    child,
                    label=f"{label}.{key}",
                    depth=depth + 1,
                )
                for key, child in sorted(value.items())
            },
            "kind": "dict",
        }
    if isinstance(value, Tensor):
        return {
            "device": str(value.device),
            "dtype": str(value.dtype),
            "kind": "tensor",
            "sha256": _tensor_sha256(value),
            "shape": list(value.shape),
        }
    if inspect.ismodule(value):
        return {"kind": "module", "name": getattr(value, "__name__", None)}
    if inspect.isclass(value):
        return {
            "kind": "type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if callable(value):
        target = getattr(value, "__func__", value)
        code = getattr(target, "__code__", None)
        return {
            "code_sha256": (
                None if code is None else _HASH_SHA256(_MARSHAL_DUMPS(code)).hexdigest()
            ),
            "kind": "callable",
            "module": getattr(target, "__module__", None),
            "qualname": getattr(target, "__qualname__", getattr(target, "__name__", None)),
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    raise GVSVerifierError(f"{label} contains unsupported callable state {type(value).__name__}")


def _callable_identity(value: object, *, label: str) -> dict[str, object]:
    if not callable(value):
        raise GVSVerifierError(f"runtime dependency {label!r} is no longer callable")
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    code_sha256: str | None = None
    if code is not None:
        try:
            code_sha256 = _HASH_SHA256(_MARSHAL_DUMPS(code)).hexdigest()
        except (TypeError, ValueError) as error:
            raise GVSVerifierError(
                f"runtime dependency {label!r} exposes invalid Python code"
            ) from error
    defaults = getattr(target, "__defaults__", None)
    kwdefaults = getattr(target, "__kwdefaults__", None)
    closure: list[object] = []
    for index, cell in enumerate(getattr(target, "__closure__", None) or ()):
        try:
            child = cell.cell_contents
        except ValueError:
            closure.append({"kind": "empty_cell"})
        else:
            closure.append(
                _behavior_value_record(child, label=f"{label}.closure[{index}]", depth=1)
            )
    instance_state: object = None
    if code is None and not inspect.isclass(value) and hasattr(value, "__dict__"):
        state = vars(value)
        if state:
            instance_state = _behavior_value_record(
                state,
                label=f"{label}.instance_state",
                depth=1,
            )
    value_type = type(value)
    return {
        "code_sha256": code_sha256,
        "closure_sha256": _sha256_json(closure),
        "dependency": label,
        "defaults_sha256": _sha256_json(
            _behavior_value_record(
                {"defaults": defaults, "kwdefaults": kwdefaults},
                label=f"{label}.defaults",
            )
        ),
        "instance_state_sha256": (None if instance_state is None else _sha256_json(instance_state)),
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "type_module": value_type.__module__,
        "type_qualname": value_type.__qualname__,
    }


def _assert_import_time_dependency_bindings(
    _expected: tuple[tuple[str, object, str, object, str], ...] = (
        ("F.linear", F, "linear", _F_LINEAR, "_F_LINEAR"),
        ("F.dropout", F, "dropout", _F_DROPOUT, "_F_DROPOUT"),
        ("F.embedding", F, "embedding", _F_EMBEDDING, "_F_EMBEDDING"),
        (
            "F.scaled_dot_product_attention",
            F,
            "scaled_dot_product_attention",
            _F_SCALED_DOT_PRODUCT_ATTENTION,
            "_F_SCALED_DOT_PRODUCT_ATTENTION",
        ),
        ("F.silu", F, "silu", _F_SILU, "_F_SILU"),
        ("hashlib.sha256", hashlib, "sha256", _HASH_SHA256, "_HASH_SHA256"),
        ("json.dumps", json, "dumps", _JSON_DUMPS, "_JSON_DUMPS"),
        ("json.loads", json, "loads", _JSON_LOADS, "_JSON_LOADS"),
        ("marshal.dumps", marshal, "dumps", _MARSHAL_DUMPS, "_MARSHAL_DUMPS"),
        ("math.sqrt", math, "sqrt", _MATH_SQRT, "_MATH_SQRT"),
        (
            "barun_model._local_bidirectional_mask",
            barun_model_module,
            "_local_bidirectional_mask",
            _MODEL_LOCAL_BIDIRECTIONAL_MASK,
            "_MODEL_LOCAL_BIDIRECTIONAL_MASK",
        ),
        (
            "barun_model._local_block_mask",
            barun_model_module,
            "_local_block_mask",
            _MODEL_LOCAL_BLOCK_MASK,
            "_MODEL_LOCAL_BLOCK_MASK",
        ),
        (
            "barun_model._local_causal_mask",
            barun_model_module,
            "_local_causal_mask",
            _MODEL_LOCAL_CAUSAL_MASK,
            "_MODEL_LOCAL_CAUSAL_MASK",
        ),
        (
            "barun_model._rotate_half",
            barun_model_module,
            "_rotate_half",
            _MODEL_ROTATE_HALF,
            "_MODEL_ROTATE_HALF",
        ),
        (
            "nn.init.kaiming_uniform_",
            nn.init,
            "kaiming_uniform_",
            _NN_INIT_KAIMING_UNIFORM,
            "_NN_INIT_KAIMING_UNIFORM",
        ),
        (
            "safetensors.torch.load",
            safetensors_torch,
            "load",
            _SAFETENSORS_LOAD,
            "_SAFETENSORS_LOAD",
        ),
        (
            "safetensors.torch.save",
            safetensors_torch,
            "save",
            _SAFETENSORS_SAVE,
            "_SAFETENSORS_SAVE",
        ),
        ("torch.arange", torch, "arange", _TORCH_ARANGE, "_TORCH_ARANGE"),
        ("torch.cat", torch, "cat", _TORCH_CAT, "_TORCH_CAT"),
        ("torch.empty", torch, "empty", _TORCH_EMPTY, "_TORCH_EMPTY"),
        ("torch.equal", torch, "equal", _TORCH_EQUAL, "_TORCH_EQUAL"),
        ("torch.Generator", torch, "Generator", _TORCH_GENERATOR, "_TORCH_GENERATOR"),
        ("torch.isfinite", torch, "isfinite", _TORCH_ISFINITE, "_TORCH_ISFINITE"),
        (
            "torch.is_floating_point",
            torch,
            "is_floating_point",
            _TORCH_IS_FLOATING_POINT,
            "_TORCH_IS_FLOATING_POINT",
        ),
        ("torch.logsumexp", torch, "logsumexp", _TORCH_LOGSUMEXP, "_TORCH_LOGSUMEXP"),
        ("torch.rsqrt", torch, "rsqrt", _TORCH_RSQRT, "_TORCH_RSQRT"),
        ("torch.sigmoid", torch, "sigmoid", _TORCH_SIGMOID, "_TORCH_SIGMOID"),
        ("torch.stack", torch, "stack", _TORCH_STACK, "_TORCH_STACK"),
        ("torch.tensor", torch, "tensor", _TORCH_TENSOR, "_TORCH_TENSOR"),
        ("torch.where", torch, "where", _TORCH_WHERE, "_TORCH_WHERE"),
        ("torch.zeros", torch, "zeros", _TORCH_ZEROS, "_TORCH_ZEROS"),
        (
            "unicodedata.normalize",
            unicodedata,
            "normalize",
            _UNICODE_NORMALIZE,
            "_UNICODE_NORMALIZE",
        ),
    ),
) -> None:
    changed = [
        label
        for label, module, attribute, expected, alias_name in _expected
        if getattr(module, attribute, None) is not expected
        or globals().get(alias_name) is not expected
    ]
    if changed:
        raise GVSVerifierError(
            f"import-time verifier dependency binding changed: {', '.join(sorted(set(changed)))}"
        )


def _assert_import_time_source_bindings(
    _verifier_path: Path = _VERIFIER_SOURCE_PATH,
    _verifier_sha256: str = _VERIFIER_SOURCE_SHA256_AT_IMPORT,
    _model_path: Path = _MODEL_SOURCE_PATH,
    _model_sha256: str = _MODEL_SOURCE_SHA256_AT_IMPORT,
) -> None:
    try:
        verifier_sha256 = _HASH_SHA256(_verifier_path.read_bytes()).hexdigest()
        model_sha256 = _HASH_SHA256(_model_path.read_bytes()).hexdigest()
    except OSError as error:
        raise GVSVerifierError("verifier/model source became unreadable after import") from error
    if verifier_sha256 != _verifier_sha256 or model_sha256 != _model_sha256:
        raise GVSVerifierError("verifier/model source bytes changed after import")


def _behavior_dependency_record() -> dict[str, object]:
    dependencies = (
        ("F.linear", _F_LINEAR),
        ("F.dropout", _F_DROPOUT),
        ("F.embedding", _F_EMBEDDING),
        ("F.scaled_dot_product_attention", _F_SCALED_DOT_PRODUCT_ATTENTION),
        ("F.silu", _F_SILU),
        ("hashlib.sha256", _HASH_SHA256),
        ("json.dumps", _JSON_DUMPS),
        ("json.loads", _JSON_LOADS),
        ("marshal.dumps", _MARSHAL_DUMPS),
        ("math.sqrt", _MATH_SQRT),
        ("barun_model._local_bidirectional_mask", _MODEL_LOCAL_BIDIRECTIONAL_MASK),
        ("barun_model._local_block_mask", _MODEL_LOCAL_BLOCK_MASK),
        ("barun_model._local_causal_mask", _MODEL_LOCAL_CAUSAL_MASK),
        ("barun_model._rotate_half", _MODEL_ROTATE_HALF),
        ("nn.init.kaiming_uniform_", _NN_INIT_KAIMING_UNIFORM),
        ("safetensors.torch.load", _SAFETENSORS_LOAD),
        ("safetensors.torch.save", _SAFETENSORS_SAVE),
        ("torch.arange", _TORCH_ARANGE),
        ("torch.cat", _TORCH_CAT),
        ("torch.empty", _TORCH_EMPTY),
        ("torch.equal", _TORCH_EQUAL),
        ("torch.Generator", _TORCH_GENERATOR),
        ("torch.isfinite", _TORCH_ISFINITE),
        ("torch.is_floating_point", _TORCH_IS_FLOATING_POINT),
        ("torch.logsumexp", _TORCH_LOGSUMEXP),
        ("torch.rsqrt", _TORCH_RSQRT),
        ("torch.sigmoid", _TORCH_SIGMOID),
        ("torch.stack", _TORCH_STACK),
        ("torch.tensor", _TORCH_TENSOR),
        ("torch.where", _TORCH_WHERE),
        ("torch.zeros", _TORCH_ZEROS),
        ("unicodedata.normalize", _UNICODE_NORMALIZE),
    )
    values = [_callable_identity(value, label=label) for label, value in dependencies]
    return {
        "dependencies": values,
        "dependency_count": len(values),
    }


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise GVSVerifierError("verifier receipt contains an oversized integer")
    return int(value)


def _canonical_json(value: object) -> str:
    try:
        return _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GVSVerifierError("verifier artifact is not strict JSON") from error


def _snapshot_json(
    value: object,
    *,
    label: str,
    active_ids: set[int] | None = None,
    budget: dict[str, int] | None = None,
    memo: dict[int, tuple[object, int, int]] | None = None,
    depth: int = 0,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise GVSVerifierError(f"{label} exceeds maximum JSON depth {_MAX_JSON_DEPTH}")
    active = set() if active_ids is None else active_ids
    snapshots = {} if memo is None else memo
    remaining = (
        {"nodes": _MAX_JSON_NODES, "text_bytes": _MAX_JSON_TEXT_BYTES} if budget is None else budget
    )
    remaining["nodes"] -= 1
    if remaining["nodes"] < 0:
        raise GVSVerifierError(f"{label} exceeds the bounded JSON node count")
    if value is None or type(value) in {bool, int, str}:
        if type(value) is int and abs(value) >= 10**_MAX_JSON_INTEGER_DIGITS:
            raise GVSVerifierError(f"{label} contains an oversized integer")
        if type(value) is str:
            if value != _UNICODE_NORMALIZE("NFC", value):
                raise GVSVerifierError(f"{label} contains non-NFC text")
            try:
                encoded = value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise GVSVerifierError(f"{label} contains invalid UTF-8 text") from error
            remaining["text_bytes"] -= len(encoded)
            if remaining["text_bytes"] < 0:
                raise GVSVerifierError(f"{label} exceeds the bounded JSON text size")
        return value
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise GVSVerifierError(f"{label} contains a JSON cycle")
        if identity in snapshots:
            snapshot, node_cost, text_cost = snapshots[identity]
            remaining["nodes"] -= node_cost - 1
            remaining["text_bytes"] -= text_cost
            if remaining["nodes"] < 0 or remaining["text_bytes"] < 0:
                raise GVSVerifierError(f"{label} exceeds its bounded JSON snapshot size")
            return copy.deepcopy(snapshot)
        active.add(identity)
        try:
            nodes_before = remaining["nodes"]
            text_before = remaining["text_bytes"]
            try:
                children = tuple(value)
            except RuntimeError as error:  # pragma: no cover - concurrent mutation guard
                raise GVSVerifierError(f"{label} changed while it was snapshotted") from error
            result = [
                _snapshot_json(
                    child,
                    label=f"{label}[]",
                    active_ids=active,
                    budget=remaining,
                    memo=snapshots,
                    depth=depth + 1,
                )
                for child in children
            ]
            snapshots[identity] = (
                result,
                nodes_before - remaining["nodes"] + 1,
                text_before - remaining["text_bytes"],
            )
            return result
        finally:
            active.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in active:
            raise GVSVerifierError(f"{label} contains a JSON cycle")
        if identity in snapshots:
            snapshot, node_cost, text_cost = snapshots[identity]
            remaining["nodes"] -= node_cost - 1
            remaining["text_bytes"] -= text_cost
            if remaining["nodes"] < 0 or remaining["text_bytes"] < 0:
                raise GVSVerifierError(f"{label} exceeds its bounded JSON snapshot size")
            return copy.deepcopy(snapshot)
        active.add(identity)
        try:
            nodes_before = remaining["nodes"]
            text_before = remaining["text_bytes"]
            result: dict[str, object] = {}
            try:
                items = tuple(value.items())
            except RuntimeError as error:  # pragma: no cover - concurrent mutation guard
                raise GVSVerifierError(f"{label} changed while it was snapshotted") from error
            for key, child in items:
                if type(key) is not str:
                    raise GVSVerifierError(f"{label} contains a non-string JSON key")
                if key != _UNICODE_NORMALIZE("NFC", key):
                    raise GVSVerifierError(f"{label} contains a non-NFC JSON key")
                try:
                    encoded_key = key.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise GVSVerifierError(f"{label} contains an invalid UTF-8 JSON key") from error
                remaining["text_bytes"] -= len(encoded_key)
                if remaining["text_bytes"] < 0:
                    raise GVSVerifierError(f"{label} exceeds the bounded JSON text size")
                result[key] = _snapshot_json(
                    child,
                    label=f"{label}.{key}",
                    active_ids=active,
                    budget=remaining,
                    memo=snapshots,
                    depth=depth + 1,
                )
            snapshots[identity] = (
                result,
                nodes_before - remaining["nodes"] + 1,
                text_before - remaining["text_bytes"],
            )
            return result
        finally:
            active.remove(identity)
    raise GVSVerifierError(f"{label} must contain exact built-in JSON values")


def _sha256_json(value: object) -> str:
    return _HASH_SHA256(_canonical_json(value).encode("utf-8")).hexdigest()


def _config_record(config: BarunConfig) -> dict[str, object]:
    if type(config) is not BarunConfig:
        raise GVSVerifierError("config must be an exact BarunConfig")
    return asdict(config)


def _strict_seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SEED:
        raise GVSVerifierError(f"seed must be an integer in [0, {_MAX_SEED}]")
    return value


@dataclass(frozen=True, slots=True)
class GVSVerifierContract:
    """Immutable architecture and optimization boundary."""

    config_sha256: str
    layers: int
    dimension: int
    rank: int
    alpha: int
    initialization_seed: int
    target_module_names: tuple[str, ...]
    trainable_parameters: int
    candidate_count: int = FROZEN_CANDIDATE_COUNT
    schema_version: str = GVS_VERIFIER_CONTRACT_VERSION
    pooling: str = "last_nonpadding_hidden_float32"
    objective: str = "multi_positive_listwise_logsumexp"
    candidate_identity: str = "lowercase_sha256_of_lowered_action_ir_semantic_canonical_v1_utf8_bytes_for_valid_candidates"
    action_identity_contract_version: str = "barun-gvs-lowered-semantic-action-ir-v1"
    valid_candidate_set: str = (
        "complete_nontruncated_schema_valid_distinct_lowered_semantic_actions_only"
    )
    token_row_contract_version: str = "barun-gvs-prompt-plus-generated-unpadded-v1"
    batch_contract_version: str = "barun-gvs-k8-right-padding-bool-mask-v1"
    learned_input_fields: tuple[str, ...] = ("input_ids", "attention_mask")
    selection_tie_break: str = "lowered_semantic_candidate_sha256_lexicographic_asc"
    adapter_dropout: str = "0/1"
    backbone_frozen: bool = True
    head_bias: bool = True
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != GVS_VERIFIER_CONTRACT_VERSION:
            raise GVSVerifierError("verifier contract schema changed")
        if len(self.config_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.config_sha256
        ):
            raise GVSVerifierError("config_sha256 must be lowercase SHA-256")
        for name in ("layers", "dimension", "rank", "alpha", "trainable_parameters"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise GVSVerifierError(f"{name} must be a positive integer")
        _strict_seed(self.initialization_seed)
        if self.candidate_count != FROZEN_CANDIDATE_COUNT:
            raise GVSVerifierError("candidate_count differs from frozen K=8")
        expected_targets = tuple(
            name
            for layer in range(self.layers)
            for name in (
                f"backbone.layers.{layer}.attn.q_proj",
                f"backbone.layers.{layer}.attn.v_proj",
            )
        )
        if self.target_module_names != expected_targets:
            raise GVSVerifierError("target modules must be every q/v projection in canonical order")
        if self.pooling != "last_nonpadding_hidden_float32":
            raise GVSVerifierError("verifier pooling contract changed")
        if self.objective != "multi_positive_listwise_logsumexp":
            raise GVSVerifierError("verifier objective contract changed")
        if (
            self.candidate_identity
            != "lowercase_sha256_of_lowered_action_ir_semantic_canonical_v1_utf8_bytes_for_valid_candidates"
        ):
            raise GVSVerifierError("verifier candidate identity contract changed")
        if self.action_identity_contract_version != "barun-gvs-lowered-semantic-action-ir-v1":
            raise GVSVerifierError("verifier action-identity contract version changed")
        if (
            self.valid_candidate_set
            != "complete_nontruncated_schema_valid_distinct_lowered_semantic_actions_only"
        ):
            raise GVSVerifierError("verifier valid-candidate set contract changed")
        if self.token_row_contract_version != "barun-gvs-prompt-plus-generated-unpadded-v1":
            raise GVSVerifierError("verifier token-row contract changed")
        if self.batch_contract_version != "barun-gvs-k8-right-padding-bool-mask-v1":
            raise GVSVerifierError("verifier batch contract changed")
        if self.learned_input_fields != ("input_ids", "attention_mask"):
            raise GVSVerifierError("verifier learned-input field boundary changed")
        if self.selection_tie_break != "lowered_semantic_candidate_sha256_lexicographic_asc":
            raise GVSVerifierError("verifier selection tie-break contract changed")
        if self.adapter_dropout != "0/1":
            raise GVSVerifierError("verifier adapter dropout must remain exactly zero")
        if not self.backbone_frozen or not self.head_bias or self.launch_authorized:
            raise GVSVerifierError(
                "verifier contract must stay frozen, biased-head, and nonauthorizing"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "action_identity_contract_version": self.action_identity_contract_version,
            "adapter_dropout": self.adapter_dropout,
            "alpha": self.alpha,
            "backbone_frozen": self.backbone_frozen,
            "batch_contract_version": self.batch_contract_version,
            "candidate_count": self.candidate_count,
            "candidate_identity": self.candidate_identity,
            "config_sha256": self.config_sha256,
            "dimension": self.dimension,
            "head_bias": self.head_bias,
            "initialization_seed": self.initialization_seed,
            "launch_authorized": self.launch_authorized,
            "layers": self.layers,
            "learned_input_fields": list(self.learned_input_fields),
            "objective": self.objective,
            "pooling": self.pooling,
            "rank": self.rank,
            "schema_version": self.schema_version,
            "selection_tie_break": self.selection_tie_break,
            "target_module_names": list(self.target_module_names),
            "token_row_contract_version": self.token_row_contract_version,
            "trainable_parameters": self.trainable_parameters,
            "valid_candidate_set": self.valid_candidate_set,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_record())


def build_verifier_contract(
    config: BarunConfig,
    *,
    rank: int = PRODUCTION_RANK,
    alpha: int = PRODUCTION_ALPHA,
    initialization_seed: int = PRODUCTION_SEED,
) -> GVSVerifierContract:
    config_record = _config_record(config)
    if type(rank) is not int or rank <= 0:
        raise GVSVerifierError("rank must be a positive integer")
    if type(alpha) is not int or alpha <= 0:
        raise GVSVerifierError("alpha must be a positive integer")
    seed = _strict_seed(initialization_seed)
    if seed > _MAX_SEED - 2 * config.n_layers:
        raise GVSVerifierError("seed leaves no room for deterministic per-module initialization")
    accounting = count_verifier_parameters(config, rank=rank)
    targets = tuple(
        name
        for layer in range(config.n_layers)
        for name in (
            f"backbone.layers.{layer}.attn.q_proj",
            f"backbone.layers.{layer}.attn.v_proj",
        )
    )
    return GVSVerifierContract(
        config_sha256=_sha256_json(config_record),
        layers=config.n_layers,
        dimension=config.dim,
        rank=rank,
        alpha=alpha,
        initialization_seed=seed,
        target_module_names=targets,
        trainable_parameters=accounting.verifier_parameters,
    )


def _seeded_lora_a(*, rank: int, input_features: int, seed: int) -> Tensor:
    generator = _TORCH_GENERATOR(device="cpu")
    generator.manual_seed(seed)
    value = _TORCH_EMPTY((rank, input_features), device="cpu", dtype=torch.float32)
    _NN_INIT_KAIMING_UNIFORM(value, a=_MATH_SQRT(5), generator=generator)
    return value


class FrozenLinearLoRA(nn.Module):
    """One frozen bias-free linear plus deterministic float32 LoRA parameters."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, seed: int) -> None:
        super().__init__()
        if type(base) is not nn.Linear or base.bias is not None:
            raise GVSVerifierError("LoRA targets must be exact bias-free nn.Linear modules")
        if type(rank) is not int or rank <= 0 or type(alpha) is not int or alpha <= 0:
            raise GVSVerifierError("LoRA rank and alpha must be positive integers")
        _strict_seed(seed)
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        device = base.weight.device
        self.lora_a = nn.Parameter(
            _seeded_lora_a(rank=rank, input_features=base.in_features, seed=seed).to(device=device)
        )
        self.lora_b = nn.Parameter(
            _TORCH_ZEROS((base.out_features, rank), device=device, dtype=torch.float32)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _assert_import_time_dependency_bindings()
        base_output = self.base(inputs)
        branch = _F_LINEAR(_F_LINEAR(inputs.float(), self.lora_a), self.lora_b)
        return base_output + branch.to(dtype=base_output.dtype) * self.scaling


def _shallow_module_view(module: nn.Module) -> nn.Module:
    """Clone module structure while deliberately sharing parameters and child modules."""

    clone = copy.copy(module)
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
    clone._modules = module._modules.copy()
    for name in (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
    ):
        value = getattr(module, name, None)
        if value is not None:
            setattr(clone, name, copy.copy(value))
    return clone


def _verifier_layer_view(
    layer: nn.Module,
    *,
    rank: int,
    alpha: int,
    q_seed: int,
    v_seed: int,
) -> nn.Module:
    view = _shallow_module_view(layer)
    attention = _shallow_module_view(layer.attn)
    attention.q_proj = FrozenLinearLoRA(
        layer.attn.q_proj,
        rank=rank,
        alpha=alpha,
        seed=q_seed,
    )
    attention.v_proj = FrozenLinearLoRA(
        layer.attn.v_proj,
        rank=rank,
        alpha=alpha,
        seed=v_seed,
    )
    view.attn = attention
    return view


class GVSVerifier(nn.Module):
    """Frozen generator plus weight-sharing verifier views with q/v LoRA branches."""

    def __init__(
        self,
        backbone: BarunLM,
        *,
        rank: int = PRODUCTION_RANK,
        alpha: int = PRODUCTION_ALPHA,
        initialization_seed: int = PRODUCTION_SEED,
    ) -> None:
        if torch.is_inference_mode_enabled():
            raise GVSVerifierError("trainable verifier construction is forbidden in inference mode")
        super().__init__()
        if type(backbone) is not BarunLM:
            raise GVSVerifierError("backbone must be an exact BarunLM instance")
        if any(type(module) is FrozenLinearLoRA for module in backbone.modules()):
            raise GVSVerifierError("backbone already contains a GVS LoRA adapter")
        self.contract = build_verifier_contract(
            backbone.config,
            rank=rank,
            alpha=alpha,
            initialization_seed=initialization_seed,
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.backbone = backbone
        adapter_seed = self.contract.initialization_seed
        self.verifier_layers = nn.ModuleList(
            _verifier_layer_view(
                layer,
                rank=rank,
                alpha=alpha,
                q_seed=adapter_seed + 2 * layer_index,
                v_seed=adapter_seed + 2 * layer_index + 1,
            )
            for layer_index, layer in enumerate(self.backbone.layers)
        )
        device = self.backbone.embedding.weight.device
        self.score_head = nn.Linear(
            self.backbone.config.dim,
            1,
            bias=True,
            device="meta",
            dtype=torch.float32,
        )
        generator = _TORCH_GENERATOR(device="cpu")
        generator.manual_seed(self.contract.initialization_seed + 2 * self.contract.layers)
        head_weight = _TORCH_EMPTY((1, self.contract.dimension), device="cpu", dtype=torch.float32)
        _NN_INIT_KAIMING_UNIFORM(head_weight, a=_MATH_SQRT(5), generator=generator)
        self.score_head.weight = nn.Parameter(head_weight.to(device=device))
        self.score_head.bias = nn.Parameter(_TORCH_ZEROS((1,), device=device, dtype=torch.float32))
        super().train(backbone.training)
        audit_verifier_architecture(self)

    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        """Audit before ``nn.Module`` can run any local or global pre-hook."""

        audit_verifier_architecture(self)
        return super().__call__(*args, **kwargs)

    def train(self, mode: bool = True) -> GVSVerifier:
        """Synchronize the generator/view modes and immediately re-audit them."""

        if type(mode) is not bool:
            raise GVSVerifierError("verifier training mode must be an exact boolean")
        result = super().train(mode)
        audit_verifier_architecture(self)
        return result

    def to(self, *args: object, **kwargs: object) -> GVSVerifier:
        """Permit device movement but forbid dtype-wide conversion of float32 adapters."""

        try:
            _device, dtype, _non_blocking, _memory_format = torch._C._nn._parse_to(  # type: ignore[attr-defined]
                *args, **kwargs
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise GVSVerifierError("invalid verifier device-movement request") from error
        if _device is not None and torch.device(_device).type == "meta":
            raise GVSVerifierError("verifier movement to the meta device is forbidden")
        if dtype is not None:
            raise GVSVerifierError(
                "verifier-wide dtype conversion is forbidden; LoRA and head must stay float32"
            )
        result = super().to(*args, **kwargs)
        audit_verifier_architecture(self)
        return result

    def half(self) -> GVSVerifier:
        raise GVSVerifierError("verifier-wide dtype conversion is forbidden")

    def bfloat16(self) -> GVSVerifier:
        raise GVSVerifierError("verifier-wide dtype conversion is forbidden")

    def float(self) -> GVSVerifier:
        raise GVSVerifierError("verifier-wide dtype conversion is forbidden")

    def double(self) -> GVSVerifier:
        raise GVSVerifierError("verifier-wide dtype conversion is forbidden")

    def type(self, dst_type: object = None) -> GVSVerifier:
        del dst_type
        raise GVSVerifierError("verifier-wide dtype conversion is forbidden")

    def to_empty(self, *args: object, **kwargs: object) -> GVSVerifier:
        del args, kwargs
        raise GVSVerifierError("to_empty would destroy the frozen generator state")

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, Tensor]:
        del args, kwargs
        raise GVSVerifierError(
            "full verifier state_dict is alias-ambiguous; export the frozen backbone and "
            "adapter/head artifact separately"
        )

    def load_state_dict(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise GVSVerifierError(
            "full verifier load_state_dict is forbidden; load the backbone before construction "
            "and use load_verifier_adapter"
        )

    def _hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.backbone.embedding(input_ids)
        checkpoint = hidden
        selector_index = 0
        for index, layer in enumerate(self.verifier_layers):
            hidden = layer(hidden, attention_mask)
            stride = self.backbone.config.residual_select_every
            if stride and (index + 1) % stride == 0:
                hidden = self.backbone.selectors[selector_index](checkpoint, hidden)
                checkpoint = hidden
                selector_index += 1
        return self.backbone.final_norm(hidden)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        _assert_import_time_dependency_bindings()
        audit_verifier_architecture(self)
        if type(input_ids) is not Tensor or type(attention_mask) is not Tensor:
            raise GVSVerifierError("input_ids and attention_mask must be exact tensors")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise GVSVerifierError("verifier inputs must have identical [batch, sequence] shape")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise GVSVerifierError("input_ids must have a Torch embedding index dtype")
        if attention_mask.dtype is not torch.bool:
            raise GVSVerifierError("attention_mask must be boolean")
        if input_ids.layout is not torch.strided or attention_mask.layout is not torch.strided:
            raise GVSVerifierError("verifier inputs must use dense strided tensor layouts")
        model_device = self.backbone.embedding.weight.device
        if input_ids.device != model_device or attention_mask.device != model_device:
            raise GVSVerifierError("inputs, mask, and verifier weights must share one device")
        input_storage = {
            _storage_identity(input_ids, label="input_ids"),
            _storage_identity(attention_mask, label="attention_mask"),
        }
        if len(input_storage) != 2:
            raise GVSVerifierError("verifier inputs must not share storage")
        state_storage = {
            _storage_identity(tensor, label=f"verifier state {name!r}")
            for name, tensor in (*self.named_parameters(), *self.named_buffers())
        }
        if input_storage & state_storage:
            raise GVSVerifierError("verifier inputs must not alias parameter or buffer storage")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise GVSVerifierError("verifier batch and sequence dimensions must be nonempty")
        if input_ids.shape[1] > self.backbone.config.max_seq_len:
            raise GVSVerifierError("verifier sequence exceeds the backbone maximum")
        if bool((input_ids < 0).any().item()) or bool(
            (input_ids >= self.backbone.config.vocab_size).any().item()
        ):
            raise GVSVerifierError("input_ids contain an index outside the backbone vocabulary")
        lengths = attention_mask.sum(dim=1)
        if bool((lengths < 1).any().item()):
            raise GVSVerifierError("every verifier sequence must contain a non-padding token")
        positions = _TORCH_ARANGE(input_ids.shape[1], device=attention_mask.device)[None, :]
        expected_mask = positions < lengths[:, None]
        if not _TORCH_EQUAL(attention_mask, expected_mask):
            raise GVSVerifierError("verifier attention_mask must be contiguous right padding")
        hidden = self._hidden_states(input_ids, attention_mask)
        rows = _TORCH_ARANGE(input_ids.shape[0], device=input_ids.device)
        pooled = hidden[rows, lengths.to(device=input_ids.device) - 1].float()
        scores = self.score_head(pooled).squeeze(-1)
        if scores.dtype is not torch.float32 or not bool(_TORCH_ISFINITE(scores).all().item()):
            raise GVSVerifierError("verifier scores must be finite float32 values")
        _assert_import_time_dependency_bindings()
        return scores


def _expected_trainable_names(verifier: GVSVerifier) -> tuple[str, ...]:
    names: list[str] = []
    for layer_index in range(verifier.contract.layers):
        for projection in ("q_proj", "v_proj"):
            prefix = f"verifier_layers.{layer_index}.attn.{projection}"
            names.extend((f"{prefix}.lora_a", f"{prefix}.lora_b"))
    names.extend(("score_head.weight", "score_head.bias"))
    return tuple(names)


def _expected_backbone_module_types(config: BarunConfig) -> dict[str, type[nn.Module]]:
    expected: dict[str, type[nn.Module]] = {
        "": BarunLM,
        "embedding": nn.Embedding,
        "layers": nn.ModuleList,
        "selectors": nn.ModuleList,
        "final_norm": RMSNorm,
        "lm_head": nn.Linear,
    }
    for index in range(config.n_layers):
        prefix = f"layers.{index}"
        expected.update(
            {
                prefix: BarunBlock,
                f"{prefix}.attn_norm": RMSNorm,
                f"{prefix}.attn": GroupedAttention,
                f"{prefix}.attn.q_proj": nn.Linear,
                f"{prefix}.attn.k_proj": nn.Linear,
                f"{prefix}.attn.v_proj": nn.Linear,
                f"{prefix}.attn.o_proj": nn.Linear,
                f"{prefix}.attn.q_norm": RMSNorm if config.qk_norm else nn.Identity,
                f"{prefix}.attn.k_norm": RMSNorm if config.qk_norm else nn.Identity,
                f"{prefix}.attn.rope": PartialRotaryEmbedding,
                f"{prefix}.ffn_norm": RMSNorm,
                f"{prefix}.ffn": BoundedSwiGLU,
                f"{prefix}.ffn.gate_up": nn.Linear,
                f"{prefix}.ffn.down": nn.Linear,
            }
        )
        if config.attention_gate:
            expected[f"{prefix}.attn.g_proj"] = nn.Linear
    selector_count = (
        config.n_layers // config.residual_select_every if config.residual_select_every else 0
    )
    for index in range(selector_count):
        prefix = f"selectors.{index}"
        expected.update(
            {
                prefix: ResidualSelector,
                f"{prefix}.norm": RMSNorm,
                f"{prefix}.score": nn.Linear,
            }
        )
    if config.mtp_loss_weight > 0:
        expected["mtp_norm"] = RMSNorm
        expected["mtp_proj"] = nn.Linear
    return expected


def _require_direct_module_slots(module: nn.Module, *, label: str) -> None:
    module_type = type(module)
    if module_type is GVSVerifier:
        expected_modules = {"backbone", "verifier_layers", "score_head"}
        expected_parameters: set[str] = set()
        expected_buffers: set[str] = set()
    elif module_type is BarunLM:
        expected_modules = {
            "embedding",
            "layers",
            "selectors",
            "final_norm",
            "lm_head",
        }
        if module.mtp_norm is not None:
            expected_modules.update({"mtp_norm", "mtp_proj"})
        expected_parameters = set()
        expected_buffers = set()
    elif module_type is BarunBlock:
        expected_modules = {"attn_norm", "attn", "ffn_norm", "ffn"}
        expected_parameters = set()
        expected_buffers = set()
    elif module_type is GroupedAttention:
        expected_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_norm",
            "k_norm",
            "rope",
        }
        if module.g_proj is not None:
            expected_modules.add("g_proj")
        expected_parameters = set()
        expected_buffers = set()
    elif module_type is BoundedSwiGLU:
        expected_modules = {"gate_up", "down"}
        expected_parameters = set()
        expected_buffers = set()
    elif module_type is ResidualSelector:
        expected_modules = {"norm", "score"}
        expected_parameters = set()
        expected_buffers = set()
    elif module_type is FrozenLinearLoRA:
        expected_modules = {"base"}
        expected_parameters = {"lora_a", "lora_b"}
        expected_buffers = set()
    elif module_type is nn.Linear:
        expected_modules = set()
        expected_parameters = {"weight", "bias"}
        expected_buffers = set()
    elif module_type is nn.Embedding:
        expected_modules = set()
        expected_parameters = {"weight"}
        expected_buffers = set()
    elif module_type is RMSNorm:
        expected_modules = set()
        expected_parameters = {"weight"} if module.weight is not None else set()
        expected_buffers = set()
    elif module_type is PartialRotaryEmbedding:
        expected_modules = set()
        expected_parameters = set()
        expected_buffers = {"cos", "sin"}
    elif module_type in {nn.ModuleList, nn.Identity}:
        expected_modules = (
            {str(index) for index in range(len(module))} if module_type is nn.ModuleList else set()
        )
        expected_parameters = set()
        expected_buffers = set()
    else:  # pragma: no cover - the exact module-name/type roster rejects this first
        raise GVSVerifierError(f"{label} has unsupported exact module type {module_type.__name__}")
    if set(module._modules) != expected_modules:
        raise GVSVerifierError(f"{label} direct child-module roster changed")
    if set(module._parameters) != expected_parameters:
        raise GVSVerifierError(f"{label} direct parameter-slot roster changed")
    if set(module._buffers) != expected_buffers:
        raise GVSVerifierError(f"{label} direct buffer-slot roster changed")
    expected_nonpersistent = {"cos", "sin"} if module_type is PartialRotaryEmbedding else set()
    if set(module._non_persistent_buffers_set) != expected_nonpersistent:
        raise GVSVerifierError(f"{label} nonpersistent-buffer roster changed")


def _audit_exact_module_graph(verifier: GVSVerifier) -> None:
    config = verifier.backbone.config
    backbone_expected = _expected_backbone_module_types(config)
    expected: dict[str, type[nn.Module]] = {"": GVSVerifier, "backbone": BarunLM}
    expected.update(
        {f"backbone.{name}": module_type for name, module_type in backbone_expected.items() if name}
    )
    expected["verifier_layers"] = nn.ModuleList
    for index in range(config.n_layers):
        prefix = f"verifier_layers.{index}"
        expected[prefix] = BarunBlock
        expected[f"{prefix}.attn"] = GroupedAttention
        expected[f"{prefix}.attn.q_proj"] = FrozenLinearLoRA
        expected[f"{prefix}.attn.v_proj"] = FrozenLinearLoRA
    expected["score_head"] = nn.Linear
    actual_rows = tuple(verifier.named_modules())
    actual = {name: module for name, module in actual_rows}
    if len(actual) != len(actual_rows) or set(actual) != set(expected):
        raise GVSVerifierError(
            "exact verifier module graph changed; "
            f"missing={sorted(set(expected) - set(actual))!r}, "
            f"extra={sorted(set(actual) - set(expected))!r}"
        )
    for name, expected_type in expected.items():
        if type(actual[name]) is not expected_type:
            raise GVSVerifierError(
                f"exact verifier module {name!r} must remain {expected_type.__qualname__}"
            )
    try:
        all_rows = verifier.named_modules(remove_duplicate=False)
    except TypeError as error:  # pragma: no cover - pinned Torch supports this API
        raise GVSVerifierError("Torch lacks exact alias-preserving module traversal") from error
    for name, module in all_rows:
        _require_direct_module_slots(module, label=name or "<root>")

    config = verifier.backbone.config
    embedding = verifier.backbone.embedding
    if (
        embedding.num_embeddings != config.vocab_size
        or embedding.embedding_dim != config.dim
        or embedding.padding_idx is not None
        or embedding.max_norm is not None
        or embedding.norm_type != 2.0
        or embedding.scale_grad_by_freq is not False
        or embedding.sparse is not False
        or type(embedding.weight) is not nn.Parameter
        or embedding.weight.shape != (config.vocab_size, config.dim)
    ):
        raise GVSVerifierError("backbone embedding attributes differ from the exact config")

    expected_linear_dimensions: dict[str, tuple[int, int, bool]] = {
        "backbone.lm_head": (config.dim, config.vocab_size, False),
        "score_head": (config.dim, 1, True),
    }
    kv_dimension = config.n_kv_heads * config.head_dim
    for index in range(config.n_layers):
        prefix = f"backbone.layers.{index}"
        expected_linear_dimensions.update(
            {
                f"{prefix}.attn.q_proj": (config.dim, config.dim, False),
                f"{prefix}.attn.k_proj": (config.dim, kv_dimension, False),
                f"{prefix}.attn.v_proj": (config.dim, kv_dimension, False),
                f"{prefix}.attn.o_proj": (config.dim, config.dim, False),
                f"{prefix}.ffn.gate_up": (config.dim, 2 * config.ffn_dim, False),
                f"{prefix}.ffn.down": (config.ffn_dim, config.dim, False),
            }
        )
        if config.attention_gate:
            expected_linear_dimensions[f"{prefix}.attn.g_proj"] = (
                config.dim,
                config.dim,
                False,
            )
    for index in range(len(verifier.backbone.selectors)):
        expected_linear_dimensions[f"backbone.selectors.{index}.score"] = (
            config.dim,
            1,
            False,
        )
    if config.mtp_loss_weight > 0:
        expected_linear_dimensions["backbone.mtp_proj"] = (config.dim, config.dim, False)
    for name, (input_features, output_features, biased) in expected_linear_dimensions.items():
        module = actual[name]
        if type(module) is not nn.Linear:
            raise GVSVerifierError(f"linear module {name!r} changed type during the audit")
        if (
            module.in_features != input_features
            or module.out_features != output_features
            or type(module.weight) is not nn.Parameter
            or module.weight.shape != (output_features, input_features)
            or (module.bias is not None) is not biased
            or (biased and type(module.bias) is not nn.Parameter)
            or (module.bias is not None and module.bias.shape != (output_features,))
        ):
            raise GVSVerifierError(f"linear module {name!r} differs from the exact config")

    for name, module in actual.items():
        if type(module) is not RMSNorm:
            continue
        if type(module.eps) is not float or module.eps != config.norm_eps:
            raise GVSVerifierError(f"RMSNorm {name!r} epsilon differs from the exact config")
        expected_dimension = (
            config.head_dim if name.endswith((".q_norm", ".k_norm")) else config.dim
        )
        affine = not (name.startswith("backbone.selectors.") and name.endswith(".norm"))
        if (module.weight is not None) is not affine or (
            module.weight is not None
            and (
                type(module.weight) is not nn.Parameter
                or module.weight.shape != (expected_dimension,)
            )
        ):
            raise GVSVerifierError(f"RMSNorm {name!r} affine state differs from the exact config")


def _storage_identity(tensor: Tensor, *, label: str) -> tuple[str, int]:
    if tensor.layout is not torch.strided or tensor.device.type == "meta":
        raise GVSVerifierError(f"{label} must use real dense strided storage")
    try:
        storage_identity = int(tensor.untyped_storage()._cdata)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise GVSVerifierError(f"{label} storage identity is unavailable") from error
    return str(tensor.device), storage_identity


def _audit_unique_tensor_storage(verifier: GVSVerifier) -> int:
    seen_objects: set[int] = set()
    seen_storage: dict[tuple[str, int], str] = {}
    unique_storage_elements = 0
    for kind, values in (
        ("parameter", verifier.named_parameters()),
        ("buffer", verifier.named_buffers()),
    ):
        for name, tensor in values:
            if id(tensor) in seen_objects:
                continue
            seen_objects.add(id(tensor))
            if kind == "parameter" and type(tensor) is not nn.Parameter:
                raise GVSVerifierError(f"parameter {name!r} must be an exact Parameter")
            if kind == "buffer" and type(tensor) is not Tensor:
                raise GVSVerifierError(f"buffer {name!r} must be an exact Tensor")
            storage_key = _storage_identity(tensor, label=f"{kind} {name!r}")
            prior = seen_storage.get(storage_key)
            if prior is not None:
                raise GVSVerifierError(f"distinct tensors {prior!r} and {name!r} share one storage")
            seen_storage[storage_key] = name
            if tensor.requires_grad:
                if (
                    not tensor.is_contiguous()
                    or tensor.storage_offset() != 0
                    or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
                ):
                    raise GVSVerifierError(
                        f"trainable parameter {name!r} must own one exact contiguous storage"
                    )
                unique_storage_elements += tensor.numel()
    for name, parameter in verifier.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if (
            type(gradient) is not Tensor
            or gradient.layout is not torch.strided
            or not gradient.is_contiguous()
            or gradient.storage_offset() != 0
            or gradient.untyped_storage().nbytes() != gradient.numel() * gradient.element_size()
        ):
            raise GVSVerifierError(
                f"trainable parameter {name!r} gradient must own exact contiguous storage"
            )
        storage_key = _storage_identity(gradient, label=f"gradient {name!r}")
        prior = seen_storage.get(storage_key)
        if prior is not None:
            raise GVSVerifierError(
                f"distinct tensors {prior!r} and gradient {name!r} share one storage"
            )
        seen_storage[storage_key] = f"gradient {name}"
    return unique_storage_elements


def _torch_execution_state_record() -> dict[str, object]:
    try:
        from torch.overrides import _get_current_function_mode_stack
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        function_modes = tuple(_get_current_function_mode_stack())
        dispatch_modes = tuple(_get_current_dispatch_mode_stack())
    except (AttributeError, RuntimeError, TypeError) as error:  # pragma: no cover
        raise GVSVerifierError("Torch execution-mode stacks are not inspectable") from error
    return {
        "cpu_autocast_dtype": str(torch.get_autocast_dtype("cpu")),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "default_device": str(torch.get_default_device()),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "dispatch_modes": [
            f"{type(mode).__module__}.{type(mode).__qualname__}" for mode in dispatch_modes
        ],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "function_modes": [
            f"{type(mode).__module__}.{type(mode).__qualname__}" for mode in function_modes
        ],
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "interop_threads": torch.get_num_interop_threads(),
        "mkldnn_deterministic": bool(torch.backends.mkldnn.deterministic),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "overwrite_module_params_on_conversion": (
            torch.__future__.get_overwrite_module_params_on_conversion()
        ),
        "swap_module_params_on_conversion": (
            torch.__future__.get_swap_module_params_on_conversion()
        ),
        "torch_threads": torch.get_num_threads(),
    }


def _reject_execution_modes() -> None:
    state = _torch_execution_state_record()
    if state["function_modes"] or state["dispatch_modes"]:
        raise GVSVerifierError("Torch Python function/dispatch modes are forbidden")
    if state["cpu_autocast_enabled"] or state["cuda_autocast_enabled"]:
        raise GVSVerifierError("autocast is outside the float32 verifier contract")
    if state["overwrite_module_params_on_conversion"]:
        raise GVSVerifierError("Torch parameter-overwrite conversion mode is forbidden")
    if torch.compiler.is_compiling():
        raise GVSVerifierError("compiled execution is outside the audited verifier contract")


def _clear_model_execution_caches() -> None:
    for label, function in (
        ("local bidirectional mask", _MODEL_LOCAL_BIDIRECTIONAL_MASK),
        ("local block mask", _MODEL_LOCAL_BLOCK_MASK),
        ("local causal mask", _MODEL_LOCAL_CAUSAL_MASK),
    ):
        cache_clear = getattr(function, "cache_clear", None)
        if not callable(cache_clear):
            raise GVSVerifierError(f"BarunLM {label} cache is not clearable")
        cache_clear()


def _capture_execution_callable_bindings() -> tuple[tuple[str, object, str, object, str], ...]:
    specs = (
        ("GVSVerifier.__call__", GVSVerifier, "__call__"),
        ("GVSVerifier.forward", GVSVerifier, "forward"),
        ("GVSVerifier._hidden_states", GVSVerifier, "_hidden_states"),
        ("GVSVerifier.train", GVSVerifier, "train"),
        ("GVSVerifier.to", GVSVerifier, "to"),
        ("GVSVerifier.state_dict", GVSVerifier, "state_dict"),
        ("GVSVerifier.load_state_dict", GVSVerifier, "load_state_dict"),
        ("FrozenLinearLoRA.forward", FrozenLinearLoRA, "forward"),
        ("BarunBlock.forward", BarunBlock, "forward"),
        ("GroupedAttention.forward", GroupedAttention, "forward"),
        ("GroupedAttention._project", GroupedAttention, "_project"),
        ("GroupedAttention._torch_attention", GroupedAttention, "_torch_attention"),
        ("GroupedAttention._finish", GroupedAttention, "_finish"),
        ("GroupedAttention._is_binary_mask", GroupedAttention, "_is_binary_mask"),
        ("GroupedAttention._is_key_mask", GroupedAttention, "_is_key_mask"),
        ("GroupedAttention._causal_mask", GroupedAttention, "_causal_mask"),
        ("GroupedAttention._key_mask_to_additive", GroupedAttention, "_key_mask_to_additive"),
        ("GroupedAttention._ensure_nonempty_rows", GroupedAttention, "_ensure_nonempty_rows"),
        ("RMSNorm.forward", RMSNorm, "forward"),
        ("PartialRotaryEmbedding.forward", PartialRotaryEmbedding, "forward"),
        ("BoundedSwiGLU.forward", BoundedSwiGLU, "forward"),
        ("ResidualSelector.forward", ResidualSelector, "forward"),
        ("nn.Module.__call__", nn.Module, "__call__"),
        ("nn.Module._call_impl", nn.Module, "_call_impl"),
        ("nn.Module._apply", nn.Module, "_apply"),
        ("nn.Module.buffers", nn.Module, "buffers"),
        ("nn.Module.children", nn.Module, "children"),
        ("nn.Module.cpu", nn.Module, "cpu"),
        ("nn.Module.cuda", nn.Module, "cuda"),
        ("nn.Module.modules", nn.Module, "modules"),
        ("nn.Module.named_buffers", nn.Module, "named_buffers"),
        ("nn.Module.named_modules", nn.Module, "named_modules"),
        ("nn.Module.named_parameters", nn.Module, "named_parameters"),
        ("nn.Module.parameters", nn.Module, "parameters"),
        ("nn.Module.to", nn.Module, "to"),
        ("nn.Module.train", nn.Module, "train"),
        ("nn.Linear.forward", nn.Linear, "forward"),
        ("nn.Embedding.forward", nn.Embedding, "forward"),
        ("nn.Identity.forward", nn.Identity, "forward"),
    )
    return tuple(
        (
            label,
            owner,
            attribute,
            getattr(owner, attribute),
            _sha256_json(_callable_identity(getattr(owner, attribute), label=label)),
        )
        for label, owner, attribute in specs
    )


def _assert_execution_callable_bindings(
    _expected: tuple[tuple[str, object, str, object, str], ...] = (
        _capture_execution_callable_bindings()
    ),
    _dependency_guard: object = _assert_import_time_dependency_bindings,
) -> None:
    changed: list[str] = []
    if globals().get("_assert_import_time_dependency_bindings") is not _dependency_guard:
        changed.append("_assert_import_time_dependency_bindings")
    for label, owner, attribute, expected, expected_sha256 in _expected:
        current = getattr(owner, attribute, None)
        if current is not expected:
            changed.append(label)
            continue
        if _sha256_json(_callable_identity(current, label=label)) != expected_sha256:
            changed.append(label)
    if changed:
        raise GVSVerifierError(
            "verifier execution callable binding changed: " + ", ".join(sorted(set(changed)))
        )


def audit_verifier_architecture(verifier: GVSVerifier) -> dict[str, object]:
    """Fail closed unless the exact intended trainable set and shapes are present."""

    _assert_import_time_dependency_bindings()
    _assert_execution_callable_bindings()
    _reject_execution_modes()
    _clear_model_execution_caches()
    if type(verifier) is not GVSVerifier:
        raise GVSVerifierError("verifier must be an exact GVSVerifier")
    if type(verifier.backbone) is not BarunLM:
        raise GVSVerifierError("verifier backbone must remain an exact BarunLM")
    if type(verifier.contract) is not GVSVerifierContract:
        raise GVSVerifierError("verifier contract must remain an exact immutable contract")
    expected_contract = build_verifier_contract(
        verifier.backbone.config,
        rank=verifier.contract.rank,
        alpha=verifier.contract.alpha,
        initialization_seed=verifier.contract.initialization_seed,
    )
    if verifier.contract.to_record() != expected_contract.to_record():
        raise GVSVerifierError("verifier contract or backbone config changed after construction")
    config = verifier.backbone.config
    selector_count = (
        config.n_layers // config.residual_select_every if config.residual_select_every else 0
    )
    if (
        len(verifier.backbone.layers) != config.n_layers
        or len(verifier.backbone.selectors) != selector_count
    ):
        raise GVSVerifierError("backbone layer/selector count differs from its config")
    for layer_index, layer in enumerate(verifier.backbone.layers):
        if layer.attn.config is not config:
            raise GVSVerifierError("backbone attention no longer aliases its exact config")
        if layer.attn.is_full is not ((layer_index + 1) % config.full_attention_every == 0):
            raise GVSVerifierError("backbone full/local attention allocation changed")
        if type(layer.dropout) is not float or layer.dropout != config.dropout:
            raise GVSVerifierError("backbone block dropout differs from its config")
        if type(layer.ffn.clip) is not float or layer.ffn.clip != config.activation_clip:
            raise GVSVerifierError("backbone feed-forward clipping differs from its config")
        if layer.attn.rope.dim != config.rope_dim:
            raise GVSVerifierError("backbone rotary dimension differs from its config")
        expected_rope_shape = (1, 1, config.max_seq_len, config.rope_dim)
        if (
            layer.attn.rope.cos.shape != expected_rope_shape
            or layer.attn.rope.sin.shape != expected_rope_shape
            or not _TORCH_IS_FLOATING_POINT(layer.attn.rope.cos)
            or layer.attn.rope.sin.dtype != layer.attn.rope.cos.dtype
        ):
            raise GVSVerifierError("backbone rotary buffers differ from the exact config")
    if (verifier.backbone.lm_head.weight is verifier.backbone.embedding.weight) is not (
        config.tie_embeddings
    ):
        raise GVSVerifierError("backbone embedding tie state differs from its config")
    if (verifier.backbone.mtp_norm is not None) is not (config.mtp_loss_weight > 0) or (
        verifier.backbone.mtp_proj is not None
    ) is not (config.mtp_loss_weight > 0):
        raise GVSVerifierError("backbone MTP module state differs from its config")
    if len(verifier.verifier_layers) != len(verifier.backbone.layers):
        raise GVSVerifierError("verifier layer-view count differs from the frozen backbone")
    for layer_index, (base_layer, layer) in enumerate(
        zip(verifier.backbone.layers, verifier.verifier_layers, strict=True)
    ):
        if layer is base_layer or type(layer) is not type(base_layer):
            raise GVSVerifierError("verifier layer must be a distinct exact-type shared view")
        if layer.attn is base_layer.attn or type(layer.attn) is not type(base_layer.attn):
            raise GVSVerifierError("verifier attention must be a distinct exact-type shared view")
        if (
            type(base_layer.attn.q_proj) is not nn.Linear
            or type(base_layer.attn.v_proj) is not nn.Linear
        ):
            raise GVSVerifierError("generator q/v projections must remain exact nn.Linear modules")
        if (
            type(layer.attn.q_proj) is not FrozenLinearLoRA
            or type(layer.attn.v_proj) is not FrozenLinearLoRA
        ):
            raise GVSVerifierError("every q/v projection must have exactly one GVS LoRA wrapper")
        for projection_name, projection, base_projection in (
            ("q_proj", layer.attn.q_proj, base_layer.attn.q_proj),
            ("v_proj", layer.attn.v_proj, base_layer.attn.v_proj),
        ):
            if (
                type(projection.rank) is not int
                or type(projection.alpha) is not int
                or type(projection.scaling) is not float
                or projection.rank != verifier.contract.rank
                or projection.alpha != verifier.contract.alpha
            ):
                raise GVSVerifierError("LoRA rank/alpha differs from the contract")
            if projection.scaling != float(projection.alpha) / float(projection.rank):
                raise GVSVerifierError("LoRA scaling differs from alpha/rank")
            if projection.base is not base_projection:
                raise GVSVerifierError(
                    f"layer {layer_index} {projection_name} does not share the generator base"
                )
            if projection.base.bias is not None or projection.base.weight.requires_grad:
                raise GVSVerifierError("LoRA base projection is not bias-free and frozen")
            if (
                projection.lora_a.dtype is not torch.float32
                or projection.lora_b.dtype is not torch.float32
            ):
                raise GVSVerifierError("LoRA trainable parameters must stay float32")
            if projection.lora_a.shape != (
                projection.rank,
                base_projection.in_features,
            ) or projection.lora_b.shape != (
                base_projection.out_features,
                projection.rank,
            ):
                raise GVSVerifierError("LoRA parameter shapes differ from the base projection")
        if (
            layer.training is not base_layer.training
            or layer.attn.training is not base_layer.attn.training
        ):
            raise GVSVerifierError("verifier views and generator modules have divergent modes")
        if layer.dropout != base_layer.dropout:
            raise GVSVerifierError("verifier layer dropout differs from the generator")
        if (
            layer.attn.config is not base_layer.attn.config
            or layer.attn.is_full is not base_layer.attn.is_full
        ):
            raise GVSVerifierError("verifier attention behavior differs from the generator")
        for shared_name in ("attn_norm", "ffn_norm", "ffn"):
            if getattr(layer, shared_name) is not getattr(base_layer, shared_name):
                raise GVSVerifierError(
                    f"verifier layer {layer_index} does not share {shared_name} with generator"
                )
        for shared_name in ("k_proj", "o_proj", "g_proj", "q_norm", "k_norm", "rope"):
            if getattr(layer.attn, shared_name) is not getattr(base_layer.attn, shared_name):
                raise GVSVerifierError(
                    f"verifier attention {layer_index} does not share {shared_name} with generator"
                )
    _audit_exact_module_graph(verifier)
    expected_names = _expected_trainable_names(verifier)
    if any(
        parameter.grad is not None
        for parameter in verifier.parameters()
        if not parameter.requires_grad
    ):
        raise GVSVerifierError("a frozen generator parameter retains a gradient tensor")
    actual_names = tuple(
        name for name, parameter in verifier.named_parameters() if parameter.requires_grad
    )
    if actual_names != expected_names:
        raise GVSVerifierError(
            f"trainable parameter roster changed; expected={expected_names!r}, actual={actual_names!r}"
        )
    for name, parameter in verifier.named_parameters():
        if parameter.requires_grad and not bool(_TORCH_ISFINITE(parameter.detach()).all().item()):
            raise GVSVerifierError(f"trainable parameter {name!r} contains a non-finite value")
        gradient = parameter.grad
        if not parameter.requires_grad or gradient is None:
            continue
        if (
            type(gradient) is not Tensor
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device
            or gradient.layout is not torch.strided
            or not bool(_TORCH_ISFINITE(gradient).all().item())
        ):
            raise GVSVerifierError(f"trainable parameter {name!r} has an invalid gradient tensor")
    trainable_count = sum(
        parameter.numel() for parameter in verifier.parameters() if parameter.requires_grad
    )
    if trainable_count != verifier.contract.trainable_parameters:
        raise GVSVerifierError("trainable parameter count differs from the exact contract")
    if (
        type(verifier.score_head) is not nn.Linear
        or verifier.score_head.in_features != verifier.contract.dimension
        or verifier.score_head.out_features != 1
        or type(verifier.score_head.weight) is not nn.Parameter
        or type(verifier.score_head.bias) is not nn.Parameter
        or verifier.score_head.weight.dtype is not torch.float32
        or verifier.score_head.bias.dtype is not torch.float32
        or not verifier.score_head.weight.requires_grad
        or not verifier.score_head.bias.requires_grad
        or verifier.score_head.weight.shape != (1, verifier.contract.dimension)
        or verifier.score_head.bias.shape != (1,)
    ):
        raise GVSVerifierError("scalar head shape changed")
    unique_trainable_storage_elements = _audit_unique_tensor_storage(verifier)
    if unique_trainable_storage_elements != trainable_count:
        raise GVSVerifierError("trainable parameter storage count differs from the exact contract")
    generator_count = sum(parameter.numel() for parameter in verifier.backbone.parameters())
    unique_system_count = sum(parameter.numel() for parameter in verifier.parameters())
    if unique_system_count != generator_count + trainable_count:
        raise GVSVerifierError("shared verifier view duplicates or omits stored parameters")
    if verifier.contract.config_sha256 == _sha256_json(_config_record(BarunConfig())):
        if generator_count != PRODUCTION_GENERATOR_PARAMETERS:
            raise GVSVerifierError("production generator parameter count changed")
        if unique_system_count != PRODUCTION_SYSTEM_PARAMETERS:
            raise GVSVerifierError("production complete-system parameter count changed")
    parameter_devices = {parameter.device for parameter in verifier.parameters()}
    buffer_devices = {buffer.device for buffer in verifier.buffers()}
    all_devices = parameter_devices | buffer_devices
    if len(all_devices) != 1 or next(iter(all_devices)).type == "meta":
        raise GVSVerifierError("all verifier parameters and buffers must share one real device")
    for kind, values in (
        ("parameter", verifier.named_parameters()),
        ("buffer", verifier.named_buffers()),
    ):
        for name, tensor in values:
            for hook_name in ("_backward_hooks", "_post_accumulate_grad_hooks"):
                if getattr(tensor, hook_name, None):
                    raise GVSVerifierError(f"{kind} {name!r} has forbidden tensor hook {hook_name}")
    for module_name, module in verifier.named_modules():
        if type(verifier.training) is not bool or type(module.training) is not bool:
            raise GVSVerifierError("verifier module training modes must be exact booleans")
        if module.training is not verifier.training:
            raise GVSVerifierError(f"module {module_name!r} diverges from the verifier mode")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise GVSVerifierError(f"module {module_name!r} has a compiled call override")
        for attribute in module.__dict__:
            class_attribute = getattr(type(module), attribute, None)
            if callable(class_attribute):
                raise GVSVerifierError(
                    f"module {module_name!r} shadows class callable {attribute!r}"
                )
        for hook_name in (
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks",
            "_forward_hooks_always_called",
            "_forward_hooks_with_kwargs",
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_load_state_dict_post_hooks",
            "_load_state_dict_pre_hooks",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
        ):
            hooks = getattr(module, hook_name, None)
            if hooks:
                raise GVSVerifierError(
                    f"module {module_name!r} has forbidden execution hook {hook_name}"
                )
    global_hook_names = (
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
        "_global_buffer_registration_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
    )
    global_hook_state = torch.nn.modules.module
    for hook_name in global_hook_names:
        if getattr(global_hook_state, hook_name, None):
            raise GVSVerifierError(f"Torch has forbidden global execution hook {hook_name}")
    result = {
        "behavior_sha256": _verifier_behavior_sha256(verifier),
        "contract_sha256": verifier.contract.sha256,
        "generator_parameter_count": generator_count,
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_names": list(actual_names),
        "unique_trainable_storage_elements": unique_trainable_storage_elements,
        "unique_system_parameter_count": unique_system_count,
    }
    _assert_execution_callable_bindings()
    _assert_import_time_dependency_bindings()
    return result


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode("ascii")
    return _HASH_SHA256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _callable_code_sha256(value: object, *, label: str) -> str:
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    if code is None:
        raise GVSVerifierError(f"{label} has no inspectable Python code object")
    return _HASH_SHA256(_MARSHAL_DUMPS(code)).hexdigest()


def _verifier_behavior_record(verifier: GVSVerifier) -> list[dict[str, str]]:
    records: dict[tuple[str, str], dict[str, str]] = {}
    for module_name, module in verifier.named_modules():
        module_type = type(module)
        class_name = f"{module_type.__module__}.{module_type.__qualname__}"
        key = (class_name, "forward")
        if key not in records:
            records[key] = {
                "callable": f"{class_name}.forward",
                "code_sha256": _callable_code_sha256(
                    module_type.forward,
                    label=f"{module_name or '<root>'}.forward",
                ),
            }
    records[(f"{GVSVerifier.__module__}.{GVSVerifier.__qualname__}", "_hidden_states")] = {
        "callable": f"{GVSVerifier.__module__}.{GVSVerifier.__qualname__}._hidden_states",
        "code_sha256": _callable_code_sha256(
            GVSVerifier._hidden_states,
            label="GVSVerifier._hidden_states",
        ),
    }
    return [records[key] for key in sorted(records)]


def _verifier_behavior_sha256(verifier: GVSVerifier) -> str:
    return _sha256_json(_verifier_behavior_record(verifier))


def _complete_state_record(verifier: nn.Module) -> list[dict[str, object]]:
    tensors: list[dict[str, object]] = []
    for kind, values in (
        ("parameter", verifier.named_parameters()),
        ("buffer", verifier.named_buffers()),
    ):
        for name, tensor in values:
            if not bool(_TORCH_ISFINITE(tensor.detach()).all().item()):
                raise GVSVerifierError(f"state tensor {name!r} contains a non-finite value")
            tensors.append(
                {
                    "device": str(tensor.device),
                    "dtype": str(tensor.dtype),
                    "kind": kind,
                    "name": name,
                    "requires_grad": bool(tensor.requires_grad),
                    "shape": list(tensor.shape),
                    "sha256": _tensor_sha256(tensor),
                }
            )
    return tensors


def _tensor_version_record(module: nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, values in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, tensor in values:
            try:
                version = tensor._version
            except RuntimeError as error:
                raise GVSVerifierError(
                    f"{kind} {name!r} has no mutation version counter"
                ) from error
            rows.append({"kind": kind, "name": name, "version": version})
    return rows


def _direct_state_topology_record(verifier: nn.Module) -> dict[str, object]:
    """Bind direct tensor slots, persistence, and all shared-module/tensor aliases."""

    module_aliases: dict[int, str] = {}
    tensor_aliases: dict[int, str] = {}
    storage_aliases: dict[tuple[str, int], str] = {}
    modules: list[dict[str, object]] = []
    tensors: list[dict[str, object]] = []
    try:
        named_modules = verifier.named_modules(remove_duplicate=False)
    except TypeError as error:  # pragma: no cover - pinned Torch supports this API
        raise GVSVerifierError("Torch lacks exact alias-preserving module traversal") from error
    for module_path, module in named_modules:
        path = module_path or "<root>"
        canonical_module_path = module_aliases.setdefault(id(module), path)
        modules.append(
            {
                "alias_of": canonical_module_path,
                "module_path": path,
                "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            }
        )
        for kind, slots in (("parameter", module._parameters), ("buffer", module._buffers)):
            for slot_name, tensor in sorted(slots.items()):
                slot_path = f"{path}.{slot_name}"
                if tensor is None:
                    tensors.append(
                        {
                            "kind": kind,
                            "persistent": (
                                None
                                if kind == "parameter"
                                else slot_name not in module._non_persistent_buffers_set
                            ),
                            "slot_path": slot_path,
                            "tensor": None,
                        }
                    )
                    continue
                canonical_tensor_path = tensor_aliases.setdefault(id(tensor), slot_path)
                storage_key = _storage_identity(tensor, label=slot_path)
                canonical_storage_path = storage_aliases.setdefault(storage_key, slot_path)
                tensors.append(
                    {
                        "alias_of": canonical_tensor_path,
                        "device": str(tensor.device),
                        "dtype": str(tensor.dtype),
                        "is_contiguous": tensor.is_contiguous(),
                        "kind": kind,
                        "persistent": (
                            None
                            if kind == "parameter"
                            else slot_name not in module._non_persistent_buffers_set
                        ),
                        "requires_grad": bool(tensor.requires_grad),
                        "sha256": _tensor_sha256(tensor),
                        "shape": list(tensor.shape),
                        "slot_path": slot_path,
                        "storage_alias_of": canonical_storage_path,
                        "storage_nbytes": tensor.untyped_storage().nbytes(),
                        "storage_offset": tensor.storage_offset(),
                        "stride": list(tensor.stride()),
                        "tensor_numel": tensor.numel(),
                    }
                )
    return {
        "module_alias_count": len(module_aliases),
        "module_slots": modules,
        "storage_alias_count": len(storage_aliases),
        "tensor_alias_count": len(tensor_aliases),
        "tensor_slots": tensors,
    }


def verifier_receipt(verifier: GVSVerifier) -> dict[str, object]:
    _assert_import_time_dependency_bindings()
    _assert_import_time_source_bindings()
    source_path = _VERIFIER_SOURCE_PATH
    source_before = source_path.read_bytes()
    model_source_path = _MODEL_SOURCE_PATH
    model_source_before = model_source_path.read_bytes()
    audit = audit_verifier_architecture(verifier)
    torch_execution_before = _torch_execution_state_record()
    tensor_versions_before = _tensor_version_record(verifier)
    complete_state_before = _complete_state_record(verifier)
    direct_topology_before = _direct_state_topology_record(verifier)
    trainable_state = [
        {
            "dtype": str(parameter.dtype),
            "name": name,
            "shape": list(parameter.shape),
            "sha256": _tensor_sha256(parameter),
        }
        for name, parameter in verifier.named_parameters()
        if parameter.requires_grad
    ]
    devices = sorted({str(parameter.device) for parameter in verifier.parameters()})
    dtypes = sorted({str(parameter.dtype) for parameter in verifier.parameters()})
    runtime = {
        "behavior_dependencies": _behavior_dependency_record(),
        "behavior_sha256": _verifier_behavior_sha256(verifier),
        "parameter_devices": devices,
        "parameter_dtypes": dtypes,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_execution_state": torch_execution_before,
        "torch_execution_state_sha256": _sha256_json(torch_execution_before),
        "torch_version": str(torch.__version__),
        "training_modes": [
            {"name": name, "training": module.training} for name, module in verifier.named_modules()
        ],
    }
    complete_state_after = _complete_state_record(verifier)
    direct_topology_after = _direct_state_topology_record(verifier)
    if (
        complete_state_after != complete_state_before
        or direct_topology_after != direct_topology_before
    ):
        raise GVSVerifierError("verifier state changed while constructing its receipt")
    if _tensor_version_record(verifier) != tensor_versions_before:
        raise GVSVerifierError("verifier tensor versions changed while constructing its receipt")
    if _torch_execution_state_record() != torch_execution_before:
        raise GVSVerifierError("Torch execution state changed while constructing its receipt")
    if (
        source_path.read_bytes() != source_before
        or model_source_path.read_bytes() != model_source_before
    ):
        raise GVSVerifierError("verifier/model source changed while constructing its receipt")
    receipt: dict[str, object] = {
        "architecture_audit": audit,
        "authorizes_cuda_or_jarvis_access": False,
        "authorizes_model_or_label_access": False,
        "backbone_checkpoint_receipt_required": True,
        "contract": verifier.contract.to_record(),
        "contract_sha256": verifier.contract.sha256,
        "complete_state": complete_state_before,
        "complete_state_sha256": _sha256_json(complete_state_before),
        "direct_state_topology": direct_topology_before,
        "direct_state_topology_sha256": _sha256_json(direct_topology_before),
        "launch_authorized": False,
        "runtime": runtime,
        "runtime_sha256": _sha256_json(runtime),
        "schema_version": GVS_VERIFIER_RECEIPT_VERSION,
        "model_source_sha256": _HASH_SHA256(model_source_before).hexdigest(),
        "source_sha256": _HASH_SHA256(source_before).hexdigest(),
        "trainable_state": trainable_state,
        "trainable_state_sha256": _sha256_json(trainable_state),
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    _assert_import_time_source_bindings()
    _assert_import_time_dependency_bindings()
    return receipt


def verify_verifier_receipt(
    verifier: GVSVerifier,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Detach and compare a receipt against complete live recomputation."""

    snapshot = _snapshot_json(receipt, label="verifier receipt")
    if type(snapshot) is not dict:
        raise GVSVerifierError("verifier receipt must be an exact JSON object")
    encoded = _canonical_json(snapshot).encode("utf-8")
    if len(encoded) > _MAX_RECEIPT_JSON_BYTES:
        raise GVSVerifierError("verifier receipt exceeds its bounded JSON size")
    expected = verifier_receipt(verifier)
    if _canonical_json(snapshot) != _canonical_json(expected):
        raise GVSVerifierError("verifier receipt differs from live recomputation")
    return snapshot


def load_verifier_receipt_json(raw: bytes, *, verifier: GVSVerifier) -> dict[str, object]:
    """Load strict bounded JSON and verify it against the live architecture/state."""

    if type(raw) is not bytes or not raw or len(raw) > _MAX_RECEIPT_JSON_BYTES:
        raise GVSVerifierError("verifier receipt bytes are empty, non-bytes, or oversized")

    def pairs_hook(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GVSVerifierError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = _JSON_LOADS(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_int=_bounded_json_integer,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GVSVerifierError(f"non-finite JSON constant {token!r}")
            ),
        )
    except GVSVerifierError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSVerifierError("verifier receipt is not strict UTF-8 JSON") from error
    return verify_verifier_receipt(verifier, payload)


def _backbone_artifact_identity(verifier: GVSVerifier) -> dict[str, object]:
    complete_state = _complete_state_record(verifier.backbone)
    topology = _direct_state_topology_record(verifier.backbone)
    for row in complete_state:
        del row["device"]
    for row in topology["tensor_slots"]:
        if row.get("tensor", object()) is not None:
            row.pop("device", None)
    return {
        "complete_state_sha256": _sha256_json(complete_state),
        "parameter_count": sum(parameter.numel() for parameter in verifier.backbone.parameters()),
        "topology_sha256": _sha256_json(topology),
    }


def _adapter_tensor_state(
    tensors: Mapping[str, Tensor],
    *,
    names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if set(tensors) != set(names):
        raise GVSVerifierError("adapter tensor names differ from the exact trainable roster")
    for name in names:
        tensor = tensors[name]
        if (
            type(tensor) is not Tensor
            or tensor.device.type != "cpu"
            or tensor.dtype is not torch.float32
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
        ):
            raise GVSVerifierError("adapter tensors must be exact contiguous CPU float32 tensors")
        if not bool(_TORCH_ISFINITE(tensor).all().item()):
            raise GVSVerifierError("adapter tensors must contain only finite values")
        rows.append(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "sha256": _tensor_sha256(tensor),
                "shape": list(tensor.shape),
            }
        )
    return rows


def _adapter_manifest(
    verifier: GVSVerifier,
    *,
    tensor_state: list[dict[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "adapter_tensor_state": tensor_state,
        "adapter_tensor_state_sha256": _sha256_json(tensor_state),
        "authorizes_cuda_or_jarvis_access": False,
        "authorizes_model_or_label_access": False,
        "backbone_checkpoint_receipt_required": True,
        "backbone_identity": _backbone_artifact_identity(verifier),
        "contract": verifier.contract.to_record(),
        "contract_sha256": verifier.contract.sha256,
        "launch_authorized": False,
        "schema_version": GVS_VERIFIER_ADAPTER_VERSION,
        "tensor_count": len(tensor_state),
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return manifest


def export_verifier_adapter(verifier: GVSVerifier) -> bytes:
    """Export only LoRA/head tensors plus an exact, nonauthorizing manifest."""

    _assert_import_time_dependency_bindings()
    audit_verifier_architecture(verifier)
    backbone_before = _backbone_artifact_identity(verifier)
    live_parameters = dict(verifier.named_parameters())
    names = _expected_trainable_names(verifier)
    tensors = {
        name: live_parameters[name].detach().to(device="cpu").contiguous().clone() for name in names
    }
    tensor_state = _adapter_tensor_state(tensors, names=names)
    manifest = _adapter_manifest(verifier, tensor_state=tensor_state)
    manifest_bytes = _canonical_json(manifest).encode("utf-8")
    if len(manifest_bytes) > _MAX_RECEIPT_JSON_BYTES:
        raise GVSVerifierError("adapter manifest exceeds its bounded JSON size")
    payload = dict(tensors)
    payload[_ADAPTER_MANIFEST_TENSOR] = _TORCH_TENSOR(
        list(manifest_bytes),
        device="cpu",
        dtype=torch.uint8,
    )
    try:
        artifact = _SAFETENSORS_SAVE(payload)
    except (SafetensorError, RuntimeError, TypeError, ValueError) as error:
        raise GVSVerifierError("adapter safetensors export failed closed") from error
    if type(artifact) is not bytes or not artifact or len(artifact) > _MAX_ADAPTER_BYTES:
        raise GVSVerifierError("adapter safetensors artifact is empty or oversized")
    audit_verifier_architecture(verifier)
    if _backbone_artifact_identity(verifier) != backbone_before:
        raise GVSVerifierError("frozen backbone changed while exporting the adapter")
    live_after = {
        name: dict(verifier.named_parameters())[name].detach().to(device="cpu").contiguous()
        for name in names
    }
    if _adapter_tensor_state(live_after, names=names) != tensor_state:
        raise GVSVerifierError("trainable verifier state changed while exporting the adapter")
    _assert_import_time_dependency_bindings()
    return artifact


def _load_adapter_manifest(tensor: Tensor) -> dict[str, object]:
    if (
        type(tensor) is not Tensor
        or tensor.device.type != "cpu"
        or tensor.dtype is not torch.uint8
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.numel() < 1
        or tensor.numel() > _MAX_RECEIPT_JSON_BYTES
    ):
        raise GVSVerifierError("adapter manifest tensor is malformed or oversized")
    raw = tensor.numpy().tobytes()

    def pairs_hook(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GVSVerifierError(f"duplicate adapter manifest key {key!r}")
            result[key] = value
        return result

    try:
        payload = _JSON_LOADS(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_int=_bounded_json_integer,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GVSVerifierError(f"non-finite adapter manifest constant {token!r}")
            ),
        )
    except GVSVerifierError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSVerifierError("adapter manifest is not strict bounded UTF-8 JSON") from error
    snapshot = _snapshot_json(payload, label="adapter manifest")
    if type(snapshot) is not dict:
        raise GVSVerifierError("adapter manifest must be an exact JSON object")
    expected_fields = {
        "adapter_tensor_state",
        "adapter_tensor_state_sha256",
        "authorizes_cuda_or_jarvis_access",
        "authorizes_model_or_label_access",
        "backbone_checkpoint_receipt_required",
        "backbone_identity",
        "contract",
        "contract_sha256",
        "launch_authorized",
        "manifest_sha256",
        "schema_version",
        "tensor_count",
    }
    if set(snapshot) != expected_fields:
        raise GVSVerifierError("adapter manifest field roster changed")
    claimed_sha256 = snapshot["manifest_sha256"]
    unsigned = dict(snapshot)
    del unsigned["manifest_sha256"]
    if claimed_sha256 != _sha256_json(unsigned):
        raise GVSVerifierError("adapter manifest self-hash is invalid")
    return snapshot


def load_verifier_adapter(verifier: GVSVerifier, artifact: bytes) -> dict[str, object]:
    """Validate and transactionally load a strict adapter/head-only safetensors artifact."""

    if torch.is_inference_mode_enabled():
        raise GVSVerifierError("adapter loading is forbidden in inference mode")
    _assert_import_time_dependency_bindings()
    audit_verifier_architecture(verifier)
    if type(artifact) is not bytes or not artifact or len(artifact) > _MAX_ADAPTER_BYTES:
        raise GVSVerifierError("adapter artifact bytes are empty, non-bytes, or oversized")
    try:
        loaded = _SAFETENSORS_LOAD(artifact)
    except (SafetensorError, RuntimeError, TypeError, ValueError) as error:
        raise GVSVerifierError("adapter artifact is not valid safetensors") from error
    try:
        canonical_artifact = _SAFETENSORS_SAVE(loaded)
    except (SafetensorError, RuntimeError, TypeError, ValueError) as error:
        raise GVSVerifierError("adapter artifact cannot be canonically reserialized") from error
    if canonical_artifact != artifact:
        raise GVSVerifierError("adapter artifact bytes are not canonical adapter-only safetensors")
    names = _expected_trainable_names(verifier)
    if set(loaded) != {*names, _ADAPTER_MANIFEST_TENSOR}:
        raise GVSVerifierError("adapter artifact contains missing or unexpected tensors")
    manifest = _load_adapter_manifest(loaded.pop(_ADAPTER_MANIFEST_TENSOR))
    if (
        manifest["schema_version"] != GVS_VERIFIER_ADAPTER_VERSION
        or manifest["launch_authorized"] is not False
        or manifest["authorizes_model_or_label_access"] is not False
        or manifest["authorizes_cuda_or_jarvis_access"] is not False
        or manifest["backbone_checkpoint_receipt_required"] is not True
    ):
        raise GVSVerifierError("adapter manifest authorization boundary changed")
    if (
        manifest["contract"] != verifier.contract.to_record()
        or manifest["contract_sha256"] != verifier.contract.sha256
    ):
        raise GVSVerifierError("adapter contract differs from the live verifier")
    if manifest["backbone_identity"] != _backbone_artifact_identity(verifier):
        raise GVSVerifierError("adapter artifact targets different frozen backbone bytes")
    tensor_state = _adapter_tensor_state(loaded, names=names)
    if (
        manifest["tensor_count"] != len(names)
        or manifest["adapter_tensor_state"] != tensor_state
        or manifest["adapter_tensor_state_sha256"] != _sha256_json(tensor_state)
    ):
        raise GVSVerifierError("adapter tensor state differs from its manifest")
    live = dict(verifier.named_parameters())
    for row in tensor_state:
        name = row["name"]
        if type(name) is not str or list(live[name].shape) != row["shape"]:
            raise GVSVerifierError("adapter tensor shape differs from the live trainable")
    staged = {
        name: loaded[name].to(device=live[name].device, dtype=torch.float32).contiguous()
        for name in names
    }
    previous = {name: live[name].detach().clone() for name in names}
    try:
        with torch.no_grad():
            for name in names:
                live[name].copy_(staged[name])
        audit_verifier_architecture(verifier)
        installed = {name: live[name].detach().to(device="cpu").contiguous() for name in names}
        if _adapter_tensor_state(installed, names=names) != tensor_state:
            raise GVSVerifierError("loaded adapter state failed exact post-copy verification")
    except (GVSVerifierError, RuntimeError, TypeError, ValueError) as error:
        with torch.no_grad():
            for name in names:
                live[name].copy_(previous[name])
        raise GVSVerifierError(
            "adapter load failed and restored the prior trainable state"
        ) from error
    _assert_import_time_dependency_bindings()
    return manifest


def _validate_candidate_identities(
    candidate_identities: tuple[tuple[str | None, ...], ...],
    *,
    valid_mask: Tensor,
    label: str,
) -> tuple[tuple[str | None, ...], ...]:
    if type(candidate_identities) is not tuple or len(candidate_identities) != valid_mask.shape[0]:
        raise GVSVerifierError(f"{label} must contain one exact tuple per score row")
    valid_rows = valid_mask.detach().to(device="cpu").tolist()
    for row_index, (identities, validity) in enumerate(
        zip(candidate_identities, valid_rows, strict=True)
    ):
        if type(identities) is not tuple or len(identities) != FROZEN_CANDIDATE_COUNT:
            raise GVSVerifierError(f"{label} row {row_index} must contain exactly eight slots")
        valid_identities: list[str] = []
        for slot, (identity, valid) in enumerate(zip(identities, validity, strict=True)):
            if valid:
                if (
                    type(identity) is not str
                    or len(identity) != 64
                    or any(character not in "0123456789abcdef" for character in identity)
                ):
                    raise GVSVerifierError(
                        f"{label} row {row_index} valid slot {slot} lacks a canonical SHA-256"
                    )
                valid_identities.append(identity)
            elif identity is not None:
                raise GVSVerifierError(
                    f"{label} row {row_index} invalid/duplicate slot {slot} must have no identity"
                )
        if len(set(valid_identities)) != len(valid_identities):
            raise GVSVerifierError(
                f"{label} row {row_index} contains duplicate valid canonical actions"
            )
    return candidate_identities


def multi_positive_listwise_loss(
    scores: Tensor,
    *,
    exact_mask: Tensor,
    valid_mask: Tensor,
    candidate_identities: tuple[tuple[str | None, ...], ...],
) -> Tensor:
    """Loss over distinct valid actions; invalid/duplicate slots receive no mass."""

    _assert_import_time_dependency_bindings()
    _assert_execution_callable_bindings()
    _reject_execution_modes()
    if (
        type(scores) is not Tensor
        or type(exact_mask) is not Tensor
        or type(valid_mask) is not Tensor
    ):
        raise GVSVerifierError("scores and masks must be exact tensors")
    if scores.ndim != 2 or scores.shape[1] != FROZEN_CANDIDATE_COUNT:
        raise GVSVerifierError("listwise scores must have shape [rows, 8]")
    if scores.shape[0] < 1:
        raise GVSVerifierError("listwise scores must contain at least one row")
    if exact_mask.shape != scores.shape or valid_mask.shape != scores.shape:
        raise GVSVerifierError("listwise masks must exactly match scores")
    if exact_mask.dtype is not torch.bool or valid_mask.dtype is not torch.bool:
        raise GVSVerifierError("listwise masks must be boolean")
    if scores.dtype is not torch.float32:
        raise GVSVerifierError("listwise scores must be float32")
    if (
        scores.layout is not torch.strided
        or exact_mask.layout is not torch.strided
        or valid_mask.layout is not torch.strided
    ):
        raise GVSVerifierError("listwise scores and masks must use dense strided layouts")
    if exact_mask.device != scores.device or valid_mask.device != scores.device:
        raise GVSVerifierError("listwise scores and masks must share one device")
    _validate_candidate_identities(
        candidate_identities,
        valid_mask=valid_mask,
        label="listwise candidate identities",
    )
    if not bool(_TORCH_ISFINITE(scores).all().item()):
        raise GVSVerifierError("listwise scores must be finite float32 values")
    if bool((exact_mask & ~valid_mask).any().item()):
        raise GVSVerifierError("an exact candidate cannot be invalid")
    positives = exact_mask.sum(dim=1)
    negatives = (valid_mask & ~exact_mask).sum(dim=1)
    if bool((positives < 1).any().item()):
        raise GVSVerifierError("every training row must retain at least one exact candidate")
    if bool((negatives < 1).any().item()):
        raise GVSVerifierError("every training row must retain at least one valid negative")
    negative_infinity = _TORCH_TENSOR(float("-inf"), device=scores.device, dtype=scores.dtype)
    valid_scores = _TORCH_WHERE(valid_mask, scores, negative_infinity)
    exact_scores = _TORCH_WHERE(exact_mask, scores, negative_infinity)
    per_row = _TORCH_LOGSUMEXP(valid_scores, dim=1) - _TORCH_LOGSUMEXP(exact_scores, dim=1)
    if not bool(_TORCH_ISFINITE(per_row).all().item()):
        raise GVSVerifierError("listwise loss produced a non-finite value")
    _assert_import_time_dependency_bindings()
    _assert_execution_callable_bindings()
    return per_row.mean()


def select_valid_candidate(
    scores: Tensor,
    valid_mask: Tensor,
    candidate_identities: tuple[tuple[str | None, ...], ...],
) -> Tensor:
    """Select by score, breaking ties by canonical identity rather than tensor slot."""

    _assert_import_time_dependency_bindings()
    _assert_execution_callable_bindings()
    _reject_execution_modes()
    if type(scores) is not Tensor or type(valid_mask) is not Tensor:
        raise GVSVerifierError("selection scores and mask must be exact tensors")
    if scores.ndim != 2 or scores.shape[1] != FROZEN_CANDIDATE_COUNT:
        raise GVSVerifierError("selection scores must have shape [rows, 8]")
    if scores.shape[0] < 1:
        raise GVSVerifierError("selection scores must contain at least one row")
    if valid_mask.shape != scores.shape or valid_mask.dtype is not torch.bool:
        raise GVSVerifierError("selection valid_mask must be boolean and match scores")
    if scores.dtype is not torch.float32:
        raise GVSVerifierError("selection scores must be float32")
    if scores.layout is not torch.strided or valid_mask.layout is not torch.strided:
        raise GVSVerifierError("selection scores and mask must use dense strided layouts")
    if valid_mask.device != scores.device:
        raise GVSVerifierError("selection scores and mask must share one device")
    identities = _validate_candidate_identities(
        candidate_identities,
        valid_mask=valid_mask,
        label="selection candidate identities",
    )
    if not bool(_TORCH_ISFINITE(scores).all().item()):
        raise GVSVerifierError("selection scores must be finite float32 values")
    if bool((valid_mask.sum(dim=1) < 1).any().item()):
        raise GVSVerifierError("every row must retain a valid candidate")
    score_rows = scores.detach().to(device="cpu").tolist()
    valid_rows = valid_mask.detach().to(device="cpu").tolist()
    selected_slots: list[int] = []
    for row_scores, row_valid, row_identities in zip(
        score_rows,
        valid_rows,
        identities,
        strict=True,
    ):
        valid_slots = [slot for slot, valid in enumerate(row_valid) if valid]
        best_score = max(row_scores[slot] for slot in valid_slots)
        tied_slots = [slot for slot in valid_slots if row_scores[slot] == best_score]
        selected_slots.append(min(tied_slots, key=lambda slot: row_identities[slot] or ""))
    selected = _TORCH_TENSOR(selected_slots, device=scores.device, dtype=torch.int64)
    _assert_import_time_dependency_bindings()
    _assert_execution_callable_bindings()
    return selected


__all__ = [
    "GVS_VERIFIER_ADAPTER_VERSION",
    "GVS_VERIFIER_CONTRACT_VERSION",
    "GVS_VERIFIER_RECEIPT_VERSION",
    "PRODUCTION_ALPHA",
    "PRODUCTION_GENERATOR_PARAMETERS",
    "PRODUCTION_RANK",
    "PRODUCTION_SEED",
    "PRODUCTION_SYSTEM_PARAMETERS",
    "PRODUCTION_VERIFIER_PARAMETERS",
    "FrozenLinearLoRA",
    "GVSVerifier",
    "GVSVerifierContract",
    "GVSVerifierError",
    "audit_verifier_architecture",
    "build_verifier_contract",
    "export_verifier_adapter",
    "load_verifier_adapter",
    "load_verifier_receipt_json",
    "multi_positive_listwise_loss",
    "select_valid_candidate",
    "verifier_receipt",
    "verify_verifier_receipt",
]
