"""CPU-testable frozen-hidden-state rankers for proposed BarunAction GVS-v1.

The rankers in this module are lightweight calibration controls inspired by SCaTR:
they score the final non-padding-token representation from the penultimate
transformer block of a *frozen* candidate generator.  This module deliberately
does not extract model features, load a checkpoint, read a dataset, train a model,
or authorize CUDA/Jarvis/model/label access.  Those boundaries require separate
hash-bound receipts.

Two architectures are fixed before outcome access:

* ``linear``: one biased scalar projection (449 parameters at BarunLM dim 448);
* ``small_mlp``: 448 -> 64 -> 16 -> 1 with ReLU (29,793 parameters).

The small MLP is a task-scale adaptation, not a reproduction of SCaTR's much
larger-model hyperparameter search.  Both arms must be evaluated alongside the
no-fit likelihood baseline and the separately proposed LoRA verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import InitVar, asdict, dataclass, fields
from enum import Enum
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules import module as _torch_module

from barunlm import config as _config_module
from barunlm.config import BarunConfig
from barunlm.evaluation.sim_program import (
    SimProgramError,
    assert_sim_program_runtime_integrity,
    canonical_json,
    module_runtime_sha256,
    runtime_callable_identity,
    sim_program_runtime_sha256,
)

GVS_CALIBRATED_RANKER_CONTRACT_VERSION = "barun-gvs-calibrated-ranker-contract-v2"
GVS_CALIBRATION_SPLIT_VERSION = "barun-gvs-calibration-component-split-v2"

PRODUCTION_FEATURE_DIMENSION = 448
PRODUCTION_PENULTIMATE_BLOCK_INDEX = 10
PRODUCTION_CANDIDATE_COUNT = 8
PRODUCTION_LINEAR_PARAMETERS = 449
PRODUCTION_SMALL_MLP_PARAMETERS = 29_793
PRODUCTION_LORA_VERIFIER_PARAMETERS = 135_617
PRODUCTION_SEED = 17

_MAX_IDENTIFIER_LENGTH = 256
_RANKER_CONTRACT_FACTORY_TOKEN = object()
_CALIBRATION_SPLIT_FACTORY_TOKEN = object()
_MODULE_HOOK_FIELDS = (
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
)
_BASE_MODULE_STATE_FIELDS = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    }
)
_GLOBAL_MODULE_HOOK_FIELDS = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_forward_pre_hooks_with_kwargs",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)


class GVSCalibratedRankerError(ValueError):
    """A calibrated-ranker architecture, split, loss, or selection is invalid."""


class CalibratedRankerKind(str, Enum):
    """The two fixed calibration controls; this is not a tuning search space."""

    LINEAR = "linear"
    SMALL_MLP = "small_mlp"


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise GVSCalibratedRankerError(f"{label} must be a nonempty string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSCalibratedRankerError(f"{label} must be valid UTF-8") from error
    if len(encoded) > _MAX_IDENTIFIER_LENGTH:
        raise GVSCalibratedRankerError(
            f"{label} must contain at most {_MAX_IDENTIFIER_LENGTH} UTF-8 bytes"
        )
    if value != unicodedata.normalize("NFC", value):
        raise GVSCalibratedRankerError(f"{label} must be NFC-normalized")
    if any(
        character.isspace() or unicodedata.category(character).startswith("C")
        for character in value
    ):
        raise GVSCalibratedRankerError(f"{label} contains forbidden whitespace or controls")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GVSCalibratedRankerError(f"{label} must be a lowercase SHA-256")
    return value


def calibrated_ranker_parameter_count(
    input_dimension: int,
    hidden_dimensions: tuple[int, ...],
) -> int:
    """Return the exact biased-Linear parameter count for one scalar ranker."""

    if type(input_dimension) is not int or input_dimension <= 0:
        raise GVSCalibratedRankerError("input_dimension must be a positive integer")
    if type(hidden_dimensions) is not tuple or any(
        type(dimension) is not int or dimension <= 0 for dimension in hidden_dimensions
    ):
        raise GVSCalibratedRankerError("hidden_dimensions must be a tuple of positive integers")
    dimensions = (input_dimension, *hidden_dimensions, 1)
    return sum(
        dimensions[index] * dimensions[index + 1] + dimensions[index + 1]
        for index in range(len(dimensions) - 1)
    )


@dataclass(frozen=True, slots=True)
class CalibratedRankerContract:
    """Frozen architecture and feature recipe for one calibration control."""

    kind: CalibratedRankerKind
    input_dimension: int
    hidden_dimensions: tuple[int, ...]
    trainable_parameters: int
    candidate_count: int
    transformer_blocks: int
    feature_block_index: int
    feature_token: str
    feature_stage: str
    activation: str
    class_balance: str
    split_rule: str
    candidate_metadata_visible: bool
    generator_frozen: bool
    authorizes_model_or_label_access: bool
    initialization_seed: int
    schema_version: str = GVS_CALIBRATED_RANKER_CONTRACT_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _RANKER_CONTRACT_FACTORY_TOKEN:
            raise TypeError(
                "CalibratedRankerContract must be constructed by build_calibrated_ranker_contract"
            )
        assert_calibrated_ranker_runtime_integrity()
        if (
            type(self.schema_version) is not str
            or self.schema_version != GVS_CALIBRATED_RANKER_CONTRACT_VERSION
        ):
            raise GVSCalibratedRankerError("unsupported calibrated-ranker contract version")
        if type(self.kind) is not CalibratedRankerKind:
            raise GVSCalibratedRankerError("kind must be a CalibratedRankerKind")
        expected_hidden = () if self.kind is CalibratedRankerKind.LINEAR else (64, 16)
        if type(self.hidden_dimensions) is not tuple or self.hidden_dimensions != expected_hidden:
            raise GVSCalibratedRankerError("hidden dimensions disagree with the fixed ranker kind")
        integer_fields = {
            "input_dimension": (self.input_dimension, PRODUCTION_FEATURE_DIMENSION),
            "trainable_parameters": (
                self.trainable_parameters,
                calibrated_ranker_parameter_count(self.input_dimension, self.hidden_dimensions),
            ),
            "candidate_count": (self.candidate_count, PRODUCTION_CANDIDATE_COUNT),
            "transformer_blocks": (self.transformer_blocks, 12),
            "feature_block_index": (
                self.feature_block_index,
                PRODUCTION_PENULTIMATE_BLOCK_INDEX,
            ),
        }
        if any(type(value) is not int for value, _ in integer_fields.values()):
            raise GVSCalibratedRankerError("contract integer fields must be exact integers")
        if self.input_dimension != PRODUCTION_FEATURE_DIMENSION:
            raise GVSCalibratedRankerError("production feature dimension must be exactly 448")
        expected_parameters = integer_fields["trainable_parameters"][1]
        if self.trainable_parameters != expected_parameters:
            raise GVSCalibratedRankerError("ranker parameter count was not recomputed exactly")
        if self.candidate_count != PRODUCTION_CANDIDATE_COUNT:
            raise GVSCalibratedRankerError("the frozen ranker contract requires exactly K=8")
        if self.transformer_blocks != 12 or self.feature_block_index != 10:
            raise GVSCalibratedRankerError(
                "feature source must be block 10 of the exact 12-block BarunLM"
            )
        if type(self.feature_token) is not str or self.feature_token != "final_nonpadding_token":
            raise GVSCalibratedRankerError("feature token rule changed")
        if (
            type(self.feature_stage) is not str
            or self.feature_stage != "post_penultimate_transformer_block"
        ):
            raise GVSCalibratedRankerError("feature stage rule changed")
        expected_activation = "identity" if self.kind is CalibratedRankerKind.LINEAR else "relu"
        if type(self.activation) is not str or self.activation != expected_activation:
            raise GVSCalibratedRankerError("ranker activation changed")
        if (
            type(self.class_balance) is not str
            or self.class_balance != "binary_cross_entropy_pos_weight_nneg_over_npos"
        ):
            raise GVSCalibratedRankerError("class-balanced loss rule changed")
        if (
            type(self.split_rule) is not str
            or self.split_rule != "sha256_ordered_component_level_3_of_4_train"
        ):
            raise GVSCalibratedRankerError("calibration split rule changed")
        if self.candidate_metadata_visible is not False:
            raise GVSCalibratedRankerError("rank and likelihood metadata must remain hidden")
        if self.generator_frozen is not True:
            raise GVSCalibratedRankerError("the candidate generator must remain frozen")
        if self.authorizes_model_or_label_access is not False:
            raise GVSCalibratedRankerError("this CPU contract cannot authorize external access")
        if type(self.initialization_seed) is not int or not 0 <= self.initialization_seed < 2**63:
            raise GVSCalibratedRankerError("initialization_seed must be a bounded integer")

    def as_record(self) -> dict[str, object]:
        audit_calibrated_ranker_contract(self)
        record = asdict(self)
        record["kind"] = self.kind.value
        record["hidden_dimensions"] = list(self.hidden_dimensions)
        record["ranker_runtime_sha256"] = calibrated_ranker_runtime_sha256()
        record["certifies_feature_provenance"] = False
        return record


def build_calibrated_ranker_contract(
    kind: CalibratedRankerKind | str,
    *,
    initialization_seed: int = PRODUCTION_SEED,
) -> CalibratedRankerContract:
    """Build one of the two exact preregistered calibration contracts."""

    assert_calibrated_ranker_runtime_integrity()
    try:
        parsed_kind = CalibratedRankerKind(kind)
    except (TypeError, ValueError) as error:
        raise GVSCalibratedRankerError("unsupported calibrated ranker kind") from error
    hidden_dimensions = () if parsed_kind is CalibratedRankerKind.LINEAR else (64, 16)
    return CalibratedRankerContract(
        kind=parsed_kind,
        input_dimension=PRODUCTION_FEATURE_DIMENSION,
        hidden_dimensions=hidden_dimensions,
        trainable_parameters=calibrated_ranker_parameter_count(
            PRODUCTION_FEATURE_DIMENSION, hidden_dimensions
        ),
        candidate_count=PRODUCTION_CANDIDATE_COUNT,
        transformer_blocks=12,
        feature_block_index=PRODUCTION_PENULTIMATE_BLOCK_INDEX,
        feature_token="final_nonpadding_token",
        feature_stage="post_penultimate_transformer_block",
        activation="identity" if parsed_kind is CalibratedRankerKind.LINEAR else "relu",
        class_balance="binary_cross_entropy_pos_weight_nneg_over_npos",
        split_rule="sha256_ordered_component_level_3_of_4_train",
        candidate_metadata_visible=False,
        generator_frozen=True,
        authorizes_model_or_label_access=False,
        initialization_seed=initialization_seed,
        _factory_token=_RANKER_CONTRACT_FACTORY_TOKEN,
    )


def audit_calibrated_ranker_contract(contract: object) -> dict[str, object]:
    """Recompute exact contract values and JSON scalar types after construction."""

    assert_calibrated_ranker_runtime_integrity()
    if type(contract) is not CalibratedRankerContract:
        raise GVSCalibratedRankerError("contract must be the exact reference class")
    expected = build_calibrated_ranker_contract(
        contract.kind,
        initialization_seed=contract.initialization_seed,
    )
    for field in fields(CalibratedRankerContract):
        actual_value = getattr(contract, field.name)
        expected_value = getattr(expected, field.name)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise GVSCalibratedRankerError("stored ranker contract changed")
    return {
        "kind": contract.kind.value,
        "parameter_count": contract.trainable_parameters,
        "authorizes_model_or_label_access": False,
        "certifies_feature_provenance": False,
        "ranker_runtime_sha256": calibrated_ranker_runtime_sha256(),
    }


class FrozenHiddenCalibratedRanker(nn.Module):
    """A scorer over already-extracted frozen BarunLM candidate representations."""

    def __init__(self, contract: CalibratedRankerContract) -> None:
        assert_calibrated_ranker_runtime_integrity()
        super().__init__()
        audit_calibrated_ranker_contract(contract)
        expected = build_calibrated_ranker_contract(
            contract.kind,
            initialization_seed=contract.initialization_seed,
        )
        self.contract = expected
        dimensions = (contract.input_dimension, *contract.hidden_dimensions, 1)
        layers: list[nn.Module] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(contract.initialization_seed)
            for index, (input_dimension, output_dimension) in enumerate(pairwise(dimensions)):
                layers.append(nn.Linear(input_dimension, output_dimension, bias=True))
                if index < len(dimensions) - 2:
                    layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        audit_calibrated_ranker(self)

    def __call__(self, features: Tensor) -> Tensor:
        # Audit before ``nn.Module._call_impl`` can run a self-removing pre-hook.
        assert_calibrated_ranker_runtime_integrity()
        audit_calibrated_ranker(self)
        return super().__call__(features)

    def forward(self, features: Tensor) -> Tensor:
        assert_calibrated_ranker_runtime_integrity()
        audit_calibrated_ranker(self)
        if type(features) is not Tensor or features.ndim != 2:
            raise GVSCalibratedRankerError("features must be an exact rank-two Tensor")
        if features.shape[0] < 1 or features.shape[1] != self.contract.input_dimension:
            raise GVSCalibratedRankerError("features have the wrong batch or hidden dimension")
        if features.device.type != "cpu" or features.layout is not torch.strided:
            raise GVSCalibratedRankerError("the reference ranker requires dense CPU features")
        if features.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise GVSCalibratedRankerError("features must use a supported floating dtype")
        feature_storage_pointer = features.untyped_storage().data_ptr()
        if any(
            feature_storage_pointer == parameter.untyped_storage().data_ptr()
            for parameter in self.parameters()
        ):
            raise GVSCalibratedRankerError("features must not alias ranker parameter storage")
        # The feature extractor is outside this trainable control. Detaching the
        # stable snapshot is part of the frozen-generator boundary: ranker loss
        # must not propagate into the source of these extracted representations.
        feature_snapshot = features.detach().clone()
        comparison_snapshot = features.detach().clone()
        if not bool(torch.isfinite(feature_snapshot).all().item()) or not bool(
            torch.isfinite(comparison_snapshot).all().item()
        ):
            raise GVSCalibratedRankerError("features contain a non-finite value")
        if not torch.equal(feature_snapshot.detach(), comparison_snapshot):
            raise GVSCalibratedRankerError("features changed during their stable snapshot")
        scores = self.network(feature_snapshot.float()).squeeze(-1)
        if not bool(torch.isfinite(scores).all().item()):
            raise GVSCalibratedRankerError("ranker produced a non-finite score")
        return scores


def audit_calibrated_ranker(ranker: object) -> dict[str, object]:
    """Recompute exact module, parameter, gradient, and storage invariants."""

    assert_calibrated_ranker_runtime_integrity()
    if any(bool(getattr(_torch_module, name, None)) for name in _GLOBAL_MODULE_HOOK_FIELDS):
        raise GVSCalibratedRankerError("global PyTorch module hooks are forbidden")
    if type(ranker) is not FrozenHiddenCalibratedRanker:
        raise GVSCalibratedRankerError("ranker must be the exact reference class")
    contract = ranker.contract
    audit_calibrated_ranker_contract(contract)
    expected_contract = build_calibrated_ranker_contract(
        contract.kind,
        initialization_seed=contract.initialization_seed,
    )
    if contract != expected_contract:  # Defensive after exact type-aware audit above.
        raise GVSCalibratedRankerError("stored ranker contract changed")
    expected_types: tuple[type[nn.Module], ...]
    if contract.kind is CalibratedRankerKind.LINEAR:
        expected_types = (nn.Linear,)
    else:
        expected_types = (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear)
    if (
        type(ranker.network) is not nn.Sequential
        or tuple(map(type, ranker.network)) != expected_types
    ):
        raise GVSCalibratedRankerError("ranker module topology changed")
    modules = tuple(ranker.modules())
    if tuple(ranker._modules) != ("network",) or tuple(ranker.network._modules) != tuple(
        str(index) for index in range(len(ranker.network))
    ):
        raise GVSCalibratedRankerError("ranker module registration changed")
    for module in modules:
        if any(bool(getattr(module, name, None)) for name in _MODULE_HOOK_FIELDS):
            raise GVSCalibratedRankerError("ranker modules cannot carry runtime hooks")
        if "forward" in vars(module) or "_call_impl" in vars(module):
            raise GVSCalibratedRankerError("ranker module callables cannot be overridden")
        if type(module.training) is not bool:
            raise GVSCalibratedRankerError("ranker training flags must be exact booleans")
        expected_state_fields = _BASE_MODULE_STATE_FIELDS
        if type(module) is FrozenHiddenCalibratedRanker:
            expected_state_fields = expected_state_fields | {"contract"}
        elif type(module) is nn.Linear:
            expected_state_fields = expected_state_fields | {"in_features", "out_features"}
        elif type(module) is nn.ReLU:
            expected_state_fields = expected_state_fields | {"inplace"}
        if set(vars(module)) != expected_state_fields:
            raise GVSCalibratedRankerError("ranker module carries uncontracted runtime state")
    linear_layers = [module for module in ranker.network if type(module) is nn.Linear]
    expected_dimensions = (contract.input_dimension, *contract.hidden_dimensions, 1)
    for layer, input_dimension, output_dimension in zip(
        linear_layers,
        expected_dimensions[:-1],
        expected_dimensions[1:],
        strict=True,
    ):
        if (
            type(layer.in_features) is not int
            or type(layer.out_features) is not int
            or layer.in_features != input_dimension
            or layer.out_features != output_dimension
        ):
            raise GVSCalibratedRankerError("ranker Linear dimensions changed")
        if layer.bias is None:
            raise GVSCalibratedRankerError("every ranker Linear must retain its bias")
        if tuple(layer._parameters) != ("weight", "bias") or layer._modules or layer._buffers:
            raise GVSCalibratedRankerError("ranker Linear registration changed")
        if layer.weight.shape != (output_dimension, input_dimension) or layer.bias.shape != (
            output_dimension,
        ):
            raise GVSCalibratedRankerError("ranker parameter shapes changed")
    relu_layers = [module for module in ranker.network if type(module) is nn.ReLU]
    if any(
        module.inplace is not False or module._parameters or module._modules or module._buffers
        for module in relu_layers
    ):
        raise GVSCalibratedRankerError("ranker ReLU contract changed")
    if (
        ranker._parameters
        or ranker._buffers
        or ranker.network._parameters
        or ranker.network._buffers
    ):
        raise GVSCalibratedRankerError("ranker carries parameters outside its Linear layers")
    named_parameters = tuple(ranker.named_parameters())
    if not named_parameters or any(
        not parameter.requires_grad for _, parameter in named_parameters
    ):
        raise GVSCalibratedRankerError("every and only ranker parameters must be trainable")
    if any(parameter.device.type != "cpu" for _, parameter in named_parameters):
        raise GVSCalibratedRankerError("the CPU reference cannot audit a moved ranker")
    expected_parameter_shapes: dict[str, tuple[int, ...]] = {}
    for index, (input_dimension, output_dimension) in enumerate(pairwise(expected_dimensions)):
        module_index = index * 2
        expected_parameter_shapes[f"network.{module_index}.weight"] = (
            output_dimension,
            input_dimension,
        )
        expected_parameter_shapes[f"network.{module_index}.bias"] = (output_dimension,)
    if tuple(name for name, _ in named_parameters) != tuple(expected_parameter_shapes):
        raise GVSCalibratedRankerError("ranker parameter names changed")
    storage_pointers = [parameter.untyped_storage().data_ptr() for _, parameter in named_parameters]
    if len(storage_pointers) != len(set(storage_pointers)):
        raise GVSCalibratedRankerError("ranker parameters must not alias storage")
    occupied_storage_pointers = set(storage_pointers)
    gradient_storage_pointers: set[int] = set()
    for name, parameter in named_parameters:
        if type(parameter) is not nn.Parameter:
            raise GVSCalibratedRankerError("ranker weights must be exact Parameters")
        if (
            parameter.shape != expected_parameter_shapes[name]
            or parameter.dtype is not torch.float32
            or parameter.layout is not torch.strided
            or not parameter.is_contiguous()
            or parameter.storage_offset() != 0
            or parameter.untyped_storage().nbytes() != parameter.numel() * parameter.element_size()
        ):
            raise GVSCalibratedRankerError(
                "ranker parameters must have exact dense float32 storage"
            )
        if not bool(torch.isfinite(parameter).all().item()):
            raise GVSCalibratedRankerError("ranker parameters contain a non-finite value")
        if any(
            bool(getattr(parameter, hook_name, None))
            for hook_name in ("_backward_hooks", "_post_accumulate_grad_hooks")
        ):
            raise GVSCalibratedRankerError("ranker parameters cannot carry gradient hooks")
        gradient = parameter.grad
        if gradient is not None:
            if (
                type(gradient) is not Tensor
                or gradient.shape != parameter.shape
                or gradient.dtype is not torch.float32
                or gradient.device.type != "cpu"
                or gradient.layout is not torch.strided
                or not gradient.is_contiguous()
                or gradient.storage_offset() != 0
                or gradient.untyped_storage().nbytes() != gradient.numel() * gradient.element_size()
                or not bool(torch.isfinite(gradient).all().item())
            ):
                raise GVSCalibratedRankerError("ranker parameter gradients are malformed")
            gradient_storage_pointer = gradient.untyped_storage().data_ptr()
            if (
                gradient_storage_pointer in occupied_storage_pointers
                or gradient_storage_pointer in gradient_storage_pointers
            ):
                raise GVSCalibratedRankerError("ranker gradients must own disjoint storage")
            gradient_storage_pointers.add(gradient_storage_pointer)
    parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
    if parameter_count != contract.trainable_parameters:
        raise GVSCalibratedRankerError("actual ranker parameter count disagrees with contract")
    if tuple(ranker.named_buffers()):
        raise GVSCalibratedRankerError("ranker cannot carry uncontracted buffers")
    return {
        "kind": contract.kind.value,
        "parameter_count": parameter_count,
        "parameter_names": [name for name, _ in named_parameters],
        "contract": contract.as_record(),
        "authorizes_model_or_label_access": False,
        "certifies_feature_provenance": False,
        "ranker_runtime_sha256": calibrated_ranker_runtime_sha256(),
    }


def class_balanced_binary_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Weighted BCE with the fixed ``negative / positive`` positive-class weight."""

    assert_calibrated_ranker_runtime_integrity()
    if type(logits) is not Tensor or type(labels) is not Tensor:
        raise GVSCalibratedRankerError("logits and labels must be exact Tensors")
    if logits.ndim != 1 or labels.shape != logits.shape or logits.numel() < 2:
        raise GVSCalibratedRankerError("logits and labels must be aligned nontrivial vectors")
    if logits.device.type != "cpu" or labels.device.type != "cpu":
        raise GVSCalibratedRankerError("the reference loss requires CPU tensors")
    if logits.layout is not torch.strided or labels.layout is not torch.strided:
        raise GVSCalibratedRankerError("the reference loss requires dense strided tensors")
    if logits.dtype is not torch.float32 or labels.dtype is not torch.bool:
        raise GVSCalibratedRankerError("logits must be float32 and labels must be boolean")
    logits_snapshot = logits.clone()
    labels_snapshot = labels.clone()
    comparison_logits = logits.detach().clone()
    if not bool(torch.isfinite(logits_snapshot).all().item()) or not bool(
        torch.isfinite(comparison_logits).all().item()
    ):
        raise GVSCalibratedRankerError("logits contain a non-finite value")
    if not torch.equal(logits_snapshot.detach(), comparison_logits) or not torch.equal(
        labels_snapshot, labels.clone()
    ):
        raise GVSCalibratedRankerError("loss inputs changed during their stable snapshot")
    positives = int(labels_snapshot.sum().item())
    negatives = labels_snapshot.numel() - positives
    if positives == 0 or negatives == 0:
        raise GVSCalibratedRankerError("each calibration batch must contain both label classes")
    positive_weight = torch.tensor(
        negatives / positives,
        dtype=torch.float32,
        device=logits_snapshot.device,
    )
    loss = F.binary_cross_entropy_with_logits(
        logits_snapshot,
        labels_snapshot.to(dtype=torch.float32),
        pos_weight=positive_weight,
    )
    if loss.ndim != 0 or loss.dtype is not torch.float32 or not bool(torch.isfinite(loss).item()):
        raise GVSCalibratedRankerError("class-balanced loss produced a non-finite scalar")
    return loss


def _component_membership_sha256(component_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(GVS_CALIBRATION_SPLIT_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    for identifier in sorted(component_ids):
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _expected_component_partitions(
    component_ids: tuple[str, ...],
    *,
    population_sha256: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = sorted(
        component_ids,
        key=lambda identifier: (
            hashlib.sha256(
                (
                    f"{GVS_CALIBRATION_SPLIT_VERSION}\x00{population_sha256}"
                    f"\x00{seed}\x00{identifier}"
                ).encode()
            ).digest(),
            identifier,
        ),
    )
    train_count = (3 * len(ordered)) // 4
    if train_count == 0 or train_count == len(ordered):
        raise GVSCalibratedRankerError("component split produced an empty partition")
    return tuple(sorted(ordered[:train_count])), tuple(sorted(ordered[train_count:]))


@dataclass(frozen=True, slots=True)
class CalibrationComponentSplit:
    """Deterministic problem/component-level 75/25 split evidence."""

    population_sha256: str
    component_membership_sha256: str
    seed: int
    train_component_ids: tuple[str, ...]
    validation_component_ids: tuple[str, ...]
    schema_version: str = GVS_CALIBRATION_SPLIT_VERSION
    authorizes_model_or_label_access: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _CALIBRATION_SPLIT_FACTORY_TOKEN:
            raise TypeError(
                "CalibrationComponentSplit must be constructed by build_calibration_component_split"
            )
        assert_calibrated_ranker_runtime_integrity()
        if type(self.schema_version) is not str or self.schema_version != (
            GVS_CALIBRATION_SPLIT_VERSION
        ):
            raise GVSCalibratedRankerError("unsupported calibration split version")
        _strict_sha256(self.population_sha256, label="population_sha256")
        _strict_sha256(
            self.component_membership_sha256,
            label="component_membership_sha256",
        )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise GVSCalibratedRankerError("split seed must be a bounded integer")
        for name, values in (
            ("train_component_ids", self.train_component_ids),
            ("validation_component_ids", self.validation_component_ids),
        ):
            if type(values) is not tuple or not values:
                raise GVSCalibratedRankerError(f"{name} must be a nonempty tuple")
            for index, value in enumerate(values):
                _strict_identifier(value, label=f"{name}[{index}]")
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise GVSCalibratedRankerError(f"{name} must be sorted and unique")
        if set(self.train_component_ids) & set(self.validation_component_ids):
            raise GVSCalibratedRankerError("calibration train/validation components overlap")
        all_component_ids = tuple(
            sorted((*self.train_component_ids, *self.validation_component_ids))
        )
        if len(all_component_ids) < 4:
            raise GVSCalibratedRankerError("at least four frozen component IDs are required")
        if _component_membership_sha256(all_component_ids) != (self.component_membership_sha256):
            raise GVSCalibratedRankerError("component membership hash mismatch")
        expected_train, expected_validation = _expected_component_partitions(
            all_component_ids,
            population_sha256=self.population_sha256,
            seed=self.seed,
        )
        if (
            self.train_component_ids != expected_train
            or self.validation_component_ids != expected_validation
        ):
            raise GVSCalibratedRankerError("component partitions violate the frozen split rule")
        if self.authorizes_model_or_label_access is not False:
            raise GVSCalibratedRankerError("a split record cannot authorize external access")

    def as_record(self) -> dict[str, object]:
        audit = audit_calibration_component_split(self)
        return {
            "schema_version": self.schema_version,
            "population_sha256": self.population_sha256,
            "component_membership_sha256": self.component_membership_sha256,
            "seed": self.seed,
            "train_component_ids": list(self.train_component_ids),
            "validation_component_ids": list(self.validation_component_ids),
            "authorizes_model_or_label_access": False,
            "certifies_feature_provenance": False,
            "ranker_runtime_sha256": audit["ranker_runtime_sha256"],
        }


def audit_calibration_component_split(split: object) -> dict[str, object]:
    """Recompute split membership, partition, and nonauthorization invariants."""

    assert_calibrated_ranker_runtime_integrity()
    if type(split) is not CalibrationComponentSplit:
        raise GVSCalibratedRankerError("split must be the exact reference class")
    all_component_ids = tuple(sorted((*split.train_component_ids, *split.validation_component_ids)))
    expected = build_calibration_component_split(
        all_component_ids,
        population_sha256=split.population_sha256,
        seed=split.seed,
    )
    if split != expected:
        raise GVSCalibratedRankerError("stored calibration split changed")
    return {
        "component_count": len(all_component_ids),
        "train_component_count": len(split.train_component_ids),
        "validation_component_count": len(split.validation_component_ids),
        "population_sha256": split.population_sha256,
        "component_membership_sha256": split.component_membership_sha256,
        "authorizes_model_or_label_access": False,
        "certifies_feature_provenance": False,
        "ranker_runtime_sha256": calibrated_ranker_runtime_sha256(),
    }


def build_calibration_component_split(
    component_ids: tuple[str, ...],
    *,
    population_sha256: str,
    seed: int = PRODUCTION_SEED,
) -> CalibrationComponentSplit:
    """Split frozen connected components without separating their rollouts."""

    assert_calibrated_ranker_runtime_integrity()
    _strict_sha256(population_sha256, label="population_sha256")
    if type(component_ids) is not tuple or len(component_ids) < 4:
        raise GVSCalibratedRankerError("at least four frozen component IDs are required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise GVSCalibratedRankerError("split seed must be a bounded integer")
    checked = tuple(
        _strict_identifier(value, label=f"component_ids[{index}]")
        for index, value in enumerate(component_ids)
    )
    if len(checked) != len(set(checked)):
        raise GVSCalibratedRankerError("component IDs must be unique")
    train_ids, validation_ids = _expected_component_partitions(
        checked,
        population_sha256=population_sha256,
        seed=seed,
    )
    return CalibrationComponentSplit(
        population_sha256=population_sha256,
        component_membership_sha256=_component_membership_sha256(checked),
        seed=seed,
        train_component_ids=train_ids,
        validation_component_ids=validation_ids,
        _factory_token=_CALIBRATION_SPLIT_FACTORY_TOKEN,
    )


def select_calibrated_candidate(
    scores: tuple[float, ...],
    candidate_sha256s: tuple[str, ...],
) -> int:
    """Select from deduplicated valid candidates without using generator rank metadata."""

    assert_calibrated_ranker_runtime_integrity()
    if (
        type(scores) is not tuple
        or type(candidate_sha256s) is not tuple
        or not 1 <= len(scores) <= PRODUCTION_CANDIDATE_COUNT
        or len(scores) != len(candidate_sha256s)
    ):
        raise GVSCalibratedRankerError("selection requires one to eight aligned candidates")
    for index, score in enumerate(scores):
        if type(score) is not float or not math.isfinite(score):
            raise GVSCalibratedRankerError(f"scores[{index}] must be a finite exact float")
    checked_ids = tuple(
        _strict_sha256(value, label=f"candidate_sha256s[{index}]")
        for index, value in enumerate(candidate_sha256s)
    )
    if len(checked_ids) != len(set(checked_ids)):
        raise GVSCalibratedRankerError("selection candidates must be canonical and deduplicated")
    # Canonical hash is the preregistered tie-break, so reordering candidates cannot change
    # the semantic winner. The returned position only addresses the caller's aligned tuple.
    return min(range(len(scores)), key=lambda index: (-scores[index], checked_ids[index]))


def assert_production_config(config: BarunConfig) -> None:
    """Fail closed unless the feature recipe targets exact BarunLM-35M topology."""

    assert_calibrated_ranker_runtime_integrity()
    if type(config) is not BarunConfig:
        raise GVSCalibratedRankerError("config must be the exact BarunConfig class")
    canonical = BarunConfig()
    first = tuple(
        (field.name, type(getattr(config, field.name)), getattr(config, field.name))
        for field in fields(BarunConfig)
    )
    second = tuple(
        (field.name, type(getattr(config, field.name)), getattr(config, field.name))
        for field in fields(BarunConfig)
    )
    expected = tuple(
        (field.name, type(getattr(canonical, field.name)), getattr(canonical, field.name))
        for field in fields(BarunConfig)
    )
    if first != second:
        raise GVSCalibratedRankerError("production config changed during its stable snapshot")
    if first != expected:
        raise GVSCalibratedRankerError("calibrated-ranker production config must be canonical")


def calibrated_ranker_runtime_sha256() -> str:
    """Bind exact source, live code, PyTorch dependencies, and lower contracts."""

    dependencies = {
        name: json.loads(canonical_json(runtime_callable_identity(value)))
        for name, value in (
            ("Tensor.clone", Tensor.clone),
            ("binary_cross_entropy_with_logits", F.binary_cross_entropy_with_logits),
            ("hashlib.sha256", hashlib.sha256),
            ("math.isfinite", math.isfinite),
            ("Module.__call__", nn.Module.__call__),
            ("Module._call_impl", nn.Module._call_impl),
            ("Module.modules", nn.Module.modules),
            ("Module.named_parameters", nn.Module.named_parameters),
            ("Linear.forward", nn.Linear.forward),
            ("ReLU.forward", nn.ReLU.forward),
            ("Sequential.forward", nn.Sequential.forward),
            ("torch.equal", torch.equal),
            ("torch.isfinite", torch.isfinite),
            ("torch.manual_seed", torch.manual_seed),
            ("torch.random.fork_rng", torch.random.fork_rng),
            ("unicodedata.category", unicodedata.category),
            ("unicodedata.normalize", unicodedata.normalize),
        )
    }
    config_source_path = getattr(_config_module, "__file__", None)
    if not isinstance(config_source_path, str):
        raise GVSCalibratedRankerError("BarunConfig module has no source identity")
    try:
        config_runtime_sha256 = module_runtime_sha256(
            vars(_config_module),
            module_name=_config_module.__name__,
            source_path=config_source_path,
            contract={"canonical_config": asdict(BarunConfig())},
        )
        contract = {
            "schema_versions": {
                "ranker": GVS_CALIBRATED_RANKER_CONTRACT_VERSION,
                "split": GVS_CALIBRATION_SPLIT_VERSION,
            },
            "production": {
                "feature_dimension": PRODUCTION_FEATURE_DIMENSION,
                "penultimate_block_index": PRODUCTION_PENULTIMATE_BLOCK_INDEX,
                "candidate_count": PRODUCTION_CANDIDATE_COUNT,
                "linear_parameters": PRODUCTION_LINEAR_PARAMETERS,
                "small_mlp_parameters": PRODUCTION_SMALL_MLP_PARAMETERS,
                "lora_verifier_parameters": PRODUCTION_LORA_VERIFIER_PARAMETERS,
                "seed": PRODUCTION_SEED,
            },
            "ranker_kinds": [kind.value for kind in CalibratedRankerKind],
            "identifier_max_utf8_bytes": _MAX_IDENTIFIER_LENGTH,
            "module_hook_fields": list(_MODULE_HOOK_FIELDS),
            "global_module_hook_fields": list(_GLOBAL_MODULE_HOOK_FIELDS),
            "base_module_state_fields": sorted(_BASE_MODULE_STATE_FIELDS),
            "torch": {
                "version": str(torch.__version__),
                "git_version": torch.version.git_version,
                "default_dtype": str(torch.get_default_dtype()),
                "default_device": str(torch.get_default_device()),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
                "build_config_sha256": hashlib.sha256(
                    torch.__config__.show().encode("utf-8")
                ).hexdigest(),
            },
            "config_runtime_sha256": config_runtime_sha256,
            "sim_program_runtime_sha256": sim_program_runtime_sha256(),
            "dependencies": dependencies,
        }
        return module_runtime_sha256(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except SimProgramError as error:
        raise GVSCalibratedRankerError("could not bind the calibrated-ranker runtime") from error


def assert_calibrated_ranker_runtime_integrity() -> None:
    assert_sim_program_runtime_integrity()
    current = calibrated_ranker_runtime_sha256()
    if current != _PINNED_CALIBRATED_RANKER_RUNTIME_SHA256:
        raise GVSCalibratedRankerError(
            "calibrated-ranker runtime differs from its import-time identity"
        )


__all__ = [
    "GVS_CALIBRATED_RANKER_CONTRACT_VERSION",
    "GVS_CALIBRATION_SPLIT_VERSION",
    "PRODUCTION_LINEAR_PARAMETERS",
    "PRODUCTION_SMALL_MLP_PARAMETERS",
    "CalibratedRankerContract",
    "CalibratedRankerKind",
    "CalibrationComponentSplit",
    "FrozenHiddenCalibratedRanker",
    "GVSCalibratedRankerError",
    "assert_calibrated_ranker_runtime_integrity",
    "assert_production_config",
    "audit_calibrated_ranker",
    "audit_calibrated_ranker_contract",
    "audit_calibration_component_split",
    "build_calibrated_ranker_contract",
    "build_calibration_component_split",
    "calibrated_ranker_parameter_count",
    "calibrated_ranker_runtime_sha256",
    "class_balanced_binary_loss",
    "select_calibrated_candidate",
]


_PINNED_CALIBRATED_RANKER_RUNTIME_SHA256 = calibrated_ranker_runtime_sha256()
