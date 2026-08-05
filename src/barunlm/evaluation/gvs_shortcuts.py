"""Fail-closed shortcut-control accounting for proposed BarunAction GVS-v1.

The proposed verifier must not succeed because correct candidates are shorter, more
syntactically regular, easier to validate, or associated with a frequent tool.  This
module freezes the CPU-only diagnostic shape for those four controls without claiming
that the evidence needed to run them already exists.

The current support bridge can rederive three candidate signals on ``D-support``:

* length is the exact number of model-visible content tokens;
* syntax is parse validity (necessarily true inside the verifier-eligible domain);
* validity is complete, nontruncated schema validity (also necessarily true there).

The original v1 inventory keeps those limitations explicit.  The CPU-only v2 contract
adds a complete structural-firewall binding for ``T-new`` and ``D-support``, replays one
certified single-fault negative per calibration row, and derives canonical candidate
tool multisets by live schema lowering.  It emits structural calibration/evaluation
scores only when both exact denominators and every D-support contrast are present.
External secret custody and single-use signer authentication remain absent, so v2 never
turns those structural scores into a scientific result or an authorization gate.  The
historical 60% value remains only a provisional diagnostic reference.

This module never reads protected prompts or labels, loads a model, contacts a remote
service, or authorizes model, label, CUDA, training, or JarvisLabs access.  Its receipts
do not certify external custody.  A later implementation must add an authenticated
External custody must still be supplied by a separate real service before scientific
use; hashes or caller booleans are not accepted as substitutes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

from . import action_ir as action_ir_module
from . import action_simulator as action_simulator_module
from . import gvs_bridge as gvs_bridge_module
from . import gvs_faults as gvs_faults_module
from . import gvs_population as gvs_population_module
from . import gvs_schema_presentation as gvs_schema_presentation_module
from . import gvs_support as gvs_support_module
from . import sim_program as sim_program_module
from .action_ir import ActionIR, parse_action_ir
from .action_simulator import (
    SIMULATOR_TOOL_REGISTRY,
    SIMULATOR_TOOL_SCHEMAS,
    ActionSimulator,
    WorldState,
    assert_action_simulator_runtime_integrity,
    compile_program_step,
)
from .gvs_bridge import CandidateSupportBridgeInput, CandidateSupportBridgeReceipt
from .gvs_faults import (
    FaultCertification,
    GVSFaultError,
    assert_gvs_fault_runtime_integrity,
    certify_single_fault,
)
from .gvs_population import (
    GVSPopulationError,
    assert_gvs_population_runtime_integrity,
    validate_population_firewall,
    validate_scan_evidence,
)
from .gvs_schema_presentation import (
    GVSSchemaPresentationError,
    lower_presented_action_ir,
)
from .gvs_support import SupportClass, SupportSampleIdentity, SupportStrata
from .sim_program import SimProgramError, module_runtime_sha256, runtime_callable_identity

GVS_SHORTCUT_ROW_SCHEMA_VERSION = "barun-gvs-shortcut-row-v1"
GVS_SHORTCUT_D_SUPPORT_SCHEMA_VERSION = "barun-gvs-shortcut-d-support-v1"
GVS_SHORTCUT_CONTROL_STATUS_SCHEMA_VERSION = "barun-gvs-shortcut-control-status-v1"
GVS_SHORTCUT_AUDIT_SCHEMA_VERSION = "barun-gvs-shortcut-audit-v1"
GVS_SHORTCUT_ARTIFACT_SCHEMA_VERSION = "barun-gvs-shortcut-artifact-v1"
GVS_SHORTCUT_PAIR_ACCURACY_SCHEMA_VERSION = "barun-gvs-shortcut-pair-accuracy-v1"
GVS_SHORTCUT_V2_CANDIDATE_SCHEMA_VERSION = "barun-gvs-shortcut-v2-candidate-v1"
GVS_SHORTCUT_V2_ROW_SCHEMA_VERSION = "barun-gvs-shortcut-v2-row-v1"
GVS_SHORTCUT_V2_CALIBRATION_ROW_SCHEMA_VERSION = "barun-gvs-shortcut-v2-calibration-row-v1"
GVS_SHORTCUT_V2_POPULATION_SCHEMA_VERSION = "barun-gvs-shortcut-v2-population-v1"
GVS_SHORTCUT_V2_RULES_SCHEMA_VERSION = "barun-gvs-shortcut-v2-rules-v1"
GVS_SHORTCUT_V2_CONTROL_RESULT_SCHEMA_VERSION = "barun-gvs-shortcut-v2-control-result-v1"
GVS_SHORTCUT_V2_AUDIT_SCHEMA_VERSION = "barun-gvs-shortcut-v2-audit-v1"
GVS_SHORTCUT_V2_ARTIFACT_SCHEMA_VERSION = "barun-gvs-shortcut-v2-artifact-v1"

PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE = Fraction(3, 5)
SHORTCUT_CALIBRATION_POPULATION = "T-new"
SHORTCUT_EVALUATION_POPULATION = "D-support"
SHORTCUT_TIE_CREDIT = Fraction(1, 2)

MAX_SHORTCUT_ROWS = 100_000
MAX_CANDIDATES_PER_ROW = 8
MAX_SHORTCUT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SHORTCUT_JSON_DEPTH = 64
MAX_SHORTCUT_JSON_INTEGER_DIGITS = 20
MAX_IDENTIFIER_UTF8_BYTES = 512
MAX_SHORTCUT_V2_PAIRS = MAX_SHORTCUT_ROWS * MAX_CANDIDATES_PER_ROW**2

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FACTORY_TOKEN = object()
_D_SUPPORT_FACTORY_TOKEN = object()
_STATUS_FACTORY_TOKEN = object()
_AUDIT_FACTORY_TOKEN = object()

_BRIDGE_INPUT_CLASS = CandidateSupportBridgeInput
_BRIDGE_RECEIPT_CLASS = CandidateSupportBridgeReceipt
_BRIDGE_ERROR_CLASS = gvs_bridge_module.GVSBridgeError
_BRIDGE_VERIFY_CALLABLE = gvs_bridge_module.verify_candidate_support_bridge
_MODULE_RUNTIME_SHA256_CALLABLE = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY_CALLABLE = runtime_callable_identity
_SIM_PROGRAM_ERROR_CLASS = SimProgramError
_ACTION_IR_CLASS = ActionIR
_FAULT_CERTIFICATION_CLASS = FaultCertification
_POPULATION_ERROR_CLASS = GVSPopulationError
_FAULT_ERROR_CLASS = GVSFaultError
_SCHEMA_PRESENTATION_ERROR_CLASS = GVSSchemaPresentationError
_ACTION_SIMULATOR_CLASS = ActionSimulator
_WORLD_STATE_CLASS = WorldState
_SIMULATOR_TOOL_REGISTRY = SIMULATOR_TOOL_REGISTRY
_SIMULATOR_TOOL_SCHEMAS = SIMULATOR_TOOL_SCHEMAS
_PARSE_ACTION_IR_CALLABLE = parse_action_ir
_COMPILE_PROGRAM_STEP_CALLABLE = compile_program_step
_LOWER_PRESENTED_ACTION_IR_CALLABLE = lower_presented_action_ir
_LOWERED_SEMANTIC_SHA256_CALLABLE = gvs_bridge_module.lowered_semantic_action_ir_sha256
_CERTIFY_SINGLE_FAULT_CALLABLE = certify_single_fault
_VALIDATE_SCAN_EVIDENCE_CALLABLE = validate_scan_evidence
_VALIDATE_POPULATION_FIREWALL_CALLABLE = validate_population_firewall
_ASSERT_POPULATION_RUNTIME_CALLABLE = assert_gvs_population_runtime_integrity
_ASSERT_FAULT_RUNTIME_CALLABLE = assert_gvs_fault_runtime_integrity
_ASSERT_SIMULATOR_RUNTIME_CALLABLE = assert_action_simulator_runtime_integrity

_T_NEW_BINDING_BLOCKER = "authenticated_recomputed_T-new_shortcut_bridge_receipt_absent"
_D_SUPPORT_BINDING_BLOCKER = "recomputed_D-support_shortcut_evidence_absent"
_D_SUPPORT_COMPLETENESS_BLOCKER = (
    "authenticated_complete_D-support_population_denominator_binding_absent"
)
_TOOL_BINDING_BLOCKER = "bridge_bound_lowered_candidate_tool_multiset_or_family_absent"
_T_NEW_INPUT_BLOCKER = "complete_firewall_bound_T-new_calibration_inputs_absent"
_D_SUPPORT_INPUT_BLOCKER = "complete_firewall_bound_D-support_evaluation_inputs_absent"
_D_SUPPORT_CONTRAST_BLOCKER = "complete_D-support_positive_negative_contrast_pairs_absent"
_EXTERNAL_CUSTODY_BLOCKER = "external_secret_custody_and_single_use_signer_authentication_absent"

_V2_CANDIDATE_FACTORY_TOKEN = object()
_V2_ROW_FACTORY_TOKEN = object()
_V2_CALIBRATION_ROW_FACTORY_TOKEN = object()
_V2_POPULATION_FACTORY_TOKEN = object()
_V2_RULES_FACTORY_TOKEN = object()
_V2_TOOL_FREQUENCY_FACTORY_TOKEN = object()
_V2_CONTROL_FACTORY_TOKEN = object()
_V2_AUDIT_FACTORY_TOKEN = object()


class GVSShortcutError(ValueError):
    """Shortcut evidence, arithmetic, or serialized receipts failed closed."""


class ShortcutControl(str, Enum):
    """The four preregistered shortcut-only controls."""

    LENGTH_ONLY = "length_only"
    SYNTAX_ONLY = "syntax_only"
    VALIDITY_ONLY = "validity_only"
    TOOL_FREQUENCY = "tool_frequency"


class ShortcutEvaluationStatus(str, Enum):
    """Outcome states; unavailable evidence is not a scientific failure."""

    NOT_EVALUATED = "not_evaluated"
    EVALUATED = "evaluated"


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSShortcutError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise GVSShortcutError(f"{label} must be a nonempty exact string")
    if value != unicodedata.normalize("NFC", value):
        raise GVSShortcutError(f"{label} must be NFC-normalized")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSShortcutError(f"{label} must be valid UTF-8") from error
    if len(encoded) > MAX_IDENTIFIER_UTF8_BYTES:
        raise GVSShortcutError(f"{label} exceeds the UTF-8 bound")
    if value != value.strip() or any(
        character in "\r\n\x00" or unicodedata.category(character).startswith("C")
        for character in value
    ):
        raise GVSShortcutError(f"{label} contains forbidden whitespace or controls")
    return value


def _strict_nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GVSShortcutError(f"{label} must be a nonnegative exact integer")
    return value


def _strict_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GVSShortcutError(f"{label} must be a positive exact integer")
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise GVSShortcutError(f"{label} must be an exact boolean")
    return value


def _strict_json(value: object, *, label: str = "$", depth: int = 0) -> None:
    if depth > MAX_SHORTCUT_JSON_DEPTH:
        raise GVSShortcutError(f"{label} exceeds maximum JSON depth")
    if value is None or type(value) in {bool, int, str}:
        if type(value) is str:
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise GVSShortcutError(f"{label} contains invalid UTF-8") from error
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise GVSShortcutError(f"{label} contains a non-finite float")
        raise GVSShortcutError(f"{label} contains a forbidden floating-point value")
    if type(value) is list:
        for index, child in enumerate(value):
            _strict_json(child, label=f"{label}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise GVSShortcutError(f"{label} contains a non-string object key")
            _strict_json(key, label=f"{label}.<key>", depth=depth + 1)
            _strict_json(child, label=f"{label}.{key}", depth=depth + 1)
        return
    raise GVSShortcutError(f"{label} contains non-JSON type {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    _strict_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GVSShortcutError("shortcut evidence is not strict canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_object(value: object, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSShortcutError(f"{label} must be an exact JSON object")
    missing = sorted(fields.difference(value))
    extra = sorted(set(value).difference(fields))
    if missing or extra:
        raise GVSShortcutError(f"{label} fields changed; missing={missing!r}, extra={extra!r}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSShortcutError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise GVSShortcutError(f"non-finite JSON constant {value!r}")


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if not digits or len(digits) > MAX_SHORTCUT_JSON_INTEGER_DIGITS:
        raise GVSShortcutError("JSON integer exceeds the shortcut artifact bound")
    return int(value)


def _reject_float(value: str) -> NoReturn:
    raise GVSShortcutError(f"floating-point JSON value {value!r} is forbidden")


def _file_sha256(path: Path) -> str:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise GVSShortcutError(f"cannot read source path {path}") from error
    if (
        not path.is_file()
        or len(payload) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise GVSShortcutError(f"source path {path} changed while hashing")
    return hashlib.sha256(payload).hexdigest()


def _source_files_record() -> dict[str, str]:
    action_ir_path = Path(getattr(action_ir_module, "__file__", ""))
    action_simulator_path = Path(getattr(action_simulator_module, "__file__", ""))
    bridge_path = Path(getattr(gvs_bridge_module, "__file__", ""))
    faults_path = Path(getattr(gvs_faults_module, "__file__", ""))
    population_path = Path(getattr(gvs_population_module, "__file__", ""))
    presentation_path = Path(getattr(gvs_schema_presentation_module, "__file__", ""))
    support_path = Path(getattr(gvs_support_module, "__file__", ""))
    return {
        "action_ir": _file_sha256(action_ir_path),
        "action_simulator": _file_sha256(action_simulator_path),
        "gvs_bridge": _file_sha256(bridge_path),
        "gvs_faults": _file_sha256(faults_path),
        "gvs_population": _file_sha256(population_path),
        "gvs_schema_presentation": _file_sha256(presentation_path),
        "gvs_shortcuts": _file_sha256(Path(__file__)),
        "gvs_support": _file_sha256(support_path),
    }


def _feature_contract() -> dict[str, object]:
    return {
        "calibration_population": SHORTCUT_CALIBRATION_POPULATION,
        "controls": {
            ShortcutControl.LENGTH_ONLY.value: {
                "available_from_current_d_support_bridge": True,
                "feature": "exact_model_visible_content_token_count",
            },
            ShortcutControl.SYNTAX_ONLY.value: {
                "available_from_current_d_support_bridge": True,
                "feature": "recomputed_parse_validity_on_verifier_eligible_candidates",
            },
            ShortcutControl.TOOL_FREQUENCY.value: {
                "available_from_current_d_support_bridge": False,
                "feature": "lowered_candidate_tool_multiset_or_family",
                "missing_binding": _TOOL_BINDING_BLOCKER,
            },
            ShortcutControl.VALIDITY_ONLY.value: {
                "available_from_current_d_support_bridge": True,
                "feature": "complete_nontruncated_schema_valid_distinct_candidate",
            },
        },
        "evaluation_population": SHORTCUT_EVALUATION_POPULATION,
        "pair_accuracy": {
            "aggregation": "pair_mean_then_sample_mean_then_component_mean",
            "arithmetic": "exact_rational",
            "tie_credit": "1/2",
        },
        "provisional_reference": {
            "denominator": PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE.denominator,
            "is_authorization_gate": False,
            "numerator": PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE.numerator,
        },
        "visible_candidate_metadata": [
            "opaque_semantic_identity_sha256",
            "content_token_count",
            "parse_valid",
            "schema_valid_nontruncated",
        ],
        "withheld_candidate_metadata": [
            "attempted_rank",
            "beam_or_sample_origin",
            "generator_likelihood",
            "list_position",
        ],
        "v2_contract": {
            "calibration_population": "T-new",
            "evaluation_population": "D-support",
            "complete_firewall_bound_denominators_required": True,
            "external_custody_authenticated": False,
            "negative_contract": "live_recertified_single_semantic_fault",
            "positive_contract": "bound_program_gold_action",
            "tool_feature": "sorted_lowered_canonical_tool_multiset",
            "status_rule": "not_evaluated_unless_complete_T-new_and_D-support_are_supplied",
        },
    }


def _assert_import_identities(
    *,
    _expected_input_class: object = _BRIDGE_INPUT_CLASS,
    _expected_receipt_class: object = _BRIDGE_RECEIPT_CLASS,
    _expected_bridge_error: object = _BRIDGE_ERROR_CLASS,
    _expected_verify_callable: object = _BRIDGE_VERIFY_CALLABLE,
    _expected_module_runtime_sha256: object = _MODULE_RUNTIME_SHA256_CALLABLE,
    _expected_runtime_callable_identity: object = _RUNTIME_CALLABLE_IDENTITY_CALLABLE,
    _expected_sim_program_error: object = _SIM_PROGRAM_ERROR_CLASS,
    _expected_action_simulator_class: object = _ACTION_SIMULATOR_CLASS,
    _expected_world_state_class: object = _WORLD_STATE_CLASS,
    _expected_simulator_tool_registry: object = _SIMULATOR_TOOL_REGISTRY,
    _expected_simulator_tool_schemas: object = _SIMULATOR_TOOL_SCHEMAS,
    _expected_population_error: object = _POPULATION_ERROR_CLASS,
    _expected_fault_error: object = _FAULT_ERROR_CLASS,
    _expected_schema_presentation_error: object = _SCHEMA_PRESENTATION_ERROR_CLASS,
) -> None:
    if _BRIDGE_INPUT_CLASS is not _expected_input_class or (
        gvs_bridge_module.CandidateSupportBridgeInput is not _expected_input_class
    ):
        raise GVSShortcutError("bridge input class changed identity")
    if _BRIDGE_RECEIPT_CLASS is not _expected_receipt_class or (
        gvs_bridge_module.CandidateSupportBridgeReceipt is not _expected_receipt_class
    ):
        raise GVSShortcutError("bridge receipt class changed identity")
    if _BRIDGE_VERIFY_CALLABLE is not _expected_verify_callable or (
        gvs_bridge_module.verify_candidate_support_bridge is not _expected_verify_callable
    ):
        raise GVSShortcutError("bridge verification callable changed identity")
    if _BRIDGE_ERROR_CLASS is not _expected_bridge_error or (
        gvs_bridge_module.GVSBridgeError is not _expected_bridge_error
    ):
        raise GVSShortcutError("bridge error class changed identity")
    if module_runtime_sha256 is not _expected_module_runtime_sha256 or (
        sim_program_module.module_runtime_sha256 is not _expected_module_runtime_sha256
    ):
        raise GVSShortcutError("runtime-hash callable changed identity")
    if runtime_callable_identity is not _expected_runtime_callable_identity or (
        sim_program_module.runtime_callable_identity is not _expected_runtime_callable_identity
    ):
        raise GVSShortcutError("runtime-callable identity helper changed identity")
    if SimProgramError is not _expected_sim_program_error or (
        sim_program_module.SimProgramError is not _expected_sim_program_error
    ):
        raise GVSShortcutError("runtime identity error class changed identity")
    if (
        _ACTION_SIMULATOR_CLASS is not _expected_action_simulator_class
        or ActionSimulator is not _expected_action_simulator_class
        or action_simulator_module.ActionSimulator is not _expected_action_simulator_class
    ):
        raise GVSShortcutError("action simulator class changed identity")
    if (
        _WORLD_STATE_CLASS is not _expected_world_state_class
        or WorldState is not _expected_world_state_class
        or action_simulator_module.WorldState is not _expected_world_state_class
    ):
        raise GVSShortcutError("world-state class changed identity")
    if (
        SIMULATOR_TOOL_REGISTRY is not _expected_simulator_tool_registry
        or _SIMULATOR_TOOL_REGISTRY is not _expected_simulator_tool_registry
        or action_simulator_module.SIMULATOR_TOOL_REGISTRY is not _expected_simulator_tool_registry
    ):
        raise GVSShortcutError("simulator tool registry changed identity")
    if (
        SIMULATOR_TOOL_SCHEMAS is not _expected_simulator_tool_schemas
        or _SIMULATOR_TOOL_SCHEMAS is not _expected_simulator_tool_schemas
        or action_simulator_module.SIMULATOR_TOOL_SCHEMAS is not _expected_simulator_tool_schemas
    ):
        raise GVSShortcutError("simulator tool schemas changed identity")
    if (
        _POPULATION_ERROR_CLASS is not _expected_population_error
        or GVSPopulationError is not _expected_population_error
        or gvs_population_module.GVSPopulationError is not _expected_population_error
    ):
        raise GVSShortcutError("population error class changed identity")
    if (
        _FAULT_ERROR_CLASS is not _expected_fault_error
        or GVSFaultError is not _expected_fault_error
        or gvs_faults_module.GVSFaultError is not _expected_fault_error
    ):
        raise GVSShortcutError("fault error class changed identity")
    if (
        _SCHEMA_PRESENTATION_ERROR_CLASS is not _expected_schema_presentation_error
        or GVSSchemaPresentationError is not _expected_schema_presentation_error
        or gvs_schema_presentation_module.GVSSchemaPresentationError
        is not _expected_schema_presentation_error
    ):
        raise GVSShortcutError("schema-presentation error class changed identity")
    identity_checks = (
        ("ActionIR", ActionIR, action_ir_module.ActionIR, _ACTION_IR_CLASS),
        (
            "FaultCertification",
            FaultCertification,
            gvs_faults_module.FaultCertification,
            _FAULT_CERTIFICATION_CLASS,
        ),
        (
            "parse_action_ir",
            parse_action_ir,
            action_ir_module.parse_action_ir,
            _PARSE_ACTION_IR_CALLABLE,
        ),
        (
            "compile_program_step",
            compile_program_step,
            action_simulator_module.compile_program_step,
            _COMPILE_PROGRAM_STEP_CALLABLE,
        ),
        (
            "lower_presented_action_ir",
            lower_presented_action_ir,
            gvs_schema_presentation_module.lower_presented_action_ir,
            _LOWER_PRESENTED_ACTION_IR_CALLABLE,
        ),
        (
            "lowered_semantic_action_ir_sha256",
            gvs_bridge_module.lowered_semantic_action_ir_sha256,
            gvs_bridge_module.lowered_semantic_action_ir_sha256,
            _LOWERED_SEMANTIC_SHA256_CALLABLE,
        ),
        (
            "certify_single_fault",
            certify_single_fault,
            gvs_faults_module.certify_single_fault,
            _CERTIFY_SINGLE_FAULT_CALLABLE,
        ),
        (
            "validate_scan_evidence",
            validate_scan_evidence,
            gvs_population_module.validate_scan_evidence,
            _VALIDATE_SCAN_EVIDENCE_CALLABLE,
        ),
        (
            "validate_population_firewall",
            validate_population_firewall,
            gvs_population_module.validate_population_firewall,
            _VALIDATE_POPULATION_FIREWALL_CALLABLE,
        ),
        (
            "assert_gvs_population_runtime_integrity",
            assert_gvs_population_runtime_integrity,
            gvs_population_module.assert_gvs_population_runtime_integrity,
            _ASSERT_POPULATION_RUNTIME_CALLABLE,
        ),
        (
            "assert_gvs_fault_runtime_integrity",
            assert_gvs_fault_runtime_integrity,
            gvs_faults_module.assert_gvs_fault_runtime_integrity,
            _ASSERT_FAULT_RUNTIME_CALLABLE,
        ),
        (
            "assert_action_simulator_runtime_integrity",
            assert_action_simulator_runtime_integrity,
            action_simulator_module.assert_action_simulator_runtime_integrity,
            _ASSERT_SIMULATOR_RUNTIME_CALLABLE,
        ),
        (
            "WorldState",
            WorldState,
            action_simulator_module.WorldState,
            _WORLD_STATE_CLASS,
        ),
    )
    changed = [
        name
        for name, imported, module_value, captured in identity_checks
        if imported is not captured or module_value is not captured
    ]
    if changed:
        raise GVSShortcutError(
            "v2 shortcut dependency changed identity: " + ", ".join(sorted(changed))
        )


def shortcut_runtime_sha256() -> str:
    """Bind source bytes, live Python code, dependencies, and the frozen contract."""

    _assert_import_identities()
    contract = {
        "artifact_schema_version": GVS_SHORTCUT_ARTIFACT_SCHEMA_VERSION,
        "audit_schema_version": GVS_SHORTCUT_AUDIT_SCHEMA_VERSION,
        "bounds": {
            "artifact_bytes": MAX_SHORTCUT_ARTIFACT_BYTES,
            "candidates_per_row": MAX_CANDIDATES_PER_ROW,
            "json_depth": MAX_SHORTCUT_JSON_DEPTH,
            "json_integer_digits": MAX_SHORTCUT_JSON_INTEGER_DIGITS,
            "rows": MAX_SHORTCUT_ROWS,
        },
        "feature_contract": _feature_contract(),
        "live_bridge_verify_callable": dict(
            _RUNTIME_CALLABLE_IDENTITY_CALLABLE(_BRIDGE_VERIFY_CALLABLE)
        ),
        "source_files": _source_files_record(),
    }
    try:
        return _MODULE_RUNTIME_SHA256_CALLABLE(
            globals(),
            module_name=__name__,
            source_path=__file__,
            contract=contract,
        )
    except _SIM_PROGRAM_ERROR_CLASS as error:
        raise GVSShortcutError("shortcut runtime identity could not be computed") from error


def assert_shortcut_runtime_integrity() -> None:
    """Reject source, dependency, import-alias, or live-code mutation."""

    _assert_import_identities()
    try:
        _ASSERT_POPULATION_RUNTIME_CALLABLE()
        _ASSERT_FAULT_RUNTIME_CALLABLE()
        _ASSERT_SIMULATOR_RUNTIME_CALLABLE()
    except (_POPULATION_ERROR_CLASS, _FAULT_ERROR_CLASS, ValueError) as error:
        raise GVSShortcutError("v2 shortcut dependency runtime changed") from error
    if _canonical_sha256(_source_files_record()) != _INITIAL_SOURCE_FILES_SHA256:
        raise GVSShortcutError("shortcut or dependency source changed after import")
    if shortcut_runtime_sha256() != _INITIAL_RUNTIME_SHA256:
        raise GVSShortcutError("shortcut loaded runtime changed after import")


@dataclass(frozen=True, slots=True)
class ExactRational:
    """One bounded exact rational used for scores and accuracies."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _strict_nonnegative_integer(self.numerator, label="numerator")
        _strict_positive_integer(self.denominator, label="denominator")
        if self.numerator > self.denominator:
            raise GVSShortcutError("an accuracy-like rational must lie in [0, 1]")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (self.numerator, self.denominator):
            raise GVSShortcutError("exact rational must be stored in lowest terms")

    @classmethod
    def from_fraction(cls, value: Fraction) -> ExactRational:
        if type(value) is not Fraction or not 0 <= value <= 1:
            raise GVSShortcutError("value must be an exact Fraction in [0, 1]")
        return cls(value.numerator, value.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_record(self) -> dict[str, int | str]:
        return {
            "denominator": self.denominator,
            "exact": f"{self.numerator}/{self.denominator}",
            "numerator": self.numerator,
        }


@dataclass(frozen=True, slots=True)
class ShortcutPairScore:
    """One positive/negative comparison with no candidate-order metadata."""

    component_id: str
    sample_id: str
    positive_candidate_sha256: str
    negative_candidate_sha256: str
    positive_score: ExactRational
    negative_score: ExactRational

    def __post_init__(self) -> None:
        _strict_identifier(self.component_id, label="component_id")
        _strict_identifier(self.sample_id, label="sample_id")
        _strict_sha256(self.positive_candidate_sha256, label="positive_candidate_sha256")
        _strict_sha256(self.negative_candidate_sha256, label="negative_candidate_sha256")
        if self.positive_candidate_sha256 == self.negative_candidate_sha256:
            raise GVSShortcutError("a shortcut pair requires distinct semantic candidates")
        if type(self.positive_score) is not ExactRational:
            raise GVSShortcutError("positive_score must be ExactRational")
        if type(self.negative_score) is not ExactRational:
            raise GVSShortcutError("negative_score must be ExactRational")

    @property
    def pair_sha256(self) -> str:
        return _canonical_sha256(
            {
                "component_id": self.component_id,
                "negative_candidate_sha256": self.negative_candidate_sha256,
                "positive_candidate_sha256": self.positive_candidate_sha256,
                "sample_id": self.sample_id,
            }
        )


@dataclass(frozen=True, slots=True)
class ComponentBalancedPairAccuracy:
    """Exact pair accuracy after equal weighting of samples and components."""

    accuracy: ExactRational
    pair_count: int
    tie_count: int
    sample_count: int
    component_count: int
    schema_version: str = GVS_SHORTCUT_PAIR_ACCURACY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SHORTCUT_PAIR_ACCURACY_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported pair-accuracy schema")
        if type(self.accuracy) is not ExactRational:
            raise GVSShortcutError("accuracy must be ExactRational")
        for name in ("pair_count", "sample_count", "component_count"):
            _strict_positive_integer(getattr(self, name), label=name)
        _strict_nonnegative_integer(self.tie_count, label="tie_count")
        if self.tie_count > self.pair_count:
            raise GVSShortcutError("tie_count cannot exceed pair_count")
        if self.component_count > self.sample_count or self.sample_count > self.pair_count:
            raise GVSShortcutError("component/sample/pair counts are inconsistent")

    def to_record(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy.to_record(),
            "aggregation": "pair_mean_then_sample_mean_then_component_mean",
            "component_count": self.component_count,
            "pair_count": self.pair_count,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "tie_count": self.tie_count,
            "tie_credit": ExactRational.from_fraction(SHORTCUT_TIE_CREDIT).to_record(),
        }


def compute_component_balanced_pair_accuracy(
    pairs: Sequence[ShortcutPairScore],
) -> ComponentBalancedPairAccuracy:
    """Compute exact accuracy without letting large components dominate.

    Every positive/negative pair is worth one, zero, or one half.  Pairs are averaged
    within a sample, sample means are averaged within a connected component, and
    component means are averaged globally.  Canonical sorting makes input order
    irrelevant; ties are never resolved using hashes or list positions.

    This is an arithmetic primitive, not a shortcut evaluation receipt.  It does not
    authenticate a population or authorize any access.
    """

    assert_shortcut_runtime_integrity()
    if type(pairs) not in {list, tuple} or not pairs:
        raise GVSShortcutError("pairs must be a nonempty exact list or tuple")
    if len(pairs) > MAX_SHORTCUT_ROWS * MAX_CANDIDATES_PER_ROW**2:
        raise GVSShortcutError("pair evidence exceeds its cardinality bound")
    first = tuple(pairs)
    second = tuple(pairs)
    if tuple(id(value) for value in first) != tuple(id(value) for value in second):
        raise GVSShortcutError("pair evidence changed during its stable snapshot")
    if any(type(pair) is not ShortcutPairScore for pair in first):
        raise GVSShortcutError("pair evidence contains the wrong type")
    ordered = tuple(sorted(first, key=lambda pair: pair.pair_sha256))
    pair_ids = tuple(pair.pair_sha256 for pair in ordered)
    if len(pair_ids) != len(set(pair_ids)):
        raise GVSShortcutError("pair evidence contains duplicate semantic comparisons")

    by_component: dict[str, dict[str, list[Fraction]]] = defaultdict(lambda: defaultdict(list))
    tie_count = 0
    for pair in ordered:
        positive = pair.positive_score.fraction
        negative = pair.negative_score.fraction
        if positive > negative:
            credit = Fraction(1, 1)
        elif positive < negative:
            credit = Fraction(0, 1)
        else:
            credit = SHORTCUT_TIE_CREDIT
            tie_count += 1
        by_component[pair.component_id][pair.sample_id].append(credit)

    component_means: list[Fraction] = []
    sample_count = 0
    for component_id in sorted(by_component):
        sample_means: list[Fraction] = []
        for sample_id in sorted(by_component[component_id]):
            credits = by_component[component_id][sample_id]
            sample_means.append(sum(credits, Fraction()) / len(credits))
            sample_count += 1
        component_means.append(sum(sample_means, Fraction()) / len(sample_means))
    accuracy = sum(component_means, Fraction()) / len(component_means)
    return ComponentBalancedPairAccuracy(
        accuracy=ExactRational.from_fraction(accuracy),
        pair_count=len(ordered),
        tie_count=tie_count,
        sample_count=sample_count,
        component_count=len(component_means),
    )


@dataclass(frozen=True, slots=True)
class ShortcutCandidateSignal:
    """One verifier-eligible semantic candidate derived from bridge evidence."""

    semantic_identity_sha256: str
    candidate_evidence_sha256: str
    content_token_count: int
    syntax_parse_valid: bool
    validity_schema_valid_nontruncated: bool
    exact_semantic_match: bool
    tool_identity: None = None
    tool_feature_binding_sha256: None = None

    def __post_init__(self) -> None:
        _strict_sha256(self.semantic_identity_sha256, label="semantic_identity_sha256")
        _strict_sha256(self.candidate_evidence_sha256, label="candidate_evidence_sha256")
        _strict_nonnegative_integer(self.content_token_count, label="content_token_count")
        if self.content_token_count > 1_000_000:
            raise GVSShortcutError("content_token_count exceeds the bridge token bound")
        if _strict_bool(self.syntax_parse_valid, label="syntax_parse_valid") is not True:
            raise GVSShortcutError("verifier-eligible candidates must be parse-valid")
        if (
            _strict_bool(
                self.validity_schema_valid_nontruncated,
                label="validity_schema_valid_nontruncated",
            )
            is not True
        ):
            raise GVSShortcutError("verifier-eligible candidates must pass validity filtering")
        _strict_bool(self.exact_semantic_match, label="exact_semantic_match")
        if self.tool_identity is not None or self.tool_feature_binding_sha256 is not None:
            raise GVSShortcutError(
                "current bridge cannot supply a candidate tool-frequency feature"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_evidence_sha256": self.candidate_evidence_sha256,
            "content_token_count": self.content_token_count,
            "exact_semantic_match": self.exact_semantic_match,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "syntax_parse_valid": self.syntax_parse_valid,
            "tool_feature_binding_sha256": self.tool_feature_binding_sha256,
            "tool_identity": self.tool_identity,
            "validity_schema_valid_nontruncated": (self.validity_schema_valid_nontruncated),
        }


@dataclass(frozen=True, slots=True)
class ShortcutRowEvidence:
    """One D-support row, with pre-outcome component and bridge bindings."""

    sample_id: str
    firewall_sha256: str
    scan_evidence_sha256: str
    population_record_sha256: str
    component_id: str
    source_commitment_sha256: str
    bridge_receipt_sha256: str
    bridge_runtime_identity_sha256: str
    rank_hidden_payload_sha256: str
    rank_hidden_batch_sha256: str
    schema_presentation_receipt_sha256: str
    gold_semantic_action_ir_sha256: str
    candidates: tuple[ShortcutCandidateSignal, ...]
    schema_version: str = GVS_SHORTCUT_ROW_SCHEMA_VERSION
    population_role: str = SHORTCUT_EVALUATION_POPULATION
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    certifies_external_custody: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ROW_FACTORY_TOKEN:
            raise TypeError("ShortcutRowEvidence must be built from a live support bridge")
        if self.schema_version != GVS_SHORTCUT_ROW_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported shortcut row schema")
        if self.population_role != SHORTCUT_EVALUATION_POPULATION:
            raise GVSShortcutError("current bridge rows must be D-support only")
        _strict_identifier(self.sample_id, label="sample_id")
        _strict_identifier(self.component_id, label="component_id")
        for name in (
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_sha256",
            "source_commitment_sha256",
            "bridge_receipt_sha256",
            "bridge_runtime_identity_sha256",
            "rank_hidden_payload_sha256",
            "rank_hidden_batch_sha256",
            "schema_presentation_receipt_sha256",
            "gold_semantic_action_ir_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if type(self.candidates) is not tuple or len(self.candidates) > MAX_CANDIDATES_PER_ROW:
            raise GVSShortcutError("shortcut candidates must be a bounded exact tuple")
        if any(type(candidate) is not ShortcutCandidateSignal for candidate in self.candidates):
            raise GVSShortcutError("shortcut row contains an invalid candidate signal")
        identities = tuple(candidate.semantic_identity_sha256 for candidate in self.candidates)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise GVSShortcutError("shortcut semantic candidates must be unique and sorted")
        for candidate in self.candidates:
            if candidate.exact_semantic_match is not (
                candidate.semantic_identity_sha256 == self.gold_semantic_action_ir_sha256
            ):
                raise GVSShortcutError("shortcut exactness differs from semantic hash equality")
        for name in (
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "certifies_external_custody",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("shortcut row evidence must remain nonauthorizing")

    @property
    def has_contrast_pair(self) -> bool:
        return any(candidate.exact_semantic_match for candidate in self.candidates) and any(
            not candidate.exact_semantic_match for candidate in self.candidates
        )

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "bridge_receipt_sha256": self.bridge_receipt_sha256,
            "bridge_runtime_identity_sha256": self.bridge_runtime_identity_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "certifies_external_custody": self.certifies_external_custody,
            "component_id": self.component_id,
            "firewall_sha256": self.firewall_sha256,
            "gold_semantic_action_ir_sha256": self.gold_semantic_action_ir_sha256,
            "population_record_sha256": self.population_record_sha256,
            "population_role": self.population_role,
            "rank_hidden_batch_sha256": self.rank_hidden_batch_sha256,
            "rank_hidden_payload_sha256": self.rank_hidden_payload_sha256,
            "sample_id": self.sample_id,
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "schema_presentation_receipt_sha256": (self.schema_presentation_receipt_sha256),
            "schema_version": self.schema_version,
            "source_commitment_sha256": self.source_commitment_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())


def derive_d_support_shortcut_row(
    bridge_input: CandidateSupportBridgeInput,
) -> ShortcutRowEvidence:
    """Recompute one support bridge and derive only rank-hidden shortcut signals."""

    assert_shortcut_runtime_integrity()
    if type(bridge_input) is not _BRIDGE_INPUT_CLASS:
        raise GVSShortcutError("bridge_input must be exact CandidateSupportBridgeInput")
    try:
        verified = _BRIDGE_VERIFY_CALLABLE(
            bridge_input.receipt,
            sample=bridge_input.sample,
            trace=bridge_input.trace,
            analysis=bridge_input.analysis,
            program=bridge_input.program,
            step_id=bridge_input.step_id,
            now=bridge_input.now,
            user_text=bridge_input.user_text,
            prompt_tools=bridge_input.prompt_tools,
            schema_presentation_receipt=bridge_input.schema_presentation_receipt,
        )
    except _BRIDGE_ERROR_CLASS as error:
        raise GVSShortcutError("D-support bridge failed complete live recomputation") from error
    if type(verified) is not _BRIDGE_RECEIPT_CLASS:
        raise GVSShortcutError("bridge verification returned the wrong receipt type")
    if verified.source_custody_authenticated is not False:
        raise GVSShortcutError("current bridge cannot certify source custody")
    payload = verified.rank_hidden_verifier_payload
    signals: list[ShortcutCandidateSignal] = []
    for candidate in payload.candidates:
        identity = candidate.candidate_identity_sha256
        if identity is None:
            continue
        if candidate.presented_action_ir_sha256 is None:
            raise GVSShortcutError("valid semantic candidate lacks a presented identity")
        evidence_record = {
            "candidate_semantic_identity_sha256": identity,
            "content_token_ids_sha256": _canonical_sha256(list(candidate.content_token_ids)),
            "generated_token_ids_sha256": _canonical_sha256(list(candidate.generated_token_ids)),
            "presented_action_ir_sha256": candidate.presented_action_ir_sha256,
            "rank_hidden_payload_sha256": verified.rank_hidden_verifier_payload_sha256,
            "schema_presentation_receipt_sha256": (verified.schema_presentation_receipt_sha256),
        }
        signals.append(
            ShortcutCandidateSignal(
                semantic_identity_sha256=identity,
                candidate_evidence_sha256=_canonical_sha256(evidence_record),
                content_token_count=len(candidate.content_token_ids),
                syntax_parse_valid=True,
                validity_schema_valid_nontruncated=True,
                exact_semantic_match=identity == verified.gold_semantic_action_ir_sha256,
            )
        )
    signals.sort(key=lambda candidate: candidate.semantic_identity_sha256)
    support = verified.support_record
    row = ShortcutRowEvidence(
        sample_id=verified.sample_id,
        firewall_sha256=support.firewall_sha256,
        scan_evidence_sha256=support.scan_evidence_sha256,
        population_record_sha256=support.population_record_sha256,
        component_id=support.component_id,
        source_commitment_sha256=verified.source_commitment_sha256,
        bridge_receipt_sha256=verified.sha256,
        bridge_runtime_identity_sha256=verified.runtime_identity_sha256,
        rank_hidden_payload_sha256=verified.rank_hidden_verifier_payload_sha256,
        rank_hidden_batch_sha256=verified.rank_hidden_verifier_batch_sha256,
        schema_presentation_receipt_sha256=verified.schema_presentation_receipt_sha256,
        gold_semantic_action_ir_sha256=verified.gold_semantic_action_ir_sha256,
        candidates=tuple(signals),
        _factory_token=_ROW_FACTORY_TOKEN,
    )
    assert_shortcut_runtime_integrity()
    return row


@dataclass(frozen=True, slots=True)
class DSupportShortcutEvidence:
    """Caller-supplied D-support rows and roots, without denominator authentication."""

    rows: tuple[ShortcutRowEvidence, ...]
    firewall_sha256: str
    scan_evidence_sha256: str
    population_record_root_sha256: str
    component_assignment_sha256: str
    bridge_receipt_root_sha256: str
    rank_hidden_payload_root_sha256: str
    semantic_evidence_root_sha256: str
    row_count: int
    component_count: int
    rows_with_contrast_pair: int
    rows_without_contrast_pair: int
    schema_version: str = GVS_SHORTCUT_D_SUPPORT_SCHEMA_VERSION
    population_role: str = SHORTCUT_EVALUATION_POPULATION
    source_custody_authenticated: bool = False
    population_completeness_authenticated: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    certifies_external_custody: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _D_SUPPORT_FACTORY_TOKEN:
            raise TypeError("DSupportShortcutEvidence must be built from live bridge inputs")
        if self.schema_version != GVS_SHORTCUT_D_SUPPORT_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported D-support shortcut evidence schema")
        if self.population_role != SHORTCUT_EVALUATION_POPULATION:
            raise GVSShortcutError("shortcut evidence must remain D-support only")
        if type(self.rows) is not tuple or not self.rows or len(self.rows) > MAX_SHORTCUT_ROWS:
            raise GVSShortcutError("D-support rows must be a nonempty bounded tuple")
        if any(type(row) is not ShortcutRowEvidence for row in self.rows):
            raise GVSShortcutError("D-support evidence contains an invalid row")
        sample_ids = tuple(row.sample_id for row in self.rows)
        if sample_ids != tuple(sorted(sample_ids)) or len(sample_ids) != len(set(sample_ids)):
            raise GVSShortcutError("D-support rows must be unique and sorted by sample ID")
        for name in (
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_root_sha256",
            "component_assignment_sha256",
            "bridge_receipt_root_sha256",
            "rank_hidden_payload_root_sha256",
            "semantic_evidence_root_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if any(row.firewall_sha256 != self.firewall_sha256 for row in self.rows):
            raise GVSShortcutError("D-support rows mix population firewalls")
        if any(row.scan_evidence_sha256 != self.scan_evidence_sha256 for row in self.rows):
            raise GVSShortcutError("D-support rows mix scan evidence")
        population_records = sorted(row.population_record_sha256 for row in self.rows)
        bridge_receipts = sorted(row.bridge_receipt_sha256 for row in self.rows)
        payloads = sorted(row.rank_hidden_payload_sha256 for row in self.rows)
        semantic_evidence = [
            {
                "candidates": [
                    {
                        "candidate_evidence_sha256": candidate.candidate_evidence_sha256,
                        "semantic_identity_sha256": candidate.semantic_identity_sha256,
                    }
                    for candidate in row.candidates
                ],
                "gold_semantic_action_ir_sha256": row.gold_semantic_action_ir_sha256,
                "sample_id": row.sample_id,
            }
            for row in self.rows
        ]
        assignments = sorted(
            (
                {
                    "component_id": row.component_id,
                    "population_record_sha256": row.population_record_sha256,
                    "sample_id": row.sample_id,
                }
                for row in self.rows
            ),
            key=lambda value: (value["component_id"], value["sample_id"]),
        )
        expected_roots = {
            "population_record_root_sha256": _canonical_sha256(population_records),
            "component_assignment_sha256": _canonical_sha256(assignments),
            "bridge_receipt_root_sha256": _canonical_sha256(bridge_receipts),
            "rank_hidden_payload_root_sha256": _canonical_sha256(payloads),
            "semantic_evidence_root_sha256": _canonical_sha256(semantic_evidence),
        }
        for name, expected in expected_roots.items():
            if getattr(self, name) != expected:
                raise GVSShortcutError(f"{name} failed exact recomputation")
        expected_counts = {
            "row_count": len(self.rows),
            "component_count": len({row.component_id for row in self.rows}),
            "rows_with_contrast_pair": sum(row.has_contrast_pair for row in self.rows),
        }
        expected_counts["rows_without_contrast_pair"] = (
            expected_counts["row_count"] - expected_counts["rows_with_contrast_pair"]
        )
        for name, expected in expected_counts.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise GVSShortcutError(f"{name} failed exact recomputation")
        for name in (
            "source_custody_authenticated",
            "population_completeness_authenticated",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "certifies_external_custody",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("D-support shortcut evidence must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "bridge_receipt_root_sha256": self.bridge_receipt_root_sha256,
            "certifies_external_custody": self.certifies_external_custody,
            "component_assignment_sha256": self.component_assignment_sha256,
            "component_count": self.component_count,
            "firewall_sha256": self.firewall_sha256,
            "population_completeness_authenticated": (self.population_completeness_authenticated),
            "population_record_root_sha256": self.population_record_root_sha256,
            "population_role": self.population_role,
            "rank_hidden_payload_root_sha256": self.rank_hidden_payload_root_sha256,
            "row_count": self.row_count,
            "rows": [row.to_record() for row in self.rows],
            "rows_with_contrast_pair": self.rows_with_contrast_pair,
            "rows_without_contrast_pair": self.rows_without_contrast_pair,
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "schema_version": self.schema_version,
            "semantic_evidence_root_sha256": self.semantic_evidence_root_sha256,
            "source_custody_authenticated": self.source_custody_authenticated,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())


def build_d_support_shortcut_evidence(
    bridge_inputs: Sequence[CandidateSupportBridgeInput],
) -> DSupportShortcutEvidence:
    """Recompute caller-supplied rows without claiming a complete D-support denominator."""

    assert_shortcut_runtime_integrity()
    if type(bridge_inputs) not in {list, tuple} or not bridge_inputs:
        raise GVSShortcutError("bridge_inputs must be a nonempty exact list or tuple")
    if len(bridge_inputs) > MAX_SHORTCUT_ROWS:
        raise GVSShortcutError("bridge_inputs exceed the D-support row bound")
    first = tuple(bridge_inputs)
    second = tuple(bridge_inputs)
    if tuple(id(value) for value in first) != tuple(id(value) for value in second):
        raise GVSShortcutError("bridge_inputs changed during their stable snapshot")
    rows = tuple(sorted(map(derive_d_support_shortcut_row, first), key=lambda row: row.sample_id))
    population_records = sorted(row.population_record_sha256 for row in rows)
    assignments = sorted(
        (
            {
                "component_id": row.component_id,
                "population_record_sha256": row.population_record_sha256,
                "sample_id": row.sample_id,
            }
            for row in rows
        ),
        key=lambda value: (value["component_id"], value["sample_id"]),
    )
    semantic_evidence = [
        {
            "candidates": [
                {
                    "candidate_evidence_sha256": candidate.candidate_evidence_sha256,
                    "semantic_identity_sha256": candidate.semantic_identity_sha256,
                }
                for candidate in row.candidates
            ],
            "gold_semantic_action_ir_sha256": row.gold_semantic_action_ir_sha256,
            "sample_id": row.sample_id,
        }
        for row in rows
    ]
    return DSupportShortcutEvidence(
        rows=rows,
        firewall_sha256=rows[0].firewall_sha256,
        scan_evidence_sha256=rows[0].scan_evidence_sha256,
        population_record_root_sha256=_canonical_sha256(population_records),
        component_assignment_sha256=_canonical_sha256(assignments),
        bridge_receipt_root_sha256=_canonical_sha256(
            sorted(row.bridge_receipt_sha256 for row in rows)
        ),
        rank_hidden_payload_root_sha256=_canonical_sha256(
            sorted(row.rank_hidden_payload_sha256 for row in rows)
        ),
        semantic_evidence_root_sha256=_canonical_sha256(semantic_evidence),
        row_count=len(rows),
        component_count=len({row.component_id for row in rows}),
        rows_with_contrast_pair=sum(row.has_contrast_pair for row in rows),
        rows_without_contrast_pair=sum(not row.has_contrast_pair for row in rows),
        _factory_token=_D_SUPPORT_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ShortcutControlStatus:
    """Availability and non-result for one named shortcut control."""

    control: ShortcutControl
    feature_derivation_implemented: bool
    d_support_feature_evidence_present: bool
    calibration_status: ShortcutEvaluationStatus
    evaluation_status: ShortcutEvaluationStatus
    missing_bindings: tuple[str, ...]
    pair_accuracy: None
    below_provisional_reference: None
    passes_gate: None
    schema_version: str = GVS_SHORTCUT_CONTROL_STATUS_SCHEMA_VERSION
    calibration_population: str = SHORTCUT_CALIBRATION_POPULATION
    evaluation_population: str = SHORTCUT_EVALUATION_POPULATION
    provisional_reference_is_gate: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    certifies_external_custody: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _STATUS_FACTORY_TOKEN:
            raise TypeError("ShortcutControlStatus must be built by the shortcut audit")
        if self.schema_version != GVS_SHORTCUT_CONTROL_STATUS_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported shortcut control-status schema")
        if type(self.control) is not ShortcutControl:
            raise GVSShortcutError("control must be an exact ShortcutControl")
        for name in ("feature_derivation_implemented", "d_support_feature_evidence_present"):
            _strict_bool(getattr(self, name), label=name)
        if self.control is ShortcutControl.TOOL_FREQUENCY and (
            self.feature_derivation_implemented or self.d_support_feature_evidence_present
        ):
            raise GVSShortcutError("tool-frequency evidence is unavailable from this bridge")
        if type(self.calibration_status) is not ShortcutEvaluationStatus or (
            self.calibration_status is not ShortcutEvaluationStatus.NOT_EVALUATED
        ):
            raise GVSShortcutError("shortcut calibration must remain not_evaluated")
        if type(self.evaluation_status) is not ShortcutEvaluationStatus or (
            self.evaluation_status is not ShortcutEvaluationStatus.NOT_EVALUATED
        ):
            raise GVSShortcutError("shortcut evaluation must remain not_evaluated")
        if type(self.missing_bindings) is not tuple or not self.missing_bindings:
            raise GVSShortcutError("not_evaluated controls require explicit missing bindings")
        for index, value in enumerate(self.missing_bindings):
            _strict_identifier(value, label=f"missing_bindings[{index}]")
        if self.missing_bindings != tuple(sorted(set(self.missing_bindings))):
            raise GVSShortcutError("missing bindings must be sorted and unique")
        if _T_NEW_BINDING_BLOCKER not in self.missing_bindings:
            raise GVSShortcutError("missing T-new calibration must be explicit")
        if self.control is ShortcutControl.TOOL_FREQUENCY and (
            _TOOL_BINDING_BLOCKER not in self.missing_bindings
        ):
            raise GVSShortcutError("tool-frequency status must name its missing bridge binding")
        if self.pair_accuracy is not None:
            raise GVSShortcutError("not_evaluated controls cannot carry pair accuracy")
        if self.below_provisional_reference is not None or self.passes_gate is not None:
            raise GVSShortcutError("not_evaluated controls cannot carry pass/fail claims")
        if self.calibration_population != SHORTCUT_CALIBRATION_POPULATION:
            raise GVSShortcutError("shortcut controls may calibrate only on T-new")
        if self.evaluation_population != SHORTCUT_EVALUATION_POPULATION:
            raise GVSShortcutError("shortcut controls may evaluate only on D-support")
        for name in (
            "provisional_reference_is_gate",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "certifies_external_custody",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("shortcut control status must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "below_provisional_reference": self.below_provisional_reference,
            "calibration_population": self.calibration_population,
            "calibration_status": self.calibration_status.value,
            "certifies_external_custody": self.certifies_external_custody,
            "control": self.control.value,
            "d_support_feature_evidence_present": self.d_support_feature_evidence_present,
            "evaluation_population": self.evaluation_population,
            "evaluation_status": self.evaluation_status.value,
            "feature_derivation_implemented": self.feature_derivation_implemented,
            "missing_bindings": list(self.missing_bindings),
            "pair_accuracy": self.pair_accuracy,
            "passes_gate": self.passes_gate,
            "provisional_reference": ExactRational.from_fraction(
                PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE
            ).to_record(),
            "provisional_reference_is_gate": self.provisional_reference_is_gate,
            "schema_version": self.schema_version,
        }


def _control_statuses(
    d_support_evidence: DSupportShortcutEvidence | None,
) -> tuple[ShortcutControlStatus, ...]:
    evidence_present = d_support_evidence is not None
    statuses: list[ShortcutControlStatus] = []
    for control in ShortcutControl:
        implemented = control is not ShortcutControl.TOOL_FREQUENCY
        missing = {_D_SUPPORT_COMPLETENESS_BLOCKER, _T_NEW_BINDING_BLOCKER}
        if not evidence_present:
            missing.add(_D_SUPPORT_BINDING_BLOCKER)
        if control is ShortcutControl.TOOL_FREQUENCY:
            missing.add(_TOOL_BINDING_BLOCKER)
        statuses.append(
            ShortcutControlStatus(
                control=control,
                feature_derivation_implemented=implemented,
                d_support_feature_evidence_present=evidence_present and implemented,
                calibration_status=ShortcutEvaluationStatus.NOT_EVALUATED,
                evaluation_status=ShortcutEvaluationStatus.NOT_EVALUATED,
                missing_bindings=tuple(sorted(missing)),
                pair_accuracy=None,
                below_provisional_reference=None,
                passes_gate=None,
                _factory_token=_STATUS_FACTORY_TOKEN,
            )
        )
    return tuple(statuses)


@dataclass(frozen=True, slots=True)
class ShortcutAuditReceipt:
    """Deterministic evidence inventory that cannot be mistaken for a result."""

    d_support_evidence: DSupportShortcutEvidence | None
    d_support_evidence_sha256: str | None
    controls: tuple[ShortcutControlStatus, ...]
    source_files_sha256: str
    runtime_identity_sha256: str
    schema_version: str = GVS_SHORTCUT_AUDIT_SCHEMA_VERSION
    authenticated_t_new_calibration_present: bool = False
    d_support_population_completeness_authenticated: bool = False
    scientific_result_available: bool = False
    provisional_reference_is_gate: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    certifies_external_custody: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _AUDIT_FACTORY_TOKEN:
            raise TypeError("ShortcutAuditReceipt must be built by build_shortcut_audit")
        if self.schema_version != GVS_SHORTCUT_AUDIT_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported shortcut audit schema")
        _strict_sha256(self.source_files_sha256, label="source_files_sha256")
        _strict_sha256(self.runtime_identity_sha256, label="runtime_identity_sha256")
        if self.source_files_sha256 != _INITIAL_SOURCE_FILES_SHA256:
            raise GVSShortcutError("shortcut audit source identity changed")
        if self.runtime_identity_sha256 != _INITIAL_RUNTIME_SHA256:
            raise GVSShortcutError("shortcut audit runtime identity changed")
        if self.d_support_evidence is None:
            if self.d_support_evidence_sha256 is not None:
                raise GVSShortcutError("absent D-support evidence cannot carry a hash")
        else:
            if type(self.d_support_evidence) is not DSupportShortcutEvidence:
                raise GVSShortcutError("D-support evidence has the wrong type")
            if self.d_support_evidence_sha256 != self.d_support_evidence.sha256:
                raise GVSShortcutError("D-support evidence hash failed recomputation")
        if type(self.controls) is not tuple or tuple(
            status.control for status in self.controls
        ) != (
            ShortcutControl.LENGTH_ONLY,
            ShortcutControl.SYNTAX_ONLY,
            ShortcutControl.VALIDITY_ONLY,
            ShortcutControl.TOOL_FREQUENCY,
        ):
            raise GVSShortcutError("shortcut control roster or order changed")
        if any(type(status) is not ShortcutControlStatus for status in self.controls):
            raise GVSShortcutError("shortcut audit contains an invalid control status")
        expected_statuses = _control_statuses(self.d_support_evidence)
        if self.controls != expected_statuses:
            raise GVSShortcutError("shortcut status inventory failed exact recomputation")
        for name in (
            "authenticated_t_new_calibration_present",
            "d_support_population_completeness_authenticated",
            "scientific_result_available",
            "provisional_reference_is_gate",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "certifies_external_custody",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("shortcut audit must remain fail-closed and nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authenticated_t_new_calibration_present": (
                self.authenticated_t_new_calibration_present
            ),
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "certifies_external_custody": self.certifies_external_custody,
            "controls": [status.to_record() for status in self.controls],
            "d_support_evidence": (
                None if self.d_support_evidence is None else self.d_support_evidence.to_record()
            ),
            "d_support_evidence_sha256": self.d_support_evidence_sha256,
            "d_support_population_completeness_authenticated": (
                self.d_support_population_completeness_authenticated
            ),
            "feature_contract": _feature_contract(),
            "provisional_reference_is_gate": self.provisional_reference_is_gate,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "schema_version": self.schema_version,
            "scientific_result_available": self.scientific_result_available,
            "source_files_sha256": self.source_files_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())

    def to_json_bytes(self) -> bytes:
        artifact = {
            "artifact_schema_version": GVS_SHORTCUT_ARTIFACT_SCHEMA_VERSION,
            "receipt": self.to_record(),
            "receipt_sha256": self.sha256,
        }
        payload = _canonical_bytes(artifact)
        if len(payload) > MAX_SHORTCUT_ARTIFACT_BYTES:
            raise GVSShortcutError("shortcut audit artifact exceeds its byte bound")
        return payload


def build_shortcut_audit(
    bridge_inputs: Sequence[CandidateSupportBridgeInput] | None = None,
) -> ShortcutAuditReceipt:
    """Build a non-result inventory; no T-new input is accepted by this contract."""

    assert_shortcut_runtime_integrity()
    if bridge_inputs is None:
        evidence = None
    else:
        if type(bridge_inputs) not in {list, tuple}:
            raise GVSShortcutError("bridge_inputs must be None, an exact list, or an exact tuple")
        if not bridge_inputs:
            raise GVSShortcutError("an explicit D-support bridge population must be nonempty")
        evidence = build_d_support_shortcut_evidence(bridge_inputs)
    receipt = ShortcutAuditReceipt(
        d_support_evidence=evidence,
        d_support_evidence_sha256=None if evidence is None else evidence.sha256,
        controls=_control_statuses(evidence),
        source_files_sha256=_INITIAL_SOURCE_FILES_SHA256,
        runtime_identity_sha256=_INITIAL_RUNTIME_SHA256,
        _factory_token=_AUDIT_FACTORY_TOKEN,
    )
    assert_shortcut_runtime_integrity()
    return receipt


def _parse_exact_rational(value: object, *, label: str) -> ExactRational:
    row = _exact_object(value, frozenset({"denominator", "exact", "numerator"}), label=label)
    rational = ExactRational(row["numerator"], row["denominator"])
    if row["exact"] != f"{rational.numerator}/{rational.denominator}":
        raise GVSShortcutError(f"{label}.exact failed recomputation")
    return rational


def _parse_candidate(value: object, *, label: str) -> ShortcutCandidateSignal:
    row = _exact_object(
        value,
        frozenset(
            {
                "candidate_evidence_sha256",
                "content_token_count",
                "exact_semantic_match",
                "semantic_identity_sha256",
                "syntax_parse_valid",
                "tool_feature_binding_sha256",
                "tool_identity",
                "validity_schema_valid_nontruncated",
            }
        ),
        label=label,
    )
    return ShortcutCandidateSignal(**row)


def _parse_row(value: object, *, label: str) -> ShortcutRowEvidence:
    row = _exact_object(
        value,
        frozenset(
            {
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "bridge_receipt_sha256",
                "bridge_runtime_identity_sha256",
                "candidates",
                "certifies_external_custody",
                "component_id",
                "firewall_sha256",
                "gold_semantic_action_ir_sha256",
                "population_record_sha256",
                "population_role",
                "rank_hidden_batch_sha256",
                "rank_hidden_payload_sha256",
                "sample_id",
                "scan_evidence_sha256",
                "schema_presentation_receipt_sha256",
                "schema_version",
                "source_commitment_sha256",
            }
        ),
        label=label,
    )
    raw_candidates = row.pop("candidates")
    if type(raw_candidates) is not list or len(raw_candidates) > MAX_CANDIDATES_PER_ROW:
        raise GVSShortcutError(f"{label}.candidates must be a bounded exact array")
    return ShortcutRowEvidence(
        candidates=tuple(
            _parse_candidate(candidate, label=f"{label}.candidates[{index}]")
            for index, candidate in enumerate(raw_candidates)
        ),
        **row,
        _factory_token=_ROW_FACTORY_TOKEN,
    )


def _parse_d_support(value: object) -> DSupportShortcutEvidence:
    row = _exact_object(
        value,
        frozenset(
            {
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "bridge_receipt_root_sha256",
                "certifies_external_custody",
                "component_assignment_sha256",
                "component_count",
                "firewall_sha256",
                "population_completeness_authenticated",
                "population_record_root_sha256",
                "population_role",
                "rank_hidden_payload_root_sha256",
                "semantic_evidence_root_sha256",
                "row_count",
                "rows",
                "rows_with_contrast_pair",
                "rows_without_contrast_pair",
                "scan_evidence_sha256",
                "schema_version",
                "source_custody_authenticated",
            }
        ),
        label="d_support_evidence",
    )
    raw_rows = row.pop("rows")
    if type(raw_rows) is not list or not raw_rows or len(raw_rows) > MAX_SHORTCUT_ROWS:
        raise GVSShortcutError("d_support_evidence.rows must be a nonempty bounded array")
    return DSupportShortcutEvidence(
        rows=tuple(
            _parse_row(value, label=f"d_support_evidence.rows[{index}]")
            for index, value in enumerate(raw_rows)
        ),
        **row,
        _factory_token=_D_SUPPORT_FACTORY_TOKEN,
    )


def _parse_control_status(value: object, *, label: str) -> ShortcutControlStatus:
    row = _exact_object(
        value,
        frozenset(
            {
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "below_provisional_reference",
                "calibration_population",
                "calibration_status",
                "certifies_external_custody",
                "control",
                "d_support_feature_evidence_present",
                "evaluation_population",
                "evaluation_status",
                "feature_derivation_implemented",
                "missing_bindings",
                "pair_accuracy",
                "passes_gate",
                "provisional_reference",
                "provisional_reference_is_gate",
                "schema_version",
            }
        ),
        label=label,
    )
    reference = _parse_exact_rational(
        row.pop("provisional_reference"),
        label=f"{label}.provisional_reference",
    )
    if reference.fraction != PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE:
        raise GVSShortcutError("provisional shortcut reference changed")
    missing = row.pop("missing_bindings")
    if type(missing) is not list:
        raise GVSShortcutError(f"{label}.missing_bindings must be an exact array")
    return ShortcutControlStatus(
        control=ShortcutControl(row.pop("control")),
        calibration_status=ShortcutEvaluationStatus(row.pop("calibration_status")),
        evaluation_status=ShortcutEvaluationStatus(row.pop("evaluation_status")),
        missing_bindings=tuple(missing),
        **row,
        _factory_token=_STATUS_FACTORY_TOKEN,
    )


def load_shortcut_audit_json(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    bridge_inputs: Sequence[CandidateSupportBridgeInput] | None = None,
) -> ShortcutAuditReceipt:
    """Load a bounded artifact and live-reverify every serialized D-support row.

    A serialized receipt that contains D-support evidence is deliberately unusable
    without the complete live bridge inputs.  Requiring them here prevents a
    self-consistent caller-authored JSON object from being mistaken for authenticated
    shortcut evidence.  An inventory-only receipt contains no rows and therefore
    requires ``bridge_inputs=None``.
    """

    assert_shortcut_runtime_integrity()
    expected_receipt_sha256 = _strict_sha256(
        expected_receipt_sha256,
        label="expected_receipt_sha256",
    )
    if type(raw) is not bytes or not raw or len(raw) > MAX_SHORTCUT_ARTIFACT_BYTES:
        raise GVSShortcutError("shortcut artifact bytes are empty, non-bytes, or oversized")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_bounded_json_integer,
        )
    except GVSShortcutError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSShortcutError("shortcut artifact is not strict bounded UTF-8 JSON") from error
    artifact = _exact_object(
        payload,
        frozenset({"artifact_schema_version", "receipt", "receipt_sha256"}),
        label="shortcut artifact",
    )
    if artifact["artifact_schema_version"] != GVS_SHORTCUT_ARTIFACT_SCHEMA_VERSION:
        raise GVSShortcutError("unsupported shortcut artifact schema")
    receipt_row = _exact_object(
        artifact["receipt"],
        frozenset(
            {
                "authenticated_t_new_calibration_present",
                "authorizes_cuda_or_jarvis_access",
                "authorizes_model_or_label_access",
                "certifies_external_custody",
                "controls",
                "d_support_evidence",
                "d_support_evidence_sha256",
                "d_support_population_completeness_authenticated",
                "feature_contract",
                "provisional_reference_is_gate",
                "runtime_identity_sha256",
                "schema_version",
                "scientific_result_available",
                "source_files_sha256",
            }
        ),
        label="shortcut receipt",
    )
    if receipt_row.pop("feature_contract") != _feature_contract():
        raise GVSShortcutError("shortcut feature contract changed")
    raw_controls = receipt_row.pop("controls")
    if type(raw_controls) is not list or len(raw_controls) != len(ShortcutControl):
        raise GVSShortcutError("shortcut controls must be the exact four-entry array")
    raw_d_support = receipt_row.pop("d_support_evidence")
    d_support = None if raw_d_support is None else _parse_d_support(raw_d_support)
    receipt = ShortcutAuditReceipt(
        d_support_evidence=d_support,
        controls=tuple(
            _parse_control_status(value, label=f"shortcut controls[{index}]")
            for index, value in enumerate(raw_controls)
        ),
        **receipt_row,
        _factory_token=_AUDIT_FACTORY_TOKEN,
    )
    embedded_sha256 = _strict_sha256(artifact["receipt_sha256"], label="receipt_sha256")
    if embedded_sha256 != receipt.sha256 or embedded_sha256 != expected_receipt_sha256:
        raise GVSShortcutError("shortcut receipt hash failed caller-bound recomputation")
    if receipt.d_support_evidence is None:
        if bridge_inputs is not None:
            raise GVSShortcutError(
                "inventory-only shortcut artifacts must not receive bridge inputs"
            )
    else:
        if type(bridge_inputs) not in {list, tuple} or not bridge_inputs:
            raise GVSShortcutError(
                "serialized D-support evidence requires complete live bridge inputs"
            )
        live_receipt = build_shortcut_audit(bridge_inputs)
        if live_receipt.to_record() != receipt.to_record():
            raise GVSShortcutError(
                "serialized shortcut evidence differs from live bridge recomputation"
            )
    assert_shortcut_runtime_integrity()
    return receipt


def _snapshot_v2_firewall_inputs(
    *,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    value = {
        "compared_release_cutoff_utc": compared_release_cutoff_utc,
        "expected_membership": expected_membership,
        "expected_scan_bindings": expected_scan_bindings,
        "firewall_frozen_at_utc": firewall_frozen_at_utc,
        "firewall_manifest": firewall_manifest,
        "population_records": population_records,
        "scan_evidence": scan_evidence,
        "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
    }
    try:
        first_raw = _canonical_bytes(value)
        second_raw = _canonical_bytes(value)
    except (RuntimeError, RecursionError) as error:
        raise GVSShortcutError("v2 firewall inputs changed during their snapshot") from error
    if first_raw != second_raw:
        raise GVSShortcutError("v2 firewall inputs changed during their stable snapshot")
    if len(first_raw) > MAX_SHORTCUT_ARTIFACT_BYTES:
        raise GVSShortcutError("v2 firewall inputs exceed the bounded snapshot size")
    detached = json.loads(
        first_raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
        parse_float=_reject_float,
        parse_int=_bounded_json_integer,
    )
    if type(detached) is not dict:
        raise GVSShortcutError("v2 firewall snapshot must be an exact JSON object")
    return detached


def derive_shortcut_v2_population_identities(
    *,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, tuple[SupportSampleIdentity, ...]]:
    """Derive exact eligible T-new and D-support denominators from one live firewall."""

    assert_shortcut_runtime_integrity()
    snapshot = _snapshot_v2_firewall_inputs(
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    try:
        checked_scan = _VALIDATE_SCAN_EVIDENCE_CALLABLE(
            snapshot["scan_evidence"],
            populations=snapshot["population_records"],
            expected_membership=snapshot["expected_membership"],
            expected_scan_bindings=snapshot["expected_scan_bindings"],
            compared_release_cutoff_utc=snapshot["compared_release_cutoff_utc"],
            scan_evidence_frozen_at_utc=snapshot["scan_evidence_frozen_at_utc"],
        )
        checked_firewall = _VALIDATE_POPULATION_FIREWALL_CALLABLE(
            snapshot["firewall_manifest"],
            populations=snapshot["population_records"],
            expected_membership=snapshot["expected_membership"],
            expected_scan_bindings=snapshot["expected_scan_bindings"],
            scan_evidence=checked_scan,
            compared_release_cutoff_utc=snapshot["compared_release_cutoff_utc"],
            scan_evidence_frozen_at_utc=snapshot["scan_evidence_frozen_at_utc"],
            firewall_frozen_at_utc=snapshot["firewall_frozen_at_utc"],
        )
    except _POPULATION_ERROR_CLASS as error:
        raise GVSShortcutError("v2 shortcut population firewall validation failed") from error

    scan_by_hash = {entry["record_sha256"]: entry for entry in checked_scan["entries"]}
    strata_by_hash = {
        assignment["record_sha256"]: assignment
        for assignment in checked_firewall["strata_assignments"]
    }
    component_by_hash: dict[str, Mapping[str, Any]] = {}
    for component in checked_firewall["graph"]["components"]:
        for record_sha256 in component["member_record_sha256s"]:
            if record_sha256 in component_by_hash:
                raise GVSShortcutError("v2 firewall assigns one record to multiple components")
            component_by_hash[record_sha256] = component

    output: dict[str, tuple[SupportSampleIdentity, ...]] = {}
    for role in (SHORTCUT_CALIBRATION_POPULATION, SHORTCUT_EVALUATION_POPULATION):
        rows: list[SupportSampleIdentity] = []
        observed_hashes: set[str] = set()
        for eligibility in checked_firewall["eligibility_assignments"]:
            if eligibility["population_role"] != role or eligibility["eligible"] is not True:
                continue
            record_sha256 = eligibility["record_sha256"]
            if record_sha256 in observed_hashes:
                raise GVSShortcutError("v2 eligible population record is duplicated")
            observed_hashes.add(record_sha256)
            try:
                scan_row = scan_by_hash[record_sha256]
                strata_row = strata_by_hash[record_sha256]
                component = component_by_hash[record_sha256]
            except KeyError as error:
                raise GVSShortcutError(
                    "v2 eligible population row is missing a structural binding"
                ) from error
            if (
                scan_row["record_id"] != eligibility["record_id"]
                or strata_row["record_id"] != eligibility["record_id"]
                or scan_row["population_role"] != role
                or strata_row["population_role"] != role
                or component["population_role"] != role
                or component["component_id"] != eligibility["component_id"]
                or component["eligible"] is not True
            ):
                raise GVSShortcutError("v2 population structural bindings disagree")
            expected_classification = {
                "action_family": strata_row["action_family"],
                "expected_outcome": strata_row["expected_outcome"],
                "strata": strata_row["strata"],
                "task_class": strata_row["task_class"],
            }
            if component["classification"] != expected_classification:
                raise GVSShortcutError("v2 component classification differs from its row")
            try:
                task_class = SupportClass(strata_row["task_class"])
            except (TypeError, ValueError) as error:
                raise GVSShortcutError("v2 population task class is invalid") from error
            rows.append(
                SupportSampleIdentity(
                    sample_id=eligibility["record_id"],
                    firewall_sha256=checked_firewall["firewall_sha256"],
                    scan_evidence_sha256=checked_scan["scan_evidence_sha256"],
                    population_record_sha256=record_sha256,
                    component_id=eligibility["component_id"],
                    source_commitment_sha256=scan_row["source_commitment_sha256"],
                    task_class=task_class,
                    expected_outcome=strata_row["expected_outcome"],
                    action_family=strata_row["action_family"],
                    strata=SupportStrata.from_mapping(strata_row["strata"]),
                )
            )
        declared = set(checked_firewall["populations"][role]["eligible_record_sha256s"])
        if observed_hashes != declared or not rows:
            raise GVSShortcutError(f"v2 {role} denominator differs from the firewall summary")
        output[role] = tuple(sorted(rows, key=lambda value: value.sample_id))
    assert_shortcut_runtime_integrity()
    return output


@dataclass(frozen=True, slots=True)
class TNewShortcutCalibrationInput:
    """One live T-new bridge plus its independently certified single-fault negative."""

    bridge_input: CandidateSupportBridgeInput
    fault_certification: FaultCertification

    def __post_init__(self) -> None:
        if type(self.bridge_input) is not _BRIDGE_INPUT_CLASS:
            raise GVSShortcutError("T-new calibration bridge input has the wrong type")
        if type(self.fault_certification) is not _FAULT_CERTIFICATION_CLASS:
            raise GVSShortcutError("T-new fault certification has the wrong type")


@dataclass(frozen=True, slots=True)
class ShortcutV2CandidateSignal:
    semantic_identity_sha256: str
    candidate_evidence_sha256: str
    content_token_count: int
    syntax_parse_valid: bool
    validity_schema_valid_nontruncated: bool
    exact_semantic_match: bool
    tool_identities: tuple[str, ...]
    tool_feature_binding_sha256: str
    schema_version: str = GVS_SHORTCUT_V2_CANDIDATE_SCHEMA_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_CANDIDATE_FACTORY_TOKEN:
            raise TypeError("ShortcutV2CandidateSignal must be live-derived")
        if self.schema_version != GVS_SHORTCUT_V2_CANDIDATE_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 shortcut candidate schema")
        _strict_sha256(self.semantic_identity_sha256, label="semantic_identity_sha256")
        _strict_sha256(self.candidate_evidence_sha256, label="candidate_evidence_sha256")
        _strict_sha256(self.tool_feature_binding_sha256, label="tool_feature_binding_sha256")
        _strict_nonnegative_integer(self.content_token_count, label="content_token_count")
        if self.content_token_count > 1_000_000:
            raise GVSShortcutError("v2 candidate token count exceeds the bridge bound")
        if (
            self.syntax_parse_valid is not True
            or self.validity_schema_valid_nontruncated is not True
        ):
            raise GVSShortcutError("v2 candidates must be live verifier-eligible candidates")
        _strict_bool(self.exact_semantic_match, label="exact_semantic_match")
        if type(self.tool_identities) is not tuple:
            raise GVSShortcutError("v2 tool identities must be an exact tuple")
        for index, tool in enumerate(self.tool_identities):
            _strict_identifier(tool, label=f"tool_identities[{index}]")
        if self.tool_identities != tuple(sorted(self.tool_identities)):
            raise GVSShortcutError("v2 tool identities must be canonically sorted")
        expected_tool_binding = _canonical_sha256(
            {
                "candidate_semantic_identity_sha256": self.semantic_identity_sha256,
                "tool_identities": list(self.tool_identities),
            }
        )
        if self.tool_feature_binding_sha256 != expected_tool_binding:
            raise GVSShortcutError("v2 candidate tool feature failed exact recomputation")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_evidence_sha256": self.candidate_evidence_sha256,
            "content_token_count": self.content_token_count,
            "exact_semantic_match": self.exact_semantic_match,
            "schema_version": self.schema_version,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "syntax_parse_valid": self.syntax_parse_valid,
            "tool_feature_binding_sha256": self.tool_feature_binding_sha256,
            "tool_identities": list(self.tool_identities),
            "validity_schema_valid_nontruncated": self.validity_schema_valid_nontruncated,
        }


@dataclass(frozen=True, slots=True)
class ShortcutV2RowEvidence:
    sample_id: str
    component_id: str
    firewall_sha256: str
    scan_evidence_sha256: str
    population_record_sha256: str
    source_commitment_sha256: str
    program_sha256: str
    prompt_sha256: str
    exact_request_sha256: str
    candidate_set_trace_sha256: str
    candidate_set_analysis_sha256: str
    bridge_receipt_sha256: str
    rank_hidden_payload_sha256: str
    schema_presentation_receipt_sha256: str
    gold_semantic_action_ir_sha256: str
    candidates: tuple[ShortcutV2CandidateSignal, ...]
    schema_version: str = GVS_SHORTCUT_V2_ROW_SCHEMA_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_ROW_FACTORY_TOKEN:
            raise TypeError("ShortcutV2RowEvidence must be live-derived")
        if self.schema_version != GVS_SHORTCUT_V2_ROW_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 shortcut row schema")
        _strict_identifier(self.sample_id, label="v2 row sample_id")
        for name in (
            "component_id",
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_sha256",
            "source_commitment_sha256",
            "program_sha256",
            "prompt_sha256",
            "exact_request_sha256",
            "candidate_set_trace_sha256",
            "candidate_set_analysis_sha256",
            "bridge_receipt_sha256",
            "rank_hidden_payload_sha256",
            "schema_presentation_receipt_sha256",
            "gold_semantic_action_ir_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if type(self.candidates) is not tuple or len(self.candidates) > 8:
            raise GVSShortcutError("v2 row candidates must be a bounded exact tuple")
        if any(type(value) is not ShortcutV2CandidateSignal for value in self.candidates):
            raise GVSShortcutError("v2 row contains an invalid candidate signal")
        identities = tuple(value.semantic_identity_sha256 for value in self.candidates)
        if identities != tuple(sorted(set(identities))):
            raise GVSShortcutError("v2 row candidate identities must be unique and sorted")
        if any(
            value.exact_semantic_match
            is not (value.semantic_identity_sha256 == self.gold_semantic_action_ir_sha256)
            for value in self.candidates
        ):
            raise GVSShortcutError("v2 candidate exactness differs from semantic hash equality")

    @property
    def positives(self) -> tuple[ShortcutV2CandidateSignal, ...]:
        return tuple(value for value in self.candidates if value.exact_semantic_match)

    @property
    def negatives(self) -> tuple[ShortcutV2CandidateSignal, ...]:
        return tuple(value for value in self.candidates if not value.exact_semantic_match)

    @property
    def has_contrast_pair(self) -> bool:
        return bool(self.positives and self.negatives)

    def to_record(self) -> dict[str, object]:
        return {
            "bridge_receipt_sha256": self.bridge_receipt_sha256,
            "candidates": [value.to_record() for value in self.candidates],
            "component_id": self.component_id,
            "candidate_set_analysis_sha256": self.candidate_set_analysis_sha256,
            "candidate_set_trace_sha256": self.candidate_set_trace_sha256,
            "exact_request_sha256": self.exact_request_sha256,
            "firewall_sha256": self.firewall_sha256,
            "gold_semantic_action_ir_sha256": self.gold_semantic_action_ir_sha256,
            "population_record_sha256": self.population_record_sha256,
            "program_sha256": self.program_sha256,
            "prompt_sha256": self.prompt_sha256,
            "rank_hidden_payload_sha256": self.rank_hidden_payload_sha256,
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "schema_presentation_receipt_sha256": (self.schema_presentation_receipt_sha256),
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "source_commitment_sha256": self.source_commitment_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())


def _verify_v2_bridge_input(
    bridge_input: CandidateSupportBridgeInput,
) -> CandidateSupportBridgeReceipt:
    if type(bridge_input) is not _BRIDGE_INPUT_CLASS:
        raise GVSShortcutError("v2 bridge input has the wrong exact type")
    try:
        verified = _BRIDGE_VERIFY_CALLABLE(
            bridge_input.receipt,
            sample=bridge_input.sample,
            trace=bridge_input.trace,
            analysis=bridge_input.analysis,
            program=bridge_input.program,
            step_id=bridge_input.step_id,
            now=bridge_input.now,
            user_text=bridge_input.user_text,
            prompt_tools=bridge_input.prompt_tools,
            schema_presentation_receipt=bridge_input.schema_presentation_receipt,
        )
    except _BRIDGE_ERROR_CLASS as error:
        raise GVSShortcutError("v2 bridge failed complete live recomputation") from error
    if type(verified) is not _BRIDGE_RECEIPT_CLASS:
        raise GVSShortcutError("v2 bridge verifier returned the wrong receipt type")
    for name in (
        "source_custody_authenticated",
        "authorizes_model_or_label_access",
        "authorizes_cuda_or_jarvis_access",
        "launch_authorized",
    ):
        if getattr(verified, name) is not False:
            raise GVSShortcutError("v2 shortcut inputs must remain nonauthorizing")
    return verified


def _v2_program_step_context(
    bridge_input: CandidateSupportBridgeInput,
) -> tuple[WorldState, ActionIR]:
    try:
        state = _WORLD_STATE_CLASS.from_dict(bridge_input.program.initial_state.to_dict())
        simulator = _ACTION_SIMULATOR_CLASS()
        for step in bridge_input.program.steps:
            action = _COMPILE_PROGRAM_STEP_CALLABLE(step)
            if step.step_id == bridge_input.step_id:
                return state, action
            state = simulator.simulate(action, state).state
    except (ValueError, TypeError, _SIM_PROGRAM_ERROR_CLASS) as error:
        raise GVSShortcutError("v2 program context failed live recomputation") from error
    raise GVSShortcutError("v2 bridge step is absent from its bound program")


def _v2_candidate_tools_by_identity(
    bridge_input: CandidateSupportBridgeInput,
    verified: CandidateSupportBridgeReceipt,
) -> dict[str, tuple[str, ...]]:
    tools_by_identity: dict[str, tuple[str, ...]] = {}
    for analysis_candidate, bridge_candidate in zip(
        bridge_input.analysis.candidates,
        verified.candidates,
        strict=True,
    ):
        identity = bridge_candidate.lowered_action_ir_sha256
        if identity is None or not analysis_candidate.schema_valid or analysis_candidate.truncated:
            continue
        if analysis_candidate.canonical_action_json is None:
            raise GVSShortcutError("v2 valid candidate lacks canonical Action IR")
        try:
            presented = _PARSE_ACTION_IR_CALLABLE(
                analysis_candidate.canonical_action_json,
                bridge_input.schema_presentation_receipt.presented_schemas,
            )
            lowered = _LOWER_PRESENTED_ACTION_IR_CALLABLE(
                presented,
                receipt=bridge_input.schema_presentation_receipt,
                canonical_schemas=SIMULATOR_TOOL_SCHEMAS,
                prompt_tools=bridge_input.prompt_tools,
                presentation=bridge_input.schema_presentation_receipt.presentation,
            )
            canonical = _PARSE_ACTION_IR_CALLABLE(
                lowered.canonical_json(),
                SIMULATOR_TOOL_REGISTRY,
            )
        except (ValueError, TypeError, _SCHEMA_PRESENTATION_ERROR_CLASS) as error:
            raise GVSShortcutError(
                "v2 candidate tool feature failed live schema lowering"
            ) from error
        if _LOWERED_SEMANTIC_SHA256_CALLABLE(canonical) != identity:
            raise GVSShortcutError("v2 lowered candidate identity failed recomputation")
        tool_identities = tuple(sorted(call.tool for call in canonical.calls))
        previous = tools_by_identity.setdefault(identity, tool_identities)
        if previous != tool_identities:
            raise GVSShortcutError("one v2 semantic identity maps to different tools")
    return tools_by_identity


def derive_shortcut_v2_row(
    bridge_input: CandidateSupportBridgeInput,
) -> ShortcutV2RowEvidence:
    """Live-rederive one rank-hidden row, including canonical candidate tools."""

    assert_shortcut_runtime_integrity()
    verified = _verify_v2_bridge_input(bridge_input)
    tools_by_identity = _v2_candidate_tools_by_identity(bridge_input, verified)
    signals: list[ShortcutV2CandidateSignal] = []
    payload = verified.rank_hidden_verifier_payload
    for candidate in payload.candidates:
        identity = candidate.candidate_identity_sha256
        if identity is None:
            continue
        try:
            tool_identities = tools_by_identity[identity]
        except KeyError as error:
            raise GVSShortcutError(
                "v2 rank-hidden candidate lacks a live-derived tool feature"
            ) from error
        evidence_record = {
            "candidate_semantic_identity_sha256": identity,
            "content_token_ids_sha256": _canonical_sha256(list(candidate.content_token_ids)),
            "generated_token_ids_sha256": _canonical_sha256(list(candidate.generated_token_ids)),
            "presented_action_ir_sha256": candidate.presented_action_ir_sha256,
            "rank_hidden_payload_sha256": verified.rank_hidden_verifier_payload_sha256,
            "schema_presentation_receipt_sha256": (verified.schema_presentation_receipt_sha256),
        }
        tool_binding = _canonical_sha256(
            {
                "candidate_semantic_identity_sha256": identity,
                "tool_identities": list(tool_identities),
            }
        )
        signals.append(
            ShortcutV2CandidateSignal(
                semantic_identity_sha256=identity,
                candidate_evidence_sha256=_canonical_sha256(evidence_record),
                content_token_count=len(candidate.content_token_ids),
                syntax_parse_valid=True,
                validity_schema_valid_nontruncated=True,
                exact_semantic_match=identity == verified.gold_semantic_action_ir_sha256,
                tool_identities=tool_identities,
                tool_feature_binding_sha256=tool_binding,
                _factory_token=_V2_CANDIDATE_FACTORY_TOKEN,
            )
        )
    signals.sort(key=lambda value: value.semantic_identity_sha256)
    support = verified.support_record
    row = ShortcutV2RowEvidence(
        sample_id=verified.sample_id,
        component_id=support.component_id,
        firewall_sha256=support.firewall_sha256,
        scan_evidence_sha256=support.scan_evidence_sha256,
        population_record_sha256=support.population_record_sha256,
        source_commitment_sha256=verified.source_commitment_sha256,
        program_sha256=verified.program_sha256,
        prompt_sha256=verified.prompt_sha256,
        exact_request_sha256=_canonical_sha256({"user_text": bridge_input.user_text}),
        candidate_set_trace_sha256=verified.candidate_set_trace_sha256,
        candidate_set_analysis_sha256=verified.candidate_set_analysis_sha256,
        bridge_receipt_sha256=verified.sha256,
        rank_hidden_payload_sha256=verified.rank_hidden_verifier_payload_sha256,
        schema_presentation_receipt_sha256=(verified.schema_presentation_receipt_sha256),
        gold_semantic_action_ir_sha256=verified.gold_semantic_action_ir_sha256,
        candidates=tuple(signals),
        _factory_token=_V2_ROW_FACTORY_TOKEN,
    )
    assert_shortcut_runtime_integrity()
    return row


@dataclass(frozen=True, slots=True)
class ShortcutV2CalibrationRowEvidence:
    row: ShortcutV2RowEvidence
    fault_certification_sha256: str
    fault_descriptor_sha256: str
    positive_semantic_identity_sha256: str
    negative_semantic_identity_sha256: str
    positive_tool_identities: tuple[str, ...]
    negative_tool_identities: tuple[str, ...]
    positive_tool_feature_binding_sha256: str
    negative_tool_feature_binding_sha256: str
    calibration_binding_sha256: str
    schema_version: str = GVS_SHORTCUT_V2_CALIBRATION_ROW_SCHEMA_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_CALIBRATION_ROW_FACTORY_TOKEN:
            raise TypeError("v2 calibration rows must be live-derived")
        if self.schema_version != GVS_SHORTCUT_V2_CALIBRATION_ROW_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 calibration-row schema")
        if type(self.row) is not ShortcutV2RowEvidence:
            raise GVSShortcutError("v2 calibration row has invalid row evidence")
        for name in (
            "fault_certification_sha256",
            "fault_descriptor_sha256",
            "positive_semantic_identity_sha256",
            "negative_semantic_identity_sha256",
            "positive_tool_feature_binding_sha256",
            "negative_tool_feature_binding_sha256",
            "calibration_binding_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if self.positive_semantic_identity_sha256 == self.negative_semantic_identity_sha256:
            raise GVSShortcutError("v2 calibration positive and negative must differ")
        candidates = {value.semantic_identity_sha256: value for value in self.row.candidates}
        try:
            positive = candidates[self.positive_semantic_identity_sha256]
            negative = candidates[self.negative_semantic_identity_sha256]
        except KeyError as error:
            raise GVSShortcutError("v2 calibration pair is absent from its row") from error
        if positive.exact_semantic_match is not True or negative.exact_semantic_match is not False:
            raise GVSShortcutError("v2 calibration labels differ from live semantic labels")
        for label, identities in (
            ("positive_tool_identities", self.positive_tool_identities),
            ("negative_tool_identities", self.negative_tool_identities),
        ):
            if type(identities) is not tuple:
                raise GVSShortcutError(f"{label} must be an exact tuple")
            for index, tool in enumerate(identities):
                _strict_identifier(tool, label=f"{label}[{index}]")
            if identities != tuple(sorted(identities)):
                raise GVSShortcutError(f"{label} must be sorted")
        if (
            positive.tool_identities != self.positive_tool_identities
            or positive.tool_feature_binding_sha256 != self.positive_tool_feature_binding_sha256
            or negative.tool_identities != self.negative_tool_identities
            or negative.tool_feature_binding_sha256 != self.negative_tool_feature_binding_sha256
        ):
            raise GVSShortcutError("v2 calibration tool identities are not candidate-bound")
        expected_binding = _canonical_sha256(
            {
                "fault_certification_sha256": self.fault_certification_sha256,
                "fault_descriptor_sha256": self.fault_descriptor_sha256,
                "negative_semantic_identity_sha256": (self.negative_semantic_identity_sha256),
                "negative_tool_feature_binding_sha256": (self.negative_tool_feature_binding_sha256),
                "positive_semantic_identity_sha256": (self.positive_semantic_identity_sha256),
                "positive_tool_feature_binding_sha256": (self.positive_tool_feature_binding_sha256),
                "program_sha256": self.row.program_sha256,
                "row_evidence_sha256": self.row.sha256,
                "sample_id": self.row.sample_id,
            }
        )
        if self.calibration_binding_sha256 != expected_binding:
            raise GVSShortcutError("v2 calibration binding failed exact recomputation")

    def to_record(self) -> dict[str, object]:
        return {
            "calibration_binding_sha256": self.calibration_binding_sha256,
            "fault_certification_sha256": self.fault_certification_sha256,
            "fault_descriptor_sha256": self.fault_descriptor_sha256,
            "negative_semantic_identity_sha256": self.negative_semantic_identity_sha256,
            "negative_tool_feature_binding_sha256": (self.negative_tool_feature_binding_sha256),
            "negative_tool_identities": list(self.negative_tool_identities),
            "positive_semantic_identity_sha256": self.positive_semantic_identity_sha256,
            "positive_tool_feature_binding_sha256": (self.positive_tool_feature_binding_sha256),
            "positive_tool_identities": list(self.positive_tool_identities),
            "row": self.row.to_record(),
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())


def _derive_v2_calibration_row(
    value: TNewShortcutCalibrationInput,
) -> ShortcutV2CalibrationRowEvidence:
    if type(value) is not TNewShortcutCalibrationInput:
        raise GVSShortcutError("T-new calibration input has the wrong exact type")
    row = derive_shortcut_v2_row(value.bridge_input)
    pre_state, gold_action = _v2_program_step_context(value.bridge_input)
    try:
        certified = _CERTIFY_SINGLE_FAULT_CALLABLE(
            pre_state,
            gold_action,
            value.fault_certification.descriptor,
        )
    except (_FAULT_ERROR_CLASS, ValueError, TypeError) as error:
        raise GVSShortcutError("T-new single-fault negative failed recertification") from error
    if type(certified) is not _FAULT_CERTIFICATION_CLASS:
        raise GVSShortcutError("T-new certifier returned the wrong receipt type")
    if _canonical_bytes(certified.to_dict()) != _canonical_bytes(
        value.fault_certification.to_dict()
    ):
        raise GVSShortcutError("T-new fault certificate differs from live recertification")
    positive_identity = _LOWERED_SEMANTIC_SHA256_CALLABLE(certified.gold_action)
    negative_identity = _LOWERED_SEMANTIC_SHA256_CALLABLE(certified.faulty_action)
    if positive_identity != row.gold_semantic_action_ir_sha256:
        raise GVSShortcutError("T-new certified positive differs from the bound program gold")
    candidates = {value.semantic_identity_sha256: value for value in row.candidates}
    try:
        positive = candidates[positive_identity]
        negative = candidates[negative_identity]
    except KeyError as error:
        raise GVSShortcutError(
            "T-new candidate set lacks its certified positive/negative pair"
        ) from error
    positive_tools = tuple(sorted(call.tool for call in certified.gold_action.calls))
    negative_tools = tuple(sorted(call.tool for call in certified.faulty_action.calls))
    if positive.tool_identities != positive_tools or negative.tool_identities != negative_tools:
        raise GVSShortcutError("T-new certified actions differ from live candidate tools")
    fault_sha256 = certified.sha256()
    descriptor_sha256 = _canonical_sha256(certified.descriptor.to_dict())
    binding = _canonical_sha256(
        {
            "fault_certification_sha256": fault_sha256,
            "fault_descriptor_sha256": descriptor_sha256,
            "negative_semantic_identity_sha256": negative_identity,
            "negative_tool_feature_binding_sha256": (negative.tool_feature_binding_sha256),
            "positive_semantic_identity_sha256": positive_identity,
            "positive_tool_feature_binding_sha256": (positive.tool_feature_binding_sha256),
            "program_sha256": row.program_sha256,
            "row_evidence_sha256": row.sha256,
            "sample_id": row.sample_id,
        }
    )
    return ShortcutV2CalibrationRowEvidence(
        row=row,
        fault_certification_sha256=fault_sha256,
        fault_descriptor_sha256=descriptor_sha256,
        positive_semantic_identity_sha256=positive_identity,
        negative_semantic_identity_sha256=negative_identity,
        positive_tool_identities=positive_tools,
        negative_tool_identities=negative_tools,
        positive_tool_feature_binding_sha256=positive.tool_feature_binding_sha256,
        negative_tool_feature_binding_sha256=negative.tool_feature_binding_sha256,
        calibration_binding_sha256=binding,
        _factory_token=_V2_CALIBRATION_ROW_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ShortcutV2PopulationEvidence:
    population_role: str
    firewall_sha256: str
    scan_evidence_sha256: str
    sample_ids: tuple[str, ...]
    population_record_sha256s: tuple[str, ...]
    component_ids: tuple[str, ...]
    row_evidence_sha256s: tuple[str, ...]
    membership_root_sha256: str
    population_record_root_sha256: str
    component_assignment_sha256: str
    row_evidence_root_sha256: str
    row_count: int
    component_count: int
    schema_version: str = GVS_SHORTCUT_V2_POPULATION_SCHEMA_VERSION
    complete_firewall_bound_denominator: bool = True
    external_custody_authenticated: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    launch_authorized: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_POPULATION_FACTORY_TOKEN:
            raise TypeError("v2 population evidence must be live-derived")
        if self.schema_version != GVS_SHORTCUT_V2_POPULATION_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 population-evidence schema")
        if self.population_role not in {
            SHORTCUT_CALIBRATION_POPULATION,
            SHORTCUT_EVALUATION_POPULATION,
        }:
            raise GVSShortcutError("v2 shortcut population role is invalid")
        _strict_sha256(self.firewall_sha256, label="firewall_sha256")
        _strict_sha256(self.scan_evidence_sha256, label="scan_evidence_sha256")
        vectors = (
            self.sample_ids,
            self.population_record_sha256s,
            self.component_ids,
            self.row_evidence_sha256s,
        )
        if any(type(value) is not tuple for value in vectors):
            raise GVSShortcutError("v2 population vectors must be exact tuples")
        if not 0 < len(self.sample_ids) <= MAX_SHORTCUT_ROWS or any(
            len(value) != len(self.sample_ids) for value in vectors
        ):
            raise GVSShortcutError("v2 population vectors have inconsistent bounds")
        if self.sample_ids != tuple(sorted(set(self.sample_ids))):
            raise GVSShortcutError("v2 population sample IDs must be unique and sorted")
        for index, sample_id in enumerate(self.sample_ids):
            _strict_identifier(sample_id, label=f"sample_ids[{index}]")
        for label, values in (
            ("population_record_sha256s", self.population_record_sha256s),
            ("component_ids", self.component_ids),
            ("row_evidence_sha256s", self.row_evidence_sha256s),
        ):
            for index, value in enumerate(values):
                _strict_sha256(value, label=f"{label}[{index}]")
        if len(set(self.population_record_sha256s)) != len(self.population_record_sha256s):
            raise GVSShortcutError("v2 population record identities must be unique")
        membership = [
            {
                "component_id": component,
                "population_record_sha256": record,
                "sample_id": sample,
            }
            for sample, record, component in zip(
                self.sample_ids,
                self.population_record_sha256s,
                self.component_ids,
                strict=True,
            )
        ]
        expected = {
            "membership_root_sha256": _canonical_sha256(membership),
            "population_record_root_sha256": _canonical_sha256(
                list(self.population_record_sha256s)
            ),
            "component_assignment_sha256": _canonical_sha256(
                [
                    {"component_id": row["component_id"], "sample_id": row["sample_id"]}
                    for row in membership
                ]
            ),
            "row_evidence_root_sha256": _canonical_sha256(list(self.row_evidence_sha256s)),
        }
        for name, expected_value in expected.items():
            _strict_sha256(getattr(self, name), label=name)
            if getattr(self, name) != expected_value:
                raise GVSShortcutError(f"{name} failed exact recomputation")
        if self.row_count != len(self.sample_ids) or type(self.row_count) is not int:
            raise GVSShortcutError("v2 population row_count failed recomputation")
        if (
            self.component_count != len(set(self.component_ids))
            or type(self.component_count) is not int
        ):
            raise GVSShortcutError("v2 population component_count failed recomputation")
        if self.complete_firewall_bound_denominator is not True:
            raise GVSShortcutError("v2 population denominator must be complete")
        for name in (
            "external_custody_authenticated",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "launch_authorized",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("v2 population evidence must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "complete_firewall_bound_denominator": (self.complete_firewall_bound_denominator),
            "component_assignment_sha256": self.component_assignment_sha256,
            "component_count": self.component_count,
            "component_ids": list(self.component_ids),
            "external_custody_authenticated": self.external_custody_authenticated,
            "firewall_sha256": self.firewall_sha256,
            "launch_authorized": self.launch_authorized,
            "membership_root_sha256": self.membership_root_sha256,
            "population_record_root_sha256": self.population_record_root_sha256,
            "population_record_sha256s": list(self.population_record_sha256s),
            "population_role": self.population_role,
            "row_count": self.row_count,
            "row_evidence_root_sha256": self.row_evidence_root_sha256,
            "row_evidence_sha256s": list(self.row_evidence_sha256s),
            "sample_ids": list(self.sample_ids),
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "schema_version": self.schema_version,
        }


def _build_v2_population_evidence(
    *,
    population_role: str,
    identities: Sequence[SupportSampleIdentity],
    rows: Sequence[ShortcutV2RowEvidence],
    row_evidence_sha256s: Sequence[str],
) -> ShortcutV2PopulationEvidence:
    if (
        type(identities) not in {list, tuple}
        or type(rows) not in {list, tuple}
        or type(row_evidence_sha256s) not in {list, tuple}
    ):
        raise GVSShortcutError("v2 population evidence inputs must be exact sequences")
    ordered_identities = tuple(sorted(identities, key=lambda value: value.sample_id))
    ordered_rows = tuple(sorted(rows, key=lambda value: value.sample_id))
    evidence_by_sample = {
        row.sample_id: sha256 for row, sha256 in zip(rows, row_evidence_sha256s, strict=True)
    }
    if len(evidence_by_sample) != len(rows):
        raise GVSShortcutError("v2 row evidence contains duplicate sample IDs")
    if tuple(value.sample_id for value in ordered_identities) != tuple(
        value.sample_id for value in ordered_rows
    ):
        raise GVSShortcutError("v2 row evidence differs from the complete denominator")
    for identity, row in zip(ordered_identities, ordered_rows, strict=True):
        if (
            row.component_id != identity.component_id
            or row.firewall_sha256 != identity.firewall_sha256
            or row.scan_evidence_sha256 != identity.scan_evidence_sha256
            or row.population_record_sha256 != identity.population_record_sha256
            or row.source_commitment_sha256 != identity.source_commitment_sha256
        ):
            raise GVSShortcutError("v2 row differs from its firewall-derived identity")
    sample_ids = tuple(value.sample_id for value in ordered_identities)
    records = tuple(value.population_record_sha256 for value in ordered_identities)
    components = tuple(value.component_id for value in ordered_identities)
    evidence = tuple(evidence_by_sample[sample_id] for sample_id in sample_ids)
    membership = [
        {
            "component_id": component,
            "population_record_sha256": record,
            "sample_id": sample,
        }
        for sample, record, component in zip(sample_ids, records, components, strict=True)
    ]
    return ShortcutV2PopulationEvidence(
        population_role=population_role,
        firewall_sha256=ordered_identities[0].firewall_sha256,
        scan_evidence_sha256=ordered_identities[0].scan_evidence_sha256,
        sample_ids=sample_ids,
        population_record_sha256s=records,
        component_ids=components,
        row_evidence_sha256s=evidence,
        membership_root_sha256=_canonical_sha256(membership),
        population_record_root_sha256=_canonical_sha256(list(records)),
        component_assignment_sha256=_canonical_sha256(
            [
                {"component_id": component, "sample_id": sample}
                for sample, component in zip(sample_ids, components, strict=True)
            ]
        ),
        row_evidence_root_sha256=_canonical_sha256(list(evidence)),
        row_count=len(sample_ids),
        component_count=len(set(components)),
        _factory_token=_V2_POPULATION_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ShortcutV2ToolFrequency:
    tool_identities: tuple[str, ...]
    positive_count: int
    negative_count: int
    positive_component_mass: ExactRational
    negative_component_mass: ExactRational
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_TOOL_FREQUENCY_FACTORY_TOKEN:
            raise TypeError("v2 tool frequencies must be calibration-derived")
        if type(self.tool_identities) is not tuple:
            raise GVSShortcutError("v2 tool-frequency identity must be an exact tuple")
        for index, value in enumerate(self.tool_identities):
            _strict_identifier(value, label=f"tool_identities[{index}]")
        if self.tool_identities != tuple(sorted(self.tool_identities)):
            raise GVSShortcutError("v2 tool-frequency identity must be sorted")
        _strict_nonnegative_integer(self.positive_count, label="positive_count")
        _strict_nonnegative_integer(self.negative_count, label="negative_count")
        if self.positive_count + self.negative_count <= 0:
            raise GVSShortcutError("v2 tool-frequency row must have calibration support")
        if type(self.positive_component_mass) is not ExactRational or (
            type(self.negative_component_mass) is not ExactRational
        ):
            raise GVSShortcutError("v2 tool-frequency masses must be exact rationals")
        if (self.positive_count == 0) is not (self.positive_component_mass.numerator == 0) or (
            self.negative_count == 0
        ) is not (self.negative_component_mass.numerator == 0):
            raise GVSShortcutError("v2 tool-frequency counts and component masses disagree")
        if (
            self.positive_component_mass.numerator == 0
            and self.negative_component_mass.numerator == 0
        ):
            raise GVSShortcutError("v2 tool-frequency component mass must be positive")

    @property
    def score(self) -> ExactRational:
        positive = self.positive_component_mass.fraction
        negative = self.negative_component_mass.fraction
        return ExactRational.from_fraction(positive / (positive + negative))

    def to_record(self) -> dict[str, object]:
        return {
            "negative_count": self.negative_count,
            "negative_component_mass": self.negative_component_mass.to_record(),
            "positive_count": self.positive_count,
            "positive_component_mass": self.positive_component_mass.to_record(),
            "score": self.score.to_record(),
            "tool_identities": list(self.tool_identities),
        }


@dataclass(frozen=True, slots=True)
class ShortcutV2CalibrationRules:
    length_preference: str
    tool_frequency_table: tuple[ShortcutV2ToolFrequency, ...]
    calibration_row_root_sha256: str
    calibration_row_count: int
    schema_version: str = GVS_SHORTCUT_V2_RULES_SCHEMA_VERSION
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_RULES_FACTORY_TOKEN:
            raise TypeError("v2 shortcut rules must be calibration-derived")
        if self.schema_version != GVS_SHORTCUT_V2_RULES_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 shortcut-rules schema")
        if self.length_preference not in {"prefer_shorter", "prefer_longer", "neutral"}:
            raise GVSShortcutError("v2 length preference is invalid")
        if type(self.tool_frequency_table) is not tuple or not self.tool_frequency_table:
            raise GVSShortcutError("v2 tool-frequency table must be nonempty")
        if any(type(value) is not ShortcutV2ToolFrequency for value in self.tool_frequency_table):
            raise GVSShortcutError("v2 tool-frequency table contains an invalid row")
        keys = tuple(value.tool_identities for value in self.tool_frequency_table)
        if keys != tuple(sorted(set(keys))):
            raise GVSShortcutError("v2 tool-frequency rows must be unique and sorted")
        _strict_sha256(self.calibration_row_root_sha256, label="calibration_row_root_sha256")
        _strict_positive_integer(self.calibration_row_count, label="calibration_row_count")
        if sum(value.positive_count for value in self.tool_frequency_table) != (
            self.calibration_row_count
        ) or sum(value.negative_count for value in self.tool_frequency_table) != (
            self.calibration_row_count
        ):
            raise GVSShortcutError("v2 tool-frequency counts differ from calibration rows")
        if (
            sum(
                (value.positive_component_mass.fraction for value in self.tool_frequency_table),
                Fraction(),
            )
            != 1
            or sum(
                (value.negative_component_mass.fraction for value in self.tool_frequency_table),
                Fraction(),
            )
            != 1
        ):
            raise GVSShortcutError("v2 tool-frequency masses are not component-normalized")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "calibration_row_count": self.calibration_row_count,
            "calibration_row_root_sha256": self.calibration_row_root_sha256,
            "length_preference": self.length_preference,
            "schema_version": self.schema_version,
            "tool_frequency_table": [value.to_record() for value in self.tool_frequency_table],
        }


def _score_v2_candidate(
    control: ShortcutControl,
    candidate: ShortcutV2CandidateSignal,
    rules: ShortcutV2CalibrationRules,
) -> ExactRational:
    if control is ShortcutControl.LENGTH_ONLY:
        if rules.length_preference == "prefer_shorter":
            value = Fraction(1, candidate.content_token_count + 1)
        elif rules.length_preference == "prefer_longer":
            value = Fraction(
                candidate.content_token_count,
                candidate.content_token_count + 1,
            )
        else:
            value = SHORTCUT_TIE_CREDIT
        return ExactRational.from_fraction(value)
    if control is ShortcutControl.SYNTAX_ONLY:
        return ExactRational(1 if candidate.syntax_parse_valid else 0, 1)
    if control is ShortcutControl.VALIDITY_ONLY:
        return ExactRational(
            1 if candidate.validity_schema_valid_nontruncated else 0,
            1,
        )
    if control is ShortcutControl.TOOL_FREQUENCY:
        table = {value.tool_identities: value.score for value in rules.tool_frequency_table}
        return table.get(
            candidate.tool_identities,
            ExactRational.from_fraction(SHORTCUT_TIE_CREDIT),
        )
    raise GVSShortcutError("unknown v2 shortcut control")


def _pair_score(
    *,
    component_id: str,
    sample_id: str,
    positive: ShortcutV2CandidateSignal,
    negative: ShortcutV2CandidateSignal,
    control: ShortcutControl,
    rules: ShortcutV2CalibrationRules,
) -> ShortcutPairScore:
    return ShortcutPairScore(
        component_id=component_id,
        sample_id=sample_id,
        positive_candidate_sha256=positive.semantic_identity_sha256,
        negative_candidate_sha256=negative.semantic_identity_sha256,
        positive_score=_score_v2_candidate(control, positive, rules),
        negative_score=_score_v2_candidate(control, negative, rules),
    )


def _v2_calibration_pairs(
    rows: Sequence[ShortcutV2CalibrationRowEvidence],
    *,
    control: ShortcutControl,
    rules: ShortcutV2CalibrationRules,
) -> tuple[ShortcutPairScore, ...]:
    pairs: list[ShortcutPairScore] = []
    for evidence in rows:
        candidates = {value.semantic_identity_sha256: value for value in evidence.row.candidates}
        pairs.append(
            _pair_score(
                component_id=evidence.row.component_id,
                sample_id=evidence.row.sample_id,
                positive=candidates[evidence.positive_semantic_identity_sha256],
                negative=candidates[evidence.negative_semantic_identity_sha256],
                control=control,
                rules=rules,
            )
        )
    return tuple(pairs)


def _v2_evaluation_pairs(
    rows: Sequence[ShortcutV2RowEvidence],
    *,
    control: ShortcutControl,
    rules: ShortcutV2CalibrationRules,
) -> tuple[ShortcutPairScore, ...]:
    pairs: list[ShortcutPairScore] = []
    for row in rows:
        for positive in row.positives:
            for negative in row.negatives:
                pairs.append(
                    _pair_score(
                        component_id=row.component_id,
                        sample_id=row.sample_id,
                        positive=positive,
                        negative=negative,
                        control=control,
                        rules=rules,
                    )
                )
                if len(pairs) > MAX_SHORTCUT_V2_PAIRS:
                    raise GVSShortcutError("v2 evaluation pair bound exceeded")
    if not pairs:
        raise GVSShortcutError("v2 D-support evaluation has no contrast pairs")
    return tuple(pairs)


def _component_balanced_tool_statistics(
    observations: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> dict[tuple[str, ...], tuple[int, int, Fraction, Fraction]]:
    """Give every component unit positive and negative mass before tool scoring."""

    by_component: dict[
        str,
        list[tuple[tuple[str, ...], tuple[str, ...]]],
    ] = defaultdict(list)
    for component_id, positive_tools, negative_tools in observations:
        by_component[component_id].append((positive_tools, negative_tools))
    if not by_component:
        raise GVSShortcutError("v2 tool statistics require calibration observations")
    component_count = len(by_component)
    counts: dict[tuple[str, ...], list[int | Fraction]] = defaultdict(
        lambda: [0, 0, Fraction(), Fraction()]
    )
    for component_id in sorted(by_component):
        component_rows = by_component[component_id]
        sample_mass = Fraction(1, component_count * len(component_rows))
        for positive_tools, negative_tools in component_rows:
            counts[positive_tools][0] += 1
            counts[negative_tools][1] += 1
            counts[positive_tools][2] += sample_mass
            counts[negative_tools][3] += sample_mass
    return {
        key: (
            int(value[0]),
            int(value[1]),
            Fraction(value[2]),
            Fraction(value[3]),
        )
        for key, value in counts.items()
    }


def _build_v2_rules(
    rows: Sequence[ShortcutV2CalibrationRowEvidence],
) -> ShortcutV2CalibrationRules:
    if type(rows) not in {list, tuple} or not rows:
        raise GVSShortcutError("v2 rules require complete T-new calibration rows")
    observations: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for evidence in rows:
        candidates = {value.semantic_identity_sha256: value for value in evidence.row.candidates}
        positive = candidates[evidence.positive_semantic_identity_sha256]
        negative = candidates[evidence.negative_semantic_identity_sha256]
        observations.append(
            (evidence.row.component_id, positive.tool_identities, negative.tool_identities)
        )
    statistics = _component_balanced_tool_statistics(observations)
    table = tuple(
        ShortcutV2ToolFrequency(
            tool_identities=key,
            positive_count=statistics[key][0],
            negative_count=statistics[key][1],
            positive_component_mass=ExactRational.from_fraction(statistics[key][2]),
            negative_component_mass=ExactRational.from_fraction(statistics[key][3]),
            _factory_token=_V2_TOOL_FREQUENCY_FACTORY_TOKEN,
        )
        for key in sorted(statistics)
    )

    def candidate_rules(preference: str) -> ShortcutV2CalibrationRules:
        return ShortcutV2CalibrationRules(
            length_preference=preference,
            tool_frequency_table=table,
            calibration_row_root_sha256=_canonical_sha256(
                [value.sha256 for value in sorted(rows, key=lambda row: row.row.sample_id)]
            ),
            calibration_row_count=len(rows),
            _factory_token=_V2_RULES_FACTORY_TOKEN,
        )

    candidates = {
        preference: candidate_rules(preference)
        for preference in ("prefer_shorter", "prefer_longer", "neutral")
    }
    accuracies = {
        preference: compute_component_balanced_pair_accuracy(
            _v2_calibration_pairs(
                rows,
                control=ShortcutControl.LENGTH_ONLY,
                rules=rules,
            )
        ).accuracy.fraction
        for preference, rules in candidates.items()
    }
    if accuracies["prefer_shorter"] > max(accuracies["prefer_longer"], accuracies["neutral"]):
        selected = "prefer_shorter"
    elif accuracies["prefer_longer"] > max(accuracies["prefer_shorter"], accuracies["neutral"]):
        selected = "prefer_longer"
    else:
        selected = "neutral"
    return candidates[selected]


def _assert_v2_cross_role_live_disjoint(
    t_rows: Sequence[ShortcutV2CalibrationRowEvidence],
    d_rows: Sequence[ShortcutV2RowEvidence],
) -> None:
    """Reject live T/D evidence that metadata-only duplicate claims failed to expose."""

    t_base_rows = tuple(value.row for value in t_rows)
    d_base_rows = tuple(d_rows)
    for field, label in (
        ("exact_request_sha256", "exact request"),
        ("prompt_sha256", "rendered prompt"),
        ("candidate_set_trace_sha256", "candidate trace"),
        ("candidate_set_analysis_sha256", "candidate analysis"),
        ("program_sha256", "program"),
        ("source_commitment_sha256", "source commitment"),
    ):
        overlap = {getattr(value, field) for value in t_base_rows}.intersection(
            getattr(value, field) for value in d_base_rows
        )
        if overlap:
            raise GVSShortcutError(
                f"v2 T-new/D-support live {label} evidence overlaps across roles"
            )


@dataclass(frozen=True, slots=True)
class ShortcutV2ControlResult:
    control: ShortcutControl
    calibration_status: ShortcutEvaluationStatus
    evaluation_status: ShortcutEvaluationStatus
    calibration_pair_accuracy: ComponentBalancedPairAccuracy | None
    evaluation_pair_accuracy: ComponentBalancedPairAccuracy | None
    rules_sha256: str | None
    missing_bindings: tuple[str, ...]
    schema_version: str = GVS_SHORTCUT_V2_CONTROL_RESULT_SCHEMA_VERSION
    calibration_population: str = SHORTCUT_CALIBRATION_POPULATION
    evaluation_population: str = SHORTCUT_EVALUATION_POPULATION
    external_custody_authenticated: bool = False
    scientific_result_available: bool = False
    passes_gate: None = None
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    launch_authorized: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_CONTROL_FACTORY_TOKEN:
            raise TypeError("v2 control results must be audit-derived")
        if self.schema_version != GVS_SHORTCUT_V2_CONTROL_RESULT_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 control-result schema")
        if type(self.control) is not ShortcutControl:
            raise GVSShortcutError("v2 control has the wrong exact type")
        if self.calibration_population != SHORTCUT_CALIBRATION_POPULATION or (
            self.evaluation_population != SHORTCUT_EVALUATION_POPULATION
        ):
            raise GVSShortcutError("v2 control population roles changed")
        if (
            type(self.calibration_status) is not ShortcutEvaluationStatus
            or type(self.evaluation_status) is not ShortcutEvaluationStatus
        ):
            raise GVSShortcutError("v2 control status has the wrong type")
        evaluated = self.evaluation_status is ShortcutEvaluationStatus.EVALUATED
        if evaluated is not (self.calibration_status is ShortcutEvaluationStatus.EVALUATED):
            raise GVSShortcutError("v2 calibration/evaluation statuses must advance together")
        if evaluated:
            if (
                type(self.calibration_pair_accuracy) is not ComponentBalancedPairAccuracy
                or type(self.evaluation_pair_accuracy) is not ComponentBalancedPairAccuracy
            ):
                raise GVSShortcutError("evaluated v2 controls require both exact accuracies")
            if self.rules_sha256 is None:
                raise GVSShortcutError("evaluated v2 controls require calibrated rules")
            _strict_sha256(self.rules_sha256, label="rules_sha256")
        elif (
            self.calibration_pair_accuracy is not None
            or self.evaluation_pair_accuracy is not None
            or self.rules_sha256 is not None
        ):
            raise GVSShortcutError("not_evaluated v2 controls cannot serialize results")
        if type(self.missing_bindings) is not tuple or not self.missing_bindings:
            raise GVSShortcutError("v2 controls require explicit blockers")
        for index, value in enumerate(self.missing_bindings):
            _strict_identifier(value, label=f"missing_bindings[{index}]")
        if self.missing_bindings != tuple(sorted(set(self.missing_bindings))):
            raise GVSShortcutError("v2 control blockers must be unique and sorted")
        if _EXTERNAL_CUSTODY_BLOCKER not in self.missing_bindings:
            raise GVSShortcutError("v2 control must expose the external-custody blocker")
        if self.external_custody_authenticated is not False or (
            self.scientific_result_available is not False
        ):
            raise GVSShortcutError("v2 structural controls cannot certify science or custody")
        if self.passes_gate is not None:
            raise GVSShortcutError("v2 shortcut controls are not authorization gates")
        for name in (
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "launch_authorized",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("v2 control result must remain nonauthorizing")

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "calibration_pair_accuracy": (
                None
                if self.calibration_pair_accuracy is None
                else self.calibration_pair_accuracy.to_record()
            ),
            "calibration_population": self.calibration_population,
            "calibration_status": self.calibration_status.value,
            "control": self.control.value,
            "evaluation_pair_accuracy": (
                None
                if self.evaluation_pair_accuracy is None
                else self.evaluation_pair_accuracy.to_record()
            ),
            "evaluation_population": self.evaluation_population,
            "evaluation_status": self.evaluation_status.value,
            "external_custody_authenticated": self.external_custody_authenticated,
            "launch_authorized": self.launch_authorized,
            "missing_bindings": list(self.missing_bindings),
            "passes_gate": self.passes_gate,
            "rules_sha256": self.rules_sha256,
            "schema_version": self.schema_version,
            "scientific_result_available": self.scientific_result_available,
        }


@dataclass(frozen=True, slots=True)
class ShortcutV2AuditReceipt:
    t_new_population_evidence: ShortcutV2PopulationEvidence | None
    d_support_population_evidence: ShortcutV2PopulationEvidence | None
    t_new_rows: tuple[ShortcutV2CalibrationRowEvidence, ...]
    d_support_rows: tuple[ShortcutV2RowEvidence, ...]
    rules: ShortcutV2CalibrationRules | None
    controls: tuple[ShortcutV2ControlResult, ...]
    blockers: tuple[str, ...]
    source_files_sha256: str
    runtime_identity_sha256: str
    schema_version: str = GVS_SHORTCUT_V2_AUDIT_SCHEMA_VERSION
    complete_t_new_calibration_supplied: bool = False
    complete_d_support_evaluation_supplied: bool = False
    d_support_contrast_complete: bool = False
    structural_result_available: bool = False
    external_custody_authenticated: bool = False
    scientific_result_available: bool = False
    authorizes_model_or_label_access: bool = False
    authorizes_cuda_or_jarvis_access: bool = False
    launch_authorized: bool = False
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _V2_AUDIT_FACTORY_TOKEN:
            raise TypeError("v2 audit receipts must be built from live evidence")
        if self.schema_version != GVS_SHORTCUT_V2_AUDIT_SCHEMA_VERSION:
            raise GVSShortcutError("unsupported v2 shortcut-audit schema")
        for name in ("source_files_sha256", "runtime_identity_sha256"):
            _strict_sha256(getattr(self, name), label=name)
        if self.source_files_sha256 != _INITIAL_SOURCE_FILES_SHA256 or (
            self.runtime_identity_sha256 != _INITIAL_RUNTIME_SHA256
        ):
            raise GVSShortcutError("v2 audit runtime differs from its import baseline")
        if type(self.t_new_rows) is not tuple or any(
            type(value) is not ShortcutV2CalibrationRowEvidence for value in self.t_new_rows
        ):
            raise GVSShortcutError("v2 audit T-new rows are invalid")
        if type(self.d_support_rows) is not tuple or any(
            type(value) is not ShortcutV2RowEvidence for value in self.d_support_rows
        ):
            raise GVSShortcutError("v2 audit D-support rows are invalid")
        if tuple(value.row.sample_id for value in self.t_new_rows) != tuple(
            sorted(value.row.sample_id for value in self.t_new_rows)
        ) or tuple(value.sample_id for value in self.d_support_rows) != tuple(
            sorted(value.sample_id for value in self.d_support_rows)
        ):
            raise GVSShortcutError("v2 audit rows must be sorted by sample ID")
        t_complete = self.t_new_population_evidence is not None
        d_complete = self.d_support_population_evidence is not None
        if t_complete and type(self.t_new_population_evidence) is not ShortcutV2PopulationEvidence:
            raise GVSShortcutError("v2 T-new population evidence has the wrong exact type")
        if (
            d_complete
            and type(self.d_support_population_evidence) is not ShortcutV2PopulationEvidence
        ):
            raise GVSShortcutError("v2 D-support population evidence has the wrong exact type")
        if self.complete_t_new_calibration_supplied is not t_complete or (
            self.complete_d_support_evaluation_supplied is not d_complete
        ):
            raise GVSShortcutError("v2 supplied-evidence flags failed recomputation")
        if t_complete is not bool(self.t_new_rows) or d_complete is not bool(self.d_support_rows):
            raise GVSShortcutError("v2 population receipts and rows must be jointly present")
        if t_complete:
            assert self.t_new_population_evidence is not None
            t_population = self.t_new_population_evidence
            if t_population.population_role != SHORTCUT_CALIBRATION_POPULATION:
                raise GVSShortcutError("v2 T-new population evidence has the wrong role")
            if (
                t_population.sample_ids != tuple(value.row.sample_id for value in self.t_new_rows)
                or t_population.component_ids
                != tuple(value.row.component_id for value in self.t_new_rows)
                or t_population.population_record_sha256s
                != tuple(value.row.population_record_sha256 for value in self.t_new_rows)
                or t_population.row_evidence_sha256s
                != tuple(value.sha256 for value in self.t_new_rows)
            ):
                raise GVSShortcutError("v2 T-new population receipt differs from its live rows")
        if d_complete:
            assert self.d_support_population_evidence is not None
            d_population = self.d_support_population_evidence
            if d_population.population_role != SHORTCUT_EVALUATION_POPULATION:
                raise GVSShortcutError("v2 D-support population evidence has the wrong role")
            if (
                d_population.sample_ids != tuple(value.sample_id for value in self.d_support_rows)
                or d_population.component_ids
                != tuple(value.component_id for value in self.d_support_rows)
                or d_population.population_record_sha256s
                != tuple(value.population_record_sha256 for value in self.d_support_rows)
                or d_population.row_evidence_sha256s
                != tuple(value.sha256 for value in self.d_support_rows)
            ):
                raise GVSShortcutError("v2 D-support population receipt differs from its live rows")
        if t_complete and d_complete:
            assert self.t_new_population_evidence is not None
            assert self.d_support_population_evidence is not None
            if (
                self.t_new_population_evidence.firewall_sha256
                != self.d_support_population_evidence.firewall_sha256
                or self.t_new_population_evidence.scan_evidence_sha256
                != self.d_support_population_evidence.scan_evidence_sha256
            ):
                raise GVSShortcutError("v2 T-new and D-support use different joint firewalls")
            _assert_v2_cross_role_live_disjoint(self.t_new_rows, self.d_support_rows)
        contrast_complete = d_complete and all(
            value.has_contrast_pair for value in self.d_support_rows
        )
        if self.d_support_contrast_complete is not contrast_complete:
            raise GVSShortcutError("v2 D-support contrast completeness failed recomputation")
        structural = t_complete and d_complete and contrast_complete
        if self.structural_result_available is not structural:
            raise GVSShortcutError("v2 structural-result flag failed recomputation")
        if structural is not (self.rules is not None):
            raise GVSShortcutError("v2 rules require both complete structural populations")
        if self.rules is not None:
            if type(self.rules) is not ShortcutV2CalibrationRules:
                raise GVSShortcutError("v2 rules have the wrong exact type")
            expected_rules = _build_v2_rules(self.t_new_rows)
            if self.rules.to_record() != expected_rules.to_record():
                raise GVSShortcutError("v2 rules differ from live calibration recomputation")
        if type(self.controls) is not tuple or tuple(
            value.control for value in self.controls
        ) != tuple(ShortcutControl):
            raise GVSShortcutError("v2 audit must contain the exact four controls")
        expected_status = (
            ShortcutEvaluationStatus.EVALUATED
            if structural
            else ShortcutEvaluationStatus.NOT_EVALUATED
        )
        if any(
            value.calibration_status is not expected_status
            or value.evaluation_status is not expected_status
            for value in self.controls
        ):
            raise GVSShortcutError("v2 control status differs from structural availability")
        if type(self.blockers) is not tuple or self.blockers != tuple(sorted(set(self.blockers))):
            raise GVSShortcutError("v2 audit blockers must be unique and sorted")
        expected_blockers = {_EXTERNAL_CUSTODY_BLOCKER}
        if not t_complete:
            expected_blockers.add(_T_NEW_INPUT_BLOCKER)
        if not d_complete:
            expected_blockers.add(_D_SUPPORT_INPUT_BLOCKER)
        elif not contrast_complete:
            expected_blockers.add(_D_SUPPORT_CONTRAST_BLOCKER)
        if self.blockers != tuple(sorted(expected_blockers)):
            raise GVSShortcutError("v2 audit blockers differ from live completeness state")
        for index, blocker in enumerate(self.blockers):
            _strict_identifier(blocker, label=f"blockers[{index}]")
        if any(value.missing_bindings != self.blockers for value in self.controls):
            raise GVSShortcutError("v2 control blockers differ from the audit blockers")
        if self.rules is not None:
            for result in self.controls:
                expected_calibration = compute_component_balanced_pair_accuracy(
                    _v2_calibration_pairs(
                        self.t_new_rows,
                        control=result.control,
                        rules=self.rules,
                    )
                )
                expected_evaluation = compute_component_balanced_pair_accuracy(
                    _v2_evaluation_pairs(
                        self.d_support_rows,
                        control=result.control,
                        rules=self.rules,
                    )
                )
                if (
                    result.rules_sha256 != self.rules.sha256
                    or result.calibration_pair_accuracy != expected_calibration
                    or result.evaluation_pair_accuracy != expected_evaluation
                ):
                    raise GVSShortcutError(
                        "v2 control result differs from live score recomputation"
                    )
        for name in (
            "external_custody_authenticated",
            "scientific_result_available",
            "authorizes_model_or_label_access",
            "authorizes_cuda_or_jarvis_access",
            "launch_authorized",
        ):
            if getattr(self, name) is not False:
                raise GVSShortcutError("v2 audit must remain nonauthorizing")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "authorizes_cuda_or_jarvis_access": self.authorizes_cuda_or_jarvis_access,
            "authorizes_model_or_label_access": self.authorizes_model_or_label_access,
            "blockers": list(self.blockers),
            "complete_d_support_evaluation_supplied": (self.complete_d_support_evaluation_supplied),
            "complete_t_new_calibration_supplied": (self.complete_t_new_calibration_supplied),
            "controls": [value.to_record() for value in self.controls],
            "d_support_contrast_complete": self.d_support_contrast_complete,
            "d_support_population_evidence": (
                None
                if self.d_support_population_evidence is None
                else self.d_support_population_evidence.to_record()
            ),
            "d_support_rows": [value.to_record() for value in self.d_support_rows],
            "external_custody_authenticated": self.external_custody_authenticated,
            "feature_contract": _feature_contract()["v2_contract"],
            "launch_authorized": self.launch_authorized,
            "rules": None if self.rules is None else self.rules.to_record(),
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "schema_version": self.schema_version,
            "scientific_result_available": self.scientific_result_available,
            "source_files_sha256": self.source_files_sha256,
            "structural_result_available": self.structural_result_available,
            "t_new_population_evidence": (
                None
                if self.t_new_population_evidence is None
                else self.t_new_population_evidence.to_record()
            ),
            "t_new_rows": [value.to_record() for value in self.t_new_rows],
        }

    def to_json_bytes(self) -> bytes:
        payload = _canonical_bytes(
            {
                "artifact_schema_version": GVS_SHORTCUT_V2_ARTIFACT_SCHEMA_VERSION,
                "receipt": self.to_record(),
                "receipt_sha256": self.sha256,
            }
        )
        if len(payload) > MAX_SHORTCUT_ARTIFACT_BYTES:
            raise GVSShortcutError("v2 shortcut artifact exceeds its byte bound")
        return payload


def _complete_v2_bridge_population(
    values: Sequence[CandidateSupportBridgeInput],
    *,
    expected: Sequence[SupportSampleIdentity],
    label: str,
) -> tuple[CandidateSupportBridgeInput, ...]:
    if type(values) not in {list, tuple} or not values or len(values) > MAX_SHORTCUT_ROWS:
        raise GVSShortcutError(f"{label} inputs must be a nonempty bounded exact sequence")
    first = tuple(values)
    second = tuple(values)
    if tuple(id(value) for value in first) != tuple(id(value) for value in second):
        raise GVSShortcutError(f"{label} inputs changed during their stable snapshot")
    if any(type(value) is not _BRIDGE_INPUT_CLASS for value in first):
        raise GVSShortcutError(f"{label} inputs contain an invalid bridge input")
    by_sample = {value.sample.sample_id: value for value in first}
    if len(by_sample) != len(first):
        raise GVSShortcutError(f"{label} inputs contain duplicate sample IDs")
    expected_by_sample = {value.sample_id: value for value in expected}
    if set(by_sample) != set(expected_by_sample):
        raise GVSShortcutError(f"{label} inputs differ from the complete firewall denominator")
    for sample_id, bridge_input in by_sample.items():
        if bridge_input.sample.to_record() != expected_by_sample[sample_id].to_record():
            raise GVSShortcutError(f"{label} sample identity differs from the firewall")
    return tuple(by_sample[sample_id] for sample_id in sorted(by_sample))


def _complete_v2_calibration_population(
    values: Sequence[TNewShortcutCalibrationInput],
    *,
    expected: Sequence[SupportSampleIdentity],
) -> tuple[TNewShortcutCalibrationInput, ...]:
    if type(values) not in {list, tuple} or not values or len(values) > MAX_SHORTCUT_ROWS:
        raise GVSShortcutError("T-new inputs must be a nonempty bounded exact sequence")
    first = tuple(values)
    second = tuple(values)
    if tuple(id(value) for value in first) != tuple(id(value) for value in second):
        raise GVSShortcutError("T-new inputs changed during their stable snapshot")
    if any(type(value) is not TNewShortcutCalibrationInput for value in first):
        raise GVSShortcutError("T-new inputs contain an invalid calibration input")
    bridge_inputs = _complete_v2_bridge_population(
        tuple(value.bridge_input for value in first),
        expected=expected,
        label="T-new",
    )
    by_sample = {value.bridge_input.sample.sample_id: value for value in first}
    return tuple(by_sample[value.sample.sample_id] for value in bridge_inputs)


def build_shortcut_v2_audit(
    *,
    t_new_inputs: Sequence[TNewShortcutCalibrationInput] | None = None,
    d_support_inputs: Sequence[CandidateSupportBridgeInput] | None = None,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> ShortcutV2AuditReceipt:
    """Build the CPU-only v2 control receipt from exact complete live populations."""

    assert_shortcut_runtime_integrity()
    identities = derive_shortcut_v2_population_identities(
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )

    t_rows: tuple[ShortcutV2CalibrationRowEvidence, ...] = ()
    t_population: ShortcutV2PopulationEvidence | None = None
    if t_new_inputs is not None:
        t_values = _complete_v2_calibration_population(
            t_new_inputs,
            expected=identities[SHORTCUT_CALIBRATION_POPULATION],
        )
        t_rows = tuple(_derive_v2_calibration_row(value) for value in t_values)
        t_population = _build_v2_population_evidence(
            population_role=SHORTCUT_CALIBRATION_POPULATION,
            identities=identities[SHORTCUT_CALIBRATION_POPULATION],
            rows=tuple(value.row for value in t_rows),
            row_evidence_sha256s=tuple(value.sha256 for value in t_rows),
        )

    d_rows: tuple[ShortcutV2RowEvidence, ...] = ()
    d_population: ShortcutV2PopulationEvidence | None = None
    if d_support_inputs is not None:
        d_values = _complete_v2_bridge_population(
            d_support_inputs,
            expected=identities[SHORTCUT_EVALUATION_POPULATION],
            label="D-support",
        )
        d_rows = tuple(derive_shortcut_v2_row(value) for value in d_values)
        d_population = _build_v2_population_evidence(
            population_role=SHORTCUT_EVALUATION_POPULATION,
            identities=identities[SHORTCUT_EVALUATION_POPULATION],
            rows=d_rows,
            row_evidence_sha256s=tuple(value.sha256 for value in d_rows),
        )

    contrast_complete = d_population is not None and all(
        value.has_contrast_pair for value in d_rows
    )
    structural = t_population is not None and d_population is not None and contrast_complete
    blockers = {_EXTERNAL_CUSTODY_BLOCKER}
    if t_population is None:
        blockers.add(_T_NEW_INPUT_BLOCKER)
    if d_population is None:
        blockers.add(_D_SUPPORT_INPUT_BLOCKER)
    elif not contrast_complete:
        blockers.add(_D_SUPPORT_CONTRAST_BLOCKER)
    ordered_blockers = tuple(sorted(blockers))

    rules = _build_v2_rules(t_rows) if structural else None
    controls: list[ShortcutV2ControlResult] = []
    for control in ShortcutControl:
        if rules is None:
            calibration_accuracy = None
            evaluation_accuracy = None
            status = ShortcutEvaluationStatus.NOT_EVALUATED
            rules_sha256 = None
        else:
            calibration_accuracy = compute_component_balanced_pair_accuracy(
                _v2_calibration_pairs(t_rows, control=control, rules=rules)
            )
            evaluation_accuracy = compute_component_balanced_pair_accuracy(
                _v2_evaluation_pairs(d_rows, control=control, rules=rules)
            )
            status = ShortcutEvaluationStatus.EVALUATED
            rules_sha256 = rules.sha256
        controls.append(
            ShortcutV2ControlResult(
                control=control,
                calibration_status=status,
                evaluation_status=status,
                calibration_pair_accuracy=calibration_accuracy,
                evaluation_pair_accuracy=evaluation_accuracy,
                rules_sha256=rules_sha256,
                missing_bindings=ordered_blockers,
                _factory_token=_V2_CONTROL_FACTORY_TOKEN,
            )
        )
    receipt = ShortcutV2AuditReceipt(
        t_new_population_evidence=t_population,
        d_support_population_evidence=d_population,
        t_new_rows=t_rows,
        d_support_rows=d_rows,
        rules=rules,
        controls=tuple(controls),
        blockers=ordered_blockers,
        source_files_sha256=_INITIAL_SOURCE_FILES_SHA256,
        runtime_identity_sha256=_INITIAL_RUNTIME_SHA256,
        complete_t_new_calibration_supplied=t_population is not None,
        complete_d_support_evaluation_supplied=d_population is not None,
        d_support_contrast_complete=contrast_complete,
        structural_result_available=structural,
        _factory_token=_V2_AUDIT_FACTORY_TOKEN,
    )
    assert_shortcut_runtime_integrity()
    return receipt


def load_shortcut_v2_audit_json(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    t_new_inputs: Sequence[TNewShortcutCalibrationInput] | None = None,
    d_support_inputs: Sequence[CandidateSupportBridgeInput] | None = None,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> ShortcutV2AuditReceipt:
    """Strict-load and then replace serialized claims with complete live recomputation."""

    assert_shortcut_runtime_integrity()
    expected_receipt_sha256 = _strict_sha256(
        expected_receipt_sha256,
        label="expected_receipt_sha256",
    )
    if type(raw) is not bytes or not raw or len(raw) > MAX_SHORTCUT_ARTIFACT_BYTES:
        raise GVSShortcutError("v2 shortcut artifact is empty, non-bytes, or oversized")
    try:
        artifact = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_bounded_json_integer,
        )
    except GVSShortcutError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise GVSShortcutError("v2 shortcut artifact is not strict bounded JSON") from error
    artifact = _exact_object(
        artifact,
        frozenset({"artifact_schema_version", "receipt", "receipt_sha256"}),
        label="v2 shortcut artifact",
    )
    if artifact["artifact_schema_version"] != GVS_SHORTCUT_V2_ARTIFACT_SCHEMA_VERSION:
        raise GVSShortcutError("unsupported v2 shortcut artifact schema")
    embedded = _strict_sha256(artifact["receipt_sha256"], label="receipt_sha256")
    if embedded != _canonical_sha256(artifact["receipt"]) or (embedded != expected_receipt_sha256):
        raise GVSShortcutError("v2 shortcut hash failed caller-bound recomputation")
    rebuilt = build_shortcut_v2_audit(
        t_new_inputs=t_new_inputs,
        d_support_inputs=d_support_inputs,
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    if artifact["receipt"] != rebuilt.to_record() or embedded != rebuilt.sha256:
        raise GVSShortcutError("v2 shortcut artifact differs from complete live recomputation")
    assert_shortcut_runtime_integrity()
    return rebuilt


__all__ = [
    "GVS_SHORTCUT_ARTIFACT_SCHEMA_VERSION",
    "GVS_SHORTCUT_AUDIT_SCHEMA_VERSION",
    "GVS_SHORTCUT_V2_ARTIFACT_SCHEMA_VERSION",
    "GVS_SHORTCUT_V2_AUDIT_SCHEMA_VERSION",
    "PROVISIONAL_SHORTCUT_PAIR_ACCURACY_REFERENCE",
    "SHORTCUT_CALIBRATION_POPULATION",
    "SHORTCUT_EVALUATION_POPULATION",
    "ComponentBalancedPairAccuracy",
    "DSupportShortcutEvidence",
    "ExactRational",
    "GVSShortcutError",
    "ShortcutAuditReceipt",
    "ShortcutCandidateSignal",
    "ShortcutControl",
    "ShortcutControlStatus",
    "ShortcutEvaluationStatus",
    "ShortcutPairScore",
    "ShortcutRowEvidence",
    "ShortcutV2AuditReceipt",
    "ShortcutV2CalibrationRowEvidence",
    "ShortcutV2CalibrationRules",
    "ShortcutV2CandidateSignal",
    "ShortcutV2ControlResult",
    "ShortcutV2PopulationEvidence",
    "ShortcutV2RowEvidence",
    "ShortcutV2ToolFrequency",
    "TNewShortcutCalibrationInput",
    "assert_shortcut_runtime_integrity",
    "build_d_support_shortcut_evidence",
    "build_shortcut_audit",
    "build_shortcut_v2_audit",
    "compute_component_balanced_pair_accuracy",
    "derive_d_support_shortcut_row",
    "derive_shortcut_v2_population_identities",
    "derive_shortcut_v2_row",
    "load_shortcut_audit_json",
    "load_shortcut_v2_audit_json",
    "shortcut_runtime_sha256",
]


_INITIAL_SOURCE_FILES_SHA256 = _canonical_sha256(_source_files_record())
_INITIAL_RUNTIME_SHA256 = shortcut_runtime_sha256()
