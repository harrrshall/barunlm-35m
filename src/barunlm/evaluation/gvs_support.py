"""CPU-only candidate-support accounting for proposed BarunAction GVS-v1.

This module does not generate candidates, load a checkpoint, train a verifier, or
authorize an experiment.  It validates already-scored candidate-support records and
computes preliminary prototype support diagnostics with exact rational arithmetic.
Decoder-trace conversion intentionally lives in the separate GVS bridge and remains
subject to its own final integration audit. Callers cannot use these metadata/scoring
APIs as a decoder receipt, bypass that bridge, or substitute nullable placeholder
trace fields. No launch is authorized until the complete bridge receipt is frozen.

Every support population is derived from eligible ``D-support`` records in a validated
GVS structural firewall and exact scan artifact. Scoring repeats that derivation and
rejects fabricated identities, excluded rows, or unrelated hashes. Primary rates use
connected components with a conservative all-member conjunction; row and candidate-slot
diagnostics remain visible. Rank zero is the frozen greedy candidate; oracle support
means at least one schema-valid Action IR exact candidate among all eight ranks.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, NoReturn

from barunlm.config import BarunConfig

from . import gvs_population as gvs_population_module
from .gvs_population import (
    ACTION_FAMILIES,
    AUDITED_BOOLEAN_STRATA,
    EXPECTED_OUTCOMES,
    GVSPopulationError,
    validate_population_firewall,
    validate_scan_evidence,
)

GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION = "barun-gvs-candidate-support-v1"
GVS_SUPPORT_POPULATION_SCHEMA_VERSION = "barun-gvs-support-population-v1"
GVS_SUPPORT_SCORE_SCHEMA_VERSION = "barun-gvs-support-score-v1"
PROTOTYPE_SUPPORT_GATE_SCHEMA_VERSION = "barun-gvs-prototype-support-gate-v1"

FROZEN_CANDIDATE_COUNT = 8
PROTOTYPE_MINIMUM_ORACLE_PASS_AT_K = Fraction(17, 20)
PROTOTYPE_MINIMUM_ORACLE_GREEDY_GAIN = Fraction(1, 10)
PROTOTYPE_MINIMUM_GREEDY_FAILURE_RECOVERY = Fraction(1, 2)
PROTOTYPE_MINIMUM_SAFETY_SUPPORT = Fraction(4, 5)

_POPULATION_ERROR_CLASS = GVSPopulationError
_VALIDATE_POPULATION_FIREWALL = validate_population_firewall
_VALIDATE_SCAN_EVIDENCE = validate_scan_evidence
_PROTOTYPE_GATE_FACTORY_TOKEN = object()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_IDENTIFIER_LENGTH = 256
_MAX_RECORD_JSON_BYTES = 64 * 1024
_MAX_JSON_DEPTH = 64


class GVSSupportError(ValueError):
    """A GVS candidate-support contract is malformed or internally inconsistent."""


class SupportClass(str, Enum):
    """The role fixed for a support-population sample before candidate scoring."""

    EFFICACY = "efficacy"
    SAFETY = "safety"


def _assert_support_contract_integrity() -> None:
    """Reject post-import threshold or population-validator substitution."""

    threshold_contract = (
        ("candidate_count", FROZEN_CANDIDATE_COUNT, 8),
        (
            "minimum_oracle_pass_at_k",
            (
                PROTOTYPE_MINIMUM_ORACLE_PASS_AT_K.numerator,
                PROTOTYPE_MINIMUM_ORACLE_PASS_AT_K.denominator,
            ),
            (17, 20),
        ),
        (
            "minimum_oracle_minus_greedy",
            (
                PROTOTYPE_MINIMUM_ORACLE_GREEDY_GAIN.numerator,
                PROTOTYPE_MINIMUM_ORACLE_GREEDY_GAIN.denominator,
            ),
            (1, 10),
        ),
        (
            "minimum_greedy_failure_recovery",
            (
                PROTOTYPE_MINIMUM_GREEDY_FAILURE_RECOVERY.numerator,
                PROTOTYPE_MINIMUM_GREEDY_FAILURE_RECOVERY.denominator,
            ),
            (1, 2),
        ),
        (
            "minimum_safety_support",
            (
                PROTOTYPE_MINIMUM_SAFETY_SUPPORT.numerator,
                PROTOTYPE_MINIMUM_SAFETY_SUPPORT.denominator,
            ),
            (4, 5),
        ),
    )
    changed = [name for name, current, expected in threshold_contract if current != expected]
    if changed:
        raise GVSSupportError(
            "prototype support threshold contract changed after import: "
            + ", ".join(sorted(changed))
        )
    identity_contract = (
        ("GVSPopulationError", GVSPopulationError, _POPULATION_ERROR_CLASS),
        (
            "gvs_population.GVSPopulationError",
            gvs_population_module.GVSPopulationError,
            _POPULATION_ERROR_CLASS,
        ),
        (
            "validate_population_firewall",
            validate_population_firewall,
            _VALIDATE_POPULATION_FIREWALL,
        ),
        (
            "gvs_population.validate_population_firewall",
            gvs_population_module.validate_population_firewall,
            _VALIDATE_POPULATION_FIREWALL,
        ),
        ("validate_scan_evidence", validate_scan_evidence, _VALIDATE_SCAN_EVIDENCE),
        (
            "gvs_population.validate_scan_evidence",
            gvs_population_module.validate_scan_evidence,
            _VALIDATE_SCAN_EVIDENCE,
        ),
    )
    replaced = [name for name, current, expected in identity_contract if current is not expected]
    if replaced:
        raise GVSSupportError(
            "support population validation dependency changed after import: "
            + ", ".join(sorted(replaced))
        )


def _strict_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GVSSupportError(f"{label} must be a positive integer")
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise GVSSupportError(f"{label} must be a boolean")
    return value


def _strict_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        raise GVSSupportError(
            f"{label} must be a nonempty string of at most {_MAX_IDENTIFIER_LENGTH} characters"
        )
    if value != unicodedata.normalize("NFC", value):
        raise GVSSupportError(f"{label} must be NFC-normalized")
    if value != value.strip() or any(character in value for character in "\r\n\x00"):
        raise GVSSupportError(f"{label} contains forbidden whitespace or control characters")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GVSSupportError(f"{label} must be a lowercase SHA-256")
    return value


def _nullable_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _strict_sha256(value, label=label)


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise GVSSupportError(f"{label} must be an exact JSON object")
    assert isinstance(value, Mapping)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise GVSSupportError(f"{label} fields changed; missing={missing!r}, extra={extra!r}")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GVSSupportError("support evidence is not strict JSON") from error


def _detach_exact_json(
    value: object,
    *,
    label: str,
    active_container_ids: set[int] | None = None,
    depth: int = 0,
) -> object:
    """Snapshot exact built-in JSON once without tuple/list normalization."""

    if depth > _MAX_JSON_DEPTH:
        raise GVSSupportError(f"{label} exceeds maximum JSON depth {_MAX_JSON_DEPTH}")
    active = set() if active_container_ids is None else active_container_ids
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is list:
        container_id = id(value)
        if container_id in active:
            raise GVSSupportError(f"{label} contains a JSON container cycle")
        active.add(container_id)
        try:
            return [
                _detach_exact_json(
                    child,
                    label=f"{label}[]",
                    active_container_ids=active,
                    depth=depth + 1,
                )
                for child in value
            ]
        finally:
            active.remove(container_id)
    if type(value) is dict:
        container_id = id(value)
        if container_id in active:
            raise GVSSupportError(f"{label} contains a JSON container cycle")
        active.add(container_id)
        try:
            detached: dict[str, object] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise GVSSupportError(f"{label} contains a non-string JSON key")
                detached[key] = _detach_exact_json(
                    child,
                    label=f"{label}.{key}",
                    active_container_ids=active,
                    depth=depth + 1,
                )
            return detached
        finally:
            active.remove(container_id)
    raise GVSSupportError(f"{label} must contain only exact built-in JSON values")


def _fraction_record(value: Fraction) -> dict[str, int | str]:
    return {
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "numerator": value.numerator,
    }


@dataclass(frozen=True, slots=True)
class VerifierParameterAccounting:
    """Exact LoRA and scalar-head parameter counts derived from projection shapes."""

    rank: int
    layers: int
    input_dimension: int
    q_projection_output_dimension: int
    v_projection_output_dimension: int
    q_projection_lora_parameters: int
    v_projection_lora_parameters: int
    scalar_head_parameters: int
    verifier_parameters: int

    def __post_init__(self) -> None:
        for name in (
            "rank",
            "layers",
            "input_dimension",
            "q_projection_output_dimension",
            "v_projection_output_dimension",
            "q_projection_lora_parameters",
            "v_projection_lora_parameters",
            "scalar_head_parameters",
            "verifier_parameters",
        ):
            _strict_positive_integer(getattr(self, name), label=name)
        expected_q = (
            self.layers * self.rank * (self.input_dimension + self.q_projection_output_dimension)
        )
        expected_v = (
            self.layers * self.rank * (self.input_dimension + self.v_projection_output_dimension)
        )
        expected_head = self.input_dimension + 1
        if self.q_projection_lora_parameters != expected_q:
            raise GVSSupportError("q-projection LoRA count disagrees with its exact dimensions")
        if self.v_projection_lora_parameters != expected_v:
            raise GVSSupportError("v-projection LoRA count disagrees with its exact dimensions")
        if self.scalar_head_parameters != expected_head:
            raise GVSSupportError("scalar-head count disagrees with its exact dimensions")
        if self.verifier_parameters != expected_q + expected_v + expected_head:
            raise GVSSupportError("verifier parameter total disagrees with its components")

    def to_record(self) -> dict[str, int]:
        return {
            "input_dimension": self.input_dimension,
            "layers": self.layers,
            "q_projection_lora_parameters": self.q_projection_lora_parameters,
            "q_projection_output_dimension": self.q_projection_output_dimension,
            "rank": self.rank,
            "scalar_head_parameters": self.scalar_head_parameters,
            "v_projection_lora_parameters": self.v_projection_lora_parameters,
            "v_projection_output_dimension": self.v_projection_output_dimension,
            "verifier_parameters": self.verifier_parameters,
        }


@dataclass(frozen=True, slots=True)
class GVSSystemParameterAccounting:
    """One stored generator plus the distinct trainable verifier parameters."""

    generator_parameters: int
    verifier: VerifierParameterAccounting
    complete_system_parameters: int

    def __post_init__(self) -> None:
        _strict_positive_integer(self.generator_parameters, label="generator_parameters")
        _strict_positive_integer(
            self.complete_system_parameters, label="complete_system_parameters"
        )
        if not isinstance(self.verifier, VerifierParameterAccounting):
            raise GVSSupportError("verifier must be exact VerifierParameterAccounting")
        if self.complete_system_parameters != (
            self.generator_parameters + self.verifier.verifier_parameters
        ):
            raise GVSSupportError("complete system count disagrees with generator plus verifier")

    def to_record(self) -> dict[str, object]:
        return {
            "complete_system_parameters": self.complete_system_parameters,
            "generator_parameters": self.generator_parameters,
            "shared_backbone_stored_once": True,
            "verifier": self.verifier.to_record(),
        }


def count_verifier_parameters(
    config: BarunConfig,
    *,
    rank: int = 8,
) -> VerifierParameterAccounting:
    """Count rank-``rank`` LoRA on every q/v projection plus a biased scalar head."""

    if not isinstance(config, BarunConfig):
        raise GVSSupportError("config must be BarunConfig")
    rank = _strict_positive_integer(rank, label="rank")
    dimension = _strict_positive_integer(config.dim, label="config.dim")
    layers = _strict_positive_integer(config.n_layers, label="config.n_layers")
    heads = _strict_positive_integer(config.n_heads, label="config.n_heads")
    kv_heads = _strict_positive_integer(config.n_kv_heads, label="config.n_kv_heads")
    if dimension % heads:
        raise GVSSupportError("config.dim must be divisible by config.n_heads")
    if heads % kv_heads:
        raise GVSSupportError("config.n_heads must be divisible by config.n_kv_heads")

    head_dimension = dimension // heads
    q_output = heads * head_dimension
    v_output = kv_heads * head_dimension
    q_parameters = layers * rank * (dimension + q_output)
    v_parameters = layers * rank * (dimension + v_output)
    scalar_head = dimension + 1
    return VerifierParameterAccounting(
        rank=rank,
        layers=layers,
        input_dimension=dimension,
        q_projection_output_dimension=q_output,
        v_projection_output_dimension=v_output,
        q_projection_lora_parameters=q_parameters,
        v_projection_lora_parameters=v_parameters,
        scalar_head_parameters=scalar_head,
        verifier_parameters=q_parameters + v_parameters + scalar_head,
    )


def count_complete_gvs_system_parameters(
    config: BarunConfig,
    *,
    generator_parameters: int,
    rank: int = 8,
) -> GVSSystemParameterAccounting:
    """Count the complete unique system without double-counting the shared backbone."""

    generator_parameters = _strict_positive_integer(
        generator_parameters, label="generator_parameters"
    )
    verifier = count_verifier_parameters(config, rank=rank)
    return GVSSystemParameterAccounting(
        generator_parameters=generator_parameters,
        verifier=verifier,
        complete_system_parameters=generator_parameters + verifier.verifier_parameters,
    )


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One immutable rank in a candidate-support record."""

    rank: int
    candidate_id: str
    output_present: bool
    raw_output_sha256: str | None
    canonical_action_sha256: str | None
    parse_valid: bool
    schema_valid: bool
    action_ir_exact: bool
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise GVSSupportError("candidate rank must be a non-negative integer")
        _strict_identifier(self.candidate_id, label="candidate_id")
        _strict_bool(self.output_present, label="output_present")
        _nullable_sha256(self.raw_output_sha256, label="raw_output_sha256")
        _nullable_sha256(self.canonical_action_sha256, label="canonical_action_sha256")
        _strict_bool(self.parse_valid, label="parse_valid")
        _strict_bool(self.schema_valid, label="schema_valid")
        _strict_bool(self.action_ir_exact, label="action_ir_exact")
        _strict_bool(self.truncated, label="truncated")
        if self.output_present is not (self.raw_output_sha256 is not None):
            raise GVSSupportError(
                "output_present must agree exactly with nullable raw_output_sha256"
            )
        if self.parse_valid and not self.output_present:
            raise GVSSupportError("parse-valid candidate must have an output")
        if self.schema_valid and not self.parse_valid:
            raise GVSSupportError("schema-valid candidate must also be parse-valid")
        if self.schema_valid is not (self.canonical_action_sha256 is not None):
            raise GVSSupportError(
                "schema_valid must agree exactly with nullable canonical_action_sha256"
            )
        if self.action_ir_exact and not self.schema_valid:
            raise GVSSupportError("Action IR exact candidate must also be schema-valid")

    def to_record(self) -> dict[str, object]:
        return {
            "action_ir_exact": self.action_ir_exact,
            "candidate_id": self.candidate_id,
            "canonical_action_sha256": self.canonical_action_sha256,
            "output_present": self.output_present,
            "parse_valid": self.parse_valid,
            "rank": self.rank,
            "raw_output_sha256": self.raw_output_sha256,
            "schema_valid": self.schema_valid,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class SupportStrata:
    """All mandatory pre-outcome boolean strata in their canonical order."""

    values: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple:
            raise GVSSupportError("support strata values must be a tuple")
        expected_names = tuple(AUDITED_BOOLEAN_STRATA)
        actual_names = tuple(name for name, _ in self.values)
        if actual_names != expected_names:
            raise GVSSupportError("support strata names or order differ from the firewall contract")
        for name, value in self.values:
            _strict_identifier(name, label="strata name")
            _strict_bool(value, label=f"strata.{name}")

    @classmethod
    def from_mapping(cls, value: object) -> SupportStrata:
        expected = frozenset(AUDITED_BOOLEAN_STRATA)
        checked = _exact_keys(value, expected, label="support strata")
        return cls(
            tuple(
                (name, _strict_bool(checked[name], label=f"strata.{name}"))
                for name in AUDITED_BOOLEAN_STRATA
            )
        )

    def to_record(self) -> dict[str, bool]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class SupportSampleIdentity:
    """One eligible D-support identity derived from a validated structural firewall."""

    sample_id: str
    firewall_sha256: str
    scan_evidence_sha256: str
    population_record_sha256: str
    component_id: str
    source_commitment_sha256: str
    task_class: SupportClass
    expected_outcome: str
    action_family: str
    strata: SupportStrata

    def __post_init__(self) -> None:
        _strict_identifier(self.sample_id, label="sample_id")
        for name in (
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_sha256",
            "component_id",
            "source_commitment_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if not isinstance(self.task_class, SupportClass):
            raise GVSSupportError("task_class must be SupportClass")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise GVSSupportError("expected_outcome differs from the population contract")
        if self.action_family not in ACTION_FAMILIES:
            raise GVSSupportError("action_family differs from the population contract")
        if not isinstance(self.strata, SupportStrata):
            raise GVSSupportError("strata must be SupportStrata")

    def to_record(self) -> dict[str, object]:
        return {
            "action_family": self.action_family,
            "component_id": self.component_id,
            "expected_outcome": self.expected_outcome,
            "firewall_sha256": self.firewall_sha256,
            "population_record_sha256": self.population_record_sha256,
            "sample_id": self.sample_id,
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "source_commitment_sha256": self.source_commitment_sha256,
            "strata": self.strata.to_record(),
            "task_class": self.task_class.value,
        }


@dataclass(frozen=True, slots=True)
class SupportPopulation:
    """The exact expected denominator and safety subset for one support score."""

    schema_version: str
    candidate_count: int
    firewall_sha256: str
    scan_evidence_sha256: str
    samples: tuple[SupportSampleIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GVS_SUPPORT_POPULATION_SCHEMA_VERSION:
            raise GVSSupportError("unsupported GVS support population schema")
        if self.candidate_count != FROZEN_CANDIDATE_COUNT or type(self.candidate_count) is not int:
            raise GVSSupportError(
                f"GVS-v1 support population requires exactly K={FROZEN_CANDIDATE_COUNT}"
            )
        _strict_sha256(self.firewall_sha256, label="firewall_sha256")
        _strict_sha256(self.scan_evidence_sha256, label="scan_evidence_sha256")
        if type(self.samples) is not tuple or not self.samples:
            raise GVSSupportError("support population samples must be a nonempty tuple")
        if any(not isinstance(sample, SupportSampleIdentity) for sample in self.samples):
            raise GVSSupportError("support population contains an invalid sample identity")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise GVSSupportError("support population has duplicate sample IDs")
        record_hashes = [sample.population_record_sha256 for sample in self.samples]
        if len(record_hashes) != len(set(record_hashes)):
            raise GVSSupportError("support population has duplicate population record hashes")
        if any(sample.firewall_sha256 != self.firewall_sha256 for sample in self.samples):
            raise GVSSupportError("support identity is bound to a different firewall")
        if any(sample.scan_evidence_sha256 != self.scan_evidence_sha256 for sample in self.samples):
            raise GVSSupportError("support identity is bound to different scan evidence")
        if not any(sample.task_class is SupportClass.EFFICACY for sample in self.samples):
            raise GVSSupportError("support population requires at least one efficacy sample")
        if not any(sample.task_class is SupportClass.SAFETY for sample in self.samples):
            raise GVSSupportError("support population requires at least one safety sample")
        by_component: dict[str, tuple[object, ...]] = {}
        for sample in self.samples:
            classification = (
                sample.task_class,
                sample.expected_outcome,
                sample.action_family,
                sample.strata,
            )
            previous = by_component.setdefault(sample.component_id, classification)
            if previous != classification:
                raise GVSSupportError("support component mixes pre-outcome classifications")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "firewall_sha256": self.firewall_sha256,
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "samples": [
                sample.to_record() for sample in sorted(self.samples, key=lambda row: row.sample_id)
            ],
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record()).encode("utf-8")).hexdigest()


def derive_support_population(
    *,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> SupportPopulation:
    """Derive the only accepted support denominator from eligible D-support records."""

    _assert_support_contract_integrity()
    snapshot = _detach_exact_json(
        {
            "compared_release_cutoff_utc": compared_release_cutoff_utc,
            "expected_membership": expected_membership,
            "expected_scan_bindings": expected_scan_bindings,
            "firewall_frozen_at_utc": firewall_frozen_at_utc,
            "firewall_manifest": firewall_manifest,
            "population_records": population_records,
            "scan_evidence": scan_evidence,
            "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
        },
        label="support derivation inputs",
    )
    if type(snapshot) is not dict:
        raise GVSSupportError("support derivation snapshot must be an exact JSON object")
    _canonical_json(snapshot)
    population_records = snapshot["population_records"]  # type: ignore[assignment]
    expected_membership = snapshot["expected_membership"]  # type: ignore[assignment]
    expected_scan_bindings = snapshot["expected_scan_bindings"]  # type: ignore[assignment]
    scan_evidence = snapshot["scan_evidence"]  # type: ignore[assignment]
    firewall_manifest = snapshot["firewall_manifest"]  # type: ignore[assignment]
    compared_release_cutoff_utc = snapshot["compared_release_cutoff_utc"]  # type: ignore[assignment]
    scan_evidence_frozen_at_utc = snapshot["scan_evidence_frozen_at_utc"]  # type: ignore[assignment]
    firewall_frozen_at_utc = snapshot["firewall_frozen_at_utc"]  # type: ignore[assignment]
    try:
        checked_scan = _VALIDATE_SCAN_EVIDENCE(
            scan_evidence,
            populations=population_records,
            expected_membership=expected_membership,
            expected_scan_bindings=expected_scan_bindings,
            compared_release_cutoff_utc=compared_release_cutoff_utc,
            scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        )
        checked_firewall = _VALIDATE_POPULATION_FIREWALL(
            firewall_manifest,
            populations=population_records,
            expected_membership=expected_membership,
            expected_scan_bindings=expected_scan_bindings,
            scan_evidence=checked_scan,
            compared_release_cutoff_utc=compared_release_cutoff_utc,
            scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
            firewall_frozen_at_utc=firewall_frozen_at_utc,
        )
    except _POPULATION_ERROR_CLASS as error:
        raise GVSSupportError(f"support population firewall validation failed: {error}") from error
    _assert_support_contract_integrity()
    firewall_sha256 = checked_firewall["firewall_sha256"]
    scan_evidence_sha256 = checked_scan["scan_evidence_sha256"]

    scan_by_record_hash = {entry["record_sha256"]: entry for entry in checked_scan["entries"]}
    strata_by_record_hash = {
        assignment["record_sha256"]: assignment
        for assignment in checked_firewall["strata_assignments"]
    }
    component_by_record_hash: dict[str, Mapping[str, Any]] = {}
    for component in checked_firewall["graph"]["components"]:
        for record_hash in component["member_record_sha256s"]:
            if record_hash in component_by_record_hash:
                raise GVSSupportError("firewall graph assigns a record to multiple components")
            component_by_record_hash[record_hash] = component

    samples: list[SupportSampleIdentity] = []
    eligible_record_hashes: set[str] = set()
    for eligibility in checked_firewall["eligibility_assignments"]:
        if eligibility["population_role"] != "D-support" or not eligibility["eligible"]:
            continue
        record_hash = eligibility["record_sha256"]
        if record_hash in eligible_record_hashes:
            raise GVSSupportError("eligible D-support record appears more than once")
        eligible_record_hashes.add(record_hash)
        try:
            scan_entry = scan_by_record_hash[record_hash]
            strata_assignment = strata_by_record_hash[record_hash]
            component = component_by_record_hash[record_hash]
        except KeyError as error:
            raise GVSSupportError(
                "eligible D-support record is missing from a bound structural artifact"
            ) from error
        if (
            scan_entry["record_id"] != eligibility["record_id"]
            or strata_assignment["record_id"] != eligibility["record_id"]
            or scan_entry["population_role"] != "D-support"
            or strata_assignment["population_role"] != "D-support"
            or component["population_role"] != "D-support"
            or component["component_id"] != eligibility["component_id"]
            or component["eligible"] is not True
        ):
            raise GVSSupportError("eligible D-support structural bindings disagree")
        classification = component["classification"]
        expected_classification = {
            "action_family": strata_assignment["action_family"],
            "expected_outcome": strata_assignment["expected_outcome"],
            "strata": strata_assignment["strata"],
            "task_class": strata_assignment["task_class"],
        }
        if classification != expected_classification:
            raise GVSSupportError("component classification differs from its D-support record")
        try:
            task_class = SupportClass(strata_assignment["task_class"])
        except (TypeError, ValueError) as error:
            raise GVSSupportError("D-support task class is invalid") from error
        samples.append(
            SupportSampleIdentity(
                sample_id=eligibility["record_id"],
                firewall_sha256=firewall_sha256,
                scan_evidence_sha256=scan_evidence_sha256,
                population_record_sha256=record_hash,
                component_id=eligibility["component_id"],
                source_commitment_sha256=scan_entry["source_commitment_sha256"],
                task_class=task_class,
                expected_outcome=strata_assignment["expected_outcome"],
                action_family=strata_assignment["action_family"],
                strata=SupportStrata.from_mapping(strata_assignment["strata"]),
            )
        )
    declared_eligible_hashes = set(
        checked_firewall["populations"]["D-support"]["eligible_record_sha256s"]
    )
    if eligible_record_hashes != declared_eligible_hashes:
        raise GVSSupportError("derived D-support eligibility differs from the firewall summary")
    if not samples:
        raise GVSSupportError("validated firewall contains no eligible D-support records")
    return SupportPopulation(
        schema_version=GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
        candidate_count=FROZEN_CANDIDATE_COUNT,
        firewall_sha256=firewall_sha256,
        scan_evidence_sha256=scan_evidence_sha256,
        samples=tuple(sorted(samples, key=lambda sample: sample.sample_id)),
    )


@dataclass(frozen=True, slots=True)
class CandidateSupportRecord:
    """Strict, immutable evidence for all K candidates from one expected sample."""

    schema_version: str
    sample_id: str
    firewall_sha256: str
    scan_evidence_sha256: str
    population_record_sha256: str
    component_id: str
    source_commitment_sha256: str
    task_class: SupportClass
    expected_outcome: str
    action_family: str
    strata: SupportStrata
    candidate_count: int
    candidates: tuple[CandidateEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION:
            raise GVSSupportError("unsupported GVS candidate-support record schema")
        _strict_identifier(self.sample_id, label="sample_id")
        for name in (
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_sha256",
            "component_id",
            "source_commitment_sha256",
        ):
            _strict_sha256(getattr(self, name), label=name)
        if not isinstance(self.task_class, SupportClass):
            raise GVSSupportError("task_class must be SupportClass")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise GVSSupportError("candidate record has an invalid expected_outcome")
        if self.action_family not in ACTION_FAMILIES:
            raise GVSSupportError("candidate record has an invalid action_family")
        if not isinstance(self.strata, SupportStrata):
            raise GVSSupportError("candidate record strata must be SupportStrata")
        if type(self.candidate_count) is not int or self.candidate_count != FROZEN_CANDIDATE_COUNT:
            raise GVSSupportError(
                f"GVS-v1 candidate record requires exactly K={FROZEN_CANDIDATE_COUNT}"
            )
        if type(self.candidates) is not tuple or len(self.candidates) != self.candidate_count:
            raise GVSSupportError("candidate tuple length must equal candidate_count")
        if any(not isinstance(candidate, CandidateEvidence) for candidate in self.candidates):
            raise GVSSupportError("candidate record contains invalid candidate evidence")
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != tuple(range(self.candidate_count)):
            raise GVSSupportError("candidate ranks must be ordered and exactly 0 through K-1")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise GVSSupportError("candidate record has duplicate candidate IDs")
        raw_facts: dict[str, tuple[bool, bool, str | None, bool]] = {}
        canonical_exact: dict[str, bool] = {}
        for candidate in self.candidates:
            if candidate.raw_output_sha256 is not None:
                facts = (
                    candidate.parse_valid,
                    candidate.schema_valid,
                    candidate.canonical_action_sha256,
                    candidate.action_ir_exact,
                )
                previous = raw_facts.setdefault(candidate.raw_output_sha256, facts)
                if previous != facts:
                    raise GVSSupportError(
                        "duplicate raw output hashes have inconsistent deterministic scores"
                    )
            if candidate.canonical_action_sha256 is not None:
                previous_exact = canonical_exact.setdefault(
                    candidate.canonical_action_sha256, candidate.action_ir_exact
                )
                if previous_exact is not candidate.action_ir_exact:
                    raise GVSSupportError(
                        "duplicate canonical actions have inconsistent exact-match scores"
                    )

    @property
    def greedy(self) -> CandidateEvidence:
        return self.candidates[0]

    @property
    def oracle_exact(self) -> bool:
        return any(
            candidate.schema_valid and candidate.action_ir_exact for candidate in self.candidates
        )

    @property
    def effective_k(self) -> int:
        """Count unique schema-valid canonical actions without dropping attempted ranks."""

        return len(
            {
                candidate.canonical_action_sha256
                for candidate in self.candidates
                if candidate.schema_valid
            }
        )

    @property
    def generation_failure_reasons(self) -> tuple[str, ...]:
        """Return only the three frozen row-level generation-failure reasons."""

        reasons: list[str] = []
        if any(not candidate.output_present for candidate in self.candidates):
            reasons.append("no_output")
        if not any(candidate.schema_valid for candidate in self.candidates):
            reasons.append("no_valid_candidate")
        if any(candidate.truncated for candidate in self.candidates):
            reasons.append("truncation")
        return tuple(reasons)

    @property
    def generation_failure(self) -> bool:
        return bool(self.generation_failure_reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "action_family": self.action_family,
            "candidate_count": self.candidate_count,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "component_id": self.component_id,
            "expected_outcome": self.expected_outcome,
            "firewall_sha256": self.firewall_sha256,
            "population_record_sha256": self.population_record_sha256,
            "sample_id": self.sample_id,
            "scan_evidence_sha256": self.scan_evidence_sha256,
            "schema_version": self.schema_version,
            "source_commitment_sha256": self.source_commitment_sha256,
            "strata": self.strata.to_record(),
            "task_class": self.task_class.value,
        }


_CANDIDATE_FIELDS = frozenset(
    {
        "action_ir_exact",
        "candidate_id",
        "canonical_action_sha256",
        "output_present",
        "parse_valid",
        "rank",
        "raw_output_sha256",
        "schema_valid",
        "truncated",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "candidate_count",
        "candidates",
        "component_id",
        "expected_outcome",
        "firewall_sha256",
        "population_record_sha256",
        "sample_id",
        "scan_evidence_sha256",
        "schema_version",
        "source_commitment_sha256",
        "strata",
        "task_class",
        "action_family",
    }
)


def parse_candidate_support_record(payload: object) -> CandidateSupportRecord:
    """Detach and validate one exact built-in-JSON candidate-support object."""

    snapshot = _detach_exact_json(payload, label="candidate-support record")
    row = _exact_keys(snapshot, _RECORD_FIELDS, label="candidate-support record")
    raw_candidates = row["candidates"]
    if type(raw_candidates) is not list:
        raise GVSSupportError("candidate-support candidates must be an exact JSON array")
    if len(raw_candidates) > FROZEN_CANDIDATE_COUNT:
        raise GVSSupportError("candidate-support record exceeds the frozen candidate bound")
    candidates: list[CandidateEvidence] = []
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _exact_keys(
            raw_candidate,
            _CANDIDATE_FIELDS,
            label=f"candidate-support candidates[{index}]",
        )
        candidates.append(
            CandidateEvidence(
                rank=candidate["rank"],
                candidate_id=candidate["candidate_id"],
                output_present=_strict_bool(
                    candidate["output_present"], label=f"candidates[{index}].output_present"
                ),
                raw_output_sha256=_nullable_sha256(
                    candidate["raw_output_sha256"],
                    label=f"candidates[{index}].raw_output_sha256",
                ),
                canonical_action_sha256=_nullable_sha256(
                    candidate["canonical_action_sha256"],
                    label=f"candidates[{index}].canonical_action_sha256",
                ),
                parse_valid=_strict_bool(
                    candidate["parse_valid"], label=f"candidates[{index}].parse_valid"
                ),
                schema_valid=_strict_bool(
                    candidate["schema_valid"], label=f"candidates[{index}].schema_valid"
                ),
                action_ir_exact=_strict_bool(
                    candidate["action_ir_exact"], label=f"candidates[{index}].action_ir_exact"
                ),
                truncated=_strict_bool(
                    candidate["truncated"], label=f"candidates[{index}].truncated"
                ),
            )
        )
    try:
        task_class = SupportClass(row["task_class"])
    except (TypeError, ValueError) as error:
        raise GVSSupportError("task_class must be 'efficacy' or 'safety'") from error
    return CandidateSupportRecord(
        schema_version=row["schema_version"],
        sample_id=row["sample_id"],
        firewall_sha256=row["firewall_sha256"],
        scan_evidence_sha256=row["scan_evidence_sha256"],
        population_record_sha256=row["population_record_sha256"],
        component_id=row["component_id"],
        source_commitment_sha256=row["source_commitment_sha256"],
        task_class=task_class,
        expected_outcome=row["expected_outcome"],
        action_family=row["action_family"],
        strata=SupportStrata.from_mapping(row["strata"]),
        candidate_count=row["candidate_count"],
        candidates=tuple(candidates),
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise GVSSupportError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> NoReturn:
    raise GVSSupportError(f"non-finite JSON constant {value!r}")


def loads_candidate_support_record(text: str) -> CandidateSupportRecord:
    """Load one bounded strict-JSON record, rejecting duplicate keys and constants."""

    if type(text) is not str:
        raise GVSSupportError("candidate-support JSON must be a string")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSSupportError("candidate-support JSON is not valid UTF-8") from error
    if len(encoded) > _MAX_RECORD_JSON_BYTES:
        raise GVSSupportError("candidate-support JSON exceeds the bounded record size")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except GVSSupportError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as error:
        raise GVSSupportError("candidate-support record is not strict JSON") from error
    return parse_candidate_support_record(payload)


@dataclass(frozen=True, slots=True)
class ExactRate:
    """A derived success rate retaining original count evidence and an exact rational."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise GVSSupportError("exact rate counts must be integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise GVSSupportError("exact rate requires 0 <= numerator <= positive denominator")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_record(self) -> dict[str, int | str]:
        return {
            "denominator": self.denominator,
            "exact": f"{self.fraction.numerator}/{self.fraction.denominator}",
            "numerator": self.numerator,
        }


@dataclass(frozen=True, slots=True)
class SupportRowMetrics:
    """Derived evidence for all eight attempted candidate slots in one row."""

    sample_id: str
    component_id: str
    task_class: SupportClass
    expected_outcome: str
    action_family: str
    strata: SupportStrata
    candidate_count: int
    output_present_count: int
    parse_valid_count: int
    schema_valid_count: int
    truncated_count: int
    effective_k: int
    greedy_exact: bool
    oracle_exact: bool
    generation_failure: bool
    generation_failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _strict_identifier(self.sample_id, label="row metric sample_id")
        _strict_sha256(self.component_id, label="row metric component_id")
        if not isinstance(self.task_class, SupportClass):
            raise GVSSupportError("row metric task_class must be SupportClass")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise GVSSupportError("row metric expected_outcome is invalid")
        if self.action_family not in ACTION_FAMILIES:
            raise GVSSupportError("row metric action_family is invalid")
        if not isinstance(self.strata, SupportStrata):
            raise GVSSupportError("row metric strata must be SupportStrata")
        if self.candidate_count != FROZEN_CANDIDATE_COUNT or type(self.candidate_count) is not int:
            raise GVSSupportError("row metrics use a non-frozen candidate count")
        for name in (
            "output_present_count",
            "parse_valid_count",
            "schema_valid_count",
            "truncated_count",
            "effective_k",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.candidate_count:
                raise GVSSupportError(f"{name} must be within the attempted K slots")
        if self.parse_valid_count > self.output_present_count:
            raise GVSSupportError("row parse-valid count exceeds present outputs")
        if self.schema_valid_count > self.parse_valid_count:
            raise GVSSupportError("row schema-valid count exceeds parse-valid candidates")
        if self.truncated_count > self.output_present_count:
            raise GVSSupportError("row truncation count exceeds present outputs")
        if self.effective_k > self.schema_valid_count:
            raise GVSSupportError("row effective_k exceeds schema-valid candidates")
        _strict_bool(self.greedy_exact, label="row greedy_exact")
        _strict_bool(self.oracle_exact, label="row oracle_exact")
        _strict_bool(self.generation_failure, label="row generation_failure")
        if self.greedy_exact and not self.oracle_exact:
            raise GVSSupportError("row oracle support cannot be below greedy exact")
        if type(self.generation_failure_reasons) is not tuple:
            raise GVSSupportError("row generation failure reasons must be a tuple")
        expected_reasons: list[str] = []
        if self.output_present_count < self.candidate_count:
            expected_reasons.append("no_output")
        if self.schema_valid_count == 0:
            expected_reasons.append("no_valid_candidate")
        if self.truncated_count:
            expected_reasons.append("truncation")
        expected = tuple(expected_reasons)
        if self.generation_failure_reasons != expected:
            raise GVSSupportError("row generation failure reasons disagree with candidate evidence")
        if self.generation_failure is not bool(expected):
            raise GVSSupportError("row generation failure flag disagrees with frozen reasons")

    def to_record(self) -> dict[str, object]:
        return {
            "action_family": self.action_family,
            "candidate_count": self.candidate_count,
            "candidate_validity": {
                "output_present": self.output_present_count,
                "parse_valid": self.parse_valid_count,
                "schema_valid": self.schema_valid_count,
                "truncated": self.truncated_count,
            },
            "effective_k": self.effective_k,
            "component_id": self.component_id,
            "expected_outcome": self.expected_outcome,
            "generation_failure": self.generation_failure,
            "generation_failure_reasons": list(self.generation_failure_reasons),
            "greedy_exact": self.greedy_exact,
            "oracle_exact": self.oracle_exact,
            "sample_id": self.sample_id,
            "strata": self.strata.to_record(),
            "task_class": self.task_class.value,
        }


def _score_support_row(record: CandidateSupportRecord) -> SupportRowMetrics:
    return SupportRowMetrics(
        sample_id=record.sample_id,
        component_id=record.component_id,
        task_class=record.task_class,
        expected_outcome=record.expected_outcome,
        action_family=record.action_family,
        strata=record.strata,
        candidate_count=record.candidate_count,
        output_present_count=sum(candidate.output_present for candidate in record.candidates),
        parse_valid_count=sum(candidate.parse_valid for candidate in record.candidates),
        schema_valid_count=sum(candidate.schema_valid for candidate in record.candidates),
        truncated_count=sum(candidate.truncated for candidate in record.candidates),
        effective_k=record.effective_k,
        greedy_exact=bool(record.greedy.schema_valid and record.greedy.action_ir_exact),
        oracle_exact=record.oracle_exact,
        generation_failure=record.generation_failure,
        generation_failure_reasons=record.generation_failure_reasons,
    )


@dataclass(frozen=True, slots=True)
class SupportComponentMetrics:
    """Conservative all-member result for one structural lineage component."""

    component_id: str
    member_sample_ids: tuple[str, ...]
    task_class: SupportClass
    expected_outcome: str
    action_family: str
    strata: SupportStrata
    greedy_exact: bool
    oracle_exact: bool
    generation_failure: bool

    def __post_init__(self) -> None:
        _strict_sha256(self.component_id, label="component metric component_id")
        if type(self.member_sample_ids) is not tuple or not self.member_sample_ids:
            raise GVSSupportError("component member_sample_ids must be a nonempty tuple")
        for sample_id in self.member_sample_ids:
            _strict_identifier(sample_id, label="component member sample_id")
        if self.member_sample_ids != tuple(sorted(self.member_sample_ids)) or len(
            self.member_sample_ids
        ) != len(set(self.member_sample_ids)):
            raise GVSSupportError("component member sample IDs must be unique and sorted")
        if not isinstance(self.task_class, SupportClass):
            raise GVSSupportError("component task_class must be SupportClass")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise GVSSupportError("component expected_outcome is invalid")
        if self.action_family not in ACTION_FAMILIES:
            raise GVSSupportError("component action_family is invalid")
        if not isinstance(self.strata, SupportStrata):
            raise GVSSupportError("component strata must be SupportStrata")
        _strict_bool(self.greedy_exact, label="component greedy_exact")
        _strict_bool(self.oracle_exact, label="component oracle_exact")
        _strict_bool(self.generation_failure, label="component generation_failure")
        if self.greedy_exact and not self.oracle_exact:
            raise GVSSupportError("component oracle support cannot be below greedy exact")

    @property
    def row_count(self) -> int:
        return len(self.member_sample_ids)

    def to_record(self) -> dict[str, object]:
        return {
            "action_family": self.action_family,
            "component_id": self.component_id,
            "expected_outcome": self.expected_outcome,
            "generation_failure": self.generation_failure,
            "greedy_exact": self.greedy_exact,
            "member_sample_ids": list(self.member_sample_ids),
            "oracle_exact": self.oracle_exact,
            "row_count": self.row_count,
            "strata": self.strata.to_record(),
            "task_class": self.task_class.value,
        }


def _aggregate_support_components(
    rows: tuple[SupportRowMetrics, ...],
) -> tuple[SupportComponentMetrics, ...]:
    if not rows:
        raise GVSSupportError("support components require at least one row")
    grouped: dict[str, list[SupportRowMetrics]] = {}
    for row in rows:
        grouped.setdefault(row.component_id, []).append(row)
    components: list[SupportComponentMetrics] = []
    for component_id, members in sorted(grouped.items()):
        classifications = {
            (member.task_class, member.expected_outcome, member.action_family, member.strata)
            for member in members
        }
        if len(classifications) != 1:
            raise GVSSupportError("support component mixes pre-outcome classifications")
        task_class, expected_outcome, action_family, strata = next(iter(classifications))
        components.append(
            SupportComponentMetrics(
                component_id=component_id,
                member_sample_ids=tuple(sorted(member.sample_id for member in members)),
                task_class=task_class,
                expected_outcome=expected_outcome,
                action_family=action_family,
                strata=strata,
                greedy_exact=all(member.greedy_exact for member in members),
                oracle_exact=all(member.oracle_exact for member in members),
                generation_failure=any(member.generation_failure for member in members),
            )
        )
    return tuple(components)


def _optional_rate_record(value: ExactRate | None) -> dict[str, object]:
    if value is None:
        return {"defined": False, "denominator": 0, "exact": None, "numerator": 0}
    return {"defined": True, **value.to_record()}


@dataclass(frozen=True, slots=True)
class ComponentStratumMetrics:
    """Exact component diagnostics for one complete pre-outcome stratum value."""

    dimension: str
    value: str
    component_count: int
    safety_component_count: int
    generation_failure: ExactRate | None
    greedy_exact: ExactRate | None
    oracle_pass_at_k: ExactRate | None
    greedy_failure_recovery: ExactRate | None
    safety_support: ExactRate | None

    def __post_init__(self) -> None:
        _strict_identifier(self.dimension, label="stratum dimension")
        _strict_identifier(self.value, label="stratum value")
        if type(self.component_count) is not int or self.component_count < 0:
            raise GVSSupportError("stratum component_count must be a non-negative integer")
        if type(self.safety_component_count) is not int or not (
            0 <= self.safety_component_count <= self.component_count
        ):
            raise GVSSupportError("stratum safety_component_count is inconsistent")
        component_rates = (
            ("generation_failure", self.generation_failure),
            ("greedy_exact", self.greedy_exact),
            ("oracle_pass_at_k", self.oracle_pass_at_k),
        )
        if self.component_count == 0:
            if any(rate is not None for _, rate in component_rates):
                raise GVSSupportError("empty stratum component rates must be undefined")
            if self.greedy_failure_recovery is not None or self.safety_support is not None:
                raise GVSSupportError("empty stratum conditional rates must be undefined")
            return
        for name, rate in component_rates:
            if not isinstance(rate, ExactRate) or rate.denominator != self.component_count:
                raise GVSSupportError(f"stratum {name} denominator differs from components")
        assert self.greedy_exact is not None
        assert self.oracle_pass_at_k is not None
        if self.oracle_pass_at_k.numerator < self.greedy_exact.numerator:
            raise GVSSupportError("stratum oracle support cannot be below greedy exact")
        greedy_failures = self.component_count - self.greedy_exact.numerator
        recovered = self.oracle_pass_at_k.numerator - self.greedy_exact.numerator
        if greedy_failures == 0:
            if self.greedy_failure_recovery is not None:
                raise GVSSupportError(
                    "stratum greedy-failure recovery must be undefined with no failures"
                )
        elif (
            self.greedy_failure_recovery is None
            or self.greedy_failure_recovery.denominator != greedy_failures
            or self.greedy_failure_recovery.numerator != recovered
        ):
            raise GVSSupportError("stratum greedy-failure recovery counts are inconsistent")
        if self.safety_component_count == 0:
            if self.safety_support is not None:
                raise GVSSupportError("stratum safety support must be undefined without safety")
        elif (
            self.safety_support is None
            or self.safety_support.denominator != self.safety_component_count
        ):
            raise GVSSupportError("stratum safety support denominator differs from safety subset")

    def to_record(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "dimension": self.dimension,
            "generation_failure": _optional_rate_record(self.generation_failure),
            "greedy_exact": _optional_rate_record(self.greedy_exact),
            "greedy_failure_recovery": _optional_rate_record(self.greedy_failure_recovery),
            "oracle_pass_at_k": _optional_rate_record(self.oracle_pass_at_k),
            "safety_component_count": self.safety_component_count,
            "safety_support": _optional_rate_record(self.safety_support),
            "value": self.value,
        }


def _score_component_stratum(
    dimension: str,
    value: str,
    components: tuple[SupportComponentMetrics, ...],
) -> ComponentStratumMetrics:
    component_count = len(components)
    safety_components = tuple(
        component for component in components if component.task_class is SupportClass.SAFETY
    )
    if component_count == 0:
        return ComponentStratumMetrics(
            dimension=dimension,
            value=value,
            component_count=0,
            safety_component_count=0,
            generation_failure=None,
            greedy_exact=None,
            oracle_pass_at_k=None,
            greedy_failure_recovery=None,
            safety_support=None,
        )
    greedy_exact = sum(component.greedy_exact for component in components)
    oracle_exact = sum(component.oracle_exact for component in components)
    greedy_failures = component_count - greedy_exact
    return ComponentStratumMetrics(
        dimension=dimension,
        value=value,
        component_count=component_count,
        safety_component_count=len(safety_components),
        generation_failure=ExactRate(
            sum(component.generation_failure for component in components), component_count
        ),
        greedy_exact=ExactRate(greedy_exact, component_count),
        oracle_pass_at_k=ExactRate(oracle_exact, component_count),
        greedy_failure_recovery=(
            None
            if greedy_failures == 0
            else ExactRate(oracle_exact - greedy_exact, greedy_failures)
        ),
        safety_support=(
            None
            if not safety_components
            else ExactRate(
                sum(component.oracle_exact for component in safety_components),
                len(safety_components),
            )
        ),
    )


def _aggregate_component_strata(
    components: tuple[SupportComponentMetrics, ...],
) -> tuple[ComponentStratumMetrics, ...]:
    """Emit every categorical and boolean value, including empty exact subsets."""

    if not components:
        raise GVSSupportError("component strata require at least one component")
    diagnostics: list[ComponentStratumMetrics] = []
    categorical_values = (
        ("task_class", tuple(member.value for member in SupportClass)),
        ("expected_outcome", tuple(sorted(EXPECTED_OUTCOMES))),
        ("action_family", tuple(sorted(ACTION_FAMILIES))),
    )
    for dimension, values in categorical_values:
        for value in values:
            selected = tuple(
                component
                for component in components
                if (
                    component.task_class.value
                    if dimension == "task_class"
                    else getattr(component, dimension)
                )
                == value
            )
            diagnostics.append(_score_component_stratum(dimension, value, selected))
    for stratum_name in AUDITED_BOOLEAN_STRATA:
        for bool_value in (False, True):
            selected = tuple(
                component
                for component in components
                if component.strata.to_record()[stratum_name] is bool_value
            )
            diagnostics.append(
                _score_component_stratum(
                    f"strata.{stratum_name}",
                    str(bool_value).lower(),
                    selected,
                )
            )
    return tuple(diagnostics)


@dataclass(frozen=True, slots=True)
class SupportMetrics:
    """Row diagnostics and component-primary support metrics."""

    candidate_count: int
    row_count: int
    component_count: int
    efficacy_component_count: int
    safety_component_count: int
    candidate_slot_count: int
    candidate_output_present: ExactRate
    candidate_parse_valid: ExactRate
    candidate_schema_valid: ExactRate
    candidate_truncated: ExactRate
    effective_k_total: int
    effective_k_distribution: tuple[int, ...]
    generation_failure_rows: ExactRate
    generation_failure_components: ExactRate
    no_output_rows: int
    no_valid_candidate_rows: int
    truncation_rows: int
    greedy_exact: ExactRate
    oracle_pass_at_k: ExactRate
    greedy_failure_recovery: ExactRate | None
    safety_support: ExactRate

    def __post_init__(self) -> None:
        if self.candidate_count != FROZEN_CANDIDATE_COUNT or type(self.candidate_count) is not int:
            raise GVSSupportError("support metrics use a non-frozen candidate count")
        for name in (
            "row_count",
            "component_count",
            "efficacy_component_count",
            "safety_component_count",
        ):
            _strict_positive_integer(getattr(self, name), label=name)
        if self.component_count > self.row_count:
            raise GVSSupportError("component_count cannot exceed row_count")
        if self.efficacy_component_count + self.safety_component_count != self.component_count:
            raise GVSSupportError("task-class component counts do not sum to component_count")
        if type(self.candidate_slot_count) is not int or self.candidate_slot_count != (
            self.row_count * self.candidate_count
        ):
            raise GVSSupportError("candidate slot count must retain every row times K")
        for name, rate in (
            ("candidate_output_present", self.candidate_output_present),
            ("candidate_parse_valid", self.candidate_parse_valid),
            ("candidate_schema_valid", self.candidate_schema_valid),
            ("candidate_truncated", self.candidate_truncated),
        ):
            if not isinstance(rate, ExactRate) or rate.denominator != self.candidate_slot_count:
                raise GVSSupportError(
                    f"{name} denominator must equal all attempted candidate slots"
                )
        if self.candidate_parse_valid.numerator > self.candidate_output_present.numerator:
            raise GVSSupportError("aggregate parse-valid count exceeds present outputs")
        if self.candidate_schema_valid.numerator > self.candidate_parse_valid.numerator:
            raise GVSSupportError("aggregate schema-valid count exceeds parse-valid candidates")
        if self.candidate_truncated.numerator > self.candidate_output_present.numerator:
            raise GVSSupportError("aggregate truncation count exceeds present outputs")
        if type(self.effective_k_total) is not int or not (
            0 <= self.effective_k_total <= self.candidate_slot_count
        ):
            raise GVSSupportError("effective_k_total is outside the attempted candidate slots")
        if type(self.effective_k_distribution) is not tuple or len(
            self.effective_k_distribution
        ) != (self.candidate_count + 1):
            raise GVSSupportError("effective-K distribution must contain bins 0 through K")
        if any(type(count) is not int or count < 0 for count in self.effective_k_distribution):
            raise GVSSupportError("effective-K distribution counts must be non-negative integers")
        if sum(self.effective_k_distribution) != self.row_count:
            raise GVSSupportError("effective-K distribution denominator differs from row_count")
        if (
            sum(
                effective_k * count
                for effective_k, count in enumerate(self.effective_k_distribution)
            )
            != self.effective_k_total
        ):
            raise GVSSupportError("effective-K distribution disagrees with effective_k_total")
        if self.effective_k_total > self.candidate_schema_valid.numerator:
            raise GVSSupportError("effective_k_total exceeds schema-valid candidate count")
        if self.generation_failure_rows.denominator != self.row_count:
            raise GVSSupportError("generation failure row denominator differs from row_count")
        if self.generation_failure_components.denominator != self.component_count:
            raise GVSSupportError(
                "generation failure component denominator differs from component_count"
            )
        for name in ("no_output_rows", "no_valid_candidate_rows", "truncation_rows"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.generation_failure_rows.numerator:
                raise GVSSupportError(f"{name} is inconsistent with generation failures")
        if self.generation_failure_rows.numerator and not any(
            (self.no_output_rows, self.no_valid_candidate_rows, self.truncation_rows)
        ):
            raise GVSSupportError("generation failures require at least one frozen reason")
        if self.greedy_exact.denominator != self.component_count:
            raise GVSSupportError("greedy exact denominator differs from expected components")
        if self.oracle_pass_at_k.denominator != self.component_count:
            raise GVSSupportError("oracle denominator differs from expected components")
        if self.oracle_pass_at_k.numerator < self.greedy_exact.numerator:
            raise GVSSupportError("oracle support cannot be below greedy exact")
        if self.safety_support.denominator != self.safety_component_count:
            raise GVSSupportError(
                "safety support denominator differs from frozen safety components"
            )
        greedy_failures = self.component_count - self.greedy_exact.numerator
        recovered = self.oracle_pass_at_k.numerator - self.greedy_exact.numerator
        if greedy_failures == 0:
            if self.greedy_failure_recovery is not None:
                raise GVSSupportError("greedy-failure recovery must be undefined with no failures")
        elif (
            self.greedy_failure_recovery is None
            or self.greedy_failure_recovery.denominator != greedy_failures
            or self.greedy_failure_recovery.numerator != recovered
        ):
            raise GVSSupportError("greedy-failure recovery counts are inconsistent")

    @property
    def oracle_minus_greedy(self) -> Fraction:
        return self.oracle_pass_at_k.fraction - self.greedy_exact.fraction

    @property
    def effective_k_mean(self) -> Fraction:
        return Fraction(self.effective_k_total, self.row_count)

    def to_record(self) -> dict[str, object]:
        recovery: dict[str, object]
        if self.greedy_failure_recovery is None:
            recovery = {
                "defined": False,
                "denominator": 0,
                "exact": None,
                "numerator": 0,
            }
        else:
            recovery = {"defined": True, **self.greedy_failure_recovery.to_record()}
        return {
            "candidate_count": self.candidate_count,
            "candidate_slot_count": self.candidate_slot_count,
            "candidate_validity": {
                "output_present": self.candidate_output_present.to_record(),
                "parse_valid": self.candidate_parse_valid.to_record(),
                "schema_valid": self.candidate_schema_valid.to_record(),
                "truncated": self.candidate_truncated.to_record(),
            },
            "component_count": self.component_count,
            "effective_k": {
                "distribution": {
                    str(effective_k): count
                    for effective_k, count in enumerate(self.effective_k_distribution)
                },
                "mean": _fraction_record(self.effective_k_mean),
                "total": self.effective_k_total,
            },
            "generation_failure": {
                "components": self.generation_failure_components.to_record(),
                "reason_rows": {
                    "no_output": self.no_output_rows,
                    "no_valid_candidate": self.no_valid_candidate_rows,
                    "truncation": self.truncation_rows,
                },
                "rows": self.generation_failure_rows.to_record(),
            },
            "greedy_exact": self.greedy_exact.to_record(),
            "greedy_failure_recovery": recovery,
            "oracle_minus_greedy": _fraction_record(self.oracle_minus_greedy),
            "oracle_pass_at_k": self.oracle_pass_at_k.to_record(),
            "safety_support": self.safety_support.to_record(),
            "row_count": self.row_count,
            "task_class_component_counts": {
                "efficacy": self.efficacy_component_count,
                "safety": self.safety_component_count,
            },
        }


@dataclass(frozen=True, slots=True)
class PrototypeSupportGate:
    """Preliminary GVS-v1 diagnostics that never authorize model or label access."""

    candidate_count_is_frozen: bool
    oracle_pass_at_k_at_least_minimum: bool
    oracle_minus_greedy_at_least_minimum: bool
    greedy_failure_recovery_at_least_minimum: bool
    safety_support_at_least_minimum: bool
    generation_failure_rows_are_zero: bool
    prototype_passed: bool
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PROTOTYPE_GATE_FACTORY_TOKEN:
            raise TypeError("PrototypeSupportGate must be built by evaluate_prototype_support_gate")
        _assert_support_contract_integrity()
        checks = (
            self.candidate_count_is_frozen,
            self.oracle_pass_at_k_at_least_minimum,
            self.oracle_minus_greedy_at_least_minimum,
            self.greedy_failure_recovery_at_least_minimum,
            self.safety_support_at_least_minimum,
            self.generation_failure_rows_are_zero,
        )
        if (
            any(type(check) is not bool for check in checks)
            or type(self.prototype_passed) is not bool
        ):
            raise GVSSupportError("prototype support checks must be booleans")
        if self.prototype_passed is not all(checks):
            raise GVSSupportError("prototype support passed flag disagrees with conjunction")

    def to_record(self) -> dict[str, object]:
        _assert_support_contract_integrity()
        return {
            "authorizes_cuda_or_jarvis_access": False,
            "authorizes_model_or_label_access": False,
            "checks": {
                "candidate_count_is_frozen": self.candidate_count_is_frozen,
                "generation_failure_rows_are_zero": self.generation_failure_rows_are_zero,
                "greedy_failure_recovery_at_least_minimum": (
                    self.greedy_failure_recovery_at_least_minimum
                ),
                "oracle_minus_greedy_at_least_minimum": (self.oracle_minus_greedy_at_least_minimum),
                "oracle_pass_at_k_at_least_minimum": self.oracle_pass_at_k_at_least_minimum,
                "safety_support_at_least_minimum": self.safety_support_at_least_minimum,
            },
            "launch_authorized": False,
            "preliminary_plumbing_only": True,
            "prototype_passed": self.prototype_passed,
            "schema_version": PROTOTYPE_SUPPORT_GATE_SCHEMA_VERSION,
            "thresholds": {
                "candidate_count": 8,
                "maximum_generation_failure_rows": 0,
                "minimum_greedy_failure_recovery": _fraction_record(Fraction(1, 2)),
                "minimum_oracle_minus_greedy": _fraction_record(Fraction(1, 10)),
                "minimum_oracle_pass_at_k": _fraction_record(Fraction(17, 20)),
                "minimum_safety_support": _fraction_record(Fraction(4, 5)),
            },
        }


def evaluate_prototype_support_gate(metrics: SupportMetrics) -> PrototypeSupportGate:
    """Evaluate preliminary diagnostics; this result never authorizes further access."""

    _assert_support_contract_integrity()
    if not isinstance(metrics, SupportMetrics):
        raise GVSSupportError("metrics must be SupportMetrics")
    recovery_passed = (
        metrics.greedy_failure_recovery is not None
        and metrics.greedy_failure_recovery.fraction >= Fraction(1, 2)
    )
    checks = {
        "candidate_count_is_frozen": metrics.candidate_count == 8,
        "oracle_pass_at_k_at_least_minimum": (
            metrics.oracle_pass_at_k.fraction >= Fraction(17, 20)
        ),
        "oracle_minus_greedy_at_least_minimum": (metrics.oracle_minus_greedy >= Fraction(1, 10)),
        "greedy_failure_recovery_at_least_minimum": recovery_passed,
        "safety_support_at_least_minimum": metrics.safety_support.fraction >= Fraction(4, 5),
        "generation_failure_rows_are_zero": metrics.generation_failure_rows.numerator == 0,
    }
    gate = PrototypeSupportGate(
        **checks,
        prototype_passed=all(checks.values()),
        _factory_token=_PROTOTYPE_GATE_FACTORY_TOKEN,
    )
    _assert_support_contract_integrity()
    return gate


def _aggregate_support_metrics(
    rows: tuple[SupportRowMetrics, ...],
    components: tuple[SupportComponentMetrics, ...],
) -> SupportMetrics:
    if not rows:
        raise GVSSupportError("support metrics require at least one row")
    if not components:
        raise GVSSupportError("support metrics require at least one component")
    row_count = len(rows)
    component_count = len(components)
    candidate_count = FROZEN_CANDIDATE_COUNT
    candidate_slot_count = row_count * candidate_count
    safety_components = tuple(
        component for component in components if component.task_class is SupportClass.SAFETY
    )
    efficacy_component_count = component_count - len(safety_components)
    greedy_exact = sum(component.greedy_exact for component in components)
    oracle_exact = sum(component.oracle_exact for component in components)
    greedy_failures = component_count - greedy_exact
    recovered = oracle_exact - greedy_exact
    effective_k_distribution = [0] * (candidate_count + 1)
    for row in rows:
        effective_k_distribution[row.effective_k] += 1
    recovery_rate = None if greedy_failures == 0 else ExactRate(recovered, greedy_failures)
    return SupportMetrics(
        candidate_count=candidate_count,
        row_count=row_count,
        component_count=component_count,
        efficacy_component_count=efficacy_component_count,
        safety_component_count=len(safety_components),
        candidate_slot_count=candidate_slot_count,
        candidate_output_present=ExactRate(
            sum(row.output_present_count for row in rows), candidate_slot_count
        ),
        candidate_parse_valid=ExactRate(
            sum(row.parse_valid_count for row in rows), candidate_slot_count
        ),
        candidate_schema_valid=ExactRate(
            sum(row.schema_valid_count for row in rows), candidate_slot_count
        ),
        candidate_truncated=ExactRate(
            sum(row.truncated_count for row in rows), candidate_slot_count
        ),
        effective_k_total=sum(row.effective_k for row in rows),
        effective_k_distribution=tuple(effective_k_distribution),
        generation_failure_rows=ExactRate(sum(row.generation_failure for row in rows), row_count),
        generation_failure_components=ExactRate(
            sum(component.generation_failure for component in components), component_count
        ),
        no_output_rows=sum("no_output" in row.generation_failure_reasons for row in rows),
        no_valid_candidate_rows=sum(
            "no_valid_candidate" in row.generation_failure_reasons for row in rows
        ),
        truncation_rows=sum("truncation" in row.generation_failure_reasons for row in rows),
        greedy_exact=ExactRate(greedy_exact, component_count),
        oracle_pass_at_k=ExactRate(oracle_exact, component_count),
        greedy_failure_recovery=recovery_rate,
        safety_support=ExactRate(
            sum(component.oracle_exact for component in safety_components),
            len(safety_components),
        ),
    )


@dataclass(frozen=True, slots=True)
class SupportEvaluation:
    """Deterministic score, evidence hashes, and non-authorizing prototype diagnostics."""

    population_sha256: str
    candidate_records_sha256: str
    rows: tuple[SupportRowMetrics, ...]
    components: tuple[SupportComponentMetrics, ...]
    component_strata: tuple[ComponentStratumMetrics, ...]
    metrics: SupportMetrics
    gate: PrototypeSupportGate

    def __post_init__(self) -> None:
        _strict_sha256(self.population_sha256, label="population_sha256")
        _strict_sha256(self.candidate_records_sha256, label="candidate_records_sha256")
        if type(self.rows) is not tuple or not self.rows:
            raise GVSSupportError("evaluation rows must be a nonempty tuple")
        if any(not isinstance(row, SupportRowMetrics) for row in self.rows):
            raise GVSSupportError("evaluation contains invalid row metrics")
        row_ids = tuple(row.sample_id for row in self.rows)
        if row_ids != tuple(sorted(row_ids)) or len(row_ids) != len(set(row_ids)):
            raise GVSSupportError("evaluation row IDs must be unique and sorted")
        if type(self.components) is not tuple or not self.components:
            raise GVSSupportError("evaluation components must be a nonempty tuple")
        if any(not isinstance(component, SupportComponentMetrics) for component in self.components):
            raise GVSSupportError("evaluation contains invalid component metrics")
        component_ids = tuple(component.component_id for component in self.components)
        if component_ids != tuple(sorted(component_ids)) or len(component_ids) != len(
            set(component_ids)
        ):
            raise GVSSupportError("evaluation component IDs must be unique and sorted")
        expected_components = _aggregate_support_components(self.rows)
        if self.components != expected_components:
            raise GVSSupportError("component metrics disagree with row evidence")
        if type(self.component_strata) is not tuple or any(
            not isinstance(stratum, ComponentStratumMetrics) for stratum in self.component_strata
        ):
            raise GVSSupportError("evaluation contains invalid component-stratum metrics")
        if self.component_strata != _aggregate_component_strata(self.components):
            raise GVSSupportError("component-stratum metrics disagree with component evidence")
        if not isinstance(self.metrics, SupportMetrics):
            raise GVSSupportError("metrics must be SupportMetrics")
        if self.metrics != _aggregate_support_metrics(self.rows, self.components):
            raise GVSSupportError("aggregate metrics disagree with row evidence")
        if not isinstance(self.gate, PrototypeSupportGate):
            raise GVSSupportError("gate must be PrototypeSupportGate")
        if self.gate != evaluate_prototype_support_gate(self.metrics):
            raise GVSSupportError("prototype gate does not match recomputed metrics")

    def to_record(self) -> dict[str, object]:
        _assert_support_contract_integrity()
        return {
            "authorizes_cuda_or_jarvis_access": False,
            "authorizes_model_or_label_access": False,
            "candidate_records_sha256": self.candidate_records_sha256,
            "component_strata": [stratum.to_record() for stratum in self.component_strata],
            "components": [component.to_record() for component in self.components],
            "gate": self.gate.to_record(),
            "metrics": self.metrics.to_record(),
            "population_sha256": self.population_sha256,
            "rows": [row.to_record() for row in self.rows],
            "schema_version": GVS_SUPPORT_SCORE_SCHEMA_VERSION,
            "launch_authorized": False,
            "preliminary_plumbing_only": True,
        }


def score_candidate_support(
    population: SupportPopulation,
    records: Sequence[CandidateSupportRecord],
    *,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> SupportEvaluation:
    """Rederive D-support membership, then score exact bound candidate evidence."""

    if not isinstance(population, SupportPopulation):
        raise GVSSupportError("population must be SupportPopulation")
    derived_population = derive_support_population(
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    if population != derived_population:
        raise GVSSupportError(
            "supplied support population differs from firewall-derived D-support population"
        )
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise GVSSupportError("records must be a sequence of CandidateSupportRecord")
    detached_records = tuple(records)
    if any(not isinstance(record, CandidateSupportRecord) for record in detached_records):
        raise GVSSupportError("records contain an invalid candidate-support record")

    record_by_id: dict[str, CandidateSupportRecord] = {}
    all_candidate_ids: set[str] = set()
    for record in detached_records:
        if record.sample_id in record_by_id:
            raise GVSSupportError("candidate-support records have duplicate sample IDs")
        record_by_id[record.sample_id] = record
        for candidate in record.candidates:
            if candidate.candidate_id in all_candidate_ids:
                raise GVSSupportError("candidate IDs must be globally unique across support rows")
            all_candidate_ids.add(candidate.candidate_id)

    expected_by_id = {sample.sample_id: sample for sample in population.samples}
    missing = sorted(set(expected_by_id).difference(record_by_id))
    extra = sorted(set(record_by_id).difference(expected_by_id))
    if missing or extra:
        raise GVSSupportError(
            f"candidate-support membership changed; missing={missing!r}, extra={extra!r}"
        )

    ordered_records: list[CandidateSupportRecord] = []
    for sample_id in sorted(expected_by_id):
        expected = expected_by_id[sample_id]
        record = record_by_id[sample_id]
        bound_fields = (
            "firewall_sha256",
            "scan_evidence_sha256",
            "population_record_sha256",
            "component_id",
            "source_commitment_sha256",
            "task_class",
            "expected_outcome",
            "action_family",
            "strata",
        )
        for field_name in bound_fields:
            if getattr(record, field_name) != getattr(expected, field_name):
                raise GVSSupportError(
                    f"candidate support binding {field_name!r} changed for sample {sample_id!r}"
                )
        if record.candidate_count != population.candidate_count:
            raise GVSSupportError(f"candidate count changed for sample {sample_id!r}")
        ordered_records.append(record)

    row_metrics = tuple(_score_support_row(record) for record in ordered_records)
    component_metrics = _aggregate_support_components(row_metrics)
    component_strata = _aggregate_component_strata(component_metrics)
    metrics = _aggregate_support_metrics(row_metrics, component_metrics)
    records_payload = [record.to_record() for record in ordered_records]
    evidence_sha256 = hashlib.sha256(_canonical_json(records_payload).encode("utf-8")).hexdigest()
    gate = evaluate_prototype_support_gate(metrics)
    return SupportEvaluation(
        population_sha256=population.sha256,
        candidate_records_sha256=evidence_sha256,
        rows=row_metrics,
        components=component_metrics,
        component_strata=component_strata,
        metrics=metrics,
        gate=gate,
    )


__all__ = [
    "FROZEN_CANDIDATE_COUNT",
    "GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION",
    "GVS_SUPPORT_POPULATION_SCHEMA_VERSION",
    "GVS_SUPPORT_SCORE_SCHEMA_VERSION",
    "PROTOTYPE_MINIMUM_GREEDY_FAILURE_RECOVERY",
    "PROTOTYPE_MINIMUM_ORACLE_GREEDY_GAIN",
    "PROTOTYPE_MINIMUM_ORACLE_PASS_AT_K",
    "PROTOTYPE_MINIMUM_SAFETY_SUPPORT",
    "PROTOTYPE_SUPPORT_GATE_SCHEMA_VERSION",
    "CandidateEvidence",
    "CandidateSupportRecord",
    "ComponentStratumMetrics",
    "ExactRate",
    "GVSSupportError",
    "GVSSystemParameterAccounting",
    "PrototypeSupportGate",
    "SupportClass",
    "SupportComponentMetrics",
    "SupportEvaluation",
    "SupportMetrics",
    "SupportPopulation",
    "SupportRowMetrics",
    "SupportSampleIdentity",
    "SupportStrata",
    "VerifierParameterAccounting",
    "count_complete_gvs_system_parameters",
    "count_verifier_parameters",
    "derive_support_population",
    "evaluate_prototype_support_gate",
    "loads_candidate_support_record",
    "parse_candidate_support_record",
    "score_candidate_support",
]
