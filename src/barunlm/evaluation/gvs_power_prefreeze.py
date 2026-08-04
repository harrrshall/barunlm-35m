"""Live-firewall-bound, outcome-blind power prefreeze evidence for GVS-v1.

This module connects the existing four-role population firewall to the exact-rational power
planner.  It accepts a caller-committed design containing only external pre-outcome assumptions,
fixed target effects, multiplicity, target power, and component support floors.  Component counts
are rederived exclusively from a completely live-validated ``T/D/S/C`` firewall; row counts and
caller-supplied component rosters are not accepted.

The result is structural prefreeze evidence, not a scientific authorization.  External assumption
sources, outcome blindness, secret custody, one-shot signing, durable retirement, and independent
power review remain unauthenticated.  Every model, label, CUDA, JarvisLabs, training, launch, and
execution authorization flag is therefore permanently false.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

from . import gvs_population as _population_module
from . import gvs_power as _power_module
from . import sim_program as _sim_program_module
from .gvs_population import (
    AUDITED_BOOLEAN_STRATA,
    POPULATION_ROLES,
    GVSPopulationError,
    assert_gvs_population_runtime_integrity,
    gvs_population_runtime_sha256,
    validate_population_firewall,
)
from .gvs_power import (
    GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION,
    ONE_SAMPLE_KIND,
    PAIRED_KIND,
    GVSPowerError,
    build_power_report,
    exact_critical_successes,
    verify_power_report,
)
from .sim_program import (
    SimProgramError,
    assert_sim_program_runtime_integrity,
    module_runtime_sha256,
    runtime_callable_identity,
)

GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION = "barun-gvs-power-prefreeze-design-v1"
GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION = "barun-gvs-power-prefreeze-receipt-v1"
GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION = "barun-gvs-power-prefreeze-artifact-v1"
GVS_POWER_PREFREEZE_RUNTIME_SCHEMA_VERSION = "barun-gvs-power-prefreeze-runtime-v1"

ANALYSIS_ROLES = ("D-support", "S-new", "C-new")
COMPONENT_UNIT = "gvs_firewall_connected_component"
SUPPORTED_SELECTOR_KINDS = frozenset(
    {
        "all_components",
        "task_class",
        "expected_outcome",
        "action_family",
        "boolean_stratum",
    }
)

MAX_ENDPOINTS = 64
MAX_SOURCES = 64
MAX_COMPONENTS = 10_000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 5_000_000
MAX_STRING_BYTES = 65_536
MAX_INPUT_JSON_BYTES = 512 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_INTEGER_ABS = 2**63 - 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")

_DESIGN_FIELDS = frozenset(
    {
        "schema_version",
        "design_id",
        "joint_firewall_sha256",
        "population_protocol_sha256",
        "protocol_sha256",
        "assumption_sources",
        "target_power",
        "multiplicity",
        "endpoints",
        "outcome_access_prohibited",
        "zero_post_freeze_attrition",
        "authorizes_model_or_label_access",
    }
)
_SOURCE_FIELDS = frozenset({"source_id", "sha256", "role", "precollection"})
_MULTIPLICITY_FIELDS = frozenset({"family_id", "familywise_alpha", "method"})
_ENDPOINT_COMMON_FIELDS = frozenset(
    {
        "endpoint_id",
        "kind",
        "population_role",
        "selector",
        "metric_id",
        "reference_id",
        "source_id",
        "minimum_component_count",
        "minimum_detectable_absolute_gain",
    }
)
_ONE_SAMPLE_ENDPOINT_FIELDS = frozenset({*_ENDPOINT_COMMON_FIELDS, "baseline_rate"})
_PAIRED_ENDPOINT_FIELDS = frozenset({*_ENDPOINT_COMMON_FIELDS, "assumed_discordance"})
_ALL_SELECTOR_FIELDS = frozenset({"kind"})
_CLASS_SELECTOR_FIELDS = frozenset({"kind", "value"})
_BOOLEAN_SELECTOR_FIELDS = frozenset({"kind", "name", "value"})

_BLOCKERS = (
    "durable_atomic_retirement_service_absent",
    "external_preoutcome_assumption_source_authentication_absent",
    "external_secret_custody_and_outcome_blindness_authentication_absent",
    "independent_power_review_absent",
    "single_use_signer_absent",
)

_HASHLIB_SHA256 = hashlib.sha256
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_JSON_DECODE_ERROR = json.JSONDecodeError
_OS_FSTAT = os.fstat
_PATH_OPEN = Path.open
_FRACTION_CLASS = Fraction
_VALIDATE_POPULATION_FIREWALL = validate_population_firewall
_ASSERT_POPULATION_RUNTIME = assert_gvs_population_runtime_integrity
_POPULATION_RUNTIME_SHA256 = gvs_population_runtime_sha256
_BUILD_POWER_REPORT = build_power_report
_VERIFY_POWER_REPORT = verify_power_report
_EXACT_CRITICAL_SUCCESSES = exact_critical_successes
_ASSERT_SIM_PROGRAM_RUNTIME = assert_sim_program_runtime_integrity
_MODULE_RUNTIME_SHA256 = module_runtime_sha256
_RUNTIME_CALLABLE_IDENTITY = runtime_callable_identity
_POPULATION_ERROR_CLASS = GVSPopulationError
_POWER_ERROR_CLASS = GVSPowerError
_SIM_PROGRAM_ERROR_CLASS = SimProgramError

_PINNED_SOURCE_FILES_SHA256 = ""
_PINNED_RUNTIME_SHA256 = ""


class GVSPowerPrefreezeError(ValueError):
    """A real-roster power-prefreeze input or receipt failed closed."""


def _exact_object(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GVSPowerPrefreezeError(f"{label} must be an exact JSON object")
    keys = set(value)
    if keys != fields:
        raise GVSPowerPrefreezeError(
            f"{label} fields changed; missing={sorted(fields - keys)!r}, "
            f"extra={sorted(keys - fields)!r}"
        )
    return value


def _strict_string(value: object, *, label: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise GVSPowerPrefreezeError(f"{label} must be an exact bounded string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise GVSPowerPrefreezeError(f"{label} must be valid UTF-8") from error
    if len(encoded) > MAX_STRING_BYTES:
        raise GVSPowerPrefreezeError(f"{label} exceeds the UTF-8 byte bound")
    return value


def _identifier(value: object, *, label: str) -> str:
    value = _strict_string(value, label=label)
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise GVSPowerPrefreezeError(f"{label} must be a bounded portable identifier")
    return value


def _sha256(value: object, *, label: str) -> str:
    value = _strict_string(value, label=label)
    if _SHA256_RE.fullmatch(value) is None:
        raise GVSPowerPrefreezeError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, label: str, maximum: int = MAX_COMPONENTS) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise GVSPowerPrefreezeError(f"{label} must be an exact integer in [1, {maximum}]")
    return value


def _snapshot_json(
    value: object,
    *,
    label: str,
    depth: int = 0,
    active: set[int] | None = None,
    budget: list[int] | None = None,
) -> object:
    if depth > MAX_JSON_DEPTH:
        raise GVSPowerPrefreezeError(f"{label} exceeds the JSON depth bound")
    active_ids = set() if active is None else active
    node_budget = [0] if budget is None else budget
    node_budget[0] += 1
    if node_budget[0] > MAX_JSON_NODES:
        raise GVSPowerPrefreezeError(f"{label} exceeds the JSON node bound")
    if value is None or type(value) in {bool, int, str}:
        if type(value) is int and abs(value) > MAX_INTEGER_ABS:
            raise GVSPowerPrefreezeError(f"{label} integer exceeds the signed-64-bit bound")
        if type(value) is str:
            _strict_string(value, label=label, nonempty=False)
        return value
    if type(value) is float:
        raise GVSPowerPrefreezeError(f"{label} floating-point values are forbidden")
    if type(value) not in {list, dict}:
        raise GVSPowerPrefreezeError(f"{label} must contain only exact built-in JSON containers")
    object_id = id(value)
    if object_id in active_ids:
        raise GVSPowerPrefreezeError(f"{label} contains a JSON cycle")
    active_ids.add(object_id)
    try:
        if type(value) is list:
            first = tuple(value)
            second = tuple(value)
            if tuple(id(item) for item in first) != tuple(id(item) for item in second):
                raise GVSPowerPrefreezeError(f"{label} changed during snapshot")
            return [
                _snapshot_json(
                    child,
                    label=f"{label}[{index}]",
                    depth=depth + 1,
                    active=active_ids,
                    budget=node_budget,
                )
                for index, child in enumerate(first)
            ]
        first_items = tuple(value.items())
        second_items = tuple(value.items())
        if tuple((key, id(child)) for key, child in first_items) != tuple(
            (key, id(child)) for key, child in second_items
        ):
            raise GVSPowerPrefreezeError(f"{label} changed during snapshot")
        result: dict[str, object] = {}
        for key, child in first_items:
            if type(key) is not str or key in result:
                raise GVSPowerPrefreezeError(f"{label} contains an invalid or duplicate key")
            _strict_string(key, label=f"{label}.<key>")
            result[key] = _snapshot_json(
                child,
                label=f"{label}.{key}",
                depth=depth + 1,
                active=active_ids,
                budget=node_budget,
            )
        return result
    finally:
        active_ids.remove(object_id)


def _canonical_bytes(value: object, *, limit: int) -> bytes:
    try:
        payload = _JSON_DUMPS(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GVSPowerPrefreezeError("power-prefreeze evidence is not strict JSON") from error
    if len(payload) > limit:
        raise GVSPowerPrefreezeError("power-prefreeze evidence exceeds its byte bound")
    return payload


def _canonical_sha256(value: object, *, limit: int = MAX_ARTIFACT_JSON_BYTES) -> str:
    return _HASHLIB_SHA256(_canonical_bytes(value, limit=limit)).hexdigest()


def _stable_input_snapshot(value: object) -> dict[str, Any]:
    first = _snapshot_json(value, label="power-prefreeze inputs")
    second = _snapshot_json(value, label="power-prefreeze inputs")
    if type(first) is not dict or type(second) is not dict:
        raise GVSPowerPrefreezeError("power-prefreeze inputs must be one exact object")
    first_bytes = _canonical_bytes(first, limit=MAX_INPUT_JSON_BYTES)
    if first_bytes != _canonical_bytes(second, limit=MAX_INPUT_JSON_BYTES):
        raise GVSPowerPrefreezeError("power-prefreeze inputs changed during stable snapshot")
    return first


def _ratio(value: object, *, label: str) -> Fraction:
    row = _exact_object(value, fields=frozenset({"numerator", "denominator"}), label=label)
    numerator = row["numerator"]
    denominator = row["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator <= 0
        or denominator <= 0
        or numerator >= denominator
        or abs(numerator) > MAX_INTEGER_ABS
        or denominator > MAX_INTEGER_ABS
    ):
        raise GVSPowerPrefreezeError(f"{label} must be an exact rational strictly in (0, 1)")
    return _FRACTION_CLASS(numerator, denominator)


def _ratio_record(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _parse_selector(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GVSPowerPrefreezeError(f"{label} must be an exact selector object")
    kind = value.get("kind")
    if kind not in SUPPORTED_SELECTOR_KINDS:
        raise GVSPowerPrefreezeError(f"{label}.kind is unsupported")
    if kind == "all_components":
        return dict(_exact_object(value, fields=_ALL_SELECTOR_FIELDS, label=label))
    if kind in {"task_class", "expected_outcome", "action_family"}:
        row = _exact_object(value, fields=_CLASS_SELECTOR_FIELDS, label=label)
        checked = _identifier(row["value"], label=f"{label}.value")
        allowed: set[str]
        if kind == "task_class":
            allowed = {"efficacy", "safety"}
        elif kind == "expected_outcome":
            allowed = {"ACTION", "ABSTAIN", "CLARIFY", "CONFIRM"}
        else:
            allowed = {
                "reminders",
                "calendars",
                "contacts",
                "notes",
                "lists",
                "maps",
                "messages",
                "media",
                "device_settings",
                "multi_family",
                "none",
            }
        if checked not in allowed:
            raise GVSPowerPrefreezeError(f"{label}.value is outside the frozen population contract")
        return {"kind": kind, "value": checked}
    row = _exact_object(value, fields=_BOOLEAN_SELECTOR_FIELDS, label=label)
    name = _identifier(row["name"], label=f"{label}.name")
    if name not in AUDITED_BOOLEAN_STRATA:
        raise GVSPowerPrefreezeError(f"{label}.name is not an audited boolean stratum")
    if type(row["value"]) is not bool:
        raise GVSPowerPrefreezeError(f"{label}.value must be an exact boolean")
    return {"kind": kind, "name": name, "value": row["value"]}


def _parse_design(value: object) -> dict[str, Any]:
    design = _exact_object(value, fields=_DESIGN_FIELDS, label="power-prefreeze design")
    if design["schema_version"] != GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION:
        raise GVSPowerPrefreezeError("unsupported power-prefreeze design schema")
    _identifier(design["design_id"], label="design_id")
    for name in ("joint_firewall_sha256", "population_protocol_sha256", "protocol_sha256"):
        _sha256(design[name], label=name)
    if design["outcome_access_prohibited"] is not True:
        raise GVSPowerPrefreezeError("outcome_access_prohibited must be true")
    if design["zero_post_freeze_attrition"] is not True:
        raise GVSPowerPrefreezeError("zero_post_freeze_attrition must be true")
    if design["authorizes_model_or_label_access"] is not False:
        raise GVSPowerPrefreezeError("power-prefreeze design cannot authorize access")
    target_power = _ratio(design["target_power"], label="target_power")
    if target_power < _FRACTION_CLASS(4, 5):
        raise GVSPowerPrefreezeError("target_power must be at least four fifths")

    raw_sources = design["assumption_sources"]
    if type(raw_sources) is not list or not 0 < len(raw_sources) <= MAX_SOURCES:
        raise GVSPowerPrefreezeError("assumption_sources must be a bounded nonempty array")
    source_ids: list[str] = []
    for index, source_value in enumerate(raw_sources):
        source = _exact_object(
            source_value,
            fields=_SOURCE_FIELDS,
            label=f"assumption_sources[{index}]",
        )
        source_id = _identifier(source["source_id"], label=f"assumption_sources[{index}].source_id")
        _sha256(source["sha256"], label=f"assumption_sources[{index}].sha256")
        if source["role"] != "external_or_historical_precollection_assumption":
            raise GVSPowerPrefreezeError(
                "assumption source role must remain external precollection"
            )
        if source["precollection"] is not True:
            raise GVSPowerPrefreezeError("assumption source precollection must be true")
        source_ids.append(source_id)
    if source_ids != sorted(set(source_ids)):
        raise GVSPowerPrefreezeError("assumption sources must be unique and canonically sorted")

    multiplicity = _exact_object(
        design["multiplicity"],
        fields=_MULTIPLICITY_FIELDS,
        label="multiplicity",
    )
    _identifier(multiplicity["family_id"], label="multiplicity.family_id")
    if multiplicity["method"] != "bonferroni":
        raise GVSPowerPrefreezeError("only Bonferroni multiplicity is supported")
    familywise_alpha = _ratio(
        multiplicity["familywise_alpha"], label="multiplicity.familywise_alpha"
    )
    if familywise_alpha > _FRACTION_CLASS(1, 20):
        raise GVSPowerPrefreezeError("familywise alpha must be at most one twentieth")

    raw_endpoints = design["endpoints"]
    if type(raw_endpoints) is not list or not 0 < len(raw_endpoints) <= MAX_ENDPOINTS:
        raise GVSPowerPrefreezeError("endpoints must be a bounded nonempty array")
    endpoint_ids: list[str] = []
    selectors: set[tuple[str, str]] = set()
    structured_hypotheses: set[tuple[str, str, str, str]] = set()
    used_sources: set[str] = set()
    observed_roles: set[str] = set()
    for index, endpoint_value in enumerate(raw_endpoints):
        if type(endpoint_value) is not dict:
            raise GVSPowerPrefreezeError(f"endpoints[{index}] must be an exact object")
        kind = endpoint_value.get("kind")
        fields = (
            _ONE_SAMPLE_ENDPOINT_FIELDS
            if kind == ONE_SAMPLE_KIND
            else _PAIRED_ENDPOINT_FIELDS
            if kind == PAIRED_KIND
            else frozenset()
        )
        if not fields:
            raise GVSPowerPrefreezeError(f"endpoints[{index}].kind is unsupported")
        endpoint = _exact_object(endpoint_value, fields=fields, label=f"endpoints[{index}]")
        endpoint_id = _identifier(endpoint["endpoint_id"], label=f"endpoints[{index}].endpoint_id")
        role = endpoint["population_role"]
        if type(role) is not str or role not in ANALYSIS_ROLES:
            raise GVSPowerPrefreezeError(f"endpoints[{index}].population_role is unsupported")
        selector = _parse_selector(endpoint["selector"], label=f"endpoints[{index}].selector")
        selector_sha256 = _canonical_sha256(selector)
        selector_key = (role, selector_sha256)
        if selector_key in selectors:
            raise GVSPowerPrefreezeError("duplicate population-role stratum selector")
        selectors.add(selector_key)
        metric_id = _identifier(endpoint["metric_id"], label=f"endpoints[{index}].metric_id")
        reference_id = _identifier(
            endpoint["reference_id"], label=f"endpoints[{index}].reference_id"
        )
        structured = (role, selector_sha256, metric_id, reference_id)
        if structured in structured_hypotheses:
            raise GVSPowerPrefreezeError("duplicate structured power hypothesis")
        structured_hypotheses.add(structured)
        source_id = _identifier(endpoint["source_id"], label=f"endpoints[{index}].source_id")
        if source_id not in source_ids:
            raise GVSPowerPrefreezeError("endpoint references an unknown assumption source")
        used_sources.add(source_id)
        _positive_integer(
            endpoint["minimum_component_count"],
            label=f"endpoints[{index}].minimum_component_count",
        )
        gain = _ratio(
            endpoint["minimum_detectable_absolute_gain"],
            label=f"endpoints[{index}].minimum_detectable_absolute_gain",
        )
        if kind == ONE_SAMPLE_KIND:
            baseline = _ratio(endpoint["baseline_rate"], label=f"endpoints[{index}].baseline_rate")
            if baseline + gain >= 1:
                raise GVSPowerPrefreezeError(
                    "one-sample baseline plus target gain must be below one"
                )
        else:
            discordance = _ratio(
                endpoint["assumed_discordance"],
                label=f"endpoints[{index}].assumed_discordance",
            )
            if gain >= discordance:
                raise GVSPowerPrefreezeError("paired target gain must be below discordance")
        endpoint_ids.append(endpoint_id)
        observed_roles.add(role)
    if endpoint_ids != sorted(set(endpoint_ids)):
        raise GVSPowerPrefreezeError("endpoint IDs must be unique and canonically sorted")
    if used_sources != set(source_ids):
        raise GVSPowerPrefreezeError("every assumption source must support an endpoint")
    if observed_roles != set(ANALYSIS_ROLES):
        raise GVSPowerPrefreezeError(
            "power-prefreeze endpoints must cover D-support, S-new, and C-new"
        )
    return design


def _selector_matches(component: Mapping[str, Any], selector: Mapping[str, object]) -> bool:
    classification = component["classification"]
    kind = selector["kind"]
    if kind == "all_components":
        return True
    if kind in {"task_class", "expected_outcome", "action_family"}:
        return classification[kind] == selector["value"]
    return classification["strata"][selector["name"]] is selector["value"]


def _derive_joint_roster_binding(firewall: Mapping[str, Any]) -> dict[str, object]:
    components = firewall["graph"]["components"]
    if type(components) is not list or not components:
        raise GVSPowerPrefreezeError("validated firewall has no component graph")
    role_records: dict[str, dict[str, object]] = {}
    for role in POPULATION_ROLES:
        role_components = [value for value in components if value["population_role"] == role]
        eligible_ids = sorted(
            value["component_id"] for value in role_components if value["eligible"] is True
        )
        excluded_ids = sorted(
            value["component_id"] for value in role_components if value["eligible"] is False
        )
        summary = firewall["eligible_component_summaries"][role]
        if summary != {
            "eligible_component_count": len(eligible_ids),
            "eligible_component_ids": eligible_ids,
            "excluded_component_count": len(excluded_ids),
            "excluded_component_ids": excluded_ids,
        }:
            raise GVSPowerPrefreezeError("firewall component summary differs from its graph")
        role_records[role] = {
            "eligible_component_count": len(eligible_ids),
            "eligible_component_ids_sha256": _canonical_sha256(eligible_ids),
            "excluded_component_count": len(excluded_ids),
            "excluded_component_ids_sha256": _canonical_sha256(excluded_ids),
            "record_count": len(firewall["populations"][role]["record_sha256s"]),
        }
    lineage_edge_counts: dict[str, int] = {}
    for edge in firewall["graph"]["lineage_edges"]:
        kind = edge["lineage_kind"]
        lineage_edge_counts[kind] = lineage_edge_counts.get(kind, 0) + 1
    return {
        "firewall_sha256": firewall["firewall_sha256"],
        "graph_sha256": firewall["graph_sha256"],
        "population_membership_sha256": firewall["population_membership_sha256"],
        "record_set_sha256": firewall["record_set_sha256"],
        "strata_sha256": firewall["strata_sha256"],
        "eligibility_sha256": firewall["eligibility_sha256"],
        "analysis_unit": COMPONENT_UNIT,
        "roles": role_records,
        "lineage_edge_counts": dict(sorted(lineage_edge_counts.items())),
        "row_counts_are_analysis_units": False,
        "component_ids_are_graph_derived": True,
    }


def _derive_component_roster(
    firewall: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> tuple[dict[str, object], dict[str, object]]:
    selector = _parse_selector(endpoint["selector"], label="endpoint.selector")
    role = endpoint["population_role"]
    selected = sorted(
        component["component_id"]
        for component in firewall["graph"]["components"]
        if component["population_role"] == role
        and component["eligible"] is True
        and _selector_matches(component, selector)
    )
    component_count = len(selected)
    minimum_count = endpoint["minimum_component_count"]
    if component_count < minimum_count:
        role_rows = len(firewall["populations"][role]["eligible_record_sha256s"])
        raise GVSPowerPrefreezeError(
            "inadequate independent component support for endpoint "
            f"{endpoint['endpoint_id']!r}: {component_count} components from {role_rows} "
            f"eligible rows, minimum {minimum_count}"
        )
    selector_sha256 = _canonical_sha256(selector)
    roster_id = f"roster-{endpoint['endpoint_id']}"
    roster = {
        "component_count": component_count,
        "eligible_component_ids_sha256": _canonical_sha256(selected),
        "eligible_only": True,
        "outcomes_absent": True,
        "population_firewall_report_sha256": firewall["firewall_sha256"],
        "population_role": role,
        "preoutcome_frozen": True,
        "roster_id": roster_id,
        "scope": f"{role}/{selector_sha256}",
        "stratum_id": f"stratum-{selector_sha256}",
        "unit": COMPONENT_UNIT,
    }
    evidence = {
        "endpoint_id": endpoint["endpoint_id"],
        "population_role": role,
        "selector": selector,
        "selector_sha256": selector_sha256,
        "component_count": component_count,
        "minimum_component_count": minimum_count,
        "eligible_component_ids_sha256": roster["eligible_component_ids_sha256"],
        "eligible_row_count_diagnostic_only": len(
            firewall["populations"][role]["eligible_record_sha256s"]
        ),
        "analysis_unit": COMPONENT_UNIT,
        "minimum_component_floor_met": True,
    }
    return roster, evidence


def _build_power_assumptions(
    design: Mapping[str, Any], firewall: Mapping[str, Any]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    component_rosters: list[dict[str, object]] = []
    roster_evidence: list[dict[str, object]] = []
    power_endpoints: list[dict[str, object]] = []
    endpoint_ids: list[str] = []
    for endpoint in design["endpoints"]:
        roster, evidence = _derive_component_roster(firewall, endpoint)
        component_rosters.append(roster)
        roster_evidence.append(evidence)
        endpoint_ids.append(endpoint["endpoint_id"])
        gain = _ratio(
            endpoint["minimum_detectable_absolute_gain"],
            label="minimum_detectable_absolute_gain",
        )
        common = {
            "alternative_direction": "candidate_greater",
            "component_roster_id": roster["roster_id"],
            "endpoint_id": endpoint["endpoint_id"],
            "kind": endpoint["kind"],
            "metric_id": endpoint["metric_id"],
            "reference_id": endpoint["reference_id"],
            "source_id": endpoint["source_id"],
        }
        if endpoint["kind"] == ONE_SAMPLE_KIND:
            baseline = _ratio(endpoint["baseline_rate"], label="baseline_rate")
            power_endpoints.append(
                {
                    **common,
                    "null_rate": _ratio_record(baseline),
                    "alternative_rate": _ratio_record(baseline + gain),
                }
            )
        else:
            power_endpoints.append(
                {
                    **common,
                    "assumed_discordance": endpoint["assumed_discordance"],
                    "assumed_gain": _ratio_record(gain),
                }
            )
    assumptions = {
        "assumption_sources": design["assumption_sources"],
        "authorizes_model_or_label_access": False,
        "component_rosters": component_rosters,
        "design_id": design["design_id"],
        "endpoints": power_endpoints,
        "hypothesis_roster": {
            "complete_for_protocol": True,
            "endpoint_ids": endpoint_ids,
            "family_id": design["multiplicity"]["family_id"],
            "protocol_sha256": design["protocol_sha256"],
        },
        "independence_contract": {
            "cross_role_component_closure_required": True,
            "effective_n_equals_eligible_components": True,
            "unit": COMPONENT_UNIT,
        },
        "multiplicity": {
            "endpoint_count": len(power_endpoints),
            "family_id": design["multiplicity"]["family_id"],
            "familywise_alpha": design["multiplicity"]["familywise_alpha"],
            "method": "bonferroni",
        },
        "outcome_access_prohibited": True,
        "population_protocol_sha256": design["population_protocol_sha256"],
        "protocol_sha256": design["protocol_sha256"],
        "schema_version": GVS_POWER_ASSUMPTIONS_SCHEMA_VERSION,
        "target_power": design["target_power"],
        "zero_post_freeze_attrition": True,
    }
    return assumptions, roster_evidence


def _paired_boundaries(component_count: int, alpha: Fraction) -> tuple[list[dict[str, int]], str]:
    digest = _HASHLIB_SHA256()
    records: list[dict[str, int]] = []
    for discordant_count in range(component_count + 1):
        cutoff = _EXACT_CRITICAL_SUCCESSES(
            trials=max(1, discordant_count),
            null_probability=_FRACTION_CLASS(1, 2),
            one_sided_alpha=alpha,
        )
        if discordant_count == 0:
            cutoff = 1
        digest.update(f"{discordant_count}:{cutoff}\n".encode("ascii"))
        records.append(
            {
                "discordant_component_count": discordant_count,
                "minimum_candidate_only_wins": cutoff,
            }
        )
    return records, digest.hexdigest()


def _derive_thresholds(
    design: Mapping[str, Any],
    report: Mapping[str, Any],
    roster_evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    endpoint_alpha = _ratio(report["endpoint_alpha"], label="endpoint_alpha")
    design_by_id = {value["endpoint_id"]: value for value in design["endpoints"]}
    evidence_by_id = {value["endpoint_id"]: value for value in roster_evidence}
    thresholds: list[dict[str, object]] = []
    for result in report["endpoints"]:
        endpoint_id = result["endpoint_id"]
        endpoint = design_by_id[endpoint_id]
        evidence = evidence_by_id[endpoint_id]
        calculation = result["calculation"]
        if calculation["meets_planning_power_target"] is not True:
            raise GVSPowerPrefreezeError(
                f"real component roster is underpowered for endpoint {endpoint_id!r}"
            )
        common: dict[str, object] = {
            **evidence,
            "kind": endpoint["kind"],
            "metric_id": endpoint["metric_id"],
            "reference_id": endpoint["reference_id"],
            "source_id": endpoint["source_id"],
            "one_sided_alpha": _ratio_record(endpoint_alpha),
            "target_power": design["target_power"],
            "minimum_detectable_absolute_gain": endpoint["minimum_detectable_absolute_gain"],
            "alternative_power": calculation["alternative_power"],
            "planning_power_target_met": True,
        }
        if endpoint["kind"] == ONE_SAMPLE_KIND:
            critical = calculation["critical_successes"]
            component_count = result["component_count"]
            common.update(
                {
                    "baseline_rate": endpoint["baseline_rate"],
                    "point_alternative_rate": result["assumptions"]["alternative_rate"],
                    "critical_successes": critical,
                    "critical_observed_rate": _ratio_record(
                        _FRACTION_CLASS(critical, component_count)
                    ),
                    "test": calculation["test"],
                }
            )
        else:
            boundaries, boundaries_sha256 = _paired_boundaries(
                result["component_count"], endpoint_alpha
            )
            if boundaries_sha256 != calculation["critical_boundaries_sha256"]:
                raise GVSPowerPrefreezeError("paired critical-boundary hash differs from planner")
            common.update(
                {
                    "assumed_discordance": endpoint["assumed_discordance"],
                    "critical_boundaries": boundaries,
                    "critical_boundaries_sha256": boundaries_sha256,
                    "test": calculation["test"],
                }
            )
        thresholds.append(common)
    return thresholds


def _receipt_hash_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = "0" * 64
    return result


def _snapshot_live_inputs(
    *,
    design: Mapping[str, Any],
    expected_design_sha256: str,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    return _stable_input_snapshot(
        {
            "design": design,
            "expected_design_sha256": expected_design_sha256,
            "population_records": population_records,
            "expected_membership": expected_membership,
            "expected_scan_bindings": expected_scan_bindings,
            "scan_evidence": scan_evidence,
            "firewall_manifest": firewall_manifest,
            "compared_release_cutoff_utc": compared_release_cutoff_utc,
            "scan_evidence_frozen_at_utc": scan_evidence_frozen_at_utc,
            "firewall_frozen_at_utc": firewall_frozen_at_utc,
        }
    )


def build_power_prefreeze(
    *,
    design: Mapping[str, Any],
    expected_design_sha256: str,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Build nonauthorizing thresholds from one live-validated real component roster."""

    _assert_runtime_integrity()
    snapshot = _snapshot_live_inputs(
        design=design,
        expected_design_sha256=expected_design_sha256,
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    checked_expected_design = _sha256(
        snapshot["expected_design_sha256"], label="expected_design_sha256"
    )
    checked_design = _parse_design(snapshot["design"])
    if _canonical_sha256(checked_design) != checked_expected_design:
        raise GVSPowerPrefreezeError("design differs from its caller commitment")
    try:
        checked_firewall = _VALIDATE_POPULATION_FIREWALL(
            snapshot["firewall_manifest"],
            populations=snapshot["population_records"],
            expected_membership=snapshot["expected_membership"],
            expected_scan_bindings=snapshot["expected_scan_bindings"],
            scan_evidence=snapshot["scan_evidence"],
            compared_release_cutoff_utc=snapshot["compared_release_cutoff_utc"],
            scan_evidence_frozen_at_utc=snapshot["scan_evidence_frozen_at_utc"],
            firewall_frozen_at_utc=snapshot["firewall_frozen_at_utc"],
        )
    except _POPULATION_ERROR_CLASS as error:
        raise GVSPowerPrefreezeError("joint population firewall failed live validation") from error
    if checked_design["joint_firewall_sha256"] != checked_firewall["firewall_sha256"]:
        raise GVSPowerPrefreezeError("design is bound to a different joint firewall")
    joint_binding = _derive_joint_roster_binding(checked_firewall)
    assumptions, roster_evidence = _build_power_assumptions(checked_design, checked_firewall)
    try:
        power_report = _BUILD_POWER_REPORT(assumptions)
        power_report = _VERIFY_POWER_REPORT(power_report, assumptions)
    except _POWER_ERROR_CLASS as error:
        raise GVSPowerPrefreezeError("exact power planner rejected the frozen design") from error
    thresholds = _derive_thresholds(checked_design, power_report, roster_evidence)
    source_files, source_files_sha256, runtime_sha256 = _assert_runtime_integrity()
    receipt: dict[str, Any] = {
        "schema_version": GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION,
        "structural_prefreeze_complete": True,
        "validated_joint_firewall": True,
        "component_roster_live_revalidated": True,
        "analysis_unit": COMPONENT_UNIT,
        "row_counts_are_analysis_units": False,
        "contains_scored_outcomes": False,
        "contains_predictions": False,
        "contains_private_labels": False,
        "zero_post_freeze_attrition": True,
        "design": checked_design,
        "design_sha256": checked_expected_design,
        "joint_roster_binding": joint_binding,
        "joint_roster_binding_sha256": _canonical_sha256(joint_binding),
        "power_assumptions": assumptions,
        "power_assumptions_sha256": _canonical_sha256(assumptions),
        "power_report": power_report,
        "power_report_sha256": power_report["report_sha256"],
        "thresholds": thresholds,
        "thresholds_sha256": _canonical_sha256(thresholds),
        "blockers": list(_BLOCKERS),
        "assumption_sources_authenticated": False,
        "outcome_blindness_authenticated": False,
        "external_secret_custody_authenticated": False,
        "independent_power_review_passed": False,
        "scientific_threshold_freeze_authorized": False,
        "authorizes_model_access": False,
        "authorizes_label_access": False,
        "authorizes_cuda": False,
        "authorizes_jarvislabs": False,
        "authorizes_training": False,
        "authorizes_launch": False,
        "authorizes_execution": False,
        "source_files": source_files,
        "source_files_sha256": source_files_sha256,
        "runtime_sha256": runtime_sha256,
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = _canonical_sha256(_receipt_hash_payload(receipt))
    _canonical_bytes(receipt, limit=MAX_ARTIFACT_JSON_BYTES)
    _assert_runtime_integrity()
    return receipt


def verify_power_prefreeze(
    receipt: object,
    *,
    design: Mapping[str, Any],
    expected_design_sha256: str,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Live-rebuild and exact-compare a purported power-prefreeze receipt."""

    _assert_runtime_integrity()
    checked = _snapshot_json(receipt, label="power-prefreeze receipt")
    if type(checked) is not dict:
        raise GVSPowerPrefreezeError("power-prefreeze receipt must be an exact object")
    received_sha256 = _sha256(checked.get("receipt_sha256"), label="receipt_sha256")
    if received_sha256 != _canonical_sha256(_receipt_hash_payload(checked)):
        raise GVSPowerPrefreezeError("power-prefreeze receipt self-hash mismatch")
    rebuilt = build_power_prefreeze(
        design=design,
        expected_design_sha256=expected_design_sha256,
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    if _canonical_bytes(checked, limit=MAX_ARTIFACT_JSON_BYTES) != _canonical_bytes(
        rebuilt, limit=MAX_ARTIFACT_JSON_BYTES
    ):
        raise GVSPowerPrefreezeError("receipt differs from complete live recomputation")
    _assert_runtime_integrity()
    return rebuilt


def power_prefreeze_to_json_bytes(receipt: object) -> bytes:
    """Serialize one exact self-hashed receipt; trust still requires live verification."""

    checked = _snapshot_json(receipt, label="power-prefreeze receipt")
    if type(checked) is not dict:
        raise GVSPowerPrefreezeError("power-prefreeze receipt must be an exact object")
    received_sha256 = _sha256(checked.get("receipt_sha256"), label="receipt_sha256")
    if received_sha256 != _canonical_sha256(_receipt_hash_payload(checked)):
        raise GVSPowerPrefreezeError("power-prefreeze receipt self-hash mismatch")
    artifact = {
        "artifact_schema_version": GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION,
        "receipt": checked,
        "receipt_sha256": received_sha256,
    }
    return _canonical_bytes(artifact, limit=MAX_ARTIFACT_JSON_BYTES) + b"\n"


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GVSPowerPrefreezeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_float(value: str) -> NoReturn:
    raise GVSPowerPrefreezeError(f"floating-point JSON value {value!r} is forbidden")


def _reject_constant(value: str) -> NoReturn:
    raise GVSPowerPrefreezeError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_integer(value: str) -> int:
    if len(value.removeprefix("-")) > 19:
        raise GVSPowerPrefreezeError("JSON integer exceeds the signed-64-bit bound")
    parsed = int(value)
    if abs(parsed) > MAX_INTEGER_ABS:
        raise GVSPowerPrefreezeError("JSON integer exceeds the signed-64-bit bound")
    return parsed


def load_power_prefreeze_json(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    design: Mapping[str, Any],
    expected_design_sha256: str,
    population_records: Mapping[str, list[Mapping[str, Any]]],
    expected_membership: Mapping[str, list[str]],
    expected_scan_bindings: Mapping[str, str],
    scan_evidence: Mapping[str, Any],
    firewall_manifest: Mapping[str, Any],
    compared_release_cutoff_utc: str,
    scan_evidence_frozen_at_utc: str,
    firewall_frozen_at_utc: str,
) -> dict[str, Any]:
    """Strict-load canonical artifact bytes, then completely live-rederive the receipt."""

    _assert_runtime_integrity()
    expected_receipt_sha256 = _sha256(expected_receipt_sha256, label="expected_receipt_sha256")
    if type(raw) is not bytes or not raw or len(raw) > MAX_ARTIFACT_JSON_BYTES:
        raise GVSPowerPrefreezeError("artifact must be nonempty bounded exact bytes")
    try:
        artifact = _JSON_LOADS(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except GVSPowerPrefreezeError:
        raise
    except (_JSON_DECODE_ERROR, UnicodeError, RecursionError, ValueError) as error:
        raise GVSPowerPrefreezeError("artifact is not strict UTF-8 JSON") from error
    artifact = _exact_object(
        artifact,
        fields=frozenset({"artifact_schema_version", "receipt", "receipt_sha256"}),
        label="power-prefreeze artifact",
    )
    if artifact["artifact_schema_version"] != GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION:
        raise GVSPowerPrefreezeError("unsupported power-prefreeze artifact schema")
    embedded = _sha256(artifact["receipt_sha256"], label="artifact receipt_sha256")
    if embedded != expected_receipt_sha256:
        raise GVSPowerPrefreezeError("artifact differs from the caller receipt commitment")
    if raw != _canonical_bytes(artifact, limit=MAX_ARTIFACT_JSON_BYTES) + b"\n":
        raise GVSPowerPrefreezeError("artifact is not exact canonical serialized evidence")
    rebuilt = verify_power_prefreeze(
        artifact["receipt"],
        design=design,
        expected_design_sha256=expected_design_sha256,
        population_records=population_records,
        expected_membership=expected_membership,
        expected_scan_bindings=expected_scan_bindings,
        scan_evidence=scan_evidence,
        firewall_manifest=firewall_manifest,
        compared_release_cutoff_utc=compared_release_cutoff_utc,
        scan_evidence_frozen_at_utc=scan_evidence_frozen_at_utc,
        firewall_frozen_at_utc=firewall_frozen_at_utc,
    )
    if rebuilt["receipt_sha256"] != embedded:
        raise GVSPowerPrefreezeError("artifact receipt differs from live recomputation")
    return rebuilt


def _file_sha256(path: Path) -> str:
    try:
        with _PATH_OPEN(path, "rb") as handle:
            before = _OS_FSTAT(handle.fileno())
            payload = handle.read(MAX_SOURCE_BYTES + 1)
            after = _OS_FSTAT(handle.fileno())
    except OSError as error:
        raise GVSPowerPrefreezeError(f"could not snapshot source {path.name!r}") from error
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_id != after_id or len(payload) != before.st_size:
        raise GVSPowerPrefreezeError(f"source {path.name!r} changed while hashing")
    if len(payload) > MAX_SOURCE_BYTES:
        raise GVSPowerPrefreezeError(f"source {path.name!r} exceeds the source byte bound")
    return _HASHLIB_SHA256(payload).hexdigest()


def _module_path(module: object, *, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str:
        raise GVSPowerPrefreezeError(f"{label} has no exact source path")
    return Path(raw).resolve(strict=True)


def _source_files() -> dict[str, str]:
    return {
        "gvs_population": _file_sha256(_module_path(_population_module, label="population")),
        "gvs_power": _file_sha256(_module_path(_power_module, label="power")),
        "gvs_power_prefreeze": _file_sha256(Path(__file__).resolve(strict=True)),
        "sim_program": _file_sha256(_module_path(_sim_program_module, label="sim program")),
    }


def _assert_import_bindings() -> None:
    checks = (
        (
            "validate_population_firewall",
            _VALIDATE_POPULATION_FIREWALL,
            _population_module.validate_population_firewall,
        ),
        (
            "assert_gvs_population_runtime_integrity",
            _ASSERT_POPULATION_RUNTIME,
            _population_module.assert_gvs_population_runtime_integrity,
        ),
        (
            "gvs_population_runtime_sha256",
            _POPULATION_RUNTIME_SHA256,
            _population_module.gvs_population_runtime_sha256,
        ),
        ("build_power_report", _BUILD_POWER_REPORT, _power_module.build_power_report),
        ("verify_power_report", _VERIFY_POWER_REPORT, _power_module.verify_power_report),
        (
            "exact_critical_successes",
            _EXACT_CRITICAL_SUCCESSES,
            _power_module.exact_critical_successes,
        ),
        (
            "module_runtime_sha256",
            _MODULE_RUNTIME_SHA256,
            _sim_program_module.module_runtime_sha256,
        ),
        (
            "runtime_callable_identity",
            _RUNTIME_CALLABLE_IDENTITY,
            _sim_program_module.runtime_callable_identity,
        ),
        (
            "assert_sim_program_runtime_integrity",
            _ASSERT_SIM_PROGRAM_RUNTIME,
            _sim_program_module.assert_sim_program_runtime_integrity,
        ),
        ("hashlib.sha256", _HASHLIB_SHA256, hashlib.sha256),
        ("json.dumps", _JSON_DUMPS, json.dumps),
        ("json.loads", _JSON_LOADS, json.loads),
        ("json.JSONDecodeError", _JSON_DECODE_ERROR, json.JSONDecodeError),
        ("os.fstat", _OS_FSTAT, os.fstat),
        ("Path.open", _PATH_OPEN, Path.open),
    )
    changed = [name for name, captured, live in checks if captured is not live]
    if changed:
        raise GVSPowerPrefreezeError(
            "power-prefreeze runtime dependency changed: " + ", ".join(sorted(changed))
        )
    if GVSPopulationError is not _POPULATION_ERROR_CLASS or (
        _population_module.GVSPopulationError is not _POPULATION_ERROR_CLASS
    ):
        raise GVSPowerPrefreezeError("population error class changed identity")
    if (
        GVSPowerError is not _POWER_ERROR_CLASS
        or _power_module.GVSPowerError is not _POWER_ERROR_CLASS
    ):
        raise GVSPowerPrefreezeError("power error class changed identity")


def power_prefreeze_runtime_sha256() -> str:
    """Bind this module, lower source/runtime identities, and the exact contract."""

    _assert_import_bindings()
    dependencies = {
        name: dict(_RUNTIME_CALLABLE_IDENTITY(value))
        for name, value in (
            ("build_power_report", _BUILD_POWER_REPORT),
            ("exact_critical_successes", _EXACT_CRITICAL_SUCCESSES),
            ("hashlib.sha256", _HASHLIB_SHA256),
            ("json.dumps", _JSON_DUMPS),
            ("json.loads", _JSON_LOADS),
            ("validate_population_firewall", _VALIDATE_POPULATION_FIREWALL),
            ("verify_power_report", _VERIFY_POWER_REPORT),
        )
    }
    contract = {
        "schema_versions": {
            "artifact": GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION,
            "design": GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION,
            "receipt": GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION,
            "runtime": GVS_POWER_PREFREEZE_RUNTIME_SCHEMA_VERSION,
        },
        "analysis_roles": list(ANALYSIS_ROLES),
        "all_population_roles": list(POPULATION_ROLES),
        "analysis_unit": COMPONENT_UNIT,
        "selector_kinds": sorted(SUPPORTED_SELECTOR_KINDS),
        "audited_boolean_strata": list(AUDITED_BOOLEAN_STRATA),
        "blockers": list(_BLOCKERS),
        "limits": {
            "artifact_json_bytes": MAX_ARTIFACT_JSON_BYTES,
            "components": MAX_COMPONENTS,
            "endpoints": MAX_ENDPOINTS,
            "input_json_bytes": MAX_INPUT_JSON_BYTES,
            "json_depth": MAX_JSON_DEPTH,
            "json_nodes": MAX_JSON_NODES,
            "sources": MAX_SOURCES,
            "source_bytes": MAX_SOURCE_BYTES,
            "string_bytes": MAX_STRING_BYTES,
        },
        "dependencies": dependencies,
        "population_runtime_sha256": _POPULATION_RUNTIME_SHA256(),
        "source_files_sha256": _canonical_sha256(_source_files()),
    }
    try:
        return _MODULE_RUNTIME_SHA256(
            globals(), module_name=__name__, source_path=__file__, contract=contract
        )
    except _SIM_PROGRAM_ERROR_CLASS as error:
        raise GVSPowerPrefreezeError("could not bind power-prefreeze runtime") from error


def _assert_runtime_integrity() -> tuple[dict[str, str], str, str]:
    _assert_import_bindings()
    try:
        _ASSERT_POPULATION_RUNTIME()
        _ASSERT_SIM_PROGRAM_RUNTIME()
    except (ValueError, _POPULATION_ERROR_CLASS, _SIM_PROGRAM_ERROR_CLASS) as error:
        raise GVSPowerPrefreezeError("lower runtime integrity check failed") from error
    source_files = _source_files()
    source_files_sha256 = _canonical_sha256(source_files)
    runtime_sha256 = power_prefreeze_runtime_sha256()
    if source_files_sha256 != _PINNED_SOURCE_FILES_SHA256:
        raise GVSPowerPrefreezeError("power-prefreeze source differs from import-time identity")
    if runtime_sha256 != _PINNED_RUNTIME_SHA256:
        raise GVSPowerPrefreezeError("power-prefreeze runtime differs from import-time identity")
    return source_files, source_files_sha256, runtime_sha256


def assert_power_prefreeze_runtime_integrity() -> None:
    """Public fail-closed runtime-integrity boundary."""

    _assert_runtime_integrity()


__all__ = [
    "ANALYSIS_ROLES",
    "COMPONENT_UNIT",
    "GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION",
    "GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION",
    "GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION",
    "GVS_POWER_PREFREEZE_RUNTIME_SCHEMA_VERSION",
    "GVSPowerPrefreezeError",
    "assert_power_prefreeze_runtime_integrity",
    "build_power_prefreeze",
    "load_power_prefreeze_json",
    "power_prefreeze_runtime_sha256",
    "power_prefreeze_to_json_bytes",
    "verify_power_prefreeze",
]


_PINNED_SOURCE_FILES_SHA256 = _canonical_sha256(_source_files())
_PINNED_RUNTIME_SHA256 = power_prefreeze_runtime_sha256()
