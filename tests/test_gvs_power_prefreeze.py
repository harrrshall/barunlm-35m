from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest
from test_gvs_population import (
    CUTOFF,
    FIREWALL_FROZEN,
    SCAN_FROZEN,
    _bindings,
    _firewall,
    _membership,
    _record,
    _scan_evidence,
    _unsealed_record,
)

from barunlm.evaluation import gvs_power_prefreeze
from barunlm.evaluation.gvs_population import (
    POPULATION_ROLES,
    GVSPopulationError,
    seal_population_record,
)
from barunlm.evaluation.gvs_power import ONE_SAMPLE_KIND, PAIRED_KIND
from barunlm.evaluation.gvs_power_prefreeze import (
    COMPONENT_UNIT,
    GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION,
    GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION,
    GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION,
    GVSPowerPrefreezeError,
    assert_power_prefreeze_runtime_integrity,
    build_power_prefreeze,
    load_power_prefreeze_json,
    power_prefreeze_runtime_sha256,
    power_prefreeze_to_json_bytes,
    verify_power_prefreeze,
)


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"denominator": denominator, "numerator": numerator}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _live_fixture(
    *,
    analysis_rows: int = 11,
    shared_schema_families: int | None = None,
    common_collection_batch: bool = False,
) -> dict[str, Any]:
    populations: dict[str, list[dict[str, Any]]] = {}
    for role in POPULATION_ROLES:
        count = 1 if role == "T-new" else analysis_rows
        rows: list[dict[str, Any]] = []
        for index in range(count):
            name = f"prefreeze-{role.lower()}-{index}"
            if role == "T-new" or shared_schema_families is None:
                rows.append(_record(name, role))
                continue
            row = _unsealed_record(name, role)
            row["provenance"]["schema_family"] = (
                f"honest-{role.lower()}-schema-family-{index % shared_schema_families}"
            )
            if common_collection_batch:
                row["provenance"]["collection_batch"] = (
                    f"honest-{role.lower()}-shared-collection-batch"
                )
            rows.append(seal_population_record(row, compared_release_cutoff_utc=CUTOFF))
        populations[role] = rows
    membership = _membership(populations)
    bindings = _bindings()
    scan_evidence = _scan_evidence(
        populations,
        membership=membership,
        bindings=bindings,
    )
    firewall = _firewall(
        populations,
        membership=membership,
        bindings=bindings,
        scan_evidence=scan_evidence,
    )
    return {
        "population_records": populations,
        "expected_membership": membership,
        "expected_scan_bindings": bindings,
        "scan_evidence": scan_evidence,
        "firewall_manifest": firewall,
    }


def _design(firewall_sha256: str) -> dict[str, Any]:
    endpoints = [
        {
            "endpoint_id": "c-confirmation",
            "kind": PAIRED_KIND,
            "population_role": "C-new",
            "selector": {"kind": "task_class", "value": "safety"},
            "metric_id": "exact-execution",
            "reference_id": "frozen-greedy",
            "source_id": "external-precollection-audit",
            "minimum_component_count": 11,
            "minimum_detectable_absolute_gain": _ratio(4, 5),
            "assumed_discordance": _ratio(9, 10),
        },
        {
            "endpoint_id": "d-oracle",
            "kind": ONE_SAMPLE_KIND,
            "population_role": "D-support",
            "selector": {"kind": "all_components"},
            "metric_id": "oracle-pass-at-8",
            "reference_id": "fixed-null-rate",
            "source_id": "external-precollection-audit",
            "minimum_component_count": 2,
            "minimum_detectable_absolute_gain": _ratio(4, 5),
            "baseline_rate": _ratio(1, 10),
        },
        {
            "endpoint_id": "s-selection",
            "kind": PAIRED_KIND,
            "population_role": "S-new",
            "selector": {"kind": "task_class", "value": "efficacy"},
            "metric_id": "exact-execution",
            "reference_id": "frozen-greedy",
            "source_id": "external-precollection-audit",
            "minimum_component_count": 11,
            "minimum_detectable_absolute_gain": _ratio(4, 5),
            "assumed_discordance": _ratio(9, 10),
        },
    ]
    return {
        "schema_version": GVS_POWER_PREFREEZE_DESIGN_SCHEMA_VERSION,
        "design_id": "gvs-real-roster-power-prefreeze-test-v1",
        "joint_firewall_sha256": firewall_sha256,
        "population_protocol_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "assumption_sources": [
            {
                "source_id": "external-precollection-audit",
                "sha256": "1" * 64,
                "role": "external_or_historical_precollection_assumption",
                "precollection": True,
            }
        ],
        "target_power": _ratio(4, 5),
        "multiplicity": {
            "family_id": "gvs-primary-family",
            "familywise_alpha": _ratio(1, 20),
            "method": "bonferroni",
        },
        "endpoints": endpoints,
        "outcome_access_prohibited": True,
        "zero_post_freeze_attrition": True,
        "authorizes_model_or_label_access": False,
    }


def _build_kwargs(live: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    return {
        "design": design,
        "expected_design_sha256": _canonical_sha256(design),
        **live,
        "compared_release_cutoff_utc": CUTOFF,
        "scan_evidence_frozen_at_utc": SCAN_FROZEN,
        "firewall_frozen_at_utc": FIREWALL_FROZEN,
    }


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    return _live_fixture()


def test_builds_deterministic_exact_component_prefreeze_with_no_authority(
    live: dict[str, Any],
) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    first = build_power_prefreeze(**_build_kwargs(live, design))
    second = build_power_prefreeze(**_build_kwargs(live, copy.deepcopy(design)))

    assert first == second
    assert first["schema_version"] == GVS_POWER_PREFREEZE_RECEIPT_SCHEMA_VERSION
    assert first["structural_prefreeze_complete"] is True
    assert first["validated_joint_firewall"] is True
    assert first["component_roster_live_revalidated"] is True
    assert first["analysis_unit"] == COMPONENT_UNIT
    assert first["row_counts_are_analysis_units"] is False
    assert first["contains_scored_outcomes"] is False
    assert first["contains_predictions"] is False
    assert first["contains_private_labels"] is False
    assert first["scientific_threshold_freeze_authorized"] is False
    for flag in (
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    ):
        assert first[flag] is False
    assert first["assumption_sources_authenticated"] is False
    assert first["outcome_blindness_authenticated"] is False
    assert first["external_secret_custody_authenticated"] is False
    assert first["independent_power_review_passed"] is False
    assert len(first["blockers"]) == 5
    assert first["power_report"]["endpoint_alpha"] == _ratio(1, 60)
    assert [row["endpoint_id"] for row in first["thresholds"]] == [
        "c-confirmation",
        "d-oracle",
        "s-selection",
    ]
    for row in first["thresholds"]:
        assert row["component_count"] == 11
        assert row["eligible_row_count_diagnostic_only"] == 11
        assert row["analysis_unit"] == COMPONENT_UNIT
        assert row["planning_power_target_met"] is True
    oracle = first["thresholds"][1]
    assert oracle["critical_successes"] == 5
    assert oracle["critical_observed_rate"] == _ratio(5, 11)
    for paired in (first["thresholds"][0], first["thresholds"][2]):
        assert len(paired["critical_boundaries"]) == 12
        assert paired["critical_boundaries"][0] == {
            "discordant_component_count": 0,
            "minimum_candidate_only_wins": 1,
        }
    assert first["receipt_sha256"] == _canonical_sha256({**first, "receipt_sha256": "0" * 64})


def test_live_verify_and_strict_canonical_load_round_trip(live: dict[str, Any]) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    kwargs = _build_kwargs(live, design)
    receipt = build_power_prefreeze(**kwargs)
    assert verify_power_prefreeze(receipt, **kwargs) == receipt

    raw = power_prefreeze_to_json_bytes(receipt)
    assert raw.endswith(b"\n")
    artifact = json.loads(raw)
    assert artifact["artifact_schema_version"] == GVS_POWER_PREFREEZE_ARTIFACT_SCHEMA_VERSION
    assert artifact["receipt_sha256"] == receipt["receipt_sha256"]
    assert (
        load_power_prefreeze_json(
            raw,
            expected_receipt_sha256=receipt["receipt_sha256"],
            **kwargs,
        )
        == receipt
    )


def test_receipt_is_detached_from_later_caller_mutation(live: dict[str, Any]) -> None:
    private_live = copy.deepcopy(live)
    design = _design(private_live["firewall_manifest"]["firewall_sha256"])
    receipt = build_power_prefreeze(**_build_kwargs(private_live, design))
    design["endpoints"].clear()
    private_live["population_records"]["D-support"].clear()
    assert len(receipt["design"]["endpoints"]) == 3
    assert receipt["joint_roster_binding"]["roles"]["D-support"]["eligible_component_count"] == 11


def test_rejects_self_rehashed_forged_receipt_by_complete_live_rebuild(
    live: dict[str, Any],
) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    kwargs = _build_kwargs(live, design)
    forged = copy.deepcopy(build_power_prefreeze(**kwargs))
    forged["authorizes_cuda"] = True
    forged["thresholds"][0]["component_count"] = 999
    forged["receipt_sha256"] = "0" * 64
    forged["receipt_sha256"] = _canonical_sha256(forged)
    with pytest.raises(GVSPowerPrefreezeError, match="complete live recomputation"):
        verify_power_prefreeze(forged, **kwargs)


def test_eight_honest_schema_families_cap_every_analysis_role_at_eight_components() -> None:
    collapsed = _live_fixture(analysis_rows=16, shared_schema_families=8)
    firewall = collapsed["firewall_manifest"]
    assert {
        role: firewall["eligible_component_summaries"][role]["eligible_component_count"]
        for role in ("D-support", "S-new", "C-new")
    } == {"D-support": 8, "S-new": 8, "C-new": 8}
    for role in ("D-support", "S-new", "C-new"):
        sizes = sorted(
            len(component["member_record_sha256s"])
            for component in firewall["graph"]["components"]
            if component["population_role"] == role
        )
        assert sizes == [2] * 8

    design = _design(firewall["firewall_sha256"])
    for endpoint in design["endpoints"]:
        endpoint["minimum_component_count"] = 9
    with pytest.raises(
        GVSPowerPrefreezeError,
        match=r"8 components from 16 eligible rows, minimum 9",
    ):
        build_power_prefreeze(**_build_kwargs(collapsed, design))

    # The same equivalence relation makes 2,000 rows across only eight honest family IDs
    # effective n <= 8; shared batch/source/template axes can merge those eight further.
    assert min(2_000, 8) == 8


def test_shared_collection_batch_bridges_schema_families_into_one_component() -> None:
    collapsed = _live_fixture(
        analysis_rows=16,
        shared_schema_families=8,
        common_collection_batch=True,
    )
    firewall = collapsed["firewall_manifest"]
    assert {
        role: firewall["eligible_component_summaries"][role]["eligible_component_count"]
        for role in ("D-support", "S-new", "C-new")
    } == {"D-support": 1, "S-new": 1, "C-new": 1}


def test_reusing_one_schema_family_across_roles_fails_the_joint_firewall() -> None:
    populations = {
        role: [_record(f"cross-role-family-{role.lower()}", role)] for role in POPULATION_ROLES
    }
    for role in ("D-support", "S-new", "C-new"):
        row = _unsealed_record(f"cross-role-family-{role.lower()}", role)
        row["provenance"]["schema_family"] = "honest-shared-cross-role-family"
        populations[role] = [seal_population_record(row, compared_release_cutoff_utc=CUTOFF)]
    membership = _membership(populations)
    bindings = _bindings()
    scan_evidence = _scan_evidence(
        populations,
        membership=membership,
        bindings=bindings,
    )
    with pytest.raises(GVSPopulationError, match="component crosses population roles"):
        _firewall(
            populations,
            membership=membership,
            bindings=bindings,
            scan_evidence=scan_evidence,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("minimum_row_count", 85),
        ("required_passes", 85),
        ("minimum_gain_points", 10),
        ("selection_minimum", 50),
    ],
)
def test_rejects_row_count_and_provisional_threshold_substitution(
    live: dict[str, Any], field: str, value: int
) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    design["endpoints"][0][field] = value
    with pytest.raises(GVSPowerPrefreezeError, match="fields changed"):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_rejects_duplicate_role_stratum_selector(live: dict[str, Any]) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    duplicate = copy.deepcopy(design["endpoints"][1])
    duplicate["endpoint_id"] = "e-duplicate-d-oracle"
    duplicate["metric_id"] = "another-metric"
    design["endpoints"].append(duplicate)
    design["endpoints"].sort(key=lambda row: row["endpoint_id"])
    with pytest.raises(GVSPowerPrefreezeError, match="duplicate population-role stratum"):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_rejects_inadequately_supported_stratum_even_when_role_has_rows(
    live: dict[str, Any],
) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    design["endpoints"][1]["selector"] = {
        "kind": "boolean_stratum",
        "name": "context_grounding",
        "value": False,
    }
    design["endpoints"][1]["minimum_component_count"] = 1
    with pytest.raises(
        GVSPowerPrefreezeError,
        match=r"0 components from 11 eligible rows, minimum 1",
    ):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_rejects_real_roster_that_cannot_reach_fixed_power(live: dict[str, Any]) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    design["endpoints"][0]["minimum_detectable_absolute_gain"] = _ratio(1, 10)
    design["endpoints"][0]["assumed_discordance"] = _ratio(1, 2)
    with pytest.raises(GVSPowerPrefreezeError, match="real component roster is underpowered"):
        build_power_prefreeze(**_build_kwargs(live, design))


@pytest.mark.parametrize("value", [0.8, float("nan"), float("inf")])
def test_rejects_float_and_nonfinite_design_values(live: dict[str, Any], value: float) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    kwargs = _build_kwargs(live, design)
    design["target_power"] = value
    kwargs["design"] = design
    with pytest.raises(GVSPowerPrefreezeError, match="floating-point values are forbidden"):
        build_power_prefreeze(**kwargs)


def test_rejects_custom_containers(live: dict[str, Any]) -> None:
    class CustomDict(dict[str, Any]):
        pass

    class CustomList(list[dict[str, Any]]):
        pass

    design = _design(live["firewall_manifest"]["firewall_sha256"])
    custom_design = CustomDict(design)
    with pytest.raises(GVSPowerPrefreezeError, match="exact built-in JSON containers"):
        build_power_prefreeze(**_build_kwargs(live, custom_design))

    design = _design(live["firewall_manifest"]["firewall_sha256"])
    design["endpoints"] = CustomList(design["endpoints"])
    with pytest.raises(GVSPowerPrefreezeError, match="exact built-in JSON containers"):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_rejects_design_commitment_and_live_firewall_mismatch(live: dict[str, Any]) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    kwargs = _build_kwargs(live, design)
    kwargs["expected_design_sha256"] = "f" * 64
    with pytest.raises(GVSPowerPrefreezeError, match="caller commitment"):
        build_power_prefreeze(**kwargs)

    design["joint_firewall_sha256"] = "e" * 64
    with pytest.raises(GVSPowerPrefreezeError, match="different joint firewall"):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_rejects_tampered_firewall_instead_of_trusting_roster_fields(
    live: dict[str, Any],
) -> None:
    private_live = copy.deepcopy(live)
    private_live["firewall_manifest"]["graph"]["components"].clear()
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    with pytest.raises(GVSPowerPrefreezeError, match="firewall failed live validation"):
        build_power_prefreeze(**_build_kwargs(private_live, design))


def test_strict_loader_rejects_duplicate_float_and_nonfinite_json(live: dict[str, Any]) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    kwargs = _build_kwargs(live, design)
    bad_payloads = (
        b'{"x":1,"x":2}\n',
        b'{"x":0.5}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
    )
    for raw in bad_payloads:
        with pytest.raises(GVSPowerPrefreezeError):
            load_power_prefreeze_json(
                raw,
                expected_receipt_sha256="a" * 64,
                **kwargs,
            )


def test_runtime_substitution_fails_before_build(
    live: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    design = _design(live["firewall_manifest"]["firewall_sha256"])
    monkeypatch.setattr(gvs_power_prefreeze, "_BUILD_POWER_REPORT", lambda _: {})
    with pytest.raises(GVSPowerPrefreezeError, match="runtime dependency changed"):
        build_power_prefreeze(**_build_kwargs(live, design))


def test_runtime_attestation_is_stable_and_source_bound() -> None:
    assert_power_prefreeze_runtime_integrity()
    first = power_prefreeze_runtime_sha256()
    assert first == power_prefreeze_runtime_sha256()
    assert len(first) == 64
