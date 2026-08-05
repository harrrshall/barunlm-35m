from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from barunlm.evaluation import gvs_population
from barunlm.evaluation.gvs_population import (
    AUDITED_BOOLEAN_STRATA,
    GVS_POPULATION_FIREWALL_SCHEMA_VERSION,
    GVS_POPULATION_RECORD_SCHEMA_VERSION,
    POPULATION_ROLES,
    GVSPopulationError,
    assert_gvs_population_runtime_integrity,
    build_population_firewall,
    build_scan_evidence,
    canonical_sha256,
    gvs_population_runtime_sha256,
    loads_population_firewall,
    loads_population_record,
    loads_scan_evidence,
    seal_population_record,
    validate_population_firewall,
    validate_population_record,
    validate_scan_evidence,
)

CUTOFF = "2026-01-01T00:00:00Z"
ROLE_ASSIGNED = "2026-02-01T00:00:00Z"
AUTHORED = "2026-02-02T00:00:00Z"
SCAN_COMPLETED = "2026-02-03T00:00:00Z"
SCAN_FROZEN = "2026-02-04T00:00:00Z"
ELIGIBILITY_DECIDED = "2026-02-05T00:00:00Z"
FIREWALL_FROZEN = "2026-02-06T00:00:00Z"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bindings() -> dict[str, str]:
    return {
        "exact_scan_manifest_sha256": _sha("exact-scan"),
        "near_scan_manifest_sha256": _sha("near-scan"),
        "scan_config_sha256": _sha("scan-config"),
        "scan_code_sha256": _sha("scan-code"),
        "lineage_config_sha256": _sha("lineage-config"),
        "firewall_code_sha256": _sha("firewall-code"),
    }


def _unsealed_record(
    name: str,
    role: str,
    *,
    eligible: bool = True,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": GVS_POPULATION_RECORD_SCHEMA_VERSION,
        "record_id": f"record-{name}",
        "population_role": role,
        "task_class": "safety" if role == "C-new" else "efficacy",
        "expected_outcome": "CONFIRM" if role == "C-new" else "ACTION",
        "action_family": "messages" if role == "C-new" else "calendars",
        "strata": {
            name: name
            in {
                "context_grounding",
                "timezone_or_relative_time",
            }
            for name in AUDITED_BOOLEAN_STRATA
        },
        "provenance": {
            "author_id": f"author-{name}",
            "source_id": f"source-{name}",
            "collection_batch": f"batch-{name}",
            "schema_family": f"schema-{name}",
            "program_template_id": f"template-{name}",
            "paraphrase_family": f"paraphrase-{name}",
            "entity_pool_ids": [f"entity-{name}"],
            "temporal_construction_id": f"temporal-{name}",
            "source_commitment_sha256": _sha(f"source-commitment-{name}"),
            "role_assigned_at_utc": ROLE_ASSIGNED,
            "authored_at_utc": AUTHORED,
            "license": "Apache-2.0",
            "consent": True,
            "no_model_assistance": role in {"S-new", "C-new"},
            "authoring_protocol_revision": "gvs-authoring-v1",
        },
        "duplicate_evidence": {
            "exact_content_sha256": _sha(f"exact-{name}"),
            "normalized_request_sha256": _sha(f"normalized-{name}"),
            "delexicalized_template_sha256": _sha(f"delexicalized-{name}"),
            "near_duplicate_lineage_id": f"near-{name}",
            "scan_completed_at_utc": SCAN_COMPLETED,
            **_bindings(),
        },
        "eligibility": {
            "eligible": eligible,
            "decision": "include" if eligible else "exclude",
            "reason_code": None if eligible else reason_code,
            "decision_stage": "pre-outcome",
            "evidence_sha256": _sha(f"eligibility-{name}"),
            "decided_at_utc": ELIGIBILITY_DECIDED,
        },
        "integrity": {"record_sha256": "0" * 64},
    }


def _record(
    name: str,
    role: str,
    *,
    eligible: bool = True,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return seal_population_record(
        _unsealed_record(name, role, eligible=eligible, reason_code=reason_code),
        compared_release_cutoff_utc=CUTOFF,
    )


def _populations() -> dict[str, list[dict[str, Any]]]:
    return {role: [_record(role.lower(), role)] for role in POPULATION_ROLES}


def _membership(populations: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    return {
        role: [str(record["record_id"]) for record in populations[role]]
        for role in POPULATION_ROLES
    }


def _scan_evidence(
    populations: dict[str, list[dict[str, Any]]],
    *,
    membership: dict[str, list[str]] | None = None,
    bindings: dict[str, str] | None = None,
    frozen_at: str = SCAN_FROZEN,
) -> dict[str, Any]:
    return build_scan_evidence(
        populations,
        expected_membership=membership or _membership(populations),
        expected_scan_bindings=bindings or _bindings(),
        compared_release_cutoff_utc=CUTOFF,
        scan_evidence_frozen_at_utc=frozen_at,
    )


def _firewall(
    populations: dict[str, list[dict[str, Any]]],
    *,
    membership: dict[str, list[str]] | None = None,
    bindings: dict[str, str] | None = None,
    scan_evidence: dict[str, Any] | None = None,
    scan_frozen_at: str = SCAN_FROZEN,
    firewall_frozen_at: str = FIREWALL_FROZEN,
) -> dict[str, Any]:
    checked_membership = membership or _membership(populations)
    checked_bindings = bindings or _bindings()
    checked_scan = scan_evidence or _scan_evidence(
        populations,
        membership=checked_membership,
        bindings=checked_bindings,
        frozen_at=scan_frozen_at,
    )
    return build_population_firewall(
        populations,
        expected_membership=checked_membership,
        expected_scan_bindings=checked_bindings,
        scan_evidence=checked_scan,
        compared_release_cutoff_utc=CUTOFF,
        scan_evidence_frozen_at_utc=scan_frozen_at,
        firewall_frozen_at_utc=firewall_frozen_at,
    )


def _manifest_fixture() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[str]],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
]:
    populations = _populations()
    membership = _membership(populations)
    bindings = _bindings()
    scan_evidence = _scan_evidence(populations, membership=membership, bindings=bindings)
    manifest = _firewall(
        populations,
        membership=membership,
        bindings=bindings,
        scan_evidence=scan_evidence,
    )
    return populations, membership, bindings, scan_evidence, manifest


def _resign_firewall(manifest: dict[str, Any]) -> None:
    manifest["firewall_sha256"] = "0" * 64
    manifest["firewall_sha256"] = canonical_sha256(manifest)


def _resign_scan_evidence(evidence: dict[str, Any]) -> None:
    evidence["scan_evidence_sha256"] = "0" * 64
    evidence["scan_evidence_sha256"] = canonical_sha256(evidence)


def test_builds_deterministic_bound_four_role_firewall() -> None:
    populations, membership, bindings, scan_evidence, manifest = _manifest_fixture()

    assert manifest["schema_version"] == GVS_POPULATION_FIREWALL_SCHEMA_VERSION
    assert manifest["structural_only"] is True
    assert manifest["authorizes_model_or_label_access"] is False
    assert manifest["requires_authenticated_custody_receipt"] is True
    assert scan_evidence["structural_only"] is True
    assert scan_evidence["authorizes_model_or_label_access"] is False
    assert set(manifest["populations"]) == set(POPULATION_ROLES)
    assert len(manifest["graph"]["components"]) == 4
    assert manifest["record_set_sha256"] == canonical_sha256(
        sorted(
            record["integrity"]["record_sha256"]
            for role in POPULATION_ROLES
            for record in populations[role]
        )
    )
    for component in manifest["graph"]["components"]:
        assert component["component_id"] == canonical_sha256(component["member_record_sha256s"])
        assert len(component["member_record_sha256s"]) == 1
        assert component["eligible"] is True
        assert set(component["classification"]) == {
            "task_class",
            "expected_outcome",
            "action_family",
            "strata",
        }
    for role in POPULATION_ROLES:
        assert manifest["eligible_component_summaries"][role]["eligible_component_count"] == 1
        assert manifest["eligible_component_summaries"][role]["excluded_component_count"] == 0
    assert manifest["scan_bindings_sha256"] == canonical_sha256(bindings)
    assert manifest["strata_sha256"] == canonical_sha256(manifest["strata_assignments"])
    assert manifest["eligibility_sha256"] == canonical_sha256(manifest["eligibility_assignments"])
    assert (
        validate_population_firewall(
            manifest,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )
        == manifest
    )


def test_input_order_does_not_change_firewall_or_component_ids() -> None:
    populations = _populations()
    populations["T-new"].append(_record("second-t", "T-new"))
    membership = _membership(populations)
    forward = _firewall(populations, membership=membership)
    reversed_populations = copy.deepcopy(populations)
    reversed_populations["T-new"].reverse()
    reverse = _firewall(reversed_populations, membership=membership)

    assert forward == reverse


def test_transitive_lineage_bridge_across_roles_fails_closed() -> None:
    populations = _populations()
    first = populations["T-new"][0]
    bridge = _unsealed_record("bridge", "T-new")
    bridge["provenance"]["author_id"] = first["provenance"]["author_id"]
    bridge["provenance"]["source_id"] = "shared-source-bridge"
    populations["T-new"].append(seal_population_record(bridge, compared_release_cutoff_utc=CUTOFF))
    support = _unsealed_record("support-bridge", "D-support")
    support["provenance"]["source_id"] = "shared-source-bridge"
    populations["D-support"][0] = seal_population_record(
        support, compared_release_cutoff_utc=CUTOFF
    )

    with pytest.raises(GVSPopulationError, match="transitive component crosses population roles"):
        _firewall(populations)


@pytest.mark.parametrize(
    "lineage_field",
    [
        "author_id",
        "source_id",
        "collection_batch",
        "schema_family",
        "program_template_id",
        "paraphrase_family",
        "entity_pool_ids",
        "temporal_construction_id",
        "source_commitment_sha256",
        "exact_content_sha256",
        "normalized_request_sha256",
        "delexicalized_template_sha256",
        "near_duplicate_lineage_id",
    ],
)
def test_every_frozen_lineage_axis_closes_components_across_roles(
    lineage_field: str,
) -> None:
    populations = _populations()
    training = populations["T-new"][0]
    support = _unsealed_record(f"shared-{lineage_field}", "D-support")
    if lineage_field in training["provenance"]:
        support["provenance"][lineage_field] = copy.deepcopy(training["provenance"][lineage_field])
    else:
        support["duplicate_evidence"][lineage_field] = training["duplicate_evidence"][lineage_field]
    populations["D-support"][0] = seal_population_record(
        support,
        compared_release_cutoff_utc=CUTOFF,
    )

    with pytest.raises(GVSPopulationError, match="component crosses population roles"):
        _firewall(populations)


def test_renamed_cluster_cannot_bypass_lineage_and_cluster_fields_are_rejected() -> None:
    populations = _populations()
    first = populations["T-new"][0]
    renamed = _unsealed_record("renamed", "T-new")
    renamed["provenance"]["program_template_id"] = first["provenance"]["program_template_id"]
    populations["T-new"].append(seal_population_record(renamed, compared_release_cutoff_utc=CUTOFF))
    manifest = _firewall(populations)
    t_components = [
        component
        for component in manifest["graph"]["components"]
        if component["population_role"] == "T-new"
    ]
    assert len(t_components) == 1
    assert sorted(t_components[0]["member_record_ids"]) == sorted(
        [first["record_id"], renamed["record_id"]]
    )

    caller_cluster = _unsealed_record("caller-cluster", "T-new")
    caller_cluster["cluster_id"] = "renamed-independent-cluster"
    with pytest.raises(GVSPopulationError, match="fields changed"):
        seal_population_record(caller_cluster, compared_release_cutoff_utc=CUTOFF)


@pytest.mark.parametrize(
    "classification",
    ["eligibility", "task_class", "expected_outcome", "action_family", "strata"],
)
def test_components_reject_every_mixed_pre_outcome_classification(
    classification: str,
) -> None:
    populations = _populations()
    first = populations["T-new"][0]
    second = _unsealed_record(f"mixed-{classification}", "T-new")
    second["provenance"]["program_template_id"] = first["provenance"]["program_template_id"]
    if classification == "eligibility":
        second["eligibility"].update(
            eligible=False,
            decision="exclude",
            reason_code="collection_quality",
        )
    elif classification == "strata":
        second["strata"]["revision"] = True
    elif classification == "task_class":
        second["task_class"] = "safety"
    elif classification == "expected_outcome":
        second["expected_outcome"] = "CLARIFY"
    else:
        second["action_family"] = "maps"
    populations["T-new"].append(seal_population_record(second, compared_release_cutoff_utc=CUTOFF))

    with pytest.raises(GVSPopulationError, match=f"mixes {classification}|mixes audited"):
        _firewall(populations)


def test_role_aware_provenance_requires_human_no_model_assistance_and_consent() -> None:
    human = _unsealed_record("human-assisted", "S-new")
    human["provenance"]["no_model_assistance"] = False
    with pytest.raises(GVSPopulationError, match="human S-new/C-new"):
        seal_population_record(human, compared_release_cutoff_utc=CUTOFF)

    training = _unsealed_record("training-assisted", "T-new")
    assert training["provenance"]["no_model_assistance"] is False
    seal_population_record(training, compared_release_cutoff_utc=CUTOFF)

    no_consent = _unsealed_record("no-consent", "T-new")
    no_consent["provenance"]["consent"] = False
    with pytest.raises(GVSPopulationError, match="consent must be true"):
        seal_population_record(no_consent, compared_release_cutoff_utc=CUTOFF)

    missing_paraphrase = _unsealed_record("missing-paraphrase", "T-new")
    del missing_paraphrase["provenance"]["paraphrase_family"]
    with pytest.raises(GVSPopulationError, match="provenance fields changed"):
        seal_population_record(missing_paraphrase, compared_release_cutoff_utc=CUTOFF)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cutoff", "cutoff < role assignment < authoring"),
        ("role_equals_authored", "cutoff < role assignment < authoring"),
        ("authored_equals_scan", "authoring < scan completion < eligibility"),
        ("scan_equals_eligibility", "authoring < scan completion < eligibility"),
    ],
)
def test_record_chronology_boundaries_are_strict(mutation: str, message: str) -> None:
    record = _unsealed_record(f"chronology-{mutation}", "T-new")
    cutoff = CUTOFF
    if mutation == "cutoff":
        cutoff = ROLE_ASSIGNED
    elif mutation == "role_equals_authored":
        record["provenance"]["authored_at_utc"] = ROLE_ASSIGNED
    elif mutation == "authored_equals_scan":
        record["duplicate_evidence"]["scan_completed_at_utc"] = AUTHORED
    else:
        record["eligibility"]["decided_at_utc"] = SCAN_COMPLETED

    with pytest.raises(GVSPopulationError, match=message):
        seal_population_record(record, compared_release_cutoff_utc=cutoff)


def test_scan_and_firewall_freeze_chronology_is_strict() -> None:
    populations = _populations()
    with pytest.raises(GVSPopulationError, match="scan completion < scan-evidence freeze"):
        _scan_evidence(populations, frozen_at=SCAN_COMPLETED)
    with pytest.raises(GVSPopulationError, match="scan completion < scan-evidence freeze"):
        _scan_evidence(populations, frozen_at=ELIGIBILITY_DECIDED)

    scan_evidence = _scan_evidence(populations)
    with pytest.raises(GVSPopulationError, match="strictly postdate every eligibility"):
        _firewall(
            populations,
            scan_evidence=scan_evidence,
            firewall_frozen_at=ELIGIBILITY_DECIDED,
        )


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_expected_membership_rejects_missing_and_extra_records(change: str) -> None:
    populations = _populations()
    membership = _membership(populations)
    if change == "missing":
        membership["T-new"].append("record-never-collected")
    else:
        populations["T-new"].append(_record("unregistered-extra", "T-new"))

    with pytest.raises(GVSPopulationError, match="membership changed"):
        _firewall(populations, membership=membership)


def test_population_role_relabeling_is_rejected() -> None:
    populations = _populations()
    relabeled = _unsealed_record("relabel", "D-support")
    populations["T-new"][0] = seal_population_record(relabeled, compared_release_cutoff_utc=CUTOFF)

    with pytest.raises(GVSPopulationError, match="is relabeled"):
        _firewall(populations)


def test_strata_relabel_tamper_fails_even_after_attacker_rehashes_manifest() -> None:
    populations, membership, bindings, scan_evidence, manifest = _manifest_fixture()
    tampered = copy.deepcopy(manifest)
    assignment = tampered["strata_assignments"][0]
    assignment["strata"]["revision"] = not assignment["strata"]["revision"]
    tampered["strata_sha256"] = canonical_sha256(tampered["strata_assignments"])
    _resign_firewall(tampered)

    with pytest.raises(GVSPopulationError, match="differs from recomputed commitments"):
        validate_population_firewall(
            tampered,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )


def test_record_and_firewall_tampering_fail_closed() -> None:
    populations, membership, bindings, scan_evidence, manifest = _manifest_fixture()
    tampered_record = copy.deepcopy(populations)
    tampered_record["T-new"][0]["task_class"] = "safety"
    with pytest.raises(GVSPopulationError, match="record hash mismatch"):
        _firewall(tampered_record, membership=membership, bindings=bindings)

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["graph"]["components"][0]["population_role"] = "tampered-role"
    with pytest.raises(GVSPopulationError, match="firewall hash mismatch"):
        validate_population_firewall(
            tampered_manifest,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )

    forged_authorization = copy.deepcopy(manifest)
    forged_authorization["authorizes_model_or_label_access"] = True
    _resign_firewall(forged_authorization)
    with pytest.raises(GVSPopulationError, match="cannot authorize"):
        validate_population_firewall(
            forged_authorization,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )


def test_scan_evidence_binds_every_source_fingerprint_lineage_and_membership() -> None:
    populations = _populations()
    membership = _membership(populations)
    evidence = _scan_evidence(populations, membership=membership)

    assert len(evidence["entries"]) == 4
    assert evidence["entries_sha256"] == canonical_sha256(evidence["entries"])
    for entry in evidence["entries"]:
        assert len(entry["source_commitment_sha256"]) == 64
        assert len(entry["lineage_commitment_sha256"]) == 64
        assert len(entry["exact_content_sha256"]) == 64
    assert (
        validate_scan_evidence(
            evidence,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=_bindings(),
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
        )
        == evidence
    )


@pytest.mark.parametrize("tamper", ["fingerprint", "missing", "extra"])
def test_rehashed_scan_evidence_content_or_membership_tamper_is_rejected(tamper: str) -> None:
    populations = _populations()
    membership = _membership(populations)
    evidence = _scan_evidence(populations, membership=membership)
    tampered = copy.deepcopy(evidence)
    if tamper == "fingerprint":
        tampered["entries"][0]["normalized_request_sha256"] = _sha("tampered-fingerprint")
    elif tamper == "missing":
        tampered["entries"].pop()
    else:
        extra = copy.deepcopy(tampered["entries"][0])
        extra["record_id"] = "record-forged-extra"
        extra["record_sha256"] = _sha("forged-extra-record")
        tampered["entries"].append(extra)
    tampered["entries"] = sorted(tampered["entries"], key=lambda entry: entry["record_sha256"])
    tampered["entries_sha256"] = canonical_sha256(tampered["entries"])
    _resign_scan_evidence(tampered)

    with pytest.raises(GVSPopulationError, match="differs from recomputed commitments"):
        validate_scan_evidence(
            tampered,
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=_bindings(),
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
        )


def test_scan_evidence_cannot_be_rehashed_into_an_authorization_artifact() -> None:
    populations = _populations()
    evidence = _scan_evidence(populations)
    evidence["authorizes_model_or_label_access"] = True
    _resign_scan_evidence(evidence)

    with pytest.raises(GVSPopulationError, match="cannot authorize"):
        validate_scan_evidence(
            evidence,
            populations=populations,
            expected_membership=_membership(populations),
            expected_scan_bindings=_bindings(),
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
        )


def test_scan_config_and_code_bindings_are_enforced_per_record() -> None:
    populations = _populations()
    changed = copy.deepcopy(populations["D-support"][0])
    changed["duplicate_evidence"]["scan_code_sha256"] = _sha("different-code")
    populations["D-support"][0] = seal_population_record(
        changed, compared_release_cutoff_utc=CUTOFF
    )

    with pytest.raises(GVSPopulationError, match="bound to a different scan_code_sha256"):
        _firewall(populations)


def test_all_mandatory_boolean_strata_are_exact_and_boolean() -> None:
    missing = _unsealed_record("missing-stratum", "T-new")
    del missing["strata"]["revision"]
    with pytest.raises(GVSPopulationError, match="strata fields changed"):
        seal_population_record(missing, compared_release_cutoff_utc=CUTOFF)

    extra = _unsealed_record("extra-stratum", "T-new")
    extra["strata"]["unregistered"] = False
    with pytest.raises(GVSPopulationError, match="strata fields changed"):
        seal_population_record(extra, compared_release_cutoff_utc=CUTOFF)

    non_boolean = _unsealed_record("non-boolean", "T-new")
    non_boolean["strata"]["revision"] = 0
    with pytest.raises(GVSPopulationError, match="must be a boolean"):
        seal_population_record(non_boolean, compared_release_cutoff_utc=CUTOFF)


def test_exclusions_are_retained_with_pre_outcome_reason_and_evidence() -> None:
    populations = _populations()
    excluded = _record(
        "excluded-training",
        "T-new",
        eligible=False,
        reason_code="simulator_roundtrip_failure",
    )
    populations["T-new"].append(excluded)
    manifest = _firewall(populations)

    assert manifest["exclusions"] == [
        assignment
        for assignment in manifest["eligibility_assignments"]
        if assignment["record_id"] == excluded["record_id"]
    ]
    assert manifest["exclusions"][0]["decision_stage"] == "pre-outcome"
    assert manifest["exclusions"][0]["reason_code"] == "simulator_roundtrip_failure"
    assert (
        manifest["exclusions"][0]["evidence_sha256"] == excluded["eligibility"]["evidence_sha256"]
    )
    assert manifest["eligible_component_summaries"]["T-new"]["excluded_component_count"] == 1


def test_caller_mutation_cannot_change_detached_record_or_firewall() -> None:
    raw = _unsealed_record("detached", "T-new")
    sealed = seal_population_record(raw, compared_release_cutoff_utc=CUTOFF)
    raw["task_class"] = "safety"
    assert sealed["task_class"] == "efficacy"

    checked = validate_population_record(sealed, compared_release_cutoff_utc=CUTOFF)
    sealed["strata"]["revision"] = not sealed["strata"]["revision"]
    assert checked["strata"]["revision"] is False

    populations = _populations()
    membership = _membership(populations)
    manifest = _firewall(populations, membership=membership)
    original = copy.deepcopy(manifest)
    populations["T-new"][0]["record_id"] = "mutated-after-build"
    membership["T-new"].append("mutated-membership")
    assert manifest == original


def test_entity_pool_ids_are_sorted_and_bounded() -> None:
    record = _unsealed_record("entity-order", "T-new")
    record["provenance"]["entity_pool_ids"] = ["entity-z", "entity-a"]
    sealed = seal_population_record(record, compared_release_cutoff_utc=CUTOFF)
    assert sealed["provenance"]["entity_pool_ids"] == ["entity-a", "entity-z"]

    unsorted = copy.deepcopy(sealed)
    unsorted["provenance"]["entity_pool_ids"].reverse()
    with pytest.raises(GVSPopulationError, match="canonically sorted"):
        validate_population_record(unsorted, compared_release_cutoff_utc=CUTOFF)

    too_many = _unsealed_record("entity-limit", "T-new")
    too_many["provenance"]["entity_pool_ids"] = [
        f"entity-{index:03d}" for index in range(gvs_population.MAX_ENTITY_POOL_IDS + 1)
    ]
    with pytest.raises(
        GVSPopulationError,
        match=f"exceeds maximum {gvs_population.MAX_ENTITY_POOL_IDS}",
    ):
        seal_population_record(too_many, compared_release_cutoff_utc=CUTOFF)

    wrong_type = _unsealed_record("entity-wrong-type", "T-new")
    wrong_type["provenance"]["entity_pool_ids"] = [1]
    with pytest.raises(GVSPopulationError, match="safe identifier"):
        seal_population_record(wrong_type, compared_release_cutoff_utc=CUTOFF)


def test_population_cardinality_limits_fail_before_component_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    populations = _populations()
    populations["T-new"].append(_record("cardinality-extra", "T-new"))
    monkeypatch.setattr(gvs_population, "MAX_RECORDS_PER_ROLE", 1)

    with pytest.raises(GVSPopulationError, match="exceeds maximum 1 records"):
        gvs_population._snapshot_role_arrays(populations, label="populations")


def test_bounded_raw_json_loaders_reject_duplicate_keys_and_nonfinite_values() -> None:
    record = _record("json-record", "T-new")
    assert (
        loads_population_record(
            json.dumps(record, allow_nan=False, sort_keys=True),
            compared_release_cutoff_utc=CUTOFF,
        )
        == record
    )
    with pytest.raises(GVSPopulationError, match="duplicate JSON key"):
        loads_population_record(
            '{"schema_version":"a","schema_version":"b"}',
            compared_release_cutoff_utc=CUTOFF,
        )
    with pytest.raises(GVSPopulationError, match="non-finite JSON constant"):
        loads_population_record('{"schema_version":NaN}', compared_release_cutoff_utc=CUTOFF)

    populations, membership, bindings, scan_evidence, manifest = _manifest_fixture()
    assert (
        loads_scan_evidence(
            json.dumps(scan_evidence, allow_nan=False, sort_keys=True),
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
        )
        == scan_evidence
    )
    assert (
        loads_population_firewall(
            json.dumps(manifest, allow_nan=False, sort_keys=True),
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )
        == manifest
    )
    with pytest.raises(GVSPopulationError, match="duplicate JSON key"):
        loads_scan_evidence(
            '{"schema_version":"a","schema_version":"b"}',
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
        )
    with pytest.raises(GVSPopulationError, match="non-finite JSON constant"):
        loads_population_firewall(
            '{"schema_version":Infinity}',
            populations=populations,
            expected_membership=membership,
            expected_scan_bindings=bindings,
            scan_evidence=scan_evidence,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )


def test_record_and_manifest_json_size_limits_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(GVSPopulationError, match="exceeds its bounded size"):
        loads_population_record(
            " " * (gvs_population.MAX_RECORD_JSON_BYTES + 1),
            compared_release_cutoff_utc=CUTOFF,
        )

    monkeypatch.setattr(gvs_population, "MAX_FIREWALL_JSON_BYTES", 32)
    with pytest.raises(GVSPopulationError, match="exceeds its bounded size"):
        gvs_population._loads_bounded_json(
            " " * 33,
            label="population_firewall",
            maximum_bytes=gvs_population.MAX_FIREWALL_JSON_BYTES,
        )


def test_population_runtime_and_contract_monkeypatches_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = _unsealed_record("forged-runtime", "T-new")
    with monkeypatch.context() as patch:
        patch.setattr(gvs_population, "assert_gvs_population_runtime_integrity", lambda: None)
        patch.setattr(gvs_population, "canonical_sha256", lambda value: "0" * 64)
        with pytest.raises(GVSPopulationError, match="runtime differs"):
            validate_population_record(forged, compared_release_cutoff_utc=CUTOFF)

    with monkeypatch.context() as patch:
        patch.setattr(gvs_population, "MAX_ENTITY_POOL_IDS", 1)
        with pytest.raises(GVSPopulationError, match="population contract"):
            seal_population_record(forged, compared_release_cutoff_utc=CUTOFF)

    with monkeypatch.context() as patch:
        patch.setattr(gvs_population.hashlib, "sha256", lambda value=b"": None)
        with pytest.raises(GVSPopulationError, match="population runtime import"):
            validate_population_record(forged, compared_release_cutoff_utc=CUTOFF)

    assert len(gvs_population_runtime_sha256()) == 64
    assert_gvs_population_runtime_integrity()


def test_custom_mapping_and_sequence_containers_are_rejected() -> None:
    populations = _populations()

    class CustomMapping(Mapping[str, Any]):
        def __init__(self, value: Mapping[str, Any]) -> None:
            self.value = value

        def __getitem__(self, key: str) -> Any:
            return self.value[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self.value)

        def __len__(self) -> int:
            return len(self.value)

    class CustomSequence(Sequence[Any]):
        def __init__(self, value: Sequence[Any]) -> None:
            self.value = value

        def __getitem__(self, index: int) -> Any:
            return self.value[index]

        def __len__(self) -> int:
            return len(self.value)

    with pytest.raises(GVSPopulationError, match="exact JSON object"):
        _firewall(CustomMapping(populations))  # type: ignore[arg-type]

    poisoned = copy.deepcopy(populations)
    poisoned["T-new"] = CustomSequence(poisoned["T-new"])  # type: ignore[assignment]
    with pytest.raises(GVSPopulationError, match="exact JSON array"):
        _firewall(poisoned)
