from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
from typing import Any

import pytest

from barunlm.config import BarunConfig
from barunlm.evaluation import gvs_support
from barunlm.evaluation.gvs_population import (
    ACTION_FAMILIES,
    AUDITED_BOOLEAN_STRATA,
    EXPECTED_OUTCOMES,
    GVS_POPULATION_RECORD_SCHEMA_VERSION,
    POPULATION_ROLES,
    build_population_firewall,
    build_scan_evidence,
    seal_population_record,
)
from barunlm.evaluation.gvs_support import (
    FROZEN_CANDIDATE_COUNT,
    GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
    GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
    CandidateSupportRecord,
    ExactRate,
    GVSSupportError,
    GVSSystemParameterAccounting,
    SupportClass,
    SupportPopulation,
    SupportSampleIdentity,
    SupportStrata,
    count_complete_gvs_system_parameters,
    count_verifier_parameters,
    derive_support_population,
    loads_candidate_support_record,
    parse_candidate_support_record,
    score_candidate_support,
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


def _strata(seed: int) -> dict[str, bool]:
    return {name: (seed + index) % 2 == 0 for index, name in enumerate(AUDITED_BOOLEAN_STRATA)}


def _unsealed_population_record(
    name: str,
    role: str,
    *,
    task_class: str = "efficacy",
    expected_outcome: str = "ACTION",
    action_family: str = "calendars",
    strata_seed: int = 0,
    eligible: bool = True,
) -> dict[str, Any]:
    reason_code = None if eligible else "collection_quality"
    return {
        "schema_version": GVS_POPULATION_RECORD_SCHEMA_VERSION,
        "record_id": f"record-{name}",
        "population_role": role,
        "task_class": task_class,
        "expected_outcome": expected_outcome,
        "action_family": action_family,
        "strata": _strata(strata_seed),
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
            "reason_code": reason_code,
            "decision_stage": "pre-outcome",
            "evidence_sha256": _sha(f"eligibility-{name}"),
            "decided_at_utc": ELIGIBILITY_DECIDED,
        },
        "integrity": {"record_sha256": "0" * 64},
    }


def _seal(record: Mapping[str, Any]) -> dict[str, Any]:
    return seal_population_record(record, compared_release_cutoff_utc=CUTOFF)


def _membership(
    populations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[str]]:
    return {
        role: [str(record["record_id"]) for record in populations[role]]
        for role in POPULATION_ROLES
    }


def _identity(
    sample_id: str = "record-sample-00",
    task_class: SupportClass = SupportClass.SAFETY,
) -> SupportSampleIdentity:
    return SupportSampleIdentity(
        sample_id=sample_id,
        firewall_sha256=_sha("firewall"),
        scan_evidence_sha256=_sha("scan"),
        population_record_sha256=_sha(f"population:{sample_id}"),
        component_id=_sha(f"component:{sample_id}"),
        source_commitment_sha256=_sha(f"source:{sample_id}"),
        task_class=task_class,
        expected_outcome="CONFIRM" if task_class is SupportClass.SAFETY else "ACTION",
        action_family="messages" if task_class is SupportClass.SAFETY else "calendars",
        strata=SupportStrata.from_mapping(_strata(0)),
    )


def _candidate(
    sample_id: str,
    rank: int,
    *,
    exact: bool = False,
    output_present: bool = True,
    parse_valid: bool = True,
    schema_valid: bool = True,
    truncated: bool = False,
) -> dict[str, object]:
    return {
        "action_ir_exact": exact,
        "candidate_id": f"{sample_id}-candidate-{rank}",
        "canonical_action_sha256": (
            _sha(f"{sample_id}:canonical:{rank}") if schema_valid else None
        ),
        "output_present": output_present,
        "parse_valid": parse_valid,
        "rank": rank,
        "raw_output_sha256": _sha(f"{sample_id}:output:{rank}") if output_present else None,
        "schema_valid": schema_valid,
        "truncated": truncated,
    }


def _record_payload(
    identity: SupportSampleIdentity,
    *,
    greedy_exact: bool,
    oracle_exact: bool,
) -> dict[str, object]:
    assert not greedy_exact or oracle_exact
    candidates = [_candidate(identity.sample_id, rank) for rank in range(FROZEN_CANDIDATE_COUNT)]
    candidates[0]["action_ir_exact"] = greedy_exact
    if oracle_exact and not greedy_exact:
        candidates[1]["action_ir_exact"] = True
    return {
        "action_family": identity.action_family,
        "candidate_count": FROZEN_CANDIDATE_COUNT,
        "candidates": candidates,
        "component_id": identity.component_id,
        "expected_outcome": identity.expected_outcome,
        "firewall_sha256": identity.firewall_sha256,
        "population_record_sha256": identity.population_record_sha256,
        "sample_id": identity.sample_id,
        "scan_evidence_sha256": identity.scan_evidence_sha256,
        "schema_version": GVS_CANDIDATE_SUPPORT_SCHEMA_VERSION,
        "source_commitment_sha256": identity.source_commitment_sha256,
        "strata": identity.strata.to_record(),
        "task_class": identity.task_class.value,
    }


@dataclass(slots=True)
class _SupportFixture:
    population_records: dict[str, list[dict[str, Any]]]
    membership: dict[str, list[str]]
    bindings: dict[str, str]
    scan_evidence: dict[str, Any]
    firewall: dict[str, Any]
    population: SupportPopulation
    records: list[CandidateSupportRecord]

    @property
    def score_kwargs(self) -> dict[str, object]:
        return {
            "population_records": self.population_records,
            "expected_membership": self.membership,
            "expected_scan_bindings": self.bindings,
            "scan_evidence": self.scan_evidence,
            "firewall_manifest": self.firewall,
            "compared_release_cutoff_utc": CUTOFF,
            "scan_evidence_frozen_at_utc": SCAN_FROZEN,
            "firewall_frozen_at_utc": FIREWALL_FROZEN,
        }


def _support_fixture(
    *,
    shared_component_pair: tuple[int, int] | None = None,
    include_excluded: bool = False,
) -> _SupportFixture:
    populations: dict[str, list[dict[str, Any]]] = {}
    for role in ("T-new", "S-new", "C-new"):
        task_class = "safety" if role == "C-new" else "efficacy"
        populations[role] = [
            _seal(
                _unsealed_population_record(
                    role.lower(),
                    role,
                    task_class=task_class,
                    expected_outcome="CONFIRM" if task_class == "safety" else "ACTION",
                    action_family="messages" if task_class == "safety" else "calendars",
                )
            )
        ]

    safety_indices = {0, 1, 2, 16, 18}
    unsealed_support: list[dict[str, Any]] = []
    for index in range(20):
        is_safety = index in safety_indices
        unsealed_support.append(
            _unsealed_population_record(
                f"sample-{index:02d}",
                "D-support",
                task_class="safety" if is_safety else "efficacy",
                expected_outcome="CONFIRM" if is_safety else "ACTION",
                action_family="messages" if is_safety else "calendars",
                strata_seed=index,
            )
        )
    if shared_component_pair is not None:
        left, right = shared_component_pair
        unsealed_support[right]["duplicate_evidence"]["near_duplicate_lineage_id"] = (
            unsealed_support[left]["duplicate_evidence"]["near_duplicate_lineage_id"]
        )
        unsealed_support[right]["strata"] = copy.deepcopy(unsealed_support[left]["strata"])
    if include_excluded:
        unsealed_support.append(
            _unsealed_population_record(
                "excluded",
                "D-support",
                eligible=False,
                strata_seed=21,
            )
        )
    populations["D-support"] = [_seal(record) for record in unsealed_support]

    membership = _membership(populations)
    bindings = _bindings()
    scan_evidence = build_scan_evidence(
        populations,
        expected_membership=membership,
        expected_scan_bindings=bindings,
        compared_release_cutoff_utc=CUTOFF,
        scan_evidence_frozen_at_utc=SCAN_FROZEN,
    )
    firewall = build_population_firewall(
        populations,
        expected_membership=membership,
        expected_scan_bindings=bindings,
        scan_evidence=scan_evidence,
        compared_release_cutoff_utc=CUTOFF,
        scan_evidence_frozen_at_utc=SCAN_FROZEN,
        firewall_frozen_at_utc=FIREWALL_FROZEN,
    )
    derive_kwargs = {
        "population_records": populations,
        "expected_membership": membership,
        "expected_scan_bindings": bindings,
        "scan_evidence": scan_evidence,
        "firewall_manifest": firewall,
        "compared_release_cutoff_utc": CUTOFF,
        "scan_evidence_frozen_at_utc": SCAN_FROZEN,
        "firewall_frozen_at_utc": FIREWALL_FROZEN,
    }
    population = derive_support_population(**derive_kwargs)
    records = []
    for identity in population.samples:
        index = int(identity.sample_id.rsplit("-", 1)[1])
        records.append(
            parse_candidate_support_record(
                _record_payload(
                    identity,
                    greedy_exact=index < 16,
                    oracle_exact=index < 18,
                )
            )
        )
    return _SupportFixture(
        population_records=populations,
        membership=membership,
        bindings=bindings,
        scan_evidence=scan_evidence,
        firewall=firewall,
        population=population,
        records=records,
    )


def _score(
    fixture: _SupportFixture,
    records: Sequence[CandidateSupportRecord] | None = None,
    population: SupportPopulation | None = None,
):
    return score_candidate_support(
        population or fixture.population,
        records if records is not None else fixture.records,
        **fixture.score_kwargs,
    )


def _assert_no_float(value: object) -> None:
    assert type(value) is not float
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_float(child)


def test_decoder_bridge_is_separate_and_still_required_for_launch() -> None:
    assert gvs_support.__doc__ is not None
    assert "separate GVS bridge" in gvs_support.__doc__
    assert "No launch is authorized until the complete bridge receipt is frozen" in (
        gvs_support.__doc__
    )


def test_canonical_gvs_parameter_accounting_is_exact() -> None:
    accounting = count_complete_gvs_system_parameters(
        BarunConfig(), generator_parameters=35_072_768
    )

    assert accounting.verifier.q_projection_output_dimension == 448
    assert accounting.verifier.v_projection_output_dimension == 64
    assert accounting.verifier.q_projection_lora_parameters == 86_016
    assert accounting.verifier.v_projection_lora_parameters == 49_152
    assert accounting.verifier.scalar_head_parameters == 449
    assert accounting.verifier.verifier_parameters == 135_617
    assert accounting.complete_system_parameters == 35_208_385
    assert accounting.to_record()["shared_backbone_stored_once"] is True
    _assert_no_float(accounting.to_record())


def test_parameter_accounting_uses_config_projection_dimensions() -> None:
    accounting = count_verifier_parameters(
        BarunConfig(dim=512, n_layers=6, n_heads=8, n_kv_heads=2), rank=4
    )

    assert accounting.q_projection_output_dimension == 512
    assert accounting.v_projection_output_dimension == 128
    assert accounting.q_projection_lora_parameters == 24_576
    assert accounting.v_projection_lora_parameters == 15_360
    assert accounting.scalar_head_parameters == 513
    assert accounting.verifier_parameters == 40_449


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_parameter_accounting_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(GVSSupportError, match="rank must be a positive integer"):
        count_verifier_parameters(BarunConfig(), rank=rank)  # type: ignore[arg-type]


@pytest.mark.parametrize("generator_parameters", [0, -1, True, 1.5])
def test_system_accounting_rejects_fake_generator_counts(generator_parameters: object) -> None:
    with pytest.raises(GVSSupportError, match="generator_parameters must be a positive integer"):
        count_complete_gvs_system_parameters(
            BarunConfig(),
            generator_parameters=generator_parameters,  # type: ignore[arg-type]
        )


def test_system_accounting_rejects_numerically_equal_float_total() -> None:
    accounting = count_complete_gvs_system_parameters(
        BarunConfig(), generator_parameters=35_072_768
    )
    with pytest.raises(
        GVSSupportError, match="complete_system_parameters must be a positive integer"
    ):
        GVSSystemParameterAccounting(
            generator_parameters=accounting.generator_parameters,
            verifier=accounting.verifier,
            complete_system_parameters=float(accounting.complete_system_parameters),  # type: ignore[arg-type]
        )


def test_candidate_support_parser_detaches_into_immutable_tuples() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    record = parse_candidate_support_record(payload)
    assert isinstance(record.candidates, tuple)
    assert record.greedy.rank == 0

    payload["sample_id"] = "mutated"
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["candidate_id"] = "mutated"
    assert record.sample_id == "record-sample-00"
    assert record.candidates[0].candidate_id == "record-sample-00-candidate-0"
    with pytest.raises(FrozenInstanceError):
        record.sample_id = "mutated"  # type: ignore[misc]


def test_candidate_support_parser_snapshots_before_validation_to_block_toctou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    original_sample_id = payload["sample_id"]
    original_candidate_id = payload["candidates"][0]["candidate_id"]
    validation_started = threading.Event()
    caller_mutated = threading.Event()
    original_strict_bool = gvs_support._strict_bool
    first_validation = True

    def pausing_strict_bool(value: object, *, label: str) -> bool:
        nonlocal first_validation
        if first_validation:
            first_validation = False
            validation_started.set()
            if not caller_mutated.wait(timeout=5):
                raise AssertionError("caller mutation did not complete during validation")
        return original_strict_bool(value, label=label)

    def mutate_caller_payload() -> None:
        if not validation_started.wait(timeout=5):
            return
        payload["sample_id"] = "record-mutated-during-validation"
        payload["candidates"][0]["candidate_id"] = "mutated-candidate-during-validation"
        caller_mutated.set()

    monkeypatch.setattr(gvs_support, "_strict_bool", pausing_strict_bool)
    mutation_thread = threading.Thread(target=mutate_caller_payload)
    mutation_thread.start()
    try:
        record = parse_candidate_support_record(payload)
    finally:
        caller_mutated.set()
        mutation_thread.join(timeout=5)

    assert not mutation_thread.is_alive()
    assert payload["sample_id"] == "record-mutated-during-validation"
    assert payload["candidates"][0]["candidate_id"] == "mutated-candidate-during-validation"
    assert record.sample_id == original_sample_id
    assert record.candidates[0].candidate_id == original_candidate_id


def test_invalid_generated_candidate_remains_measurable_as_incorrect() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    candidate = payload["candidates"][0]
    candidate.update(
        parse_valid=False,
        schema_valid=False,
        canonical_action_sha256=None,
    )

    record = parse_candidate_support_record(payload)
    assert not record.greedy.parse_valid
    assert not record.greedy.schema_valid
    assert not record.greedy.action_ir_exact


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parse_valid", 1, "must be a boolean"),
        ("schema_valid", "yes", "must be a boolean"),
        ("action_ir_exact", None, "must be a boolean"),
        ("output_present", 1, "must be a boolean"),
        ("truncated", "no", "must be a boolean"),
        ("rank", True, "rank must be a non-negative integer"),
        ("raw_output_sha256", "A" * 64, "must be a lowercase SHA-256"),
        ("canonical_action_sha256", "A" * 64, "must be a lowercase SHA-256"),
    ],
)
def test_candidate_support_rejects_malformed_candidate_fields(
    field: str, value: object, message: str
) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][0][field] = value
    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize(
    ("parse_valid", "schema_valid", "action_ir_exact", "message"),
    [
        (False, True, False, "schema-valid candidate must also be parse-valid"),
        (True, False, True, "Action IR exact candidate must also be schema-valid"),
    ],
)
def test_candidate_support_rejects_inconsistent_validity_claims(
    parse_valid: bool,
    schema_valid: bool,
    action_ir_exact: bool,
    message: str,
) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    candidate = payload["candidates"][0]
    candidate.update(
        parse_valid=parse_valid,
        schema_valid=schema_valid,
        action_ir_exact=action_ir_exact,
    )
    if not schema_valid:
        candidate["canonical_action_sha256"] = None
    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"output_present": False},
            "output_present must agree exactly with nullable raw_output_sha256",
        ),
        (
            {"raw_output_sha256": None},
            "output_present must agree exactly with nullable raw_output_sha256",
        ),
        (
            {"canonical_action_sha256": None},
            "schema_valid must agree exactly with nullable canonical_action_sha256",
        ),
        (
            {"schema_valid": False},
            "schema_valid must agree exactly with nullable canonical_action_sha256",
        ),
    ],
)
def test_candidate_support_rejects_inconsistent_nullable_hash_binding(
    changes: dict[str, object], message: str
) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][0].update(changes)
    with pytest.raises(GVSSupportError, match=message):
        parse_candidate_support_record(payload)


def test_truncation_without_output_preserves_both_failure_reasons() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][0].update(
        output_present=False,
        raw_output_sha256=None,
        canonical_action_sha256=None,
        parse_valid=False,
        schema_valid=False,
        truncated=True,
    )
    assert parse_candidate_support_record(payload).generation_failure_reasons == (
        "no_output",
        "truncation",
    )


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_candidate_support_requires_exact_record_schema(operation: str) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    if operation == "missing":
        del payload["source_commitment_sha256"]
    else:
        payload["unfrozen_aggregate"] = 1
    with pytest.raises(GVSSupportError, match="fields changed"):
        parse_candidate_support_record(payload)


def test_json_loader_rejects_duplicate_keys_and_non_finite_constants() -> None:
    with pytest.raises(GVSSupportError, match="duplicate JSON key"):
        loads_candidate_support_record('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(GVSSupportError, match="non-finite JSON constant"):
        loads_candidate_support_record('{"candidate_count":NaN}')


def test_candidate_support_rejects_duplicate_candidate_ids() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][1]["candidate_id"] = payload["candidates"][0]["candidate_id"]
    with pytest.raises(GVSSupportError, match="duplicate candidate IDs"):
        parse_candidate_support_record(payload)


def test_duplicate_raw_and_canonical_outputs_remain_all_eight_attempts() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    first = payload["candidates"][0]
    second = payload["candidates"][1]
    second["raw_output_sha256"] = first["raw_output_sha256"]
    second["canonical_action_sha256"] = first["canonical_action_sha256"]
    record = parse_candidate_support_record(payload)
    assert len(record.candidates) == FROZEN_CANDIDATE_COUNT
    assert record.effective_k == 7
    assert not record.generation_failure


def test_duplicate_raw_output_cannot_claim_a_different_canonical_action() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][1]["raw_output_sha256"] = payload["candidates"][0]["raw_output_sha256"]
    with pytest.raises(GVSSupportError, match="inconsistent deterministic scores"):
        parse_candidate_support_record(payload)


def test_duplicate_canonical_action_cannot_claim_different_exactness() -> None:
    payload = _record_payload(_identity(), greedy_exact=True, oracle_exact=True)
    payload["candidates"][1]["canonical_action_sha256"] = payload["candidates"][0][
        "canonical_action_sha256"
    ]
    with pytest.raises(GVSSupportError, match="inconsistent exact-match scores"):
        parse_candidate_support_record(payload)


def test_candidate_support_rejects_rank_reordering() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidates"][0]["rank"] = 1
    with pytest.raises(GVSSupportError, match="candidate ranks"):
        parse_candidate_support_record(payload)


@pytest.mark.parametrize("candidate_count", [7, 9, True])
def test_candidate_support_rejects_non_frozen_k(candidate_count: object) -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=False)
    payload["candidate_count"] = candidate_count
    with pytest.raises(GVSSupportError, match="exactly K=8"):
        parse_candidate_support_record(payload)


def test_population_rejects_duplicate_ids_and_empty_support_classes() -> None:
    efficacy = _identity(task_class=SupportClass.EFFICACY)
    safety = replace(
        _identity(task_class=SupportClass.SAFETY),
        population_record_sha256=_sha("different-record"),
    )
    with pytest.raises(GVSSupportError, match="duplicate sample IDs"):
        SupportPopulation(
            GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
            FROZEN_CANDIDATE_COUNT,
            efficacy.firewall_sha256,
            efficacy.scan_evidence_sha256,
            (efficacy, safety),
        )
    with pytest.raises(GVSSupportError, match="at least one safety"):
        SupportPopulation(
            GVS_SUPPORT_POPULATION_SCHEMA_VERSION,
            FROZEN_CANDIDATE_COUNT,
            efficacy.firewall_sha256,
            efficacy.scan_evidence_sha256,
            (efficacy,),
        )


def test_support_population_is_derived_only_from_eligible_d_support() -> None:
    fixture = _support_fixture(include_excluded=True)
    assert len(fixture.population.samples) == 20
    assert {sample.sample_id for sample in fixture.population.samples} == {
        f"record-sample-{index:02d}" for index in range(20)
    }
    assert "record-excluded" not in {sample.sample_id for sample in fixture.population.samples}
    assert all(
        sample.firewall_sha256 == fixture.firewall["firewall_sha256"]
        for sample in fixture.population.samples
    )
    assert all(
        sample.scan_evidence_sha256 == fixture.scan_evidence["scan_evidence_sha256"]
        for sample in fixture.population.samples
    )


def test_support_derivation_rejects_tuple_to_list_normalization() -> None:
    fixture = _support_fixture()
    membership = copy.deepcopy(fixture.membership)
    membership["D-support"] = tuple(membership["D-support"])  # type: ignore[assignment]

    with pytest.raises(GVSSupportError, match="exact built-in JSON"):
        derive_support_population(
            population_records=fixture.population_records,
            expected_membership=membership,
            expected_scan_bindings=fixture.bindings,
            scan_evidence=fixture.scan_evidence,
            firewall_manifest=fixture.firewall,
            compared_release_cutoff_utc=CUTOFF,
            scan_evidence_frozen_at_utc=SCAN_FROZEN,
            firewall_frozen_at_utc=FIREWALL_FROZEN,
        )


def test_exact_support_metrics_and_boundary_prototype_pass() -> None:
    fixture = _support_fixture()
    evaluation = _score(fixture)

    assert evaluation.metrics.row_count == 20
    assert evaluation.metrics.component_count == 20
    assert evaluation.metrics.greedy_exact == ExactRate(16, 20)
    assert evaluation.metrics.oracle_pass_at_k == ExactRate(18, 20)
    assert evaluation.metrics.oracle_minus_greedy == Fraction(1, 10)
    assert evaluation.metrics.greedy_failure_recovery == ExactRate(2, 4)
    assert evaluation.metrics.safety_support == ExactRate(4, 5)
    assert evaluation.metrics.candidate_slot_count == 160
    assert evaluation.metrics.candidate_output_present == ExactRate(160, 160)
    assert evaluation.metrics.candidate_parse_valid == ExactRate(160, 160)
    assert evaluation.metrics.candidate_schema_valid == ExactRate(160, 160)
    assert evaluation.metrics.candidate_truncated == ExactRate(0, 160)
    assert evaluation.metrics.effective_k_total == 160
    assert evaluation.metrics.effective_k_mean == Fraction(8, 1)
    assert evaluation.metrics.effective_k_distribution == (0, 0, 0, 0, 0, 0, 0, 0, 20)
    assert evaluation.metrics.generation_failure_rows == ExactRate(0, 20)
    assert evaluation.metrics.generation_failure_components == ExactRate(0, 20)
    assert evaluation.gate.prototype_passed
    gate = evaluation.gate.to_record()
    assert gate["authorizes_model_or_label_access"] is False
    assert gate["prototype_passed"] is True
    assert "passed" not in gate
    assert evaluation.to_record()["population_sha256"] == fixture.population.sha256
    _assert_no_float(evaluation.to_record())


def test_repeated_rows_do_not_inflate_primary_component_denominators() -> None:
    fixture = _support_fixture(shared_component_pair=(0, 1))
    evaluation = _score(fixture)

    assert evaluation.metrics.row_count == 20
    assert evaluation.metrics.candidate_slot_count == 160
    assert evaluation.metrics.component_count == 19
    assert evaluation.metrics.greedy_exact == ExactRate(15, 19)
    assert evaluation.metrics.oracle_pass_at_k == ExactRate(17, 19)
    repeated = [component for component in evaluation.components if component.row_count == 2]
    assert len(repeated) == 1
    assert repeated[0].member_sample_ids == (
        "record-sample-00",
        "record-sample-01",
    )


def test_multi_row_component_uses_conservative_all_member_conjunction() -> None:
    fixture = _support_fixture(shared_component_pair=(16, 18))
    evaluation = _score(fixture)
    component = next(component for component in evaluation.components if component.row_count == 2)
    member_rows = [row for row in evaluation.rows if row.sample_id in component.member_sample_ids]

    assert [row.oracle_exact for row in member_rows] == [True, False]
    assert not component.greedy_exact
    assert not component.oracle_exact
    assert evaluation.metrics.oracle_pass_at_k == ExactRate(17, 19)
    assert evaluation.metrics.safety_support == ExactRate(3, 4)


def test_every_categorical_and_boolean_stratum_is_component_scored() -> None:
    evaluation = _score(_support_fixture(shared_component_pair=(0, 1)))
    by_key = {
        (stratum.dimension, stratum.value): stratum for stratum in evaluation.component_strata
    }
    expected_keys = {
        *(("task_class", value) for value in ("efficacy", "safety")),
        *(("expected_outcome", value) for value in EXPECTED_OUTCOMES),
        *(("action_family", value) for value in ACTION_FAMILIES),
        *(
            (f"strata.{name}", value)
            for name in AUDITED_BOOLEAN_STRATA
            for value in ("false", "true")
        ),
    }
    assert set(by_key) == expected_keys
    assert len(by_key) == len(evaluation.component_strata)

    for dimension in (
        "task_class",
        "expected_outcome",
        "action_family",
        *(f"strata.{name}" for name in AUDITED_BOOLEAN_STRATA),
    ):
        assert (
            sum(
                stratum.component_count
                for stratum in evaluation.component_strata
                if stratum.dimension == dimension
            )
            == evaluation.metrics.component_count
        )
    for stratum in evaluation.component_strata:
        record = stratum.to_record()
        assert "threshold" not in record
        assert "passed" not in record
        if stratum.component_count:
            assert stratum.greedy_exact is not None
            assert stratum.greedy_exact.denominator == stratum.component_count
            assert stratum.oracle_pass_at_k is not None
            assert stratum.oracle_pass_at_k.denominator == stratum.component_count
        else:
            assert stratum.greedy_exact is None
            assert record["greedy_exact"]["exact"] is None


def test_duplicate_outputs_reduce_effective_k_without_dropping_slots() -> None:
    fixture = _support_fixture()
    records = list(fixture.records)
    candidates = list(records[19].candidates)
    candidates[7] = replace(
        candidates[7],
        raw_output_sha256=candidates[6].raw_output_sha256,
        canonical_action_sha256=candidates[6].canonical_action_sha256,
    )
    records[19] = replace(records[19], candidates=tuple(candidates))

    evaluation = _score(fixture, records)
    assert evaluation.metrics.candidate_slot_count == 160
    assert evaluation.metrics.candidate_schema_valid == ExactRate(160, 160)
    assert evaluation.metrics.effective_k_total == 159
    assert evaluation.metrics.effective_k_distribution[7] == 1
    assert evaluation.metrics.effective_k_distribution[8] == 19
    assert not evaluation.rows[19].generation_failure
    assert evaluation.gate.prototype_passed


def test_parse_invalid_slot_is_measured_but_is_not_a_generation_failure() -> None:
    fixture = _support_fixture()
    records = list(fixture.records)
    candidates = list(records[19].candidates)
    candidates[7] = replace(
        candidates[7],
        canonical_action_sha256=None,
        parse_valid=False,
        schema_valid=False,
        action_ir_exact=False,
    )
    records[19] = replace(records[19], candidates=tuple(candidates))

    evaluation = _score(fixture, records)
    assert evaluation.metrics.candidate_output_present == ExactRate(160, 160)
    assert evaluation.metrics.candidate_parse_valid == ExactRate(159, 160)
    assert evaluation.metrics.candidate_schema_valid == ExactRate(159, 160)
    assert evaluation.metrics.generation_failure_rows == ExactRate(0, 20)
    assert evaluation.rows[19].generation_failure_reasons == ()
    assert evaluation.gate.prototype_passed


@pytest.mark.parametrize(
    ("failure_kind", "expected_reasons"),
    [
        ("no_output", ("no_output",)),
        ("no_valid_candidate", ("no_valid_candidate",)),
        ("truncation", ("truncation",)),
    ],
)
def test_only_frozen_row_generation_failures_fail_the_prototype_gate(
    failure_kind: str, expected_reasons: tuple[str, ...]
) -> None:
    fixture = _support_fixture()
    records = list(fixture.records)
    row = records[19]
    candidates = list(row.candidates)
    if failure_kind == "no_output":
        candidates[7] = replace(
            candidates[7],
            output_present=False,
            raw_output_sha256=None,
            canonical_action_sha256=None,
            parse_valid=False,
            schema_valid=False,
            action_ir_exact=False,
        )
    elif failure_kind == "no_valid_candidate":
        candidates = [
            replace(
                candidate,
                canonical_action_sha256=None,
                schema_valid=False,
                action_ir_exact=False,
            )
            for candidate in candidates
        ]
    else:
        candidates[7] = replace(candidates[7], truncated=True)
    records[19] = replace(row, candidates=tuple(candidates))

    evaluation = _score(fixture, records)
    row_metrics = evaluation.rows[19]
    assert row_metrics.generation_failure
    assert row_metrics.generation_failure_reasons == expected_reasons
    assert evaluation.metrics.generation_failure_rows == ExactRate(1, 20)
    assert evaluation.metrics.generation_failure_components == ExactRate(1, 20)
    assert not evaluation.gate.generation_failure_rows_are_zero
    assert not evaluation.gate.prototype_passed
    reason_counts = evaluation.metrics.to_record()["generation_failure"]["reason_rows"]
    assert reason_counts[failure_kind] == 1


def test_support_score_is_independent_of_input_record_order() -> None:
    fixture = _support_fixture()
    assert _score(fixture) == _score(fixture, list(reversed(fixture.records)))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_support_score_rejects_changed_membership(mutation: str) -> None:
    fixture = _support_fixture()
    if mutation == "missing":
        poisoned = fixture.records[:-1]
    elif mutation == "extra":
        poisoned = [
            *fixture.records,
            parse_candidate_support_record(
                _record_payload(
                    _identity("record-unexpected", SupportClass.EFFICACY),
                    greedy_exact=False,
                    oracle_exact=False,
                )
            ),
        ]
    else:
        poisoned = [*fixture.records, fixture.records[0]]
    with pytest.raises(GVSSupportError, match="membership changed|duplicate sample IDs"):
        _score(fixture, poisoned)


def test_excluded_d_support_row_cannot_be_mixed_into_score_membership() -> None:
    fixture = _support_fixture(include_excluded=True)
    included = fixture.population.samples[0]
    excluded_identity = replace(
        included,
        sample_id="record-excluded",
        population_record_sha256=fixture.population_records["D-support"][-1]["integrity"][
            "record_sha256"
        ],
    )
    excluded = parse_candidate_support_record(
        _record_payload(excluded_identity, greedy_exact=True, oracle_exact=True)
    )
    mixed = [*fixture.records[:-1], excluded]

    with pytest.raises(GVSSupportError, match="membership changed"):
        _score(fixture, mixed)


def test_score_rederivation_rejects_a_fabricated_support_population() -> None:
    fixture = _support_fixture()
    samples = list(fixture.population.samples)
    samples[0] = replace(samples[0], component_id=_sha("fabricated-component"))
    fabricated = replace(fixture.population, samples=tuple(samples))

    with pytest.raises(GVSSupportError, match="firewall-derived"):
        _score(fixture, population=fabricated)


@pytest.mark.parametrize(
    "field_name",
    [
        "firewall_sha256",
        "scan_evidence_sha256",
        "population_record_sha256",
        "component_id",
        "source_commitment_sha256",
        "task_class",
        "expected_outcome",
        "action_family",
        "strata",
    ],
)
def test_support_score_rejects_every_changed_identity_binding(field_name: str) -> None:
    fixture = _support_fixture()
    records = list(fixture.records)
    first = records[0]
    replacements: dict[str, object] = {
        "firewall_sha256": _sha("different-firewall"),
        "scan_evidence_sha256": _sha("different-scan"),
        "population_record_sha256": _sha("different-population-record"),
        "component_id": _sha("different-component"),
        "source_commitment_sha256": _sha("different-source"),
        "task_class": SupportClass.EFFICACY,
        "expected_outcome": "ABSTAIN",
        "action_family": "maps",
        "strata": SupportStrata.from_mapping(
            {name: not value for name, value in first.strata.values}
        ),
    }
    records[0] = replace(first, **{field_name: replacements[field_name]})

    with pytest.raises(GVSSupportError, match=field_name):
        _score(fixture, records)


def test_support_score_rejects_globally_duplicate_candidate_ids() -> None:
    fixture = _support_fixture()
    records = list(fixture.records)
    candidates = list(records[1].candidates)
    candidates[0] = replace(candidates[0], candidate_id=records[0].candidates[0].candidate_id)
    records[1] = replace(records[1], candidates=tuple(candidates))
    with pytest.raises(GVSSupportError, match="globally unique"):
        _score(fixture, records)


def test_undefined_recovery_is_explicit_and_fails_prototype_gate() -> None:
    fixture = _support_fixture()
    all_greedy: list[CandidateSupportRecord] = []
    for record in fixture.records:
        candidates = list(record.candidates)
        candidates[0] = replace(candidates[0], action_ir_exact=True)
        all_greedy.append(replace(record, candidates=tuple(candidates)))

    evaluation = _score(fixture, all_greedy)
    assert evaluation.metrics.greedy_failure_recovery is None
    recovery = evaluation.metrics.to_record()["greedy_failure_recovery"]
    assert recovery == {"defined": False, "denominator": 0, "exact": None, "numerator": 0}
    assert not evaluation.gate.greedy_failure_recovery_at_least_minimum
    assert not evaluation.gate.prototype_passed
    assert evaluation.gate.to_record()["authorizes_model_or_label_access"] is False


def test_denominators_cannot_be_supplied_or_rewritten() -> None:
    with pytest.raises(GVSSupportError, match="positive denominator"):
        ExactRate(0, 0)
    fixture = _support_fixture()
    metrics = _score(fixture).metrics
    with pytest.raises(GVSSupportError, match="safety support denominator"):
        replace(metrics, safety_support=ExactRate(1, 3))
    with pytest.raises(GVSSupportError, match="all attempted candidate slots"):
        replace(metrics, candidate_schema_valid=ExactRate(159, 159))
    with pytest.raises(GVSSupportError, match="expected components"):
        replace(metrics, greedy_exact=ExactRate(16, 21))


def test_safety_support_below_threshold_fails_prototype_gate() -> None:
    fixture = _support_fixture()
    records = copy.deepcopy(fixture.records)
    safety_exact = records[2]
    candidates = list(safety_exact.candidates)
    candidates[0] = replace(candidates[0], action_ir_exact=False)
    records[2] = replace(safety_exact, candidates=tuple(candidates))

    evaluation = _score(fixture, records)
    assert evaluation.metrics.safety_support == ExactRate(3, 5)
    assert not evaluation.gate.safety_support_at_least_minimum
    assert not evaluation.gate.prototype_passed


def test_record_json_round_trip_is_stable() -> None:
    payload = _record_payload(_identity(), greedy_exact=False, oracle_exact=True)
    record = loads_candidate_support_record(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    )
    assert record.to_record() == payload
    assert isinstance(record, CandidateSupportRecord)


def test_prototype_gate_is_factory_only_and_explicitly_nonauthorizing() -> None:
    with pytest.raises(TypeError, match="evaluate_prototype_support_gate"):
        gvs_support.PrototypeSupportGate(
            candidate_count_is_frozen=True,
            oracle_pass_at_k_at_least_minimum=True,
            oracle_minus_greedy_at_least_minimum=True,
            greedy_failure_recovery_at_least_minimum=True,
            safety_support_at_least_minimum=True,
            generation_failure_rows_are_zero=True,
            prototype_passed=True,
        )

    evaluation = _score(_support_fixture())
    for record in (evaluation.gate.to_record(), evaluation.to_record()):
        assert record["authorizes_model_or_label_access"] is False
        assert record["authorizes_cuda_or_jarvis_access"] is False
        assert record["launch_authorized"] is False
        assert record["preliminary_plumbing_only"] is True


def test_outcome_chosen_prototype_threshold_replacement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _score(_support_fixture())
    monkeypatch.setattr(
        gvs_support,
        "PROTOTYPE_MINIMUM_ORACLE_PASS_AT_K",
        Fraction(0, 1),
    )
    with pytest.raises(GVSSupportError, match="threshold contract changed after import"):
        gvs_support.evaluate_prototype_support_gate(evaluation.metrics)
    with pytest.raises(GVSSupportError, match="threshold contract changed after import"):
        evaluation.gate.to_record()


def test_population_validator_runtime_substitution_fails_before_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _support_fixture()
    monkeypatch.setattr(gvs_support, "validate_scan_evidence", lambda *_args, **_kwargs: {})
    with pytest.raises(GVSSupportError, match="validation dependency changed after import"):
        derive_support_population(**fixture.score_kwargs)
