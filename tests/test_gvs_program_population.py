from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import pytest

import barunlm.evaluation.gvs_program_population as program_population_module
from barunlm.evaluation.action_simulator import verify_program_round_trip
from barunlm.evaluation.gvs_program_population import (
    EXPECTED_OUTCOMES,
    FEATURE_STRATA,
    HUMAN_ROLES_EXCLUDED,
    MAX_PLAN_JSON_BYTES,
    MAX_RECORDS_PER_ROLE,
    MAX_SPEC_JSON_BYTES,
    MAX_TOTAL_RECORDS,
    OPERATION_STRATA,
    OUTCOME_STRATA,
    PROGRAM_POPULATION_PLAN_VERSION,
    REQUIRED_STRATA,
    SEMANTIC_OPERATIONS,
    TRAINING_ROLES,
    GVSProgramPopulationError,
    ProgramPopulationSpec,
    RolePlanSpec,
    StratumFloor,
    build_program_population_plan,
    loads_program_population_plan,
    loads_program_population_spec,
    make_program_population_spec,
    planner_runtime_sha256,
)

SOURCE_COMMITMENT = "1" * 64
TEMPORAL_OPERATIONS = {
    "CREATE_CALENDAR_EVENT",
    "CREATE_REMINDER",
    "RESCHEDULE_CALENDAR_EVENT",
    "UPDATE_REMINDER",
}
AUTHORIZATION_KEYS = {
    "authorizes_model_access",
    "authorizes_label_access",
    "authorizes_cuda",
    "authorizes_jarvislabs",
    "authorizes_training",
    "authorizes_launch",
    "authorizes_execution",
}
FORBIDDEN_PACKET_KEYS = {
    "answer",
    "case_id",
    "completion",
    "final_request",
    "gold",
    "label",
    "logits",
    "model",
    "model_id",
    "membership_sha256",
    "packet_id",
    "prediction",
    "prompt",
    "request",
    "request_text",
    "role_assignment_sha256",
    "schema_alias_family_id",
    "source_commitment_sha256",
    "target",
    "teacher",
    "tool_call",
}


def _floors(**overrides: int) -> tuple[StratumFloor, ...]:
    values = {stratum: 1 for stratum in REQUIRED_STRATA}
    values.update(overrides)
    return tuple(StratumFloor(stratum, values[stratum]) for stratum in REQUIRED_STRATA)


def _spec(
    *,
    count: int = 24,
    source_sha256: str = SOURCE_COMMITMENT,
    floors: tuple[StratumFloor, ...] | None = None,
) -> ProgramPopulationSpec:
    selected = floors or _floors(
        context_grounding=4,
        revision=3,
        disfluency=3,
        distractor_tools=3,
        timezone_or_relative_time=4,
        multi_action=4,
        identity_schema=8,
        renamed_schema=6,
        unsafe_or_adversarial=2,
    )
    return make_program_population_spec(
        source_commitment_sha256=source_sha256,
        roles=(
            RolePlanSpec("T-new", count, selected),
            RolePlanSpec("D-support", count, selected),
        ),
    )


@pytest.fixture(scope="module")
def plan():
    return build_program_population_plan(_spec())


def _walk_json(value: object):
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _walk_json(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_json(child)


def test_contract_surface_has_all_semantics_and_bounded_proposed_scale() -> None:
    assert len(SEMANTIC_OPERATIONS) == 13
    assert set(EXPECTED_OUTCOMES) == {"ACTION", "CONFIRM", "CLARIFY", "ABSTAIN"}
    assert set(OPERATION_STRATA) == {f"operation:{kind}" for kind in SEMANTIC_OPERATIONS}
    assert set(OUTCOME_STRATA) == {f"outcome:{outcome}" for outcome in EXPECTED_OUTCOMES}
    assert set(FEATURE_STRATA) == {
        "context_grounding",
        "revision",
        "disfluency",
        "distractor_tools",
        "timezone_or_relative_time",
        "multi_action",
        "identity_schema",
        "renamed_schema",
        "unsafe_or_adversarial",
    }
    assert MAX_RECORDS_PER_ROLE >= 24_000
    assert MAX_TOTAL_RECORDS >= 26_000
    assert MAX_PLAN_JSON_BYTES > MAX_SPEC_JSON_BYTES
    minimum_floors = _floors()
    large = make_program_population_spec(
        source_commitment_sha256=SOURCE_COMMITMENT,
        roles=(
            RolePlanSpec("T-new", 24_000, minimum_floors),
            RolePlanSpec("D-support", 2_000, minimum_floors),
        ),
    )
    assert [role.count for role in large.roles] == [24_000, 2_000]

    t_outcomes = program_population_module._allocate_outcomes(SOURCE_COMMITMENT, large.roles[0])
    d_outcomes = program_population_module._allocate_outcomes(SOURCE_COMMITMENT, large.roles[1])
    assert Counter(t_outcomes.values()) == Counter(
        {"ACTION": 6_000, "CONFIRM": 6_000, "CLARIFY": 6_000, "ABSTAIN": 6_000}
    )
    assert Counter(d_outcomes.values()) == Counter(
        {"ACTION": 500, "CONFIRM": 500, "CLARIFY": 500, "ABSTAIN": 500}
    )


def test_plan_is_deterministic_and_has_stable_golden_membership(plan) -> None:
    rebuilt = build_program_population_plan(_spec())
    assert rebuilt.canonical_json() == plan.canonical_json()
    assert (
        plan.membership_sha256 == "c95151afb8c79fbeb010cdb47657b9aa6aaa77937daa970ab40d544c7d694fdd"
    )
    assert plan.records[0].skeleton.record_id == "gvs.t.0d1376201585a5c6b5f0dffe"
    assert plan.records[0].program_sha256 == (
        "a315a1b98c4aae2e2caef79ddc02a1a3a5110d74c7b8fde1884c58ae2a7e2f32"
    )
    assert plan.records[24].skeleton.record_id == "gvs.d.6d962488578a9befaa0722af"
    assert plan.records[24].program_sha256 == (
        "31db986795a8fa41e6844e5c13acc34aaca369317513795a047763f2bed93201"
    )


def test_membership_is_frozen_before_packets_and_families_are_disjoint(plan) -> None:
    assert plan.to_dict()["role_assignment_precedes_text_authoring"] is True
    assert plan.to_dict()["contains_authored_request_text"] is False
    memberships = plan.memberships
    assert set(memberships) == set(TRAINING_ROLES)
    assert not (set(memberships["T-new"]) & set(memberships["D-support"]))

    for field in (
        "record_id",
        "program_cluster_id",
        "authoring_slot_id",
        "entity_pool_slot_id",
        "temporal_construction_slot_id",
    ):
        by_role = {
            role: {
                getattr(record.skeleton, field)
                for record in plan.records
                if record.skeleton.role == role
            }
            for role in TRAINING_ROLES
        }
        assert len(by_role["T-new"]) == 24
        assert len(by_role["D-support"]) == 24
        assert by_role["T-new"].isdisjoint(by_role["D-support"])
    assert plan.to_dict()["authored_provenance_complete"] is False
    assert plan.to_dict()["joint_duplicate_closure_complete"] is False


def test_every_role_meets_all_caller_floors_and_coverage(plan) -> None:
    for role_spec in plan.spec.roles:
        counts: Counter[str] = Counter()
        for record in plan.records:
            if record.skeleton.role == role_spec.role:
                counts.update(record.skeleton.strata)
        for floor in role_spec.floors:
            assert counts[floor.stratum] >= floor.minimum
        assert {
            stratum.removeprefix("operation:")
            for stratum in counts
            if stratum.startswith("operation:")
        } == set(SEMANTIC_OPERATIONS)
        assert {
            stratum.removeprefix("outcome:") for stratum in counts if stratum.startswith("outcome:")
        } == set(EXPECTED_OUTCOMES)


def test_special_strata_are_semantically_real_and_schema_modes_partition(plan) -> None:
    for role in TRAINING_ROLES:
        role_records = [record for record in plan.records if record.skeleton.role == role]
        identity = [
            record for record in role_records if "identity_schema" in record.skeleton.strata
        ]
        renamed = [record for record in role_records if "renamed_schema" in record.skeleton.strata]
        assert len(identity) + len(renamed) == len(role_records)
        assert not any(
            "identity_schema" in record.skeleton.strata
            and "renamed_schema" in record.skeleton.strata
            for record in role_records
        )
        for record in role_records:
            step = record.program.steps[0]
            if "multi_action" in record.skeleton.strata:
                assert len(step.operations) == 2
                assert step.mode is not None and step.mode.value == "PARALLEL"
            if "timezone_or_relative_time" in record.skeleton.strata:
                assert record.skeleton.primary_operation in TEMPORAL_OPERATIONS
                fact_names = {fact["name"] for fact in record.authoring_packet["semantic_facts"]}
                assert {
                    "reference_time",
                    "reference_timezone",
                    "temporal_expression_rule",
                } <= fact_names
            if "unsafe_or_adversarial" in record.skeleton.strata:
                assert record.skeleton.expected_outcome == "ABSTAIN"
                assert step.operations == ()
            expected_presentation = (
                "renamed" if "renamed_schema" in record.skeleton.strata else "identity"
            )
            assert record.authoring_packet["schema_presentation"] == expected_presentation


def test_programs_and_worlds_are_exact_and_live_round_trip_verified(plan) -> None:
    for record in plan.records:
        assert record.program.sha256() == record.program_sha256
        assert record.program.initial_state.sha256() == record.world_state_sha256
        receipt = verify_program_round_trip(record.program)
        assert receipt.passed is True
        assert receipt.sha256() == record.round_trip_receipt_sha256


def test_packets_are_immutable_structured_facts_not_requests_or_answers(plan) -> None:
    for record in plan.records:
        packet = record.authoring_packet
        assert isinstance(packet, MappingProxyType)
        assert not (set(packet) & FORBIDDEN_PACKET_KEYS)
        assert set(packet) == program_population_module._AUTHORING_PACKET_FIELDS
        assert packet["contains_authored_request_text"] is False
        assert packet["contains_model_generated_text"] is False
        assert packet["contains_teacher_answer"] is False
        assert packet["author_must_create_original_request_after_membership_freeze"] is True
        assert packet["semantic_facts"]
        assert packet["prohibited_answer_leakage"]
        assert record.authoring_packet_sha256 == program_population_module._sha256_json(
            program_population_module._mutable_json(packet),
            limit=MAX_SPEC_JSON_BYTES,
        )
        serialized = json.dumps(program_population_module._mutable_json(packet))
        assert record.skeleton.record_id not in serialized
        assert record.skeleton.role_assignment_sha256 not in serialized
        assert plan.membership_sha256 not in serialized
        with pytest.raises(TypeError):
            packet["request_text"] = "injected"  # type: ignore[index]


def test_every_authorization_surface_is_false_and_human_roles_are_excluded(plan) -> None:
    payload = plan.to_dict()
    assert payload["schema_version"] == PROGRAM_POPULATION_PLAN_VERSION
    assert payload["structural_only"] is True
    assert payload["pre_authoring_only"] is True
    assert payload["authored_provenance_complete"] is False
    assert payload["joint_duplicate_closure_complete"] is False
    assert payload["populations"] == list(TRAINING_ROLES)
    assert payload["excluded_human_populations"] == list(HUMAN_ROLES_EXCLUDED)
    for value in _walk_json(payload):
        if type(value) is dict:
            for key in AUTHORIZATION_KEYS & set(value):
                assert value[key] is False


def test_bounded_spec_and_plan_loaders_round_trip_with_live_reconstruction(plan) -> None:
    spec_text = json.dumps(plan.spec.to_dict(), sort_keys=True, separators=(",", ":"))
    loaded_spec = loads_program_population_spec(spec_text)
    assert loaded_spec.to_dict() == plan.spec.to_dict()
    loaded_plan = loads_program_population_plan(plan.canonical_json())
    assert loaded_plan.canonical_json() == plan.canonical_json()
    assert loaded_plan.sha256() == plan.sha256()


@pytest.mark.parametrize(
    "injection",
    [
        {"random_seed": 17},
        {"wall_clock": "now"},
        {"network_url": "https://invalid.example"},
        {"filesystem_path": "/tmp/hidden"},
        {"model_id": "teacher"},
        {"prediction": {"decision": "CALL"}},
        {"label": "CONFIRM"},
        {"cuda_device": 0},
        {"jarvis_instance": 1},
    ],
)
def test_spec_loader_rejects_hidden_runtime_model_and_outcome_inputs(
    injection: dict[str, object],
) -> None:
    payload = _spec().to_dict()
    payload.update(injection)
    with pytest.raises(GVSProgramPopulationError, match="fields are not exact"):
        loads_program_population_spec(json.dumps(payload))


def test_builder_rejects_mutable_custom_and_forged_inputs() -> None:
    floors = _floors()
    roles = (
        RolePlanSpec("T-new", 20, floors),
        RolePlanSpec("D-support", 20, floors),
    )
    with pytest.raises(GVSProgramPopulationError, match="immutable tuple"):
        make_program_population_spec(
            source_commitment_sha256=SOURCE_COMMITMENT,
            roles=list(roles),  # type: ignore[arg-type]
        )
    with pytest.raises(GVSProgramPopulationError, match="exact tuple"):
        RolePlanSpec("T-new", 20, list(floors))  # type: ignore[arg-type]

    class CustomTuple(tuple):
        pass

    with pytest.raises(GVSProgramPopulationError, match="exact tuple"):
        RolePlanSpec("T-new", 20, CustomTuple(floors))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be created"):
        ProgramPopulationSpec(SOURCE_COMMITMENT, roles)
    with pytest.raises(GVSProgramPopulationError, match="exact ProgramPopulationSpec"):
        build_program_population_plan(object())  # type: ignore[arg-type]


def test_private_factory_tokens_cannot_forge_record_or_plan_commitments(plan) -> None:
    record = plan.records[0]
    with pytest.raises(GVSProgramPopulationError, match="program_sha256"):
        program_population_module.ProgramPopulationRecord(
            skeleton=record.skeleton,
            program=record.program,
            authoring_packet=record.authoring_packet,
            program_sha256="0" * 64,
            world_state_sha256=record.world_state_sha256,
            round_trip_receipt_sha256=record.round_trip_receipt_sha256,
            authoring_packet_sha256=record.authoring_packet_sha256,
            record_sha256=record.record_sha256,
            _factory_token=program_population_module._RECORD_FACTORY_TOKEN,
        )

    with pytest.raises(GVSProgramPopulationError, match="membership_sha256"):
        program_population_module.ProgramPopulationPlan(
            spec=plan.spec,
            records=plan.records,
            planner_source_sha256=plan.planner_source_sha256,
            planner_runtime_sha256=plan.planner_runtime_sha256,
            simulator_program_runtime_sha256=plan.simulator_program_runtime_sha256,
            simulator_action_runtime_sha256=plan.simulator_action_runtime_sha256,
            membership_sha256="0" * 64,
            roster_sha256=plan.roster_sha256,
            record_set_sha256=plan.record_set_sha256,
            artifact_body_sha256=plan.artifact_body_sha256,
            _factory_token=program_population_module._PLAN_FACTORY_TOKEN,
        )


def test_builder_detaches_one_spec_snapshot_from_later_caller_mutation() -> None:
    caller_spec = _spec()
    built = build_program_population_plan(caller_spec)
    assert built.spec is not caller_spec
    object.__setattr__(caller_spec, "source_commitment_sha256", "2" * 64)
    assert built.spec.source_commitment_sha256 == SOURCE_COMMITMENT
    assert built.to_dict()["source_commitment_sha256"] == SOURCE_COMMITMENT


def test_runtime_import_substitution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(program_population_module, "Path", object)
    with pytest.raises(GVSProgramPopulationError, match="import Path differs"):
        build_program_population_plan(_spec())


def test_runtime_builtin_substitution_cannot_bias_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        program_population_module,
        "min",
        lambda values, **kwargs: tuple(values)[-1],
        raising=False,
    )
    with pytest.raises(GVSProgramPopulationError, match="builtin min differs"):
        build_program_population_plan(_spec())


@pytest.mark.parametrize(
    ("count", "overrides", "match"),
    [
        (15, {"unsafe_or_adversarial": 10}, "require"),
        (15, {"identity_schema": 8, "renamed_schema": 8}, "disjoint roster"),
        (15, {"multi_action": 15}, "operational-row capacity"),
        (15, {"timezone_or_relative_time": 14}, "require"),
    ],
)
def test_infeasible_floor_sets_fail_before_program_construction(
    count: int, overrides: dict[str, int], match: str
) -> None:
    with pytest.raises(GVSProgramPopulationError, match=match):
        RolePlanSpec("T-new", count, _floors(**overrides))


def test_missing_duplicate_unknown_and_human_role_floors_fail_closed() -> None:
    floors = _floors()
    with pytest.raises(GVSProgramPopulationError, match="canonical order"):
        RolePlanSpec("T-new", 20, floors[:-1])
    with pytest.raises(GVSProgramPopulationError, match="canonical order"):
        RolePlanSpec("T-new", 20, (*floors[:-1], floors[0]))
    with pytest.raises(GVSProgramPopulationError, match="unknown stratum"):
        StratumFloor("model_prediction", 1)
    with pytest.raises(GVSProgramPopulationError, match="only T-new and D-support"):
        RolePlanSpec("S-new", 20, floors)
    with pytest.raises(GVSProgramPopulationError, match="only T-new and D-support"):
        RolePlanSpec("C-new", 20, floors)


def test_strict_json_rejects_duplicates_floats_nonobjects_and_unknown_plan_fields(plan) -> None:
    with pytest.raises(GVSProgramPopulationError, match="duplicate JSON key"):
        loads_program_population_spec('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(GVSProgramPopulationError, match="floating-point"):
        loads_program_population_spec('{"value":1.5}')
    with pytest.raises(GVSProgramPopulationError, match="top-level"):
        loads_program_population_spec("[]")
    injected = plan.canonical_json()[:-1] + ',"model_prediction":"forbidden"}'
    with pytest.raises(GVSProgramPopulationError, match="fields are not exact"):
        loads_program_population_plan(injected)


def test_deep_tampering_and_cross_role_lineage_fail_live_reconstruction(plan) -> None:
    payload = copy.deepcopy(plan.to_dict())
    payload["records"][0]["authoring_packet"]["prediction"] = "forbidden"
    payload["rosters"]["D-support"][0]["authoring_slot_id"] = payload["rosters"]["T-new"][0][
        "authoring_slot_id"
    ]
    with pytest.raises(GVSProgramPopulationError, match="live deterministic reconstruction"):
        loads_program_population_plan(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def test_source_runtime_roster_and_record_hashes_are_bound(plan) -> None:
    payload = plan.to_dict()
    assert len(payload["planner_source_sha256"]) == 64
    assert payload["planner_runtime_sha256"] == planner_runtime_sha256()
    assert len(payload["membership_sha256"]) == 64
    assert len(payload["roster_sha256"]) == 64
    assert len(payload["record_set_sha256"]) == 64
    assert len(payload["artifact_body_sha256"]) == 64
    assert len(plan.sha256()) == 64


def test_runtime_imports_have_no_third_party_or_remote_surface() -> None:
    source_path = Path(__file__).parents[1] / "src/barunlm/evaluation/gvs_program_population.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    absolute_roots = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert absolute_roots <= {
        "__future__",
        "builtins",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "re",
        "sys",
        "types",
        "typing",
    }
