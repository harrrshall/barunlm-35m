from __future__ import annotations

import copy
import inspect
import json
import re
from collections import Counter

import pytest

import barunlm.evaluation.gvs_human_collection as human_module
from barunlm.evaluation.gvs_human_collection import (
    AUDITED_STRATA,
    GVS_AUTHOR_EXPORT_VERSION,
    GVS_HUMAN_COLLECTION_PACKAGE_VERSION,
    GVS_LABEL_TASK_VERSION,
    GVS_PRIVATE_LABEL_ENVELOPE_VERSION,
    GVS_PROMPT_INTAKE_VERSION,
    HUMAN_POPULATION_ROLES,
    MAX_ASSIGNMENTS,
    MAX_PROMPT_INTAKE_JSON_BYTES,
    PROVENANCE_REQUIREMENTS,
    ROLE_CLASS_REQUIREMENTS,
    TASK_CLASSES,
    GVSHumanCollectionError,
    build_human_collection_package,
    build_label_task,
    build_private_label_envelope,
    build_prompt_intake,
    canonical_json,
    export_author_tasks,
    loads_author_export,
    loads_human_collection_package,
    loads_label_task,
    loads_private_label_envelope,
    loads_prompt_intake,
    validate_author_export,
    validate_human_collection_package,
    validate_label_task,
    validate_private_label_envelope,
    validate_prompt_intake,
)

CUTOFF = "2026-08-03T00:00:00Z"
SOURCE_SHA256 = "a" * 64
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
AUTHORIZATION_FIELDS = {
    "authorizes_model_access",
    "authorizes_label_access",
    "authorizes_cuda",
    "authorizes_jarvislabs",
    "authorizes_training",
    "authorizes_launch",
    "authorizes_execution",
}
NONAUTHENTICATED_CLAIMS = {
    "human_provenance_authenticated",
    "chronology_authenticated",
    "source_custody_authenticated",
    "private_label_custody_authenticated",
    "single_use_signer_available",
    "durable_atomic_retirement_available",
    "joint_duplicate_closure_complete",
    "effective_cluster_counts_authenticated",
    "semantic_content_authentication_complete",
    "population_record_bridge_complete",
    "power_thresholds_frozen_from_real_roster",
}


def _strata(index: int, *, safety: bool) -> dict[str, bool]:
    return {
        name: safety if name == "unsafe_or_adversarial" else bool((index >> (position % 8)) & 1)
        for position, name in enumerate(AUDITED_STRATA)
    }


def _assignments() -> list[dict[str, object]]:
    assignments: list[dict[str, object]] = []
    for role in HUMAN_POPULATION_ROLES:
        prefix = "s" if role == "S-new" else "c"
        ordinal = 0
        for task_class in TASK_CLASSES:
            for class_index in range(ROLE_CLASS_REQUIREMENTS[role][task_class]):
                assignments.append(
                    {
                        "assignment_id": f"{prefix}.{ordinal:04d}",
                        "population_role": role,
                        "task_class": task_class,
                        "required_strata": _strata(
                            class_index,
                            safety=task_class == "safety",
                        ),
                    }
                )
                ordinal += 1
    return assignments


@pytest.fixture(scope="module")
def assignments() -> list[dict[str, object]]:
    return _assignments()


@pytest.fixture(scope="module")
def package(assignments: list[dict[str, object]]) -> dict[str, object]:
    return build_human_collection_package(
        assignments,
        compared_release_cutoff_utc=CUTOFF,
        assignment_source_sha256=SOURCE_SHA256,
    )


def _walk(value: object):
    yield value
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif type(value) is list:
        for child in value:
            yield from _walk(child)


def _tool_schemas() -> list[dict[str, object]]:
    return [
        {
            "name": "create_calendar_event",
            "description": "Create a calendar event in the sandbox.",
            "arguments": {
                "title": {
                    "type": "string",
                    "description": "Calendar event title.",
                }
            },
            "required": ["title"],
            "additional_arguments": False,
            "side_effecting": True,
        }
    ]


def _prompt() -> dict[str, object]:
    return {
        "request": "Schedule a review called Architecture review",
        "context": {"calendar": []},
        "reference_timestamp": "2026-08-04T05:30:00+05:30",
        "reference_fold": 0,
        "timezone": "Asia/Kolkata",
        "tool_schemas": _tool_schemas(),
    }


def _provenance() -> dict[str, object]:
    return {
        "author_id": "human-author-001",
        "source_id": "human-source-001",
        "collection_batch": "human-batch-001",
        "schema_family": "declared-schema-family-001",
        "program_template_id": "declared-template-lineage-001",
        "paraphrase_family": "declared-paraphrase-lineage-001",
        "entity_pool_ids": ["declared-entity-pool-001"],
        "temporal_construction_id": "declared-temporal-lineage-001",
        "role_assigned_at_utc": "2026-08-04T00:00:00Z",
        "authored_at_utc": "2026-08-04T00:01:00Z",
        "license": "CC-BY-4.0",
        "consent": True,
        "no_model_assistance": True,
        "authoring_protocol_revision": "human-protocol-v1",
    }


def _attestations() -> dict[str, bool]:
    return {
        "independent_human_authorship": True,
        "no_model_assistance": True,
        "original_work": True,
        "not_copied_from_benchmark": True,
        "prompt_authored_before_label": True,
    }


@pytest.fixture(scope="module")
def prompt_intake(package: dict[str, object]) -> dict[str, object]:
    return build_prompt_intake(
        package,
        assignment_id="s.0000",
        prompt=_prompt(),
        provenance_declarations=_provenance(),
        author_attestations=_attestations(),
        prompt_received_at_utc="2026-08-04T00:02:00Z",
    )


@pytest.fixture(scope="module")
def label_task(package: dict[str, object], prompt_intake: dict[str, object]) -> dict[str, object]:
    return build_label_task(
        prompt_intake,
        package=package,
        created_at_utc="2026-08-04T00:03:00Z",
    )


def _action(title: str = "Architecture review") -> dict[str, object]:
    return {
        "calls": [{"args": {"title": title}, "tool": "create_calendar_event"}],
        "decision": "CALL",
        "mode": "SINGLE",
    }


def _annotation(
    *,
    labeler_action: object | None = None,
    reviewer_action: object | None = None,
    disagreement: bool = False,
) -> dict[str, object]:
    left = _action() if labeler_action is None else labeler_action
    right = copy.deepcopy(left) if reviewer_action is None else reviewer_action
    result: dict[str, object] = {
        "labeler_id": "human-labeler-001",
        "labeler_independent_human_attestation": True,
        "labeler_no_model_assistance": True,
        "labeler_no_compared_prediction_access": True,
        "labeled_at_utc": "2026-08-04T00:04:00Z",
        "labeler_action_ir": left,
        "reviewer_id": "human-reviewer-001",
        "reviewer_independent_human_attestation": True,
        "reviewer_no_model_assistance": True,
        "reviewer_no_compared_prediction_access": True,
        "reviewed_at_utc": "2026-08-04T00:05:00Z",
        "reviewer_action_ir": right,
        "disagreement": disagreement,
        "adjudicator_id": None,
        "adjudicator_independent_human_attestation": None,
        "adjudicator_no_model_assistance": None,
        "adjudicator_no_compared_prediction_access": None,
        "adjudicated_at_utc": None,
        "adjudicated_action_ir": None,
    }
    if disagreement:
        result.update(
            {
                "adjudicator_id": "human-adjudicator-001",
                "adjudicator_independent_human_attestation": True,
                "adjudicator_no_model_assistance": True,
                "adjudicator_no_compared_prediction_access": True,
                "adjudicated_at_utc": "2026-08-04T00:05:30Z",
                "adjudicated_action_ir": copy.deepcopy(left),
            }
        )
    return result


@pytest.fixture(scope="module")
def envelope(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> dict[str, object]:
    return build_private_label_envelope(
        prompt_intake,
        label_task,
        _annotation(),
        package=package,
        sealed_at_utc="2026-08-04T00:06:00Z",
    )


def test_proposed_contract_is_explicit_about_nominal_counts_and_lineage() -> None:
    assert MAX_ASSIGNMENTS == 6_000
    assert ROLE_CLASS_REQUIREMENTS == {
        "S-new": {"efficacy": 1_200, "safety": 800, "total": 2_000},
        "C-new": {"efficacy": 2_500, "safety": 1_500, "total": 4_000},
    }
    assert set(AUDITED_STRATA) == {
        "context_grounding",
        "revision",
        "disfluency",
        "distractor_tools",
        "timezone_or_relative_time",
        "multi_action",
        "renamed_schema",
        "unseen_schema",
        "unsafe_or_adversarial",
    }
    assert set(PROVENANCE_REQUIREMENTS) >= {
        "author_id",
        "source_id",
        "collection_batch",
        "schema_family",
        "program_template_id",
        "paraphrase_family",
        "entity_pool_ids",
        "temporal_construction_id",
        "no_model_assistance",
    }


def test_complete_package_meets_only_nominal_task_quotas(package: dict[str, object]) -> None:
    assert package["schema_version"] == GVS_HUMAN_COLLECTION_PACKAGE_VERSION
    requirements = package["requirements"]
    assert requirements["nominal_role_class_task_counts"] == ROLE_CLASS_REQUIREMENTS
    assert requirements["effective_cluster_counts_require_joint_lineage_closure"] is True
    assert requirements["joint_duplicate_scan_required"] is True
    assert requirements["coordinator_bound_schema_presentation_required"] is True
    assert requirements["schema_to_simulator_population_bridge_required"] is True
    assert len(package["coordinator_manifest"]) == 6_000
    counts = Counter(
        (entry["population_role"], entry["task_class"]) for entry in package["coordinator_manifest"]
    )
    assert counts == Counter(
        {
            (role, task_class): ROLE_CLASS_REQUIREMENTS[role][task_class]
            for role in HUMAN_POPULATION_ROLES
            for task_class in TASK_CLASSES
        }
    )
    assert package["claims"]["effective_cluster_counts_authenticated"] is False
    assert package["claims"]["joint_duplicate_closure_complete"] is False
    assert package["claims"]["power_thresholds_frozen_from_real_roster"] is False


def test_package_is_deterministic_and_caller_detached(
    package: dict[str, object], assignments: list[dict[str, object]]
) -> None:
    reversed_assignments = copy.deepcopy(list(reversed(assignments)))
    rebuilt = build_human_collection_package(
        reversed_assignments,
        compared_release_cutoff_utc=CUTOFF,
        assignment_source_sha256=SOURCE_SHA256,
    )
    assert canonical_json(
        rebuilt, maximum_nodes=human_module.MAX_PACKAGE_JSON_NODES
    ) == canonical_json(package, maximum_nodes=human_module.MAX_PACKAGE_JSON_NODES)
    assert (
        package["integrity"]["package_sha256"]
        == "13eb2f97f607d85116e7257b819cbbcfbbcba944a3478e5f463cbdd393367530"
    )
    reversed_assignments[0]["task_class"] = "corrupted"
    assert package["coordinator_manifest"][0]["task_class"] in TASK_CLASSES


def test_author_export_has_no_private_ids_hashes_answers_or_generated_text(
    package: dict[str, object],
) -> None:
    export = export_author_tasks(package)
    assert export["schema_version"] == GVS_AUTHOR_EXPORT_VERSION
    private_ids = {entry["assignment_id"] for entry in package["coordinator_manifest"]}
    texts = {value for value in _walk(export) if type(value) is str}
    assert texts.isdisjoint(private_ids)
    assert not any(SHA256_RE.fullmatch(value) for value in texts)
    for flag in (
        "contains_internal_identifiers",
        "contains_integrity_hashes",
        "contains_answer_or_label",
        "contains_model_generated_text",
        *sorted(AUTHORIZATION_FIELDS),
    ):
        assert export[flag] is False
    for batch in export["task_batches"]:
        task = batch["task"]
        assert set(task["required_strata"]) == set(AUDITED_STRATA)
        assert task["contains_internal_identifiers"] is False
        assert task["contains_integrity_hashes"] is False
        assert task["contains_answer_or_label"] is False
        assert task["contains_model_generated_text"] is False
        assert "population_role" not in task
        assert "assignment_id" not in task
        assert "author_id" not in task
        assert "source_id" not in task


def test_author_export_batches_sum_to_exact_raw_task_quotas(package: dict[str, object]) -> None:
    totals = Counter()
    for batch in package["author_export"]["task_batches"]:
        task = batch["task"]
        totals[task["task_class"]] += batch["quantity"]
    assert totals == Counter(
        {
            task_class: sum(
                ROLE_CLASS_REQUIREMENTS[role][task_class] for role in HUMAN_POPULATION_ROLES
            )
            for task_class in TASK_CLASSES
        }
    )


def test_build_rejects_missing_assignment_duplicate_id_and_bad_safety_stratum(
    assignments: list[dict[str, object]],
) -> None:
    with pytest.raises(GVSHumanCollectionError, match="exact array"):
        build_human_collection_package(
            assignments[:-1],
            compared_release_cutoff_utc=CUTOFF,
            assignment_source_sha256=SOURCE_SHA256,
        )

    duplicate = copy.deepcopy(assignments)
    duplicate[-1]["assignment_id"] = duplicate[0]["assignment_id"]
    with pytest.raises(GVSHumanCollectionError, match="globally unique"):
        build_human_collection_package(
            duplicate,
            compared_release_cutoff_utc=CUTOFF,
            assignment_source_sha256=SOURCE_SHA256,
        )

    bad_safety = copy.deepcopy(assignments)
    row = next(value for value in bad_safety if value["task_class"] == "safety")
    row["required_strata"]["unsafe_or_adversarial"] = False
    with pytest.raises(GVSHumanCollectionError, match="true exactly for safety"):
        build_human_collection_package(
            bad_safety,
            compared_release_cutoff_utc=CUTOFF,
            assignment_source_sha256=SOURCE_SHA256,
        )

    bad_efficacy = copy.deepcopy(assignments)
    row = next(value for value in bad_efficacy if value["task_class"] == "efficacy")
    row["required_strata"]["unsafe_or_adversarial"] = True
    with pytest.raises(GVSHumanCollectionError, match="true exactly for safety"):
        build_human_collection_package(
            bad_efficacy,
            compared_release_cutoff_utc=CUTOFF,
            assignment_source_sha256=SOURCE_SHA256,
        )


def test_package_validation_reconstructs_manifest_and_false_claims(
    package: dict[str, object],
) -> None:
    changed = copy.deepcopy(package)
    changed["coordinator_manifest"][0]["task_batch_index"] += 1
    with pytest.raises(GVSHumanCollectionError, match="different author task"):
        validate_human_collection_package(changed)

    changed = copy.deepcopy(package)
    changed["claims"]["human_provenance_authenticated"] = True
    with pytest.raises(GVSHumanCollectionError, match="exact reconstruction"):
        validate_human_collection_package(changed)

    changed = copy.deepcopy(package)
    changed["author_export"]["contains_answer_or_label"] = True
    with pytest.raises(GVSHumanCollectionError, match="must remain false"):
        validate_human_collection_package(changed)


def test_package_rejects_private_id_collision_with_author_text(
    assignments: list[dict[str, object]],
) -> None:
    colliding = copy.deepcopy(assignments)
    colliding[0]["assignment_id"] = "efficacy"
    with pytest.raises(GVSHumanCollectionError, match="collides with private assignment"):
        build_human_collection_package(
            colliding,
            compared_release_cutoff_utc=CUTOFF,
            assignment_source_sha256=SOURCE_SHA256,
        )


def test_runtime_rebinding_cannot_forge_authorization(
    package: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = {name: False for name in package["claims"]}
    forged["structural_validation_only"] = True
    for name in AUTHORIZATION_FIELDS:
        forged[name] = True
    monkeypatch.setattr(human_module, "_claims", lambda: forged)
    with pytest.raises(GVSHumanCollectionError, match="runtime callable binding changed"):
        validate_human_collection_package(package)


def test_runtime_rebinding_cannot_replace_hash_pattern(
    package: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = human_module._SHA256_RE

    class PermissivePattern:
        pattern = original.pattern

        @staticmethod
        def fullmatch(value: object):
            return original.fullmatch(value) or (value == "not-a-sha")

    monkeypatch.setattr(human_module, "_SHA256_RE", PermissivePattern())
    with pytest.raises(GVSHumanCollectionError, match="runtime contract object binding changed"):
        validate_human_collection_package(package)


def test_runtime_rebinding_cannot_replace_counter(
    package: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    class EvilCounter(Counter):
        def __eq__(self, other: object) -> bool:
            return True

    monkeypatch.setattr(human_module, "_COUNTER_CLASS", EvilCounter)
    with pytest.raises(GVSHumanCollectionError, match="runtime dependency binding changed"):
        validate_human_collection_package(package)


def test_package_and_author_export_bounded_load_round_trip(package: dict[str, object]) -> None:
    package_text = canonical_json(package, maximum_nodes=human_module.MAX_PACKAGE_JSON_NODES)
    assert loads_human_collection_package(package_text) == package
    export = package["author_export"]
    export_text = canonical_json(export, maximum_nodes=human_module.MAX_PACKAGE_JSON_NODES)
    assert loads_author_export(export_text) == export
    assert validate_author_export(export) == export
    with pytest.raises(GVSHumanCollectionError, match="duplicate key"):
        loads_author_export('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(GVSHumanCollectionError, match="non-finite"):
        loads_author_export('{"schema_version":NaN}')


def test_prompt_intake_contains_prompt_and_declarations_but_no_label(
    package: dict[str, object], prompt_intake: dict[str, object]
) -> None:
    assert prompt_intake["schema_version"] == GVS_PROMPT_INTAKE_VERSION
    assert prompt_intake["assignment_id"] == "s.0000"
    assert prompt_intake["prompt"] == _prompt()
    assert prompt_intake["provenance_declarations"] == _provenance()
    assert "annotation" not in prompt_intake
    assert "label" not in prompt_intake
    assert "action_ir" not in prompt_intake
    assert validate_prompt_intake(prompt_intake, package=package) == prompt_intake
    for claim in NONAUTHENTICATED_CLAIMS:
        assert prompt_intake["claims"][claim] is False
    for field in AUTHORIZATION_FIELDS:
        assert prompt_intake["claims"][field] is False


def test_prompt_intake_is_deterministic_detached_and_bounded(
    package: dict[str, object], prompt_intake: dict[str, object]
) -> None:
    raw_prompt = _prompt()
    raw_provenance = _provenance()
    raw_attestations = _attestations()
    rebuilt = build_prompt_intake(
        package,
        assignment_id="s.0000",
        prompt=raw_prompt,
        provenance_declarations=raw_provenance,
        author_attestations=raw_attestations,
        prompt_received_at_utc="2026-08-04T00:02:00Z",
    )
    assert rebuilt == prompt_intake
    raw_prompt["request"] = "mutated"
    raw_provenance["author_id"] = "mutated"
    raw_attestations["no_model_assistance"] = False
    assert rebuilt == prompt_intake
    text = canonical_json(prompt_intake)
    assert loads_prompt_intake(text, package=package) == prompt_intake
    with pytest.raises(GVSHumanCollectionError, match="duplicate key"):
        loads_prompt_intake('{"schema_version":"x","schema_version":"y"}', package=package)
    oversized = json.dumps({"padding": "x" * MAX_PROMPT_INTAKE_JSON_BYTES})
    with pytest.raises(GVSHumanCollectionError, match="exceeds maximum"):
        loads_prompt_intake(oversized, package=package)


def test_prompt_intake_requires_declared_no_model_assistance_and_ordered_times(
    package: dict[str, object],
) -> None:
    provenance = _provenance()
    provenance["no_model_assistance"] = False
    with pytest.raises(GVSHumanCollectionError, match="no_model_assistance"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=_prompt(),
            provenance_declarations=provenance,
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )

    attestations = _attestations()
    attestations["no_model_assistance"] = False
    with pytest.raises(GVSHumanCollectionError, match="must be exact true"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=_prompt(),
            provenance_declarations=_provenance(),
            author_attestations=attestations,
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )

    provenance = _provenance()
    provenance["authored_at_utc"] = "2026-08-02T23:59:00Z"
    with pytest.raises(GVSHumanCollectionError, match="declared chronology"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=_prompt(),
            provenance_declarations=provenance,
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )

    provenance = _provenance()
    provenance["license"] = " CC-BY-4.0\n"
    with pytest.raises(GVSHumanCollectionError, match="license must be NFC"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=_prompt(),
            provenance_declarations=provenance,
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )


def test_prompt_contract_rejects_model_visible_or_label_fields_and_invalid_tool_schema(
    package: dict[str, object],
) -> None:
    changed = _prompt()
    changed["model_input"] = "not accepted"
    with pytest.raises(GVSHumanCollectionError, match="fields changed"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=changed,
            provenance_declarations=_provenance(),
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )
    changed = _prompt()
    changed["answer"] = _action()
    with pytest.raises(GVSHumanCollectionError, match="fields changed"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=changed,
            provenance_declarations=_provenance(),
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )
    changed = _prompt()
    changed["tool_schemas"][0]["unknown"] = True
    with pytest.raises(GVSHumanCollectionError, match="tool_schemas is invalid"):
        build_prompt_intake(
            package,
            assignment_id="s.0000",
            prompt=changed,
            provenance_declarations=_provenance(),
            author_attestations=_attestations(),
            prompt_received_at_utc="2026-08-04T00:02:00Z",
        )


def test_label_task_is_prompt_derived_identifier_free_and_postdates_prompt(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    assert label_task["schema_version"] == GVS_LABEL_TASK_VERSION
    assert label_task["prompt"] == prompt_intake["prompt"]
    assert "population_role" not in label_task
    assert "task_class" not in label_task
    assert "required_strata" not in label_task
    assert "assignment_id" not in label_task
    assert "prompt_intake_sha256" not in label_task
    assert "answer" not in label_task
    assert label_task["packager_added_answer"] is False
    assert label_task["packager_added_private_coordinator_identifiers"] is False
    assert label_task["packager_added_integrity_hashes"] is False
    assert label_task["packager_added_model_generated_text"] is False
    assert label_task["prompt_content_authenticated"] is False
    for field in AUTHORIZATION_FIELDS:
        assert label_task[field] is False
    assert (
        validate_label_task(label_task, prompt_intake=prompt_intake, package=package) == label_task
    )
    assert (
        loads_label_task(canonical_json(label_task), prompt_intake=prompt_intake, package=package)
        == label_task
    )
    with pytest.raises(GVSHumanCollectionError, match="strictly after"):
        build_label_task(
            prompt_intake,
            package=package,
            created_at_utc="2026-08-04T00:02:00Z",
        )


def test_label_task_exact_reconstruction_rejects_prompt_or_policy_tampering(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    changed = copy.deepcopy(label_task)
    changed["prompt"]["request"] = "Changed after authoring"
    with pytest.raises(GVSHumanCollectionError, match="differs from"):
        validate_label_task(changed, prompt_intake=prompt_intake, package=package)
    changed = copy.deepcopy(label_task)
    changed["authorizes_label_access"] = True
    with pytest.raises(GVSHumanCollectionError, match="differs from"):
        validate_label_task(changed, prompt_intake=prompt_intake, package=package)


def test_private_label_is_separate_hash_bound_and_still_nonauthorizing(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
    envelope: dict[str, object],
) -> None:
    assert envelope["schema_version"] == GVS_PRIVATE_LABEL_ENVELOPE_VERSION
    assert envelope["assignment_id"] == prompt_intake["assignment_id"]
    assert envelope["annotation"]["labeler_action_ir"] == _action()
    assert "annotation" not in package
    assert "annotation" not in package["author_export"]
    assert "annotation" not in prompt_intake
    assert "annotation" not in label_task
    assert (
        validate_private_label_envelope(
            envelope,
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )
        == envelope
    )
    assert (
        loads_private_label_envelope(
            canonical_json(envelope),
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )
        == envelope
    )
    for claim in NONAUTHENTICATED_CLAIMS:
        assert envelope["claims"][claim] is False
    for field in AUTHORIZATION_FIELDS:
        assert envelope["claims"][field] is False


def test_private_envelope_rejects_label_model_assistance_and_bad_chronology(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    annotation = _annotation()
    annotation["labeler_no_model_assistance"] = False
    with pytest.raises(GVSHumanCollectionError, match="labeler_no_model_assistance"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            annotation,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )
    annotation = _annotation()
    annotation["reviewer_no_compared_prediction_access"] = False
    with pytest.raises(GVSHumanCollectionError, match="no_compared_prediction_access"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            annotation,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )
    annotation = _annotation()
    annotation["labeled_at_utc"] = "2026-08-04T00:02:30Z"
    with pytest.raises(GVSHumanCollectionError, match="declared label chronology"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            annotation,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )


def test_private_envelope_validates_action_ir_and_independent_reviewers(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    annotation = _annotation(labeler_action={"decision": "CALL"})
    with pytest.raises(GVSHumanCollectionError, match="Action IR"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            annotation,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )
    annotation = _annotation()
    annotation["reviewer_id"] = annotation["labeler_id"]
    with pytest.raises(GVSHumanCollectionError, match="must be distinct"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            annotation,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )


def test_disagreement_requires_independent_adjudication(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    annotation = _annotation(
        reviewer_action={"decision": "ABSTAIN"},
        disagreement=True,
    )
    envelope = build_private_label_envelope(
        prompt_intake,
        label_task,
        annotation,
        package=package,
        sealed_at_utc="2026-08-04T00:06:00Z",
    )
    assert envelope["annotation"]["adjudicated_action_ir"] == _action()

    missing = _annotation(
        reviewer_action={"decision": "ABSTAIN"},
        disagreement=True,
    )
    missing["adjudicator_id"] = None
    with pytest.raises(GVSHumanCollectionError, match="adjudicator_id"):
        build_private_label_envelope(
            prompt_intake,
            label_task,
            missing,
            package=package,
            sealed_at_utc="2026-08-04T00:06:00Z",
        )


def test_envelope_hash_or_binding_tamper_fails_closed(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
    envelope: dict[str, object],
) -> None:
    changed = copy.deepcopy(envelope)
    changed["annotation"]["labeler_action_ir"]["calls"][0]["args"]["title"] = "Tampered"
    changed["annotation"]["reviewer_action_ir"]["calls"][0]["args"]["title"] = "Tampered"
    with pytest.raises(GVSHumanCollectionError, match="hash mismatch"):
        validate_private_label_envelope(
            changed,
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )
    changed = copy.deepcopy(envelope)
    changed["prompt_intake_sha256"] = "f" * 64
    with pytest.raises(GVSHumanCollectionError, match="binding changed"):
        validate_private_label_envelope(
            changed,
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )
    changed = copy.deepcopy(envelope)
    changed["claims"]["private_label_custody_authenticated"] = True
    with pytest.raises(GVSHumanCollectionError, match="hash mismatch|nonauthorizing"):
        validate_private_label_envelope(
            changed,
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )


def test_loaders_reject_duplicate_nonfinite_and_nonobject_documents(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
) -> None:
    with pytest.raises(GVSHumanCollectionError, match="duplicate key"):
        loads_label_task(
            '{"schema_version":"x","schema_version":"y"}',
            prompt_intake=prompt_intake,
            package=package,
        )
    with pytest.raises(GVSHumanCollectionError, match="non-finite"):
        loads_private_label_envelope(
            '{"x":Infinity}',
            package=package,
            prompt_intake=prompt_intake,
            label_task=label_task,
        )
    with pytest.raises(GVSHumanCollectionError, match="exact JSON object"):
        loads_prompt_intake("[]", package=package)


def test_module_is_cpu_only_and_has_no_external_io_surface() -> None:
    source = inspect.getsource(human_module)
    for forbidden in (
        "import torch",
        "import requests",
        "import subprocess",
        "import socket",
        "urllib",
        "open(",
        "Path(",
    ):
        assert forbidden not in source
    public_source = "\n".join(
        inspect.getsource(getattr(human_module, name))
        for name in (
            "build_human_collection_package",
            "build_prompt_intake",
            "build_label_task",
            "build_private_label_envelope",
        )
    )
    assert "Jarvis" not in public_source
    assert "torch" not in public_source


def test_every_artifact_retains_explicit_false_authorization(
    package: dict[str, object],
    prompt_intake: dict[str, object],
    label_task: dict[str, object],
    envelope: dict[str, object],
) -> None:
    for field in AUTHORIZATION_FIELDS:
        assert package["claims"][field] is False
        assert package["author_export"][field] is False
        assert prompt_intake["claims"][field] is False
        assert label_task[field] is False
        assert envelope["claims"][field] is False
