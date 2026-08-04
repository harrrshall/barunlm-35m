from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import barunlm.evaluation.gvs_schema_partition as partition_module
from barunlm.evaluation.action_simulator import SIMULATOR_SCHEMA_SHA256, SIMULATOR_TOOL_SCHEMAS
from barunlm.evaluation.gvs_schema_partition import (
    MAX_ARTIFACT_BYTES,
    SCHEMA_PARTITION_ROLES,
    GVSSchemaPartitionError,
    SchemaPartitionReceipt,
    build_schema_partition,
    load_schema_partition_json,
    make_schema_family_spec,
    make_schema_partition_spec,
    verify_schema_partition,
)
from barunlm.evaluation.gvs_schema_presentation import (
    RENAMED_PRESENTATION,
    ArgumentNameMap,
    SchemaPresentation,
    SchemaPresentationReceipt,
    ToolNameMap,
)

_PREFIXES = {
    "T-new": "train",
    "D-support": "support",
    "S-new": "selection",
    "C-new": "confirm",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _presentation(role: str, *, suffix: str = "a") -> SchemaPresentation:
    prefix = f"{_PREFIXES[role]}{suffix}"
    tools: list[ToolNameMap] = []
    for tool_index, schema in enumerate(sorted(SIMULATOR_TOOL_SCHEMAS, key=lambda row: row.name)):
        arguments = tuple(
            ArgumentNameMap(
                canonical_name=name,
                presented_name=f"{prefix}_arg_{tool_index:02d}_{argument_index:02d}",
            )
            for argument_index, name in enumerate(sorted(schema.arguments))
        )
        tools.append(
            ToolNameMap(
                canonical_name=schema.name,
                presented_name=f"{prefix}_tool_{tool_index:02d}",
                arguments=arguments,
            )
        )
    return SchemaPresentation(
        presentation_id=f"{prefix}-schema-family",
        mode=RENAMED_PRESENTATION,
        tools=tuple(tools),
    )


def _family(role: str, *, suffix: str = "a", lineage: str | None = None):
    prefix = f"{_PREFIXES[role]}{suffix}"
    return make_schema_family_spec(
        role=role,
        family_id=f"family.{prefix}",
        lineage_ids=(lineage or f"lineage.{prefix}",),
        family_source_sha256=_digest(f"source:{prefix}"),
        presentation=_presentation(role, suffix=suffix),
    )


def _with_presentation(family, presentation: SchemaPresentation):
    return make_schema_family_spec(
        role=family.role,
        family_id=family.family_id,
        lineage_ids=family.lineage_ids,
        family_source_sha256=family.family_source_sha256,
        presentation=presentation,
    )


def _spec():
    # Deliberately shuffled; the factory freezes canonical role/family order.
    return make_schema_partition_spec(
        source_commitment_sha256=_digest("schema partition source v1"),
        families=(
            _family("C-new"),
            _family("T-new"),
            _family("S-new"),
            _family("D-support"),
        ),
    )


@pytest.fixture(scope="module")
def partition() -> tuple[object, SchemaPartitionReceipt]:
    spec = _spec()
    receipt = build_schema_partition(spec, expected_spec_sha256=spec.sha256())
    return spec, receipt


def test_complete_partition_exposes_role_specific_verified_receipts(partition) -> None:
    spec, receipt = partition
    assert tuple(family.role for family in spec.families) == SCHEMA_PARTITION_ROLES
    assert tuple(receipt.schema_presentation_receipts_by_role) == SCHEMA_PARTITION_ROLES
    for role in SCHEMA_PARTITION_ROLES:
        role_receipts = receipt.receipts_for_role(role)
        assert len(role_receipts) == 1
        assert type(role_receipts[0]) is SchemaPresentationReceipt
        assert role_receipts[0].presentation.mode == RENAMED_PRESENTATION
        assert role_receipts[0].authorizes_model_or_label_access is False
        assert role_receipts[0].authorizes_cuda_or_jarvis_access is False


def test_partition_claims_only_structural_separation_and_never_authorizes(partition) -> None:
    _, receipt = partition
    record = receipt.to_dict()
    assert record["canonical_simulator_schema_sha256"] == SIMULATOR_SCHEMA_SHA256
    assert record["pre_authoring_partition_only"] is False
    assert record["pre_authoring_intent_declared"] is True
    assert record["partition_contains_request_or_label_text"] is False
    assert record["role_assignment_precedes_request_authoring"] is False
    assert record["role_assignment_precedes_request_authoring_declared"] is True
    assert record["role_assignment_precedes_request_authoring_authenticated"] is False
    assert record["structural_role_disjoint_schema_names_verified"] is True
    assert record["structural_role_disjoint_argument_names_verified"] is True
    assert record["heldout_schema_names_supported"] is False
    assert record["heldout_argument_names_supported"] is False
    assert record["heldout_claim_scope"] == ("none_without_external_custody_and_population_binding")
    assert record["novel_semantics_supported"] is False
    assert record["model_performance_measured"] is False
    assert record["external_temporal_custody_authenticated"] is False
    assert record["descriptions_semantically_audited"] is False
    for field in (
        "authorizes_model_access",
        "authorizes_label_access",
        "authorizes_cuda",
        "authorizes_jarvislabs",
        "authorizes_training",
        "authorizes_launch",
        "authorizes_execution",
    ):
        assert record[field] is False


def test_every_family_proves_total_bijections_and_semantic_round_trips(partition) -> None:
    _, receipt = partition
    expected_argument_count = sum(len(schema.arguments) for schema in SIMULATOR_TOOL_SCHEMAS)
    for family in receipt.family_receipts:
        assert family.tool_bijection_count == len(SIMULATOR_TOOL_SCHEMAS) == 13
        assert family.argument_bijection_count == expected_argument_count == 33
        assert len(family.semantic_round_trip_records) == len(SIMULATOR_TOOL_SCHEMAS)
        assert {row["canonical_tool"] for row in family.semantic_round_trip_records} == {
            schema.name for schema in SIMULATOR_TOOL_SCHEMAS
        }
        assert all(row["semantic_equal"] is True for row in family.semantic_round_trip_records)
        assert all(
            row["canonical_action_sha256"] == row["lowered_action_sha256"]
            for row in family.semantic_round_trip_records
        )


def test_complete_live_verification_and_deterministic_rebuild(partition) -> None:
    spec, receipt = partition
    rebuilt = verify_schema_partition(
        receipt,
        spec=spec,
        expected_spec_sha256=spec.sha256(),
    )
    assert rebuilt.canonical_json() == receipt.canonical_json()
    assert rebuilt.to_json_bytes() == receipt.to_json_bytes()


def test_strict_loader_live_rederives_exact_canonical_artifact(partition) -> None:
    _, receipt = partition
    loaded = load_schema_partition_json(
        receipt.to_json_bytes(),
        expected_partition_sha256=receipt.sha256(),
    )
    assert loaded.canonical_json() == receipt.canonical_json()
    assert loaded.artifact_body_sha256 == receipt.artifact_body_sha256


def test_deterministic_golden_vectors(partition) -> None:
    spec, receipt = partition
    # These roots omit runtime/platform identity and therefore remain portable golden vectors.
    assert spec.sha256() == "4e65e1bd475bbdc72398ebbfb071220c4d2a21af6a054c8dacb25650239ebed7"
    assert (
        receipt.family_receipts[0].semantic_round_trip_sha256
        == "960b96acb8721d9d2eb1123cacbb27e0743bda697c40693bcdaa80d54de3c775"
    )
    assert (
        receipt.family_receipts[0].family.sha256()
        == "a2f1adb90a780a730d82426a3d480d2357b1903d919b161bf6fd54dab841ab90"
    )


def test_cross_role_lineage_overlap_fails_before_build() -> None:
    shared = "lineage.shared"
    with pytest.raises(GVSSchemaPartitionError, match="cross-role lineage overlap"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("source"),
            families=(
                _family("T-new", lineage=shared),
                _family("D-support", lineage=shared),
                _family("S-new"),
                _family("C-new"),
            ),
        )


def test_cross_role_complete_map_overlap_fails_before_build() -> None:
    train = _family("T-new")
    support = make_schema_family_spec(
        role="D-support",
        family_id="family.support-copy",
        lineage_ids=("lineage.support-copy",),
        family_source_sha256=_digest("support-copy"),
        presentation=replace(train.presentation, presentation_id="support-copy-schema"),
    )
    with pytest.raises(GVSSchemaPartitionError, match="complete rename-map overlap"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("source"),
            families=(train, support, _family("S-new"), _family("C-new")),
        )


def test_duplicate_lineage_and_complete_map_within_one_role_fail() -> None:
    train = _family("T-new")
    repeated_lineage = _family("T-new", suffix="b", lineage=train.lineage_ids[0])
    with pytest.raises(GVSSchemaPartitionError, match="duplicate lineage membership"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("same-role-lineage"),
            families=(
                train,
                repeated_lineage,
                _family("D-support"),
                _family("S-new"),
                _family("C-new"),
            ),
        )

    repeated_map = make_schema_family_spec(
        role="T-new",
        family_id="family.train-map-copy",
        lineage_ids=("lineage.train-map-copy",),
        family_source_sha256=_digest("train-map-copy"),
        presentation=replace(
            train.presentation,
            presentation_id="train-map-copy-schema-family",
        ),
    )
    with pytest.raises(GVSSchemaPartitionError, match="duplicate complete rename map"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("same-role-map"),
            families=(
                train,
                repeated_map,
                _family("D-support"),
                _family("S-new"),
                _family("C-new"),
            ),
        )


def test_family_source_hash_cannot_be_reused_across_roles() -> None:
    train = _family("T-new")
    support = _family("D-support")
    support = make_schema_family_spec(
        role=support.role,
        family_id=support.family_id,
        lineage_ids=support.lineage_ids,
        family_source_sha256=train.family_source_sha256,
        presentation=support.presentation,
    )
    with pytest.raises(GVSSchemaPartitionError, match="source hashes must be globally unique"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("same-source"),
            families=(train, support, _family("S-new"), _family("C-new")),
        )


def test_cross_role_map_edge_and_presented_name_overlap_fail() -> None:
    train = _family("T-new")
    support = _family("D-support")
    train_tool = train.presentation.tools[0]

    edge_arguments = list(support.presentation.tools[0].arguments)
    edge_arguments[0] = replace(
        edge_arguments[0],
        presented_name=train_tool.arguments[0].presented_name,
    )
    edge_tools = list(support.presentation.tools)
    edge_tools[0] = replace(edge_tools[0], arguments=tuple(edge_arguments))
    edge_family = _with_presentation(
        support,
        replace(support.presentation, tools=tuple(edge_tools)),
    )
    with pytest.raises(GVSSchemaPartitionError, match="rename-map edge overlap"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("edge"),
            families=(train, edge_family, _family("S-new"), _family("C-new")),
        )

    name_tools = list(support.presentation.tools)
    # Reuse a name from a different canonical tool, so only the presented-name set overlaps.
    name_tools[0] = replace(
        name_tools[0], presented_name=train.presentation.tools[1].presented_name
    )
    name_family = _with_presentation(
        support,
        replace(support.presentation, tools=tuple(name_tools)),
    )
    with pytest.raises(GVSSchemaPartitionError, match="presented-name overlap"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("name"),
            families=(train, name_family, _family("S-new"), _family("C-new")),
        )


def test_incomplete_tool_or_argument_bijection_fails_closed() -> None:
    family = _family("T-new")
    bad_tools = list(family.presentation.tools)
    bad_tools[0] = replace(bad_tools[0], arguments=bad_tools[0].arguments[:-1])
    bad_family = _with_presentation(
        family,
        replace(family.presentation, tools=tuple(bad_tools)),
    )
    spec = make_schema_partition_spec(
        source_commitment_sha256=_digest("bad bijection"),
        families=(bad_family, _family("D-support"), _family("S-new"), _family("C-new")),
    )
    with pytest.raises(GVSSchemaPartitionError, match="failed presentation verification"):
        build_schema_partition(spec, expected_spec_sha256=spec.sha256())


def test_mutation_after_commitment_and_receipt_forgery_fail_closed(partition) -> None:
    mutable_spec = _spec()
    commitment = mutable_spec.sha256()
    object.__setattr__(
        mutable_spec.families[0].presentation,
        "presentation_id",
        "mutated-after-commitment",
    )
    with pytest.raises(GVSSchemaPartitionError, match="differs from its commitment"):
        build_schema_partition(mutable_spec, expected_spec_sha256=commitment)

    spec, receipt = partition
    original = receipt.family_receipts[0].semantic_round_trip_sha256
    try:
        object.__setattr__(receipt.family_receipts[0], "semantic_round_trip_sha256", "0" * 64)
        with pytest.raises(GVSSchemaPartitionError, match="live recomputation"):
            verify_schema_partition(
                receipt,
                spec=spec,
                expected_spec_sha256=spec.sha256(),
            )
    finally:
        object.__setattr__(receipt.family_receipts[0], "semantic_round_trip_sha256", original)


def test_nested_round_trip_evidence_is_immutable(partition) -> None:
    _, receipt = partition
    row = receipt.family_receipts[0].semantic_round_trip_records[0]
    with pytest.raises(TypeError):
        row["semantic_equal"] = False  # type: ignore[index]


def test_partition_factory_rejects_custom_members_before_property_access() -> None:
    class Trap:
        @property
        def role(self):
            raise AssertionError("custom role property must not be accessed")

    with pytest.raises(GVSSchemaPartitionError, match="exact tuple of SchemaFamilySpec"):
        make_schema_partition_spec(
            source_commitment_sha256=_digest("custom-member"),
            families=(Trap(),),  # type: ignore[arg-type]
        )


def test_loader_rejects_tampering_noncanonical_json_duplicates_and_bounds(partition) -> None:
    _, receipt = partition
    raw = receipt.to_json_bytes()
    payload = json.loads(raw)
    payload["partition"]["authorizes_cuda"] = True
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(GVSSchemaPartitionError, match="canonical live evidence"):
        load_schema_partition_json(tampered, expected_partition_sha256=receipt.sha256())

    noncanonical = raw[:-1] + b" \n"
    with pytest.raises(GVSSchemaPartitionError, match="canonical live evidence"):
        load_schema_partition_json(noncanonical, expected_partition_sha256=receipt.sha256())

    duplicate = raw.replace(
        b'{"artifact_schema_version":',
        b'{"artifact_schema_version":"duplicate","artifact_schema_version":',
        1,
    )
    with pytest.raises(GVSSchemaPartitionError, match="duplicate JSON key"):
        load_schema_partition_json(duplicate, expected_partition_sha256=receipt.sha256())

    with pytest.raises(GVSSchemaPartitionError, match="bounded exact bytes"):
        load_schema_partition_json(
            b"{" + b" " * MAX_ARTIFACT_BYTES,
            expected_partition_sha256=receipt.sha256(),
        )
    with pytest.raises(GVSSchemaPartitionError, match="floating-point JSON"):
        load_schema_partition_json(b'{"value":1.0}', expected_partition_sha256="0" * 64)


def test_loader_rejects_unknown_fields_wrong_hash_and_wrong_version(partition) -> None:
    _, receipt = partition
    payload = json.loads(receipt.to_json_bytes())
    payload["extra"] = False
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(GVSSchemaPartitionError, match="fields are not exact"):
        load_schema_partition_json(raw, expected_partition_sha256=receipt.sha256())
    with pytest.raises(GVSSchemaPartitionError, match="expected commitment"):
        load_schema_partition_json(
            receipt.to_json_bytes(),
            expected_partition_sha256="0" * 64,
        )

    payload = json.loads(receipt.to_json_bytes())
    payload["artifact_schema_version"] = "wrong"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(GVSSchemaPartitionError, match="unsupported"):
        load_schema_partition_json(raw, expected_partition_sha256=receipt.sha256())


def test_runtime_dependency_substitution_fails_before_schema_work(monkeypatch) -> None:
    spec = _spec()
    monkeypatch.setattr(partition_module, "_BUILD_SCHEMA_PRESENTATION", lambda **_: None)
    with pytest.raises(GVSSchemaPartitionError, match="runtime dependency"):
        build_schema_partition(spec, expected_spec_sha256=spec.sha256())


def test_loader_support_dependency_substitution_fails_closed(monkeypatch, partition) -> None:
    _, receipt = partition
    real_loads = json.loads
    monkeypatch.setattr(
        partition_module,
        "_JSON_LOADS",
        lambda value, **kwargs: real_loads(value, **kwargs),
    )
    with pytest.raises(GVSSchemaPartitionError, match="json.loads changed identity"):
        load_schema_partition_json(
            receipt.to_json_bytes(),
            expected_partition_sha256=receipt.sha256(),
        )


def test_runtime_regex_substitution_fails_before_build(monkeypatch) -> None:
    import re

    spec = _spec()
    monkeypatch.setattr(partition_module, "_IDENTIFIER_RE", re.compile(r".+\Z"))
    with pytest.raises(GVSSchemaPartitionError, match="runtime differs"):
        build_schema_partition(spec, expected_spec_sha256=spec.sha256())


def test_simulator_schema_runtime_substitution_fails_closed(monkeypatch) -> None:
    spec = _spec()
    monkeypatch.setattr(partition_module._simulator_module, "SIMULATOR_SCHEMA_SHA256", "0" * 64)
    with pytest.raises(GVSSchemaPartitionError, match="schema hash alias changed"):
        build_schema_partition(spec, expected_spec_sha256=spec.sha256())


def test_source_hasher_detects_toctou_metadata_change(monkeypatch) -> None:
    real_fstat = partition_module._OS_FSTAT
    calls = 0

    def changing_fstat(file_descriptor: int):
        nonlocal calls
        calls += 1
        value = real_fstat(file_descriptor)
        if calls == 1:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(partition_module, "_OS_FSTAT", changing_fstat)
    with pytest.raises(GVSSchemaPartitionError, match="changed while hashing"):
        partition_module._file_sha256(Path(partition_module.__file__))


def test_source_and_runtime_hashes_are_complete_and_bound(partition) -> None:
    _, receipt = partition
    assert set(receipt.source_files) == {
        "action_simulator",
        "gvs_schema_partition",
        "gvs_schema_presentation",
    }
    assert all(len(value) == 64 for value in receipt.source_files.values())
    assert len(receipt.source_files_sha256) == 64
    assert len(receipt.schema_partition_runtime_sha256) == 64
    assert len(receipt.action_simulator_runtime_sha256) == 64
    assert len(receipt.schema_presentation_runtime_sha256) == 64
    assert (
        receipt.artifact_body_sha256
        == hashlib.sha256(
            json.dumps(
                receipt.body_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
