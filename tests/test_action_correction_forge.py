from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from barunlm.datasets import action_correction_forge as subject
from barunlm.evaluation.action_ir import parse_action_ir
from barunlm.evaluation.action_simulator import simulator_tool_registry_snapshot
from barunlm.evaluation.sim_program import SEMANTIC_OPERATION_SPECS

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/barunlm/datasets/action_correction_forge.py"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def train_rows() -> tuple[subject.ForgeSource, ...]:
    return subject.build_fixture_population("T-synth")


@pytest.fixture(scope="module")
def screen_rows() -> tuple[subject.ForgeSource, ...]:
    return subject.build_fixture_population("D-internal")


def test_prototype_is_explicitly_nonmaterialized_and_nonauthorizing() -> None:
    record = subject.prototype_record()

    assert record["run_id"] == "20260805-0230-action-correction-forge-screen-s17"
    assert record["fixture_rows_per_stratum_ceiling"] == 2
    assert record["claims"] == {
        "row_level_independence": False,
        "effective_component_count": False,
        "population_quota_satisfied": False,
        "model_quality_measured": False,
    }
    assert record["flags"]
    assert all(value is False for value in record["flags"].values())
    assert record["flags"] == dict(subject.PROTOTYPE_FLAGS)
    assert record["flags"]["population_materialized"] is False
    assert record["flags"]["model_access_authorized"] is False
    assert record["flags"]["cuda_authorized"] is False
    assert record["flags"]["jarvislabs_resource_creation_authorized"] is False


def test_module_imports_only_standard_library_and_the_four_permitted_cpu_contracts() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported_modules.add(node.module)

    allowed = {
        "__future__",
        "dataclasses",
        "functools",
        "hashlib",
        "json",
        "types",
        "typing",
        "unicodedata",
        "barunlm.evaluation.action_ir",
        "barunlm.evaluation.action_simulator",
        "barunlm.evaluation.sim_program",
    }
    assert imported_modules <= allowed
    assert not any("gvs_" in name for name in imported_modules)
    assert not any("planir" in name for name in imported_modules)
    assert not any("temporal_counterfactual" in name for name in imported_modules)
    assert not any(
        name.startswith(("torch", "requests", "urllib", "socket")) for name in imported_modules
    )


def test_registered_strata_are_exactly_thirteen_operations_plus_three_decisions() -> None:
    assert len(subject.SEMANTIC_OPERATION_KINDS) == 13
    assert subject.SEMANTIC_OPERATION_KINDS == tuple(sorted(SEMANTIC_OPERATION_SPECS))
    assert subject.CONTROL_DECISION_STRATA == ("ABSTAIN", "CLARIFY", "CONFIRM")
    assert len(subject.DESCRIPTIVE_STRATA) == 16
    assert len(set(subject.DESCRIPTIVE_STRATA)) == 16


def test_role_families_are_finite_exclusive_and_fixed_before_render(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    train = subject.ROLE_FAMILIES["T-synth"]
    screen = subject.ROLE_FAMILIES["D-internal"]
    dimensions = (
        "renderer_family",
        "entity_family",
        "temporal_family",
        "schema_family",
        "context_family",
        "source_family",
    )
    assert all(getattr(train, name) != getattr(screen, name) for name in dimensions)
    for row in (*train_rows, *screen_rows):
        plan = subject.ROLE_FAMILIES[row.role]
        assert row.lineage.family_assignments_fixed_before_render is True
        assert row.lineage.row_is_independent_component is False
        assert row.lineage.effective_component_claimed is False
        assert row.lineage.power_claimed is False
        assert all(getattr(row.lineage, name) == getattr(plan, name) for name in dimensions)
        assert row.source_id not in row.lineage.program_skeleton_family

    second = subject.build_forge_source("T-synth", "CREATE_NOTE", 1)
    first = subject.build_forge_source("T-synth", "CREATE_NOTE", 0)
    assert second.source_id != first.source_id
    assert second.lineage == first.lineage


def test_program_first_fixtures_cover_every_stratum_and_compile_to_strict_gold(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    schemas = simulator_tool_registry_snapshot()
    for role, rows in (("T-synth", train_rows), ("D-internal", screen_rows)):
        assert tuple(row.stratum for row in rows) == subject.DESCRIPTIVE_STRATA
        assert {row.role for row in rows} == {role}
        assert len({row.source_id for row in rows}) == 16
        for row in rows:
            reparsed = parse_action_ir(row.gold_target_json, schemas)
            assert reparsed.canonical_json() == row.gold_target_json
            assert (
                json.dumps(
                    json.loads(row.gold_target_json),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                == row.gold_target_json
            )
            assert row.reference_status in {"reference_applied", "reference_control"}
            assert SHA256.fullmatch(row.round_trip_sha256)
            if row.stratum in subject.SEMANTIC_OPERATION_KINDS:
                assert row.program.steps[0].operations[0].kind == row.stratum
            elif row.stratum == "CONFIRM":
                assert row.program.steps[0].operations[0].kind == "PROPOSE_MESSAGE"
                assert row.gold_action_ir.decision.value == "CONFIRM"
            else:
                assert row.program.steps[0].operations == ()


def test_rendered_prompts_are_canonical_and_role_presentations_are_distinct(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    train_prompt = json.loads(train_rows[0].direct_prompt_json)
    screen_prompt = json.loads(screen_rows[0].direct_prompt_json)
    assert set(train_prompt) == {
        "action_ir_contract",
        "instruction",
        "request",
        "context",
        "tools",
    }
    assert set(screen_prompt) == {
        "contract",
        "task",
        "user_request",
        "supplied_context",
        "tool_schemas",
    }
    assert len(train_prompt["tools"]) == 13
    assert len(screen_prompt["tool_schemas"]) == 13
    assert train_rows[0].direct_prompt_json != screen_rows[0].direct_prompt_json
    for row in (*train_rows, *screen_rows):
        decoded = json.loads(row.direct_prompt_json)
        assert (
            json.dumps(
                decoded,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            == row.direct_prompt_json
        )


def test_fingerprints_bind_exact_normalized_delexicalized_program_and_gold(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    for row in (*train_rows, *screen_rows):
        fingerprints = row.fingerprints.to_dict()
        assert fingerprints["version"] == subject.FINGERPRINT_VERSION
        assert all(
            SHA256.fullmatch(value) for key, value in fingerprints.items() if key != "version"
        )
        assert (
            row.fingerprints.exact_sha256
            == hashlib.sha256(row.direct_prompt_json.encode("utf-8")).hexdigest()
        )
        assert (
            row.fingerprints.gold_action_ir_sha256
            == hashlib.sha256(row.gold_target_json.encode("utf-8")).hexdigest()
        )
        assert row.fingerprints.program_sha256 == row.program.sha256()
        assert row.fingerprints.world_state_sha256 == row.program.initial_state.sha256()
    assert len({row.fingerprints.exact_sha256 for row in (*train_rows, *screen_rows)}) == 32


def test_a_b_c_views_share_source_and_complete_target_with_one_b_mask(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    for source in (*train_rows, *screen_rows):
        a, b, c = subject.build_matched_views(source)
        assert (a.arm, b.arm, c.arm) == ("A", "B", "C")
        assert {view.source_id for view in (a, b, c)} == {source.source_id}
        assert {view.target_json for view in (a, b, c)} == {source.gold_target_json}
        assert a.input_json == source.direct_prompt_json
        assert b.input_json.count(subject.MASK_TOKEN) == 1
        assert b.masked_path is not None
        for view in (a, b, c):
            assert (
                parse_action_ir(
                    view.target_json, simulator_tool_registry_snapshot()
                ).canonical_json()
                == view.target_json
            )
            assert (
                json.dumps(
                    json.loads(view.input_json),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                == view.input_json
            )


def test_c_hash_rule_is_frozen_and_yields_exact_or_certified_single_fault(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    for rows in (train_rows, screen_rows):
        assignments = [subject.build_view_c(source).c_assignment for source in rows]
        assert assignments.count("exact") == 8
        assert assignments.count("single_fault") == 8
    for source in (*train_rows, *screen_rows):
        view = subject.build_view_c(source)
        assert view.c_assignment == subject.c_assignment(source.source_id)
        assert view.diagnostics is not None
        if view.c_assignment == "single_fault":
            certificate = view.fault_certificate
            assert certificate is not None
            assert certificate.source_id == source.source_id
            assert certificate.mutation_count == 1
            assert certificate.semantic_outcome_distinct is True
            assert certificate.model_visible is False
        else:
            assert view.fault_certificate is None
            assert json.loads(view.input_json)["draft"] == json.loads(source.gold_target_json)


def test_every_stratum_has_a_working_single_fault_certificate(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    for source in (*train_rows, *screen_rows):
        draft, certificate = subject.certified_single_fault(source)
        assert SHA256.fullmatch(certificate.draft_sha256)
        assert certificate.draft_sha256 == hashlib.sha256(draft.encode("utf-8")).hexdigest()
        assert certificate.mutation_count == 1
        assert certificate.semantic_outcome_distinct is True
        assert draft != source.gold_target_json
    abstain = subject.build_forge_source("T-synth", "ABSTAIN")
    draft, certificate = subject.certified_single_fault(abstain)
    assert json.loads(draft) == {"decision": "CALL"}
    assert certificate.draft_parse_valid is True
    assert certificate.draft_schema_valid is False


def test_diagnostics_are_draft_only_and_never_expose_gold_fault_or_permission(
    train_rows: tuple[subject.ForgeSource, ...],
    screen_rows: tuple[subject.ForgeSource, ...],
) -> None:
    forbidden = {
        "execution_permitted",
        "expected",
        "fault_kind",
        "gold",
        "label",
        "oracle",
        "reference_answer",
        "target",
    }
    allowed = {
        "parse_valid",
        "schema_valid",
        "policy_status",
        "simulator_status",
        "error_codes",
    }
    for source in (*train_rows, *screen_rows):
        view = subject.build_view_c(source)
        assert view.diagnostics is not None
        visible = view.diagnostics.model_visible_dict()
        assert set(visible) == allowed
        assert forbidden.isdisjoint(visible)
        assert "policy_conformant_proposal" not in visible
        assert isinstance(view.diagnostics.policy_conformant_proposal, bool)
        serialized = json.dumps(visible, sort_keys=True)
        assert not any(term in serialized for term in forbidden)
        assert json.loads(view.input_json)["diagnostics"] == visible


def test_diagnostics_distinguish_json_schema_policy_and_confirmation_from_draft_only() -> None:
    source = subject.build_forge_source("D-internal", "PAUSE_MEDIA")
    invalid_json = subject.derive_draft_diagnostics("not json", source.program.initial_state)
    invalid_schema = subject.derive_draft_diagnostics(
        '{"decision":"CALL"}', source.program.initial_state
    )
    gold = subject.derive_draft_diagnostics(source.gold_target_json, source.program.initial_state)
    fault, _ = subject.certified_single_fault(source)
    blocked = subject.derive_draft_diagnostics(fault, source.program.initial_state)

    assert (invalid_json.parse_valid, invalid_json.schema_valid) == (False, False)
    assert (invalid_schema.parse_valid, invalid_schema.schema_valid) == (True, False)
    assert gold.policy_status == "requires_confirmation"
    assert gold.policy_conformant_proposal is True
    assert blocked.policy_status == "blocked"
    assert blocked.policy_conformant_proposal is False
    assert blocked.error_codes == ("policy_decision_mismatch",)


@pytest.mark.parametrize("rows_per_stratum", [0, 3, 1024, 4096, 16384, True])
def test_fixture_builder_cannot_materialize_planned_populations(rows_per_stratum: int) -> None:
    with pytest.raises(subject.ActionCorrectionForgeError, match="fixture-only ceiling"):
        subject.build_fixture_population("T-synth", rows_per_stratum=rows_per_stratum)


def test_unknown_roles_strata_and_large_ordinals_fail_closed() -> None:
    with pytest.raises(subject.ActionCorrectionForgeError, match="unknown forge role"):
        subject.build_forge_source("selection", "CREATE_NOTE")  # type: ignore[arg-type]
    with pytest.raises(subject.ActionCorrectionForgeError, match="unknown descriptive stratum"):
        subject.build_forge_source("T-synth", "SEND_EMAIL")
    with pytest.raises(subject.ActionCorrectionForgeError, match="fixture-only quota"):
        subject.build_forge_source("T-synth", "CREATE_NOTE", 2)


def test_golden_source_identity_is_stable() -> None:
    source = subject.build_forge_source("T-synth", "CREATE_NOTE")
    assert source.source_id == "acf.t.create-note.00.5021b079f0f9"
    assert source.fingerprints.exact_sha256 == (
        "6db9394703f72d7fd25b0f64cbdcd8ffe7c30b2dd459053a65cad138c3aa202b"
    )
    assert source.fingerprints.gold_action_ir_sha256 == (
        "346620ad8f2d2ae246b44fa25c71c249c39b8d49cb4f13cf160cfd11d52e7903"
    )
