from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

import barunlm.evaluation.action_correction_runtime as runtime
from barunlm.datasets.action_correction_forge import build_forge_source
from barunlm.evaluation.action_correction_contract import CANONICAL_ABSTAIN, DIAGNOSTIC_FIELDS

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/barunlm/evaluation/action_correction_runtime.py"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.fixture(scope="module")
def train_prompt() -> str:
    return build_forge_source("T-synth", "ABSTAIN").direct_prompt_json


@pytest.fixture(scope="module")
def screen_prompt() -> str:
    return build_forge_source("D-internal", "ABSTAIN").direct_prompt_json


def _visible(prompt: runtime.CorrectionPrompt) -> dict[str, object]:
    payload = json.loads(prompt.prompt_json)
    assert type(payload) is dict
    return payload


def _all_mapping_keys(value: object) -> set[str]:
    if type(value) is dict:
        keys = set(value)
        for child in value.values():
            keys.update(_all_mapping_keys(child))
        return keys
    if type(value) is list:
        keys: set[str] = set()
        for child in value:
            keys.update(_all_mapping_keys(child))
        return keys
    return set()


def test_runtime_boundary_is_cpu_only_and_every_authorization_flag_is_false() -> None:
    boundary = runtime.runtime_boundary()

    assert boundary["schema_version"] == "barun-action-correction-raw-runtime-v1"
    assert boundary["draft_transport_version"] == ("barun-action-correction-draft-raw-transport-v1")
    assert boundary["model_visible_diagnostic_fields"] == list(DIAGNOSTIC_FIELDS)
    assert boundary["fallback_order"] == ["pass2", "pass1", "canonical_abstain"]
    assert boundary["flags"] == dict(runtime.RUNTIME_FLAGS)
    assert boundary["flags"]
    assert all(value is False for value in boundary["flags"].values())


def test_runtime_has_no_model_network_or_closed_research_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    allowed = {
        "__future__",
        "action_correction_contract",
        "action_ir",
        "action_simulator",
        "dataclasses",
        "enum",
        "hashlib",
        "inspect",
        "json",
        "sim_program",
        "types",
        "typing",
    }
    assert imported <= allowed
    assert not any(
        fragment in name.casefold()
        for name in imported
        for fragment in (
            "torch",
            "cuda",
            "socket",
            "requests",
            "urllib",
            "http",
            "wandb",
            "huggingface",
            "jarvis",
            "dataset",
            "gvs",
            "planir",
            "temporal_counterfactual",
        )
    )


def test_fresh_import_has_no_network_io_capability_modules_or_audit_events() -> None:
    script = r"""import json
import sys

socket_events = []


def audit(event, args):
    if event.startswith("socket."):
        socket_events.append(event)


sys.addaudithook(audit)
import barunlm.evaluation.action_correction_runtime  # noqa: F401, E402

network_roots = (
    "_socket",
    "socket",
    "urllib.request",
    "http.client",
    "requests",
    "httpx",
    "aiohttp",
)
capability_modules = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in network_roots)
)
urllib_modules = sorted(name for name in sys.modules if name.startswith("urllib"))
print(
    json.dumps(
        {
            "capability_modules": capability_modules,
            "socket_events": socket_events,
            "urllib_modules": urllib_modules,
            "subprocess_loaded": "subprocess" in sys.modules,
        },
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(completed.stdout)

    assert evidence["capability_modules"] == []
    assert evidence["socket_events"] == []
    assert evidence["subprocess_loaded"] is False
    # CPython 3.10/3.11 pathlib may import URL *parsing* helpers.  Those modules do
    # not expose request or socket I/O and are the complete allowed urllib surface.
    assert set(evidence["urllib_modules"]) <= {"urllib", "urllib.parse"}


@pytest.mark.parametrize(
    ("fixture_name", "expected_shape"),
    [("train_prompt", "T-synth"), ("screen_prompt", "D-internal")],
)
def test_both_existing_canonical_original_prompt_shapes_extract_state(
    request: pytest.FixtureRequest,
    fixture_name: str,
    expected_shape: str,
) -> None:
    original = request.getfixturevalue(fixture_name)
    prompt = runtime.build_correction_prompt(
        original_prompt_json=original,
        draft_raw=CANONICAL_ABSTAIN,
    )

    assert prompt.original_shape == expected_shape
    assert (
        prompt.state_sha256
        == build_forge_source(
            expected_shape,
            "ABSTAIN",  # type: ignore[arg-type]
        ).program.initial_state.sha256()
    )
    assert prompt.pass_output.eligible is True


@pytest.mark.parametrize(
    "draft",
    [
        CANONICAL_ABSTAIN,
        ' \n { "decision" : "ABSTAIN" }\t',
        '{"decision":"CLARIFY","missing":["cafe\u0301"]}',
    ],
)
def test_draft_raw_json_string_round_trip_preserves_canonical_noncanonical_and_unicode(
    train_prompt: str,
    draft: str,
) -> None:
    prompt = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=draft,
    )
    visible = _visible(prompt)

    assert set(visible) == {
        "schema_version",
        "mode",
        "original",
        "draft_raw",
        "diagnostics",
    }
    assert visible["schema_version"] == runtime.DRAFT_TRANSPORT_VERSION
    assert visible["draft_raw"] == draft
    assert prompt.draft_raw == draft
    assert prompt.draft_utf8_bytes == len(draft.encode("utf-8"))
    assert prompt.draft_sha256 == hashlib.sha256(draft.encode("utf-8")).hexdigest()
    assert _canonical(visible) == prompt.prompt_json
    assert set(visible["diagnostics"]) == set(DIAGNOSTIC_FIELDS)  # type: ignore[arg-type]
    assert prompt.diagnostics.parse_valid is True
    assert prompt.diagnostics.schema_valid is True

    if "cafe" in draft:
        assert unicodedata.normalize("NFC", draft) != draft
        assert visible["draft_raw"] != unicodedata.normalize("NFC", draft)


@pytest.mark.parametrize(
    ("draft", "error_code"),
    [
        ("not json", "invalid_json"),
        ('{"decision":"ABSTAIN","decision":"ABSTAIN"}', "duplicate_key"),
        ('{"decision":"ABSTAIN"} suffix', "invalid_json"),
        ("", "invalid_json"),
        ("[]", "top_level_not_object"),
    ],
)
def test_invalid_duplicate_suffix_empty_and_nonobject_drafts_are_preserved_and_diagnosed(
    train_prompt: str,
    draft: str,
    error_code: str,
) -> None:
    prompt = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=draft,
    )

    assert _visible(prompt)["draft_raw"] == draft
    assert prompt.diagnostics.parse_valid is False
    assert prompt.diagnostics.schema_valid is False
    assert prompt.diagnostics.policy_status == "unavailable"
    assert prompt.diagnostics.simulator_status == "unavailable"
    assert prompt.diagnostics.error_codes == (error_code,)
    assert prompt.pass_output.eligible is False
    if not draft:
        assert prompt.pass_output.raw is None


def test_schema_policy_and_simulator_evidence_are_derived_from_raw_and_original_state() -> None:
    note = build_forge_source("T-synth", "CREATE_NOTE")
    invalid_schema = runtime.build_correction_prompt(
        original_prompt_json=note.direct_prompt_json,
        draft_raw='{"decision":"CALL"}',
    )
    blocked_payload = json.loads(note.gold_target_json)
    blocked_payload["decision"] = "CALL"
    blocked = runtime.build_correction_prompt(
        original_prompt_json=note.direct_prompt_json,
        draft_raw=_canonical(blocked_payload),
    )

    route = build_forge_source("D-internal", "LOOK_UP_ROUTE")
    accepted = runtime.build_correction_prompt(
        original_prompt_json=route.direct_prompt_json,
        draft_raw=route.gold_target_json,
    )
    missing_route_payload = json.loads(route.gold_target_json)
    missing_route_payload["calls"][0]["args"]["destination_id"] = "place.screen.00.missing"
    rejected = runtime.build_correction_prompt(
        original_prompt_json=route.direct_prompt_json,
        draft_raw=_canonical(missing_route_payload),
    )

    assert invalid_schema.diagnostics.parse_valid is True
    assert invalid_schema.diagnostics.schema_valid is False
    assert invalid_schema.diagnostics.policy_status == "unavailable"
    assert invalid_schema.diagnostics.simulator_status == "unavailable"
    assert invalid_schema.diagnostics.error_codes == ("missing_field",)

    assert blocked.diagnostics.parse_valid is True
    assert blocked.diagnostics.schema_valid is True
    assert blocked.diagnostics.policy_status == "blocked"
    assert blocked.diagnostics.simulator_status == "rejected"
    assert blocked.diagnostics.error_codes == ("policy_decision_mismatch",)
    assert blocked.pass_output.policy_conformant_proposal is False

    assert accepted.diagnostics.policy_status == "conformant"
    assert accepted.diagnostics.simulator_status == "accepted"
    assert accepted.diagnostics.error_codes == ()
    assert accepted.pass_output.eligible is True

    assert rejected.diagnostics.policy_status == "conformant"
    assert rejected.diagnostics.simulator_status == "rejected"
    assert rejected.diagnostics.error_codes == ("route_missing",)
    # The frozen fallback predicate is proposal conformance, not simulator success.
    assert rejected.pass_output.eligible is True


def test_model_visible_diagnostics_are_exactly_five_draft_only_fields(train_prompt: str) -> None:
    prompt = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=CANONICAL_ABSTAIN,
    )
    visible = _visible(prompt)
    diagnostics = visible["diagnostics"]
    assert type(diagnostics) is dict
    assert set(diagnostics) == set(DIAGNOSTIC_FIELDS)
    assert set(diagnostics) == {
        "parse_valid",
        "schema_valid",
        "policy_status",
        "simulator_status",
        "error_codes",
    }
    forbidden = {
        "gold",
        "target",
        "label",
        "oracle",
        "fault",
        "certificate",
        "execution_permitted",
        "policy_conformant_proposal",
    }
    assert forbidden.isdisjoint(diagnostics)
    assert forbidden.isdisjoint(_all_mapping_keys(visible))

    audit = prompt.audit_record()
    assert all(value is False for value in audit["flags"].values())  # type: ignore[union-attr]
    assert "prompt_json" not in audit
    assert "draft_raw" not in audit


@pytest.mark.parametrize("token", runtime.RESERVED_TOKENIZER_CONTROL_TOKENS)
def test_every_pinned_tokenizer_control_token_fails_closed(
    train_prompt: str,
    token: str,
) -> None:
    draft = _canonical({"decision": "CLARIFY", "missing": [token.upper()]})

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
        runtime.build_correction_prompt(
            original_prompt_json=train_prompt,
            draft_raw=draft,
        )

    assert caught.value.code == "reserved_token"
    assert caught.value.path == "$.draft_raw"


def test_raw_type_surrogate_and_utf8_byte_limit_fail_closed(train_prompt: str) -> None:
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as wrong_type:
        runtime.build_correction_prompt(
            original_prompt_json=train_prompt,
            draft_raw=b'{"decision":"ABSTAIN"}',  # type: ignore[arg-type]
        )
    assert wrong_type.value.code == "type_mismatch"

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as surrogate:
        runtime.build_correction_prompt(
            original_prompt_json=train_prompt,
            draft_raw='{"decision":"CLARIFY","missing":["\ud800"]}',
        )
    assert surrogate.value.code == "invalid_utf8"

    exact_limit = "é" * (runtime.MAX_DRAFT_UTF8_BYTES // 2)
    accepted = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=exact_limit,
    )
    assert accepted.draft_utf8_bytes == runtime.MAX_DRAFT_UTF8_BYTES
    assert _visible(accepted)["draft_raw"] == exact_limit

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as oversized:
        runtime.build_correction_prompt(
            original_prompt_json=train_prompt,
            draft_raw=exact_limit + "é",
        )
    assert oversized.value.code == "size_limit"


def test_original_prompt_must_be_exact_canonical_known_shape_without_hidden_answers(
    train_prompt: str,
) -> None:
    payload = json.loads(train_prompt)
    payload["gold"] = CANONICAL_ABSTAIN
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as extra:
        runtime.build_correction_prompt(
            original_prompt_json=_canonical(payload),
            draft_raw=CANONICAL_ABSTAIN,
        )
    assert extra.value.code == "prompt_shape"

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as noncanonical:
        runtime.build_correction_prompt(
            original_prompt_json=" " + train_prompt,
            draft_raw=CANONICAL_ABSTAIN,
        )
    assert noncanonical.value.code == "noncanonical_original"

    duplicate = '{"action_ir_contract":"ACTION_IR_V1","action_ir_contract":"ACTION_IR_V1"}'
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as duplicate_error:
        runtime.build_correction_prompt(
            original_prompt_json=duplicate,
            draft_raw=CANONICAL_ABSTAIN,
        )
    assert duplicate_error.value.code == "duplicate_key"

    payload = json.loads(train_prompt)
    payload["request"] += " <assistant>"
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as injection:
        runtime.build_correction_prompt(
            original_prompt_json=_canonical(payload),
            draft_raw=CANONICAL_ABSTAIN,
        )
    assert injection.value.code == "reserved_token"
    assert injection.value.path == "$.original"

    payload = json.loads(train_prompt)
    del payload["context"]["state"]["settings"]
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as state_error:
        runtime.build_correction_prompt(
            original_prompt_json=_canonical(payload),
            draft_raw=CANONICAL_ABSTAIN,
        )
    assert state_error.value.code == "invalid_state"


def test_pass_output_is_self_derived_and_invalid_pass2_falls_back_to_pass1(
    train_prompt: str,
) -> None:
    signature = inspect.signature(runtime.select_two_pass_outputs)
    assert {
        "parse_valid",
        "schema_valid",
        "policy_conformant_proposal",
        "gold",
        "target",
        "fault",
        "certificate",
    }.isdisjoint(signature.parameters)

    selected = runtime.select_two_pass_outputs(
        original_prompt_json=train_prompt,
        pass1_raw=' {"decision" : "ABSTAIN"} ',
        pass1_termination=runtime.PassTermination.COMPLETE,
        pass2_raw='{"decision":',
        pass2_termination=runtime.PassTermination.COMPLETE,
    )

    assert selected.source == "pass1"
    assert selected.raw == ' {"decision" : "ABSTAIN"} '
    assert selected.pass1.pass_output.eligible is True
    assert selected.pass2.pass_output.eligible is False

    with pytest.raises(TypeError):
        runtime.select_two_pass_outputs(  # type: ignore[call-arg]
            original_prompt_json=train_prompt,
            pass1_raw=CANONICAL_ABSTAIN,
            pass1_termination=runtime.PassTermination.COMPLETE,
            pass2_raw=CANONICAL_ABSTAIN,
            pass2_termination=runtime.PassTermination.COMPLETE,
            parse_valid=True,
        )


def test_valid_pass2_wins_and_exact_noncanonical_output_is_not_reserialized(
    train_prompt: str,
) -> None:
    pass2 = '\n{ "decision" : "ABSTAIN" }\n'
    selected = runtime.select_two_pass_outputs(
        original_prompt_json=train_prompt,
        pass1_raw=CANONICAL_ABSTAIN,
        pass1_termination=runtime.PassTermination.COMPLETE,
        pass2_raw=pass2,
        pass2_termination=runtime.PassTermination.COMPLETE,
    )

    assert selected.source == "pass2"
    assert selected.raw == pass2


def test_missing_failure_and_truncation_have_deterministic_ineligible_behavior(
    train_prompt: str,
) -> None:
    missing = runtime.assess_pass_output(
        original_prompt_json=train_prompt,
        raw=None,
        termination=runtime.PassTermination.MISSING,
    )
    failure = runtime.assess_pass_output(
        original_prompt_json=train_prompt,
        raw=None,
        termination=runtime.PassTermination.FAILURE,
    )
    truncated = runtime.assess_pass_output(
        original_prompt_json=train_prompt,
        raw=CANONICAL_ABSTAIN,
        termination=runtime.PassTermination.TRUNCATED,
    )

    assert missing.raw is None and missing.diagnostics is None
    assert failure.raw is None and failure.diagnostics is None
    assert missing.pass_output.eligible is False
    assert failure.pass_output.eligible is False
    assert truncated.raw == CANONICAL_ABSTAIN
    assert truncated.diagnostics is not None
    assert truncated.diagnostics.parse_valid is True
    assert truncated.pass_output.eligible is False
    assert truncated.pass_output.parse_valid is False

    selected = runtime.select_two_pass_outputs(
        original_prompt_json=train_prompt,
        pass1_raw=CANONICAL_ABSTAIN,
        pass1_termination=runtime.PassTermination.TRUNCATED,
        pass2_raw=None,
        pass2_termination=runtime.PassTermination.FAILURE,
    )
    assert selected.source == "canonical_abstain"
    assert selected.raw == CANONICAL_ABSTAIN

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as contradictory:
        runtime.assess_pass_output(
            original_prompt_json=train_prompt,
            raw="unexpected partial output",
            termination=runtime.PassTermination.FAILURE,
        )
    assert contradictory.value.code == "termination_shape"

    with pytest.raises(TypeError, match="exact PassTermination"):
        runtime.assess_pass_output(
            original_prompt_json=train_prompt,
            raw=CANONICAL_ABSTAIN,
            termination="complete",  # type: ignore[arg-type]
        )


def test_two_invalid_complete_passes_use_canonical_abstain(train_prompt: str) -> None:
    selected = runtime.select_two_pass_outputs(
        original_prompt_json=train_prompt,
        pass1_raw="",
        pass1_termination=runtime.PassTermination.COMPLETE,
        pass2_raw='{"decision":"CALL"}',
        pass2_termination=runtime.PassTermination.COMPLETE,
    )

    assert selected.source == "canonical_abstain"
    assert selected.raw == '{"decision":"ABSTAIN"}'


def test_runtime_has_no_gold_target_or_forge_dependency_in_public_inputs() -> None:
    public_functions = (
        runtime.build_correction_prompt,
        runtime.assess_pass_output,
        runtime.select_two_pass_outputs,
    )
    forbidden = ("gold", "target", "label", "oracle", "fault", "certificate")
    for function in public_functions:
        names = tuple(inspect.signature(function).parameters)
        assert not any(term in name for name in names for term in forbidden)

    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("forge" in module or "datasets" in module for module in imports)


def test_runtime_integrity_guard_itself_passes_and_is_closure_captured() -> None:
    runtime._assert_runtime_integrity()
    closure_values = tuple(
        cell.cell_contents for cell in (runtime.runtime_boundary.__closure__ or ())
    )

    assert runtime._assert_runtime_integrity in closure_values
    assert "hostile same-UID" in runtime.runtime_boundary()["integrity_boundary"]


def test_authorization_global_rebinding_fails_before_boundary_or_audit_returns(
    train_prompt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=CANONICAL_ABSTAIN,
    )
    monkeypatch.setattr(runtime, "RUNTIME_FLAGS", {"launch_authorized": True})

    for operation in (runtime.runtime_boundary, prompt.audit_record):
        with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
            operation()
        assert caught.value.code == "runtime_identity"


def test_derive_rebinding_cannot_turn_invalid_raw_into_an_eligible_pass(
    train_prompt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forged_derive(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("the forged derivation must never run")

    monkeypatch.setattr(runtime, "_derive_draft", forged_derive)
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
        runtime.assess_pass_output(
            original_prompt_json=train_prompt,
            raw="not json",
            termination=runtime.PassTermination.COMPLETE,
        )

    assert caught.value.code == "runtime_identity"
    assert called is False


def test_selector_rebinding_cannot_bypass_pass2_pass1_abstain_fallback(
    train_prompt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forged_selector(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("the forged selector must never run")

    monkeypatch.setattr(runtime, "select_two_pass_output", forged_selector)
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
        runtime.select_two_pass_outputs(
            original_prompt_json=train_prompt,
            pass1_raw=None,
            pass1_termination=runtime.PassTermination.MISSING,
            pass2_raw=None,
            pass2_termination=runtime.PassTermination.FAILURE,
        )

    assert caught.value.code == "runtime_identity"
    assert called is False


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("ActionSimulator", object),
        ("decode_json_object", lambda raw: {}),
        ("validate_action_ir", lambda decoded, schemas: object()),
        ("CANONICAL_ABSTAIN", '{"decision":"CALL"}'),
        ("RUNTIME_VERSION", "forged-runtime"),
        ("_assert_runtime_integrity", lambda: None),
    ],
)
def test_checked_dependency_constant_and_guard_rebinding_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(runtime, name, replacement)

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
        runtime.runtime_boundary()

    assert caught.value.code == "runtime_identity"


def test_contract_dependency_and_source_identity_rebinding_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runtime._contract_module,
        "CANONICAL_ABSTAIN",
        '{"decision":"CALL"}',
    )
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as contract_error:
        runtime.runtime_boundary()
    assert contract_error.value.code == "runtime_identity"
    monkeypatch.undo()

    substitute = tmp_path / "action_correction_runtime.py"
    substitute.write_text(MODULE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(runtime, "__file__", str(substitute))
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as source_error:
        runtime.runtime_boundary()
    assert source_error.value.code == "runtime_identity"


def test_direct_correction_prompt_constructor_rederives_every_evidence_field(
    train_prompt: str,
) -> None:
    valid = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=CANONICAL_ABSTAIN,
    )

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as bad_hash:
        runtime.CorrectionPrompt(
            prompt_json=valid.prompt_json,
            original_shape=valid.original_shape,
            draft_raw=valid.draft_raw,
            draft_utf8_bytes=valid.draft_utf8_bytes,
            draft_sha256="0" * 64,
            state_sha256=valid.state_sha256,
            diagnostics=valid.diagnostics,
            pass_output=valid.pass_output,
        )
    assert bad_hash.value.code == "evidence_mismatch"

    forged_payload = json.loads(valid.prompt_json)
    forged_payload["diagnostics"] = {
        "parse_valid": False,
        "schema_valid": False,
        "policy_status": "unavailable",
        "simulator_status": "unavailable",
        "error_codes": ["invalid_json"],
    }
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as forged_diagnostics:
        runtime.CorrectionPrompt(
            prompt_json=_canonical(forged_payload),
            original_shape=valid.original_shape,
            draft_raw=valid.draft_raw,
            draft_utf8_bytes=valid.draft_utf8_bytes,
            draft_sha256=valid.draft_sha256,
            state_sha256=valid.state_sha256,
            diagnostics=valid.diagnostics,
            pass_output=valid.pass_output,
        )
    assert forged_diagnostics.value.code == "diagnostic_mismatch"


def test_low_level_correction_prompt_construction_and_tamper_cannot_forge_audit(
    train_prompt: str,
) -> None:
    valid = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=CANONICAL_ABSTAIN,
    )
    forged = object.__new__(runtime.CorrectionPrompt)
    for name in (
        "prompt_json",
        "original_shape",
        "draft_raw",
        "draft_utf8_bytes",
        "draft_sha256",
        "state_sha256",
        "diagnostics",
        "pass_output",
    ):
        object.__setattr__(forged, name, getattr(valid, name))
    object.__setattr__(forged, "state_sha256", "f" * 64)

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as forged_error:
        forged.audit_record()
    assert forged_error.value.code == "evidence_mismatch"
    assert forged_error.value.path == "$.state_sha256"

    incomplete = object.__new__(runtime.CorrectionPrompt)
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as missing_error:
        incomplete.audit_record()
    assert missing_error.value.code == "evidence_mismatch"


def test_audit_record_rederives_after_success_and_rejects_post_call_tamper(
    train_prompt: str,
) -> None:
    prompt = runtime.build_correction_prompt(
        original_prompt_json=train_prompt,
        draft_raw=CANONICAL_ABSTAIN,
    )
    first = prompt.audit_record()
    assert first["draft_sha256"] == prompt.draft_sha256
    assert all(value is False for value in first["flags"].values())  # type: ignore[union-attr]

    object.__setattr__(
        prompt,
        "diagnostics",
        runtime.DraftDiagnostics(False, False, "unavailable", "unavailable", ("invalid_json",)),
    )
    with pytest.raises(runtime.ActionCorrectionRuntimeError) as tamper_error:
        prompt.audit_record()
    assert tamper_error.value.code == "evidence_mismatch"
    assert tamper_error.value.path == "$.diagnostics"
    assert first["draft_sha256"] != "0" * 64


def test_dependency_export_rebinding_is_translated_to_stable_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime._action_simulator_module, "ActionSimulator", object)

    with pytest.raises(runtime.ActionCorrectionRuntimeError) as caught:
        runtime.runtime_boundary()

    assert caught.value.code == "runtime_identity"
    assert caught.value.path == "$"


def test_pass_and_selection_values_are_explicitly_transport_only_not_receipts(
    train_prompt: str,
) -> None:
    assessment = runtime.assess_pass_output(
        original_prompt_json=train_prompt,
        raw=CANONICAL_ABSTAIN,
        termination=runtime.PassTermination.COMPLETE,
    )
    selection = runtime.select_two_pass_outputs(
        original_prompt_json=train_prompt,
        pass1_raw=CANONICAL_ABSTAIN,
        pass1_termination=runtime.PassTermination.COMPLETE,
        pass2_raw=None,
        pass2_termination=runtime.PassTermination.MISSING,
    )
    boundary = runtime.runtime_boundary()

    assert not hasattr(assessment, "audit_record")
    assert not hasattr(selection, "audit_record")
    assert boundary["accepted_audit_surfaces"] == ["CorrectionPrompt.audit_record"]
    assert boundary["transport_only_not_receipts"] == ["PassAssessment", "TwoPassSelection"]
