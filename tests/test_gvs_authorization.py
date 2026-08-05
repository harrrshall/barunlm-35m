from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from barunlm.evaluation import gvs_authorization as authorization_module
from barunlm.evaluation.gvs_authorization import (
    GVS_AUTHORIZATION_BINDING_FIELDS,
    GVS_AUTHORIZATION_RECEIPT_VERSION,
    GVSAuthorizationError,
    assert_gvs_authorization_runtime_integrity,
    claim_gvs_population_once,
    gvs_authorization_runtime_sha256,
    load_gvs_authorization_receipt_json,
    load_gvs_population_claim_ledger_json,
    seal_gvs_authorization_receipt,
    validate_gvs_authorization_receipt,
    validate_gvs_population_claim_ledger,
)

RECEIPT_KEY = b"receipt-key-for-tests-only-32-bytes-minimum"
LEDGER_KEY = b"ledger-key-for-tests-only-32-bytes-minimum-"
GATE_IDS = ("all_denominators_complete", "no_catastrophic_actions", "primary_effect_passed")


def _bindings(character: str = "a") -> dict[str, Any]:
    result: dict[str, Any] = {name: character * 64 for name in GVS_AUTHORIZATION_BINDING_FIELDS}
    result["eos_token_id"] = 2
    result["pad_token_id"] = 0
    return result


def _receipt(
    *,
    bindings: dict[str, Any] | None = None,
    phase: str = "selection",
    failed_gate: str | None = None,
) -> dict[str, Any]:
    gates = {name: name != failed_gate for name in GATE_IDS}
    value = {
        "schema_version": GVS_AUTHORIZATION_RECEIPT_VERSION,
        "phase": phase,
        "run_id": "20260804-1200-gvs-test-s17",
        "created_at_utc": "2026-08-04T06:30:00Z",
        "key_id": "gvs-receipt-key-v1",
        "bindings": copy.deepcopy(bindings or _bindings()),
        "gate_spec_sha256": "b" * 64,
        "gate_results": gates,
        "passed": all(gates.values()),
        "authorizes_model_cuda_training_or_jarvis": False,
        "authorizes_private_label_access": False,
        "authorization_runtime_sha256": "0" * 64,
        "receipt_sha256": "0" * 64,
        "receipt_hmac_sha256": "0" * 64,
    }
    return seal_gvs_authorization_receipt(value, signing_key=RECEIPT_KEY)


def _validate(receipt: dict[str, Any], bindings: dict[str, Any] | None = None) -> dict[str, Any]:
    return validate_gvs_authorization_receipt(
        receipt,
        signing_key=RECEIPT_KEY,
        expected_key_id="gvs-receipt-key-v1",
        expected_phase=receipt["phase"],
        expected_run_id="20260804-1200-gvs-test-s17",
        expected_bindings=bindings or _bindings(),
        expected_gate_spec_sha256="b" * 64,
        expected_gate_ids=GATE_IDS,
    )


def _claim(
    receipt: dict[str, Any],
    *,
    bindings: dict[str, Any] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    claim_id: str = "claim-1",
    claimed_at: str = "2026-08-04T06:31:00Z",
    append=None,
) -> dict[str, Any]:
    prior = [] if ledger is None else ledger

    def acknowledge(event_bytes: bytes, _head: str) -> str:
        event = json.loads(event_bytes)
        assert type(event) is dict
        return event["event_sha256"]

    return claim_gvs_population_once(
        authorization_receipt=receipt,
        receipt_signing_key=RECEIPT_KEY,
        expected_receipt_key_id="gvs-receipt-key-v1",
        expected_phase=receipt["phase"],
        expected_run_id="20260804-1200-gvs-test-s17",
        expected_bindings=bindings or _bindings(),
        expected_gate_spec_sha256="b" * 64,
        expected_gate_ids=GATE_IDS,
        prior_durable_ledger=prior,
        ledger_signing_key=LEDGER_KEY,
        expected_ledger_key_id="gvs-ledger-key-v1",
        claim_id=claim_id,
        scoring_session_id="scoring-session-1",
        accessor_id="isolated-scorer-1",
        purpose="one-shot GVS population scoring",
        claimed_at_utc=claimed_at,
        durable_compare_and_append=acknowledge if append is None else append,
    )


def _resign_claim_event(event: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(event)
    result["event_sha256"] = "0" * 64
    result["event_hmac_sha256"] = "0" * 64
    result["event_sha256"] = authorization_module._sha256_json(
        authorization_module._event_hash_payload(result),
        domain=authorization_module._EVENT_HASH_DOMAIN,
    )
    result["event_hmac_sha256"] = authorization_module._hmac_sha256(
        LEDGER_KEY,
        authorization_module._event_mac_payload(result),
        domain=authorization_module._EVENT_HMAC_DOMAIN,
    )
    return result


def test_receipt_seals_and_validates_every_exact_binding() -> None:
    bindings = _bindings()
    receipt = _receipt(bindings=bindings)
    validated = _validate(receipt, bindings)
    assert validated == receipt
    assert validated is not receipt
    assert validated["passed"] is True
    assert validated["authorizes_model_cuda_training_or_jarvis"] is False
    assert validated["authorizes_private_label_access"] is False
    assert validated["authorization_runtime_sha256"] == gvs_authorization_runtime_sha256()
    assert validated["receipt_sha256"] != validated["receipt_hmac_sha256"]
    assert set(validated["bindings"]) == GVS_AUTHORIZATION_BINDING_FIELDS


def test_receipt_rejects_tampering_forgery_and_gate_roster_drift() -> None:
    receipt = _receipt()
    altered = copy.deepcopy(receipt)
    altered["bindings"]["generator_checkpoint_sha256"] = "c" * 64
    with pytest.raises(GVSAuthorizationError, match="bindings differ"):
        _validate(altered)

    forged = copy.deepcopy(receipt)
    forged["receipt_hmac_sha256"] = forged["receipt_sha256"]
    with pytest.raises(GVSAuthorizationError, match="HMAC mismatch"):
        _validate(forged)

    with pytest.raises(GVSAuthorizationError, match="different frozen gate roster"):
        validate_gvs_authorization_receipt(
            receipt,
            signing_key=RECEIPT_KEY,
            expected_key_id="gvs-receipt-key-v1",
            expected_phase="selection",
            expected_run_id="20260804-1200-gvs-test-s17",
            expected_bindings=_bindings(),
            expected_gate_spec_sha256="b" * 64,
            expected_gate_ids=("all_denominators_complete", "primary_effect_passed"),
        )


def test_failed_gate_receipt_is_preserved_but_cannot_claim() -> None:
    receipt = _receipt(failed_gate="primary_effect_passed")
    validated = validate_gvs_authorization_receipt(
        receipt,
        signing_key=RECEIPT_KEY,
        expected_key_id="gvs-receipt-key-v1",
        expected_phase="selection",
        expected_run_id="20260804-1200-gvs-test-s17",
        expected_bindings=_bindings(),
        expected_gate_spec_sha256="b" * 64,
        expected_gate_ids=GATE_IDS,
        require_passed=False,
    )
    assert validated["passed"] is False
    with pytest.raises(GVSAuthorizationError, match="failed gate"):
        _claim(receipt)


def test_complete_population_claim_is_appended_and_globally_retired() -> None:
    receipt = _receipt()
    committed: list[dict[str, Any]] = []

    def append(event_bytes, expected_head):
        assert expected_head == "0" * 64
        assert type(event_bytes) is bytes
        event = json.loads(event_bytes)
        committed.append(event)
        return event["event_sha256"]

    event = _claim(receipt, append=append)
    assert event == committed[0]
    assert event["partial_retry_authorized"] is False
    assert event["authorizes_model_cuda_training_or_jarvis"] is False
    assert event["authorizes_private_label_access"] is False
    assert event["proves_external_atomicity"] is False
    assert event["authorization_runtime_sha256"] == gvs_authorization_runtime_sha256()
    assert event["event_sha256"] != event["event_hmac_sha256"]
    assert event["population_manifest_sha256"] == _bindings()["population_manifest_sha256"]
    assert event["prompt_collection_sha256"] == _bindings()["prompt_collection_sha256"]
    assert event["label_commitment_root_sha256"] == _bindings()["label_commitment_root_sha256"]
    assert (
        validate_gvs_population_claim_ledger(
            committed,
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )
        == event["event_sha256"]
    )
    with pytest.raises(GVSAuthorizationError, match="already.*retired"):
        _claim(
            receipt,
            ledger=committed,
            claim_id="claim-2",
            claimed_at="2026-08-04T06:32:00Z",
        )


@pytest.mark.parametrize(
    ("reused_binding", "message"),
    [
        ("prompt_collection_sha256", "prompt collection.*already.*retired"),
        ("label_commitment_root_sha256", "label commitment.*already.*retired"),
    ],
)
def test_repackaged_population_cannot_replay_prompt_or_label_commitments(
    reused_binding: str,
    message: str,
) -> None:
    first_bindings = _bindings("a")
    first = _claim(_receipt(bindings=first_bindings), bindings=first_bindings)
    repackaged_bindings = _bindings("c")
    repackaged_bindings[reused_binding] = first_bindings[reused_binding]
    repackaged_receipt = _receipt(
        bindings=repackaged_bindings,
        phase="confirmation",
    )

    with pytest.raises(GVSAuthorizationError, match=message):
        _claim(
            repackaged_receipt,
            bindings=repackaged_bindings,
            ledger=[first],
            claim_id=f"claim-repackaged-{reused_binding}",
            claimed_at="2026-08-04T06:32:00Z",
        )


@pytest.mark.parametrize(
    ("reused_field", "message"),
    [
        ("prompt_collection_sha256", "prompt collection.*already.*retired"),
        ("label_commitment_root_sha256", "label commitment.*already.*retired"),
    ],
)
def test_full_ledger_validation_rejects_resigned_retirement_replays(
    reused_field: str,
    message: str,
) -> None:
    first_bindings = _bindings("a")
    first = _claim(_receipt(bindings=first_bindings), bindings=first_bindings)
    second_bindings = _bindings("c")
    second = _claim(
        _receipt(bindings=second_bindings, phase="confirmation"),
        bindings=second_bindings,
        ledger=[first],
        claim_id="claim-second",
        claimed_at="2026-08-04T06:32:00Z",
    )
    second[reused_field] = first[reused_field]
    second = _resign_claim_event(second)

    with pytest.raises(GVSAuthorizationError, match=message):
        validate_gvs_population_claim_ledger(
            [first, second],
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )


def test_confirmation_requires_prior_same_run_selection_claim() -> None:
    bindings = _bindings("c")
    receipt = _receipt(bindings=bindings, phase="confirmation")

    with pytest.raises(GVSAuthorizationError, match="earlier same-run selection"):
        _claim(receipt, bindings=bindings)


@pytest.mark.parametrize("later_phase", ["d_support", "selection"])
def test_run_phase_claims_cannot_repeat_or_move_backwards(later_phase: str) -> None:
    first_bindings = _bindings("a")
    first = _claim(_receipt(bindings=first_bindings), bindings=first_bindings)
    later_bindings = _bindings("c")
    later_receipt = _receipt(bindings=later_bindings, phase=later_phase)

    with pytest.raises(GVSAuthorizationError, match="strictly advance once per run"):
        _claim(
            later_receipt,
            bindings=later_bindings,
            ledger=[first],
            claim_id=f"claim-later-{later_phase}",
            claimed_at="2026-08-04T06:32:00Z",
        )


def test_full_ledger_validation_enforces_phase_and_time_chronology() -> None:
    first_bindings = _bindings("a")
    first = _claim(_receipt(bindings=first_bindings), bindings=first_bindings)

    confirmation_first = copy.deepcopy(first)
    confirmation_first["phase"] = "confirmation"
    confirmation_first = _resign_claim_event(confirmation_first)
    with pytest.raises(GVSAuthorizationError, match="earlier same-run selection"):
        validate_gvs_population_claim_ledger(
            [confirmation_first],
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )

    second_bindings = _bindings("c")
    second = _claim(
        _receipt(bindings=second_bindings, phase="confirmation"),
        bindings=second_bindings,
        ledger=[first],
        claim_id="claim-chronology-second",
        claimed_at="2026-08-04T06:32:00Z",
    )
    nonadvancing_phase = copy.deepcopy(second)
    nonadvancing_phase["phase"] = "selection"
    nonadvancing_phase = _resign_claim_event(nonadvancing_phase)
    with pytest.raises(GVSAuthorizationError, match="strictly advance once per run"):
        validate_gvs_population_claim_ledger(
            [first, nonadvancing_phase],
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )

    nonadvancing_time = copy.deepcopy(second)
    nonadvancing_time["claimed_at_utc"] = first["claimed_at_utc"]
    nonadvancing_time = _resign_claim_event(nonadvancing_time)
    with pytest.raises(GVSAuthorizationError, match="timestamps must strictly increase"):
        validate_gvs_population_claim_ledger(
            [first, nonadvancing_time],
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )


def test_crash_after_external_append_still_retires_population_on_retry() -> None:
    receipt = _receipt()
    durable: list[dict[str, Any]] = []

    def append_then_crash(event_bytes, expected_head):
        assert expected_head == "0" * 64
        durable.append(json.loads(event_bytes))
        raise RuntimeError("connection dropped after commit")

    with pytest.raises(GVSAuthorizationError, match="append failed"):
        _claim(receipt, append=append_then_crash)
    assert len(durable) == 1
    with pytest.raises(GVSAuthorizationError, match="already.*retired"):
        _claim(
            receipt,
            ledger=durable,
            claim_id="claim-retry",
            claimed_at="2026-08-04T06:32:00Z",
        )


def test_wrong_ack_after_external_append_is_unknown_but_still_retires_on_refresh() -> None:
    receipt = _receipt()
    durable: list[dict[str, Any]] = []

    def append_then_wrong_ack(event_bytes, expected_head):
        assert expected_head == "0" * 64
        durable.append(json.loads(event_bytes))
        return "f" * 64

    with pytest.raises(GVSAuthorizationError, match="different head"):
        _claim(receipt, append=append_then_wrong_ack)
    with pytest.raises(GVSAuthorizationError, match="already.*retired"):
        _claim(
            receipt,
            ledger=durable,
            claim_id="claim-after-wrong-ack",
            claimed_at="2026-08-04T06:32:00Z",
        )


def test_callback_receives_immutable_bytes_and_must_acknowledge_exact_head() -> None:
    receipt = _receipt()
    with pytest.raises(GVSAuthorizationError, match="committed head"):
        _claim(receipt, append=lambda event_bytes, expected_head: False)
    with pytest.raises(GVSAuthorizationError, match="different head"):
        _claim(receipt, append=lambda event_bytes, expected_head: "f" * 64)

    captured: list[dict[str, Any]] = []

    def inspect_immutable_payload(event_bytes, expected_head):
        assert type(event_bytes) is bytes
        with pytest.raises(TypeError):
            event_bytes[0] = 0
        event = json.loads(event_bytes)
        captured.append(event)
        return event["event_sha256"]

    returned = _claim(receipt, append=inspect_immutable_payload)
    assert captured[0] == returned
    assert returned["purpose"] == "one-shot GVS population scoring"
    assert returned["proves_external_atomicity"] is False
    assert returned["authorizes_private_label_access"] is False


def test_callback_cannot_change_snapshotted_receipt_or_prior_ledger() -> None:
    receipt = _receipt()
    prior: list[dict[str, Any]] = []
    original_population = receipt["bindings"]["population_manifest_sha256"]

    def mutate_callers_after_snapshot(event_bytes, expected_head):
        assert expected_head == "0" * 64
        receipt["bindings"]["population_manifest_sha256"] = "f" * 64
        prior.append({"attacker": "late mutation"})
        event = json.loads(event_bytes)
        return event["event_sha256"]

    result = _claim(receipt, ledger=prior, append=mutate_callers_after_snapshot)
    assert result["population_manifest_sha256"] == original_population
    assert receipt["bindings"]["population_manifest_sha256"] == "f" * 64
    assert prior == [{"attacker": "late mutation"}]


def test_lying_callback_cannot_turn_protocol_plumbing_into_authorization() -> None:
    receipt = _receipt()

    def no_op_lie(event_bytes, expected_head):
        del expected_head
        return json.loads(event_bytes)["event_sha256"]

    result = _claim(receipt, append=no_op_lie)
    assert result["proves_external_atomicity"] is False
    assert result["authorizes_private_label_access"] is False


def test_ledger_rejects_hash_hmac_order_time_and_population_tampering() -> None:
    first_receipt = _receipt(bindings=_bindings("a"))
    first = _claim(first_receipt, bindings=_bindings("a"))
    second_receipt = _receipt(bindings=_bindings("c"), phase="confirmation")
    second = _claim(
        second_receipt,
        bindings=_bindings("c"),
        ledger=[first],
        claim_id="claim-2",
        claimed_at="2026-08-04T06:32:00Z",
    )
    ledger = [first, second]
    validate_gvs_population_claim_ledger(
        ledger,
        signing_key=LEDGER_KEY,
        expected_key_id="gvs-ledger-key-v1",
    )
    tampered = copy.deepcopy(ledger)
    tampered[0]["purpose"] = "changed"
    with pytest.raises(GVSAuthorizationError, match="self-hash mismatch"):
        validate_gvs_population_claim_ledger(
            tampered,
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )
    with pytest.raises(GVSAuthorizationError, match="sequence|chain"):
        validate_gvs_population_claim_ledger(
            list(reversed(ledger)),
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )


def test_strict_serialized_loaders_reject_duplicate_keys_nan_and_wrong_topology() -> None:
    receipt = _receipt()
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    assert load_gvs_authorization_receipt_json(encoded) == receipt
    assert load_gvs_population_claim_ledger_json("[]") == []
    with pytest.raises(GVSAuthorizationError, match="duplicate key"):
        load_gvs_authorization_receipt_json('{"a":1,"a":2}')
    with pytest.raises(GVSAuthorizationError, match="forbidden constant"):
        load_gvs_authorization_receipt_json('{"a":NaN}')
    with pytest.raises(GVSAuthorizationError, match="valid UTF-8|invalid UTF-8"):
        load_gvs_authorization_receipt_json('{"a":"\\ud800"}')
    with pytest.raises(GVSAuthorizationError, match="event list"):
        load_gvs_population_claim_ledger_json("{}")
    with pytest.raises(GVSAuthorizationError, match="strict JSON"):
        load_gvs_population_claim_ledger_json('[],"injected":true')
    for wrong_type in (bytearray(b"[]"), memoryview(b"[]"), 7, None):
        with pytest.raises(GVSAuthorizationError, match="exact bytes or text"):
            load_gvs_population_claim_ledger_json(wrong_type)
    with pytest.raises(GVSAuthorizationError, match="exact bytes or text"):
        load_gvs_authorization_receipt_json(bytearray(encoded.encode("utf-8")))


def test_exact_builtins_and_token_ids_are_required() -> None:
    receipt = _receipt()
    bad = copy.deepcopy(receipt)
    bad["bindings"]["eos_token_id"] = 2.0
    with pytest.raises(GVSAuthorizationError, match="nonnegative int"):
        seal_gvs_authorization_receipt(bad, signing_key=RECEIPT_KEY)
    same = copy.deepcopy(receipt)
    same["bindings"]["eos_token_id"] = 0
    with pytest.raises(GVSAuthorizationError, match="must be distinct"):
        seal_gvs_authorization_receipt(same, signing_key=RECEIPT_KEY)
    oversized = copy.deepcopy(receipt)
    oversized["bindings"]["eos_token_id"] = 2**31
    with pytest.raises(GVSAuthorizationError, match="bounded nonnegative int"):
        seal_gvs_authorization_receipt(oversized, signing_key=RECEIPT_KEY)
    with pytest.raises(GVSAuthorizationError, match="exact list"):
        validate_gvs_population_claim_ledger(
            (),
            signing_key=LEDGER_KEY,
            expected_key_id="gvs-ledger-key-v1",
        )


def test_expected_controls_require_exact_types_and_nonauthorizing_flags() -> None:
    receipt = _receipt()

    class StringSubclass(str):
        pass

    with pytest.raises(GVSAuthorizationError, match="expected_phase"):
        validate_gvs_authorization_receipt(
            receipt,
            signing_key=RECEIPT_KEY,
            expected_key_id="gvs-receipt-key-v1",
            expected_phase=StringSubclass("selection"),
            expected_run_id="20260804-1200-gvs-test-s17",
            expected_bindings=_bindings(),
            expected_gate_spec_sha256="b" * 64,
            expected_gate_ids=GATE_IDS,
        )
    failed = _receipt(failed_gate="primary_effect_passed")
    with pytest.raises(GVSAuthorizationError, match="exact bool"):
        validate_gvs_authorization_receipt(
            failed,
            signing_key=RECEIPT_KEY,
            expected_key_id="gvs-receipt-key-v1",
            expected_phase="selection",
            expected_run_id="20260804-1200-gvs-test-s17",
            expected_bindings=_bindings(),
            expected_gate_spec_sha256="b" * 64,
            expected_gate_ids=GATE_IDS,
            require_passed=0,
        )
    authorizing = copy.deepcopy(receipt)
    authorizing["authorizes_private_label_access"] = True
    with pytest.raises(GVSAuthorizationError, match="cannot authorize private-label"):
        seal_gvs_authorization_receipt(authorizing, signing_key=RECEIPT_KEY)

    noncanonical_time = copy.deepcopy(receipt)
    noncanonical_time["created_at_utc"] = "2026-08-04T06:30:00.0Z"
    with pytest.raises(GVSAuthorizationError, match="canonical UTC representation"):
        seal_gvs_authorization_receipt(noncanonical_time, signing_key=RECEIPT_KEY)
    with pytest.raises(GVSAuthorizationError, match="control"):
        claim_gvs_population_once(
            authorization_receipt=receipt,
            receipt_signing_key=RECEIPT_KEY,
            expected_receipt_key_id="gvs-receipt-key-v1",
            expected_phase="selection",
            expected_run_id="20260804-1200-gvs-test-s17",
            expected_bindings=_bindings(),
            expected_gate_spec_sha256="b" * 64,
            expected_gate_ids=GATE_IDS,
            prior_durable_ledger=[],
            ledger_signing_key=LEDGER_KEY,
            expected_ledger_key_id="gvs-ledger-key-v1",
            claim_id="control-purpose",
            scoring_session_id="scoring-session-1",
            accessor_id="isolated-scorer-1",
            purpose="line one\nline two",
            claimed_at_utc="2026-08-04T06:31:00Z",
            durable_compare_and_append=lambda event_bytes, head: "0" * 64,
        )


def test_runtime_and_import_monkeypatches_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt()
    forged = copy.deepcopy(receipt)
    forged["receipt_hmac_sha256"] = "0" * 64

    with monkeypatch.context() as patch:
        patch.setattr(authorization_module.hmac, "compare_digest", lambda left, right: True)
        with pytest.raises(GVSAuthorizationError, match="runtime import"):
            _validate(forged)

    with monkeypatch.context() as patch:
        patch.setattr(authorization_module.math, "isfinite", lambda value: True)
        with pytest.raises(GVSAuthorizationError, match="runtime import"):
            load_gvs_authorization_receipt_json('{"value":1e999}')

    with monkeypatch.context() as patch:
        patch.setattr(authorization_module, "GVS_PHASES", frozenset({"evil"}))
        with pytest.raises(GVSAuthorizationError, match="authorization contract|runtime differs"):
            seal_gvs_authorization_receipt(receipt, signing_key=RECEIPT_KEY)

    with monkeypatch.context() as patch:
        patch.setattr(
            authorization_module,
            "assert_gvs_authorization_runtime_integrity",
            lambda: None,
        )
        patch.setattr(authorization_module, "_HMAC_COMPARE_DIGEST", lambda left, right: True)
        with pytest.raises(GVSAuthorizationError, match="runtime import|runtime differs"):
            _validate(forged)

    class PermissivePattern:
        pattern = authorization_module._SHA256_RE.pattern

        @staticmethod
        def fullmatch(value):
            return object()

    with monkeypatch.context() as patch:
        patch.setattr(authorization_module, "_SHA256_RE", PermissivePattern())
        with pytest.raises(GVSAuthorizationError, match="authorization contract"):
            _validate(forged)

    with monkeypatch.context() as patch:
        patch.setattr(
            authorization_module,
            "validate_gvs_authorization_receipt",
            lambda *args, **kwargs: receipt,
        )
        with pytest.raises(GVSAuthorizationError, match="runtime differs"):
            _claim(receipt)
    assert_gvs_authorization_runtime_integrity()


def test_callback_runtime_mutation_is_detected_after_unknown_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()

    def mutate_runtime_after_possible_commit(event_bytes, expected_head):
        del expected_head
        monkeypatch.setattr(authorization_module, "GVS_PHASES", frozenset({"evil"}))
        return json.loads(event_bytes)["event_sha256"]

    with pytest.raises(GVSAuthorizationError, match="authorization contract|runtime differs"):
        _claim(receipt, append=mutate_runtime_after_possible_commit)
    monkeypatch.undo()

    def replace_public_guard(event_bytes, expected_head):
        del expected_head
        monkeypatch.setattr(
            authorization_module,
            "assert_gvs_authorization_runtime_integrity",
            lambda: None,
        )
        return json.loads(event_bytes)["event_sha256"]

    with pytest.raises(GVSAuthorizationError, match="callback boundary"):
        _claim(receipt, append=replace_public_guard)
    monkeypatch.undo()
    assert_gvs_authorization_runtime_integrity()
