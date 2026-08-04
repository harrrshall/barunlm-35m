from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import multiprocessing
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from barunlm.evaluation.gvs_authorization import (
    GVS_AUTHORIZATION_BINDING_FIELDS,
    GVS_AUTHORIZATION_RECEIPT_VERSION,
    claim_gvs_population_once,
    seal_gvs_authorization_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / "infra" / "gvs_retirement" / "sqlite_retirement.py"
SERVICE_SPEC = importlib.util.spec_from_file_location("barun_gvs_retirement", SERVICE_PATH)
assert SERVICE_SPEC is not None and SERVICE_SPEC.loader is not None
SERVICE_MODULE = importlib.util.module_from_spec(SERVICE_SPEC)
sys.modules[SERVICE_SPEC.name] = SERVICE_MODULE
SERVICE_SPEC.loader.exec_module(SERVICE_MODULE)

CLAIM_EVENT_VERSION = SERVICE_MODULE.CLAIM_EVENT_VERSION
ZERO_SHA256 = SERVICE_MODULE.ZERO_SHA256
RetirementServiceError = SERVICE_MODULE.RetirementServiceError
SQLiteRetirementLedger = SERVICE_MODULE.SQLiteRetirementLedger

EVENT_HASH_DOMAIN = b"barun-gvs-population-claim-event-v3/hash"
EVENT_HMAC_DOMAIN = b"barun-gvs-population-claim-event-v3/hmac"


class TestHMACAuthenticator:
    """Ephemeral test double; production keys belong in external custody."""

    __test__ = False

    def __init__(self, key: bytes, *, key_id: str = "gvs-ledger-key-v1") -> None:
        self._key = key
        self._key_id = key_id

    def verify_event_hmac(
        self,
        *,
        key_id: str,
        authenticated_payload: bytes,
        presented_hmac_sha256: str,
    ) -> bool:
        if key_id != self._key_id:
            return False
        expected = hmac.new(self._key, authenticated_payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, presented_hmac_sha256)


class RaisingAuthenticator:
    def verify_event_hmac(
        self,
        *,
        key_id: str,
        authenticated_payload: bytes,
        presented_hmac_sha256: str,
    ) -> bool:
        del key_id, authenticated_payload, presented_hmac_sha256
        raise OSError("provider detail must not cross the storage boundary")


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def resign_event(event: dict[str, Any], key: bytes) -> bytes:
    event = json.loads(canonical(event))
    event["event_sha256"] = ZERO_SHA256
    event["event_hmac_sha256"] = ZERO_SHA256
    event["event_sha256"] = hashlib.sha256(
        EVENT_HASH_DOMAIN + b"\x00" + canonical(event)
    ).hexdigest()
    event["event_hmac_sha256"] = hmac.new(
        key,
        EVENT_HMAC_DOMAIN + b"\x00" + canonical(event),
        hashlib.sha256,
    ).hexdigest()
    return canonical(event)


def event_bytes(
    key: bytes,
    *,
    sequence: int = 0,
    previous_head: str = ZERO_SHA256,
    claim_id: str = "claim-0",
    phase: str = "selection",
    run_id: str = "run-0",
    population: str = "1" * 64,
    prompts: str = "2" * 64,
    labels: str = "3" * 64,
    claimed_at: str = "2026-08-05T00:00:01Z",
    key_id: str = "gvs-ledger-key-v1",
) -> bytes:
    event: dict[str, Any] = {
        "schema_version": CLAIM_EVENT_VERSION,
        "sequence": sequence,
        "claim_id": claim_id,
        "phase": phase,
        "run_id": run_id,
        "population_manifest_sha256": population,
        "prompt_collection_sha256": prompts,
        "label_commitment_root_sha256": labels,
        "authorization_receipt_sha256": "4" * 64,
        "scoring_session_id": f"session-{claim_id}",
        "accessor_id": "external-scorer",
        "purpose": "one-shot whole-population scoring",
        "claimed_at_utc": claimed_at,
        "previous_event_sha256": previous_head,
        "key_id": key_id,
        "partial_retry_authorized": False,
        "authorizes_model_cuda_training_or_jarvis": False,
        "authorizes_private_label_access": False,
        "proves_external_atomicity": False,
        "authorization_runtime_sha256": "5" * 64,
        "event_sha256": ZERO_SHA256,
        "event_hmac_sha256": ZERO_SHA256,
    }
    return resign_event(event, key)


def private_state_path(tmp_path: Path, name: str = "retirement.sqlite3") -> Path:
    tmp_path.chmod(0o700)
    return tmp_path / name


def service(tmp_path: Path, key: bytes) -> SQLiteRetirementLedger:
    return SQLiteRetirementLedger(
        database_path=private_state_path(tmp_path),
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )


def _concurrent_append_worker(
    database_path: str,
    key: bytes,
    payload: bytes,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    ledger = SQLiteRetirementLedger(
        database_path=Path(database_path),
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    start.wait(timeout=10)
    try:
        head = ledger.compare_and_append(payload, ZERO_SHA256)
    except RetirementServiceError:
        results.put(("rejected", None))
    else:
        results.put(("committed", head))


def _append_then_exit_worker(database_path: str, key: bytes, payload: bytes) -> None:
    ledger = SQLiteRetirementLedger(
        database_path=Path(database_path),
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    ledger.compare_and_append(payload, ZERO_SHA256)
    os._exit(0)


def test_callback_commits_exact_whole_population_event_and_survives_reopen(
    tmp_path: Path,
) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    payload = event_bytes(key)
    expected_event = json.loads(payload)

    assert ledger.inspect().head_sha256 == ZERO_SHA256
    assert ledger.compare_and_append(payload, ZERO_SHA256) == expected_event["event_sha256"]
    assert ledger.read_ledger() == [expected_event]
    assert ledger.inspect().next_sequence == 1
    assert ledger.inspect().head_sha256 == expected_event["event_sha256"]
    assert ledger.inspect().ledger_key_id == "gvs-ledger-key-v1"
    assert ledger.database_path.stat().st_mode & 0o077 == 0

    reopened = SQLiteRetirementLedger(
        database_path=ledger.database_path,
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    assert reopened.read_ledger() == [expected_event]


def test_existing_authorization_claim_uses_service_as_its_exact_callback(tmp_path: Path) -> None:
    receipt_key = secrets.token_bytes(32)
    ledger_key = secrets.token_bytes(32)
    bindings: dict[str, Any] = {field: "a" * 64 for field in GVS_AUTHORIZATION_BINDING_FIELDS}
    bindings["eos_token_id"] = 2
    bindings["pad_token_id"] = 0
    gates = {
        "all_denominators_complete": True,
        "no_catastrophic_actions": True,
        "primary_effect_passed": True,
    }
    receipt = seal_gvs_authorization_receipt(
        {
            "schema_version": GVS_AUTHORIZATION_RECEIPT_VERSION,
            "phase": "selection",
            "run_id": "20260805-0100-gvs-retirement-test-s17",
            "created_at_utc": "2026-08-05T00:30:00Z",
            "key_id": "gvs-receipt-key-v1",
            "bindings": bindings,
            "gate_spec_sha256": "b" * 64,
            "gate_results": gates,
            "passed": True,
            "authorizes_model_cuda_training_or_jarvis": False,
            "authorizes_private_label_access": False,
            "authorization_runtime_sha256": ZERO_SHA256,
            "receipt_sha256": ZERO_SHA256,
            "receipt_hmac_sha256": ZERO_SHA256,
        },
        signing_key=receipt_key,
    )
    ledger = service(tmp_path, ledger_key)

    returned = claim_gvs_population_once(
        authorization_receipt=receipt,
        receipt_signing_key=receipt_key,
        expected_receipt_key_id="gvs-receipt-key-v1",
        expected_phase="selection",
        expected_run_id="20260805-0100-gvs-retirement-test-s17",
        expected_bindings=bindings,
        expected_gate_spec_sha256="b" * 64,
        expected_gate_ids=tuple(sorted(gates)),
        prior_durable_ledger=[],
        ledger_signing_key=ledger_key,
        expected_ledger_key_id="gvs-ledger-key-v1",
        claim_id="claim-real-callback",
        scoring_session_id="scoring-session-real-callback",
        accessor_id="isolated-scorer-real-callback",
        purpose="one-shot whole-population scoring",
        claimed_at_utc="2026-08-05T00:30:01Z",
        durable_compare_and_append=ledger.compare_and_append,
    )

    assert ledger.read_ledger() == [returned]
    assert returned["authorizes_private_label_access"] is False
    assert returned["proves_external_atomicity"] is False


def test_runtime_state_is_required_to_be_private_absolute_and_outside_repository(
    tmp_path: Path,
) -> None:
    key = secrets.token_bytes(32)
    with pytest.raises(RetirementServiceError, match="absolute"):
        SQLiteRetirementLedger(
            database_path=Path("relative.sqlite3"),
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )

    repository_state = ROOT / ".forbidden-retirement-test.sqlite3"
    with pytest.raises(RetirementServiceError, match="outside"):
        SQLiteRetirementLedger(
            database_path=repository_state,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )
    assert not repository_state.exists()

    tmp_path.chmod(0o755)
    with pytest.raises(RetirementServiceError, match="group/world"):
        SQLiteRetirementLedger(
            database_path=tmp_path / "overexposed.sqlite3",
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )


def test_caller_cannot_hide_the_actual_source_repository_with_a_false_root(
    tmp_path: Path,
) -> None:
    key = secrets.token_bytes(32)
    declared_root = tmp_path / "declared-repository"
    declared_root.mkdir(mode=0o700)
    actual_repository_state = Path(tempfile.mkdtemp(prefix=".gvs-retirement-audit-", dir=ROOT))
    actual_repository_state.chmod(0o700)
    database_path = actual_repository_state / "retirement.sqlite3"
    try:
        with pytest.raises(RetirementServiceError, match="actual source repository"):
            SQLiteRetirementLedger(
                database_path=database_path,
                authenticator=TestHMACAuthenticator(key),
                repository_root=declared_root,
            )
        assert not database_path.exists()
    finally:
        shutil.rmtree(actual_repository_state)


def test_existing_database_hard_link_is_rejected_before_sqlite_access(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    database_path = private_state_path(tmp_path)
    target = tmp_path / "unrelated-private-file"
    target.touch(mode=0o600)
    try:
        os.link(target, database_path)
    except OSError as error:
        pytest.skip(f"hard links unavailable on this filesystem: {error}")

    with pytest.raises(RetirementServiceError, match="exactly one hard link"):
        SQLiteRetirementLedger(
            database_path=database_path,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )
    assert target.stat().st_size == 0


def test_database_and_parent_symlinks_are_rejected(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    real_parent = tmp_path / "real-private-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-private-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RetirementServiceError, match="parent must not traverse a symlink"):
        SQLiteRetirementLedger(
            database_path=linked_parent / "retirement.sqlite3",
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )

    target = tmp_path / "target.sqlite3"
    target.touch(mode=0o600)
    linked_database = tmp_path / "linked.sqlite3"
    linked_database.symlink_to(target)
    with pytest.raises(RetirementServiceError, match="must not be a symlink"):
        SQLiteRetirementLedger(
            database_path=linked_database,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )


def test_parent_and_database_owner_and_exact_modes_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = secrets.token_bytes(32)
    tmp_path.chmod(0o600)
    with pytest.raises(RetirementServiceError, match="mode 0700"):
        SQLiteRetirementLedger(
            database_path=tmp_path / "wrong-parent-mode.sqlite3",
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )

    tmp_path.chmod(0o700)
    database_path = tmp_path / "wrong-file-mode.sqlite3"
    database_path.touch(mode=0o700)
    with pytest.raises(RetirementServiceError, match="mode 0600"):
        SQLiteRetirementLedger(
            database_path=database_path,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )

    database_stat = database_path.stat()
    stat_fields = list(database_stat)
    stat_fields[4] = database_stat.st_uid + 1
    with pytest.raises(RetirementServiceError, match="owned by the effective user"):
        SERVICE_MODULE._validate_database_stat(os.stat_result(stat_fields))

    database_path.unlink()
    effective_uid = os.geteuid()
    monkeypatch.setattr(SERVICE_MODULE.os, "geteuid", lambda: effective_uid + 1)
    with pytest.raises(RetirementServiceError, match="owned by the effective user"):
        SQLiteRetirementLedger(
            database_path=database_path,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )


def test_connect_time_database_path_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = secrets.token_bytes(32)
    database_path = private_state_path(tmp_path)
    displaced_path = tmp_path / "displaced.sqlite3"
    real_connect = SERVICE_MODULE.sqlite3.connect
    swapped = False

    def swapping_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(database_path, displaced_path)
            database_path.touch(mode=0o600)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(SERVICE_MODULE.sqlite3, "connect", swapping_connect)
    with pytest.raises(RetirementServiceError, match="changed during SQLite open"):
        SQLiteRetirementLedger(
            database_path=database_path,
            authenticator=TestHMACAuthenticator(key),
            repository_root=ROOT,
        )


def test_service_rejects_ordinary_runtime_authenticator_rebinding(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    with pytest.raises(AttributeError, match="sealed"):
        ledger._authenticator = TestHMACAuthenticator(secrets.token_bytes(32))


def test_service_uses_full_synchronous_rollback_journal_and_private_file(tmp_path: Path) -> None:
    ledger = service(tmp_path, secrets.token_bytes(32))
    connection = sqlite3.connect(ledger.database_path)
    try:
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("PRAGMA application_id").fetchone() == (0x42525631,)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "reused_field",
    [
        "population_manifest_sha256",
        "prompt_collection_sha256",
        "label_commitment_root_sha256",
    ],
)
def test_population_prompt_and_label_commitments_are_each_globally_one_shot(
    tmp_path: Path, reused_field: str
) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    first = event_bytes(key)
    first_event = json.loads(first)
    first_head = ledger.compare_and_append(first, ZERO_SHA256)
    second_event = json.loads(
        event_bytes(
            key,
            sequence=1,
            previous_head=first_head,
            claim_id="claim-1",
            phase="selection",
            run_id="run-1",
            population="6" * 64,
            prompts="7" * 64,
            labels="8" * 64,
            claimed_at="2026-08-05T00:00:02Z",
        )
    )
    second_event[reused_field] = first_event[reused_field]
    second = resign_event(second_event, key)

    with pytest.raises(RetirementServiceError, match="already retired"):
        ledger.compare_and_append(second, first_head)
    assert ledger.inspect().head_sha256 == first_head
    assert ledger.inspect().next_sequence == 1
    assert ledger.read_ledger() == [first_event]


def test_replay_and_compare_head_mismatch_fail_without_partial_state(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    payload = event_bytes(key)
    head = ledger.compare_and_append(payload, ZERO_SHA256)

    with pytest.raises(RetirementServiceError, match="head mismatch"):
        ledger.compare_and_append(payload, ZERO_SHA256)
    with pytest.raises(RetirementServiceError, match="disagree"):
        ledger.compare_and_append(payload, head)
    assert ledger.inspect().head_sha256 == head
    assert ledger.inspect().next_sequence == 1


def test_exact_canonical_bounded_json_and_false_flags_are_enforced(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    payload = event_bytes(key)

    with pytest.raises(RetirementServiceError, match="immutable bytes"):
        ledger.compare_and_append(bytearray(payload), ZERO_SHA256)  # type: ignore[arg-type]
    with pytest.raises(RetirementServiceError, match="canonical JSON"):
        ledger.compare_and_append(json.dumps(json.loads(payload)).encode(), ZERO_SHA256)
    duplicate = payload[:-1] + b',"claim_id":"duplicate"}'
    with pytest.raises(RetirementServiceError, match="duplicate"):
        ledger.compare_and_append(duplicate, ZERO_SHA256)
    with pytest.raises(RetirementServiceError, match="byte bound"):
        ledger.compare_and_append(b"{" + b" " * (2 * 1024 * 1024), ZERO_SHA256)

    unsafe = json.loads(payload)
    unsafe["authorizes_private_label_access"] = True
    with pytest.raises(RetirementServiceError, match="must remain false"):
        ledger.compare_and_append(resign_event(unsafe, key), ZERO_SHA256)
    assert ledger.inspect().next_sequence == 0


def test_wrong_or_raising_external_authenticator_fails_without_secret_detail(
    tmp_path: Path,
) -> None:
    key = secrets.token_bytes(32)
    payload = event_bytes(key)
    wrong = service(tmp_path, secrets.token_bytes(32))
    with pytest.raises(RetirementServiceError, match="failed closed"):
        wrong.compare_and_append(payload, ZERO_SHA256)
    assert wrong.inspect().next_sequence == 0

    other_dir = tmp_path / "raising"
    other_dir.mkdir(mode=0o700)
    raising = SQLiteRetirementLedger(
        database_path=other_dir / "retirement.sqlite3",
        authenticator=RaisingAuthenticator(),
        repository_root=ROOT,
    )
    with pytest.raises(RetirementServiceError, match="failed closed") as captured:
        raising.compare_and_append(payload, ZERO_SHA256)
    assert "provider detail" not in str(captured.value)
    assert raising.inspect().next_sequence == 0


def test_key_phase_and_time_chronology_are_independently_enforced(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    selection = event_bytes(key)
    selection_head = ledger.compare_and_append(selection, ZERO_SHA256)

    wrong_key = event_bytes(
        key,
        sequence=1,
        previous_head=selection_head,
        claim_id="claim-wrong-key",
        phase="selection",
        run_id="other-run",
        population="6" * 64,
        prompts="7" * 64,
        labels="8" * 64,
        claimed_at="2026-08-05T00:00:02Z",
        key_id="rotated-key",
    )
    with pytest.raises(RetirementServiceError, match="authentication"):
        ledger.compare_and_append(wrong_key, selection_head)

    confirmation = event_bytes(
        key,
        sequence=1,
        previous_head=selection_head,
        claim_id="claim-confirmation",
        phase="confirmation",
        run_id="run-0",
        population="6" * 64,
        prompts="7" * 64,
        labels="8" * 64,
        claimed_at="2026-08-05T00:00:02Z",
    )
    assert (
        ledger.compare_and_append(confirmation, selection_head)
        == json.loads(confirmation)["event_sha256"]
    )

    stale_time = event_bytes(
        key,
        sequence=2,
        previous_head=json.loads(confirmation)["event_sha256"],
        claim_id="claim-stale",
        phase="selection",
        run_id="run-2",
        population="9" * 64,
        prompts="a" * 64,
        labels="b" * 64,
        claimed_at="2026-08-05T00:00:01Z",
    )
    with pytest.raises(RetirementServiceError, match="strictly increase"):
        ledger.compare_and_append(stale_time, json.loads(confirmation)["event_sha256"])


def test_two_processes_racing_the_same_head_commit_exactly_one(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    database_path = private_state_path(tmp_path)
    SQLiteRetirementLedger(
        database_path=database_path,
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    first = event_bytes(key, claim_id="race-a", population="1" * 64)
    second = event_bytes(
        key,
        claim_id="race-b",
        run_id="run-b",
        population="6" * 64,
        prompts="7" * 64,
        labels="8" * 64,
        claimed_at="2026-08-05T00:00:02Z",
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_append_worker,
            args=(str(database_path), key, payload, start, results),
        )
        for payload in (first, second)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert sorted(outcome[0] for outcome in outcomes) == ["committed", "rejected"]
    ledger = SQLiteRetirementLedger(
        database_path=database_path,
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    assert ledger.inspect().next_sequence == 1
    assert len(ledger.read_ledger()) == 1


def test_commit_remains_after_writer_exits_without_graceful_cleanup(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    database_path = private_state_path(tmp_path)
    payload = event_bytes(key)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_append_then_exit_worker,
        args=(str(database_path), key, payload),
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 0

    ledger = SQLiteRetirementLedger(
        database_path=database_path,
        authenticator=TestHMACAuthenticator(key),
        repository_root=ROOT,
    )
    assert ledger.read_ledger() == [json.loads(payload)]


def test_read_ledger_detects_external_event_corruption(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    ledger = service(tmp_path, key)
    payload = event_bytes(key)
    ledger.compare_and_append(payload, ZERO_SHA256)

    connection = sqlite3.connect(ledger.database_path)
    try:
        corrupted = payload.replace(b"whole-population", b"whole_population")
        connection.execute("UPDATE claim_events SET canonical_event = ?", (corrupted,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RetirementServiceError):
        ledger.read_ledger()
